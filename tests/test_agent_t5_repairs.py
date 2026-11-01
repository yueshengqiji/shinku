"""固定真实对照暴露的协议边界；合成输入、真实文件工具和隔离 SDK，不访问供应商。"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import threading
import difflib

import pytest

from shinku.agent.verification import SmartCompletionVerifier
from shinku.agent.task_state import validate_contract
from shinku.agent_runtime import AgentTurnRequest
from shinku.agent_runtime.demo import demo_request, demo_service
from shinku.angelos.host import ServiceHost
from shinku.hosts.tool_host import ToolHost
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry
from shinku.tools.workspace import build_workspace_tools
from test_agent_conversation_loop import native_sdk
from test_agent_runtime_boundary import settled


EDIT_REQUEST = ('保留已有注释和本地修改，把 backend.py 的 State.timeout 和 client.py 的 TIMEOUT 从10改成30。'
                '展示实际差异，并读取实际文件核对。其他内容保持原样，不运行命令。')


def edit_contract():
    return {'goal': '修改两个超时值并核对', 'deliverables': ['两个文件的真实差异及读取结果'],
        'scope': ['保留其他字节，不执行命令'], 'requirements': [
            {'id': name, 'description': '超时值为30', 'effect': 'write', 'targets': [name],
             'checks': {'contains': 'timeout = 30' if name == 'backend.py' else 'TIMEOUT = 30'}}
            for name in ('backend.py', 'client.py')]}


@contextmanager
def workspace_service(root, *, blocked=(), mode='full'):
    service, _, runtime = demo_service(root)
    tools = {h.tool_type: h for h in build_workspace_tools(root,
        authorize=lambda ctx, write: ctx.get('agent_permission_mode') == 'full' or not write)}
    service.executor.tool_host = ToolHost(ToolRegistry(tools))
    original = service.authorize
    service.authorize = lambda *args: replace(original(*args), allowed_tools=frozenset(tools),
        permission_mode=mode, policy=ExecutionPolicy(blocked=frozenset(blocked),
            risk_by_tool={n: 'low' for n in tools}),
        budget_limits={'max_requests': 16, 'max_tokens': 600000, 'max_tools': 32, 'max_active_seconds': 30})
    # 兼容宿主继续验收旧契约；新默认会话的直接交付另有整链路测试。
    host = ServiceHost(service, requester='fixture', reply_delivery=False)
    try:
        yield service, tools, runtime, host.port('repair')
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize('text,expected', [
    (EDIT_REQUEST, ['backend.py', 'client.py']),
    ('把 `backend.py` 的 `State.timeout` 改为30', ['backend.py']),
    ('修改文件（backend.py）中的 State.timeout', ['backend.py']),
    ('修改 "backend.py" 内的 State.timeout', ['backend.py']),
    ("Change backend.py's State.timeout", ['backend.py']),
    ('修改文件 State.timeout', ['State.timeout']),
    ('修改 `State.timeout`', ['State.timeout']),
    ('修改文件 "State`local.timeout"', ['State`local.timeout']),
    ('修改 archive.zip 中的 config.json', ['archive.zip', 'config.json']),
    ('修改 archive.zip 中的 "State.timeout"', ['archive.zip', 'State.timeout']),
    ('修改 backend.py 的 文件 State.timeout', ['backend.py', 'State.timeout']),
    ('修改 ./State.timeout', ['State.timeout']),
    ('读取 backend.py，然后修改文件 State.timeout', ['backend.py', 'State.timeout']),
    ('修改 backend.py 和 State.timeout 两个文件', ['backend.py', 'State.timeout']),
    ('修改 backend.py 的 State.timeout，再读取 other.py', ['backend.py', 'other.py']),
])
def test_member_binding_does_not_erase_real_file(text, expected):
    assert SmartCompletionVerifier._extract_targets(text) == expected


def test_valid_two_file_contract_and_literal_third_file_still_checked():
    verifier = SmartCompletionVerifier()
    contract = validate_contract(edit_contract())
    assert verifier.target_coverage_gaps(contract, [EDIT_REQUEST]) == []
    assert verifier.target_coverage_gaps(contract, [EDIT_REQUEST + '还要修改文件 State.timeout。']) == [
        {'target': 'State.timeout', 'effects': ['write']}]


@pytest.mark.parametrize('declaration', ['tool', 'final'])
def test_contract_diagnostic_identifies_actual_missing_target(tmp_path, declaration):
    with workspace_service(tmp_path) as (service, tools, runtime, port):
        calls = []
        def respond(**kw):
            calls.append(kw)
            if len(calls) == 1:
                contract = edit_contract()
                contract['requirements'] = contract['requirements'][:1]
                if declaration == 'tool':
                    return {'tool_call': {'type': 'declare_task_contract', 'task_contract': contract}}
                return {'kind': 'final', 'text': '尚未执行。', 'task_contract': contract}
            assert 'client.py' in kw['user_prompt'] and 'uncovered_targets' in kw['user_prompt']
            assert 'requirements' in kw['user_prompt']
            return {'kind': 'wait', 'text': '诊断已读取，本测试停止。'}
        runtime.call_chat_json = respond
        handle = port.submit(AgentTurnRequest('diagnostic', EDIT_REQUEST))
        settled(port, handle.task_id)
        assert len(calls) == 2
        assert port.result(handle.task_id)['contract'] is None


def test_real_sdk_two_file_preview_apply_readback(tmp_path):
    originals = {'backend.py': '# 本地已有注释必须保留\nclass State:\n    timeout = 10\n',
        'client.py': '# 用户已有修改必须保留\nLOCAL_EDIT = "preserve-me"\nTIMEOUT = 10\n'}
    for name, text in originals.items():
        (tmp_path / name).write_text(text)
    with workspace_service(tmp_path) as (service, tools, _, port):
        def respond(payload, index):
            if index == 1:
                schema = next(t['function']['parameters'] for t in payload['tools']
                    if t['function']['name'] == 'declare_task_contract')['properties']['task_contract']
                assert schema['properties']['requirements']['items']['properties']['effect']['type'] == 'string'
                name, args = 'declare_task_contract', {'task_contract': edit_contract()}
            elif index <= 9:
                path = 'backend.py' if index <= 5 else 'client.py'
                stage = (index - 2) % 4
                if stage in (0, 3):
                    name, args = 'workspace_read_lines', {'path': path}
                    if stage == 3:
                        assert (tmp_path / path).read_text() == originals[path].replace('= 10', '= 30')
                elif stage == 1:
                    name, args = 'workspace_preview_patch', {'path': path,
                        'expected_sha256': hashlib.sha256((tmp_path / path).read_bytes()).hexdigest(),
                        'edits': [{'old': '= 10', 'new': '= 30'}]}
                else:
                    ident, plan = next((i, p) for i, p in tools['workspace_apply_patch'].access.previews.items() if p[1] == path)
                    name, args = 'workspace_apply_patch', {'path': path, 'expected_sha256': plan[2], 'preview_id': ident}
            else:
                diff = '\n'.join(''.join(difflib.unified_diff(originals[p].splitlines(keepends=True),
                    (tmp_path / p).read_text().splitlines(keepends=True), fromfile=p, tofile=p)) for p in originals)
                return {'role': 'assistant', 'content': json.dumps({'kind': 'final',
                    'text': 'backend.py 与 client.py 已修改并读取核对，注释和本地修改保留。\n' + diff,
                    'task_outcome': 'completed'})}, 'stop'
            return {'role': 'assistant', 'tool_calls': [{'id': f'call_{index}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'tool_calls'
        with native_sdk(respond) as (runtime, payloads):
            original_env = runtime._env
            runtime._env = lambda key, default='': ('offline.example.invalid:deepseek-flash'
                if key == 'NATIVE_TOOL_PROVIDER_ALLOWLIST' else original_env(key, default))
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            handle = port.submit(AgentTurnRequest('edit', EDIT_REQUEST))
            settled(port, handle.task_id)
            result = port.result(handle.task_id)
            assert result['execution']['status'] == 'completed', result
            assert result['verification']['accepted']
            assert '-    timeout = 10' in result['execution_report']['text']
            assert '+TIMEOUT = 30' in result['execution_report']['text']
            assert len(payloads) == 10
            for name, text in originals.items():
                assert (tmp_path / name).read_text() == text.replace('= 10', '= 30')


def test_capability_catalog_distinguishes_denied_and_failed_calls(tmp_path):
    with workspace_service(tmp_path, blocked=['workspace_apply_patch'], mode='read') as (service, tools, runtime, port):
        calls = []
        def respond(**kw):
            calls.append(kw)
            assert 'Current task tool availability' in kw['user_prompt']
            if len(calls) == 1:
                assert 'workspace_search' in kw['user_prompt']
                return {'tool_call': {'type': 'workspace_search', 'query': ''}}
            if len(calls) == 2:
                return {'tool_call': {'type': 'read_self_state', 'scope': 'tools'}}
            task = next(iter(service._sessions.values())).context['agent_task_state']
            snap = task['tool_calls'][-1]['data']['state_updates']['self_state']['tools']
            assert 'workspace_search' in snap['available_names']
            assert 'workspace_apply_patch' in snap['registered_names']
            assert 'workspace_apply_patch' in snap['not_allowed_names']
            assert snap['permission_mode'] == 'read' and snap['business_health'] == 'not_verified'
            assert any(c['name'] == 'workspace_search' and c['status'] != 'ok' for c in snap['recent_calls'])
            return {'kind': 'final', 'text': '已注册搜索；本次搜索参数错误，不能据此判定缺少搜索。写入未获准。',
                'task_outcome': 'completed',
                'task_contract': {'goal': '按宿主事实回答', 'deliverables': ['能力说明'],
                    'requirements': [{'id': 'state', 'description': '当前目录', 'effect': 'tool:read_self_state'}]}}
        runtime.call_chat_json = respond
        h = port.submit(AgentTurnRequest('catalog', '你有什么工具，还需要补什么？'))
        assert settled(port, h.task_id)['status'] == 'completed', json.dumps(port.result(h.task_id), ensure_ascii=False)
        assert len(calls) == 3
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('content', [
    '{"kind":"tool","text":"a\nb"}',
    '{"kind":"final","text":"a\nb","tool_call":null}',
    '{"kind":"final","text":"a\nb","task_contract":{}}',
    '{"kind":"final","text":"a\nb","kind":"tool"}',
    '{"kind":"final","text":"a\nb"',
    '{"kind":"final","text":"a\u0000b"}',
    '{"kind":"final","text":"a\nb","task_outcome":[]}',
    '   ', '正文', '{"kind":"final","text":"合法正文"}',
])
def test_final_format_recovery_never_repairs_calls_or_incomplete_objects(content):
    from shinku.llm.final_json import recover_final_text_json
    assert recover_final_text_json(content) is None


def test_final_format_recovery_requires_explicit_host_scope():
    from shinku.llm.response_protocol import observe_response, recover_final_text
    content = '{"kind":"final","text":"第一行\n第二行\t后续","task_outcome":"completed"}'
    assert recover_final_text(content) is None
    with observe_response():
        assert recover_final_text(content) is None
    with observe_response(allow_final_text_json=True) as observation:
        assert recover_final_text(content)['text'] == '第一行\n第二行\t后续'
        assert observation.events == [{'stage': 'final_text_json_recovered'}]


@pytest.mark.parametrize('extra', ['', ',"task_contract":{"goal":"旧目标不得覆盖当前契约"}'])
def test_sdk_readonly_final_newlines_deliver_without_extra_call_and_keep_original(tmp_path, extra):
    (tmp_path / 'notes.txt').write_text('约束一：保留已有修改。\n约束二：不得重复发送。\n')
    raw = '{"kind":"final","text":"约束一：保留已有修改。\n约束二：不得重复发送。","task_outcome":"completed"}'
    raw = raw[:-1] + extra + '}'
    contract = {'goal': '读取并说明约束', 'deliverables': ['约束说明'], 'requirements': [
        {'id': 'source', 'description': '完整读取', 'effect': 'read', 'targets': ['notes.txt']}]}
    with workspace_service(tmp_path, mode='read') as (service, _, _, port):
        def respond(payload, index):
            if index == 1:
                name, args = 'declare_task_contract', {'task_contract': contract}
            elif index == 2:
                name, args = 'workspace_read_lines', {'path': 'notes.txt'}
            else:
                return {'role': 'assistant', 'content': raw}, 'stop'
            return {'role': 'assistant', 'tool_calls': [{'id': f'call_{index}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'tool_calls'
        with native_sdk(respond) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            h = port.submit(AgentTurnRequest('final-format', '读取 notes.txt 并说明约束'))
            assert settled(port, h.task_id)['status'] == 'completed', port.result(h.task_id)
            result = port.result(h.task_id)
            assert result['execution_report']['text'] == json.loads(raw, strict=False)['text']
            assert len(calls) == 3 and result['budget']['requests'] == 3
            state = service._sessions[h.task_id].checkpoint.state
            assert state.native_tool_history.messages[-1]['content'] == raw
            assert state.progress['answer_format_recovery']['original_provider_message_retained']
            assert state.context['agent_task_state']['contract']['goal'] == contract['goal']
            assert state.progress['answer_drafts'][-1]['text'] == result['execution_report']['text']


@pytest.mark.parametrize('outcome', ['partial', 'completed'])
def test_sdk_final_newlines_after_verified_change_keep_outcome_and_no_extra_call(tmp_path, outcome):
    """实际写入与独立回执使只读条件为假；格式恢复不能重复写入或绕过完成门。"""
    raw = '{"kind":"final","text":"合成文件已经创建。\n后续确认仍有范围限制。","task_outcome":"' + outcome + '"}'
    service, tool, _ = demo_service(tmp_path)
    host = ServiceHost(service, requester='fixture')
    port = host.port('format')
    contract = demo_request(tool).contract
    try:
        def respond(payload, index):
            if index > 2:
                return {'role': 'assistant', 'content': raw}, 'stop'
            name, args = ('declare_task_contract', {'task_contract': contract}) if index == 1 else ('demo_file', {'path': str(tool.path)})
            return {'role': 'assistant', 'tool_calls': [{'id': f'call_{index}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'tool_calls'
        with native_sdk(respond) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            handle = port.submit(AgentTurnRequest('stateful-final-format', '创建合成文件并核对内容：' + str(tool.path)))
            settled(port, handle.task_id)
            result = port.result(handle.task_id)
            state = service._sessions[handle.task_id].checkpoint.state
            assert len(state.tool_calls) == 2 and all(c['status'] == 'ok' for c in state.tool_calls), json.dumps(
                [m for m in state.native_tool_history.messages if m.get('role') == 'tool'], ensure_ascii=False)
            assert result['execution']['reason'] != 'conversation_output_invalid', result
            assert result['execution_report']['text'] == json.loads(raw, strict=False)['text']
            assert result['execution']['status'] == outcome
            assert len(calls) == 3
            assert state.native_tool_history.messages[-1]['content'] == raw
            assert len(state.tool_calls) == 2 and tool.calls == 1
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize('extra', [',"tool_call":{"type":"demo_file"}', ',"task_contract":{}', ',"kind":"tool"'])
def test_stateful_final_recovery_rejects_execution_metadata_and_keeps_original_write_once(tmp_path, extra):
    raw = '{"kind":"final","text":"第一行\n第二行","task_outcome":"completed"' + extra + '}'
    service, tool, _ = demo_service(tmp_path)
    port = ServiceHost(service, requester='fixture').port('bad-format')
    try:
        def respond(payload, index):
            if index > 2:
                return {'role': 'assistant', 'content': raw}, 'stop'
            name, args = ('declare_task_contract', {'task_contract': demo_request(tool).contract}) if index == 1 else ('demo_file', {'path': str(tool.path)})
            return {'role': 'assistant', 'tool_calls': [{'id': f'call_{index}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'tool_calls'
        with native_sdk(respond) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            handle = port.submit(AgentTurnRequest('reject-metadata', '创建合成文件并核对内容：' + str(tool.path)))
            settled(port, handle.task_id)
            result = port.result(handle.task_id)
            assert result['execution']['status'] == 'failed'
            assert len(calls) == 3 and tool.calls == 1
            state = service._sessions[handle.task_id].checkpoint.state
            assert 'answer_format_recovery' not in state.progress
            assert state.native_tool_history.messages[-1]['content'] == raw
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_recovered_stateful_report_still_rejects_unperformed_requirement(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    port = ServiceHost(service, requester='fixture', reply_delivery=False).port('missing-effect')
    missing = tmp_path / 'missing.txt'
    contract = demo_request(tool).contract
    contract['requirements'].append({'id': 'missing', 'description': '第二个产物', 'effect': 'write', 'targets': [str(missing)]})
    raw = '{"kind":"final","text":"两个产物已完成。\n可以交付。","task_outcome":"completed"}'
    try:
        def respond(payload, index):
            if index > 3:
                return {'role': 'assistant', 'content': json.dumps({'kind':'final','text':'第二个产物未生成。','task_outcome':'failed'})}, 'stop'
            if index == 3:
                return {'role': 'assistant', 'content': raw}, 'stop'
            name, args = ('declare_task_contract', {'task_contract': contract}) if index == 1 else ('demo_file', {'path': str(tool.path)})
            return {'role': 'assistant', 'tool_calls': [{'id': f'call_{index}', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'tool_calls'
        with native_sdk(respond) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            handle = port.submit(AgentTurnRequest('missing-effect', '创建并核对两个合成文件：' + str(tool.path) + '、' + str(missing)))
            settled(port, handle.task_id)
            result = port.result(handle.task_id)
            assert result['execution']['status'] == 'failed'
            assert '第二个产物未生成' in result['execution_report']['text']
            # accepted 认可的是这份如实的失败报告；业务是否成功看 execution.status。
            assert tool.calls == 1 and not missing.exists() and len(calls) == 4
            assert service._sessions[handle.task_id].checkpoint.state.progress['answer_format_recovery']
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_terminal_observation_waits_for_sealed_result_and_hides_old_failure(tmp_path):
    with workspace_service(tmp_path) as (service, _, runtime, port):
        runtime.call_chat_json = lambda **kw: {'kind': 'final', 'text': '普通说明。', 'task_outcome': 'completed'}
        original = service.executor.run
        paused, release = threading.Event(), threading.Event()
        def run(**kw):
            result = original(**kw)
            paused.set()
            assert release.wait(5)
            return result
        service.executor.run = run
        try:
            h = port.submit(AgentTurnRequest('seal', '你好'))
            assert paused.wait(5)
            service._write(service._sessions[h.task_id].scope, h.task_id, h.attempt,
                lambda task: task.update(stop_reason='old_attempt_error'))
            view = port.status(h.task_id)
            assert view['status'] == 'running' and view['reason'] == ''
            assert view['execution']['finalization_pending']
            assert port.result(h.task_id)['execution']['result_available'] is False
        finally:
            release.set()
        assert settled(port, h.task_id)['status'] == 'completed'
        assert port.result(h.task_id)['execution']['result_available'] is True


@pytest.mark.parametrize('changed', ['task_id', 'scope', 'version', 'past_version'])
def test_sealed_result_requires_current_binding_even_with_valid_hash(tmp_path, changed):
    from shinku.agent_runtime import AgentBoundaryError
    from shinku.agent_runtime.contracts import canonical_hash
    with workspace_service(tmp_path) as (service, _, runtime, port):
        runtime.call_chat_json = lambda **kw: {'kind': 'final', 'text': '说明。', 'task_outcome': 'completed'}
        h = port.submit(AgentTurnRequest('binding', '你好'))
        settled(port, h.task_id)
        scope = service._sessions[h.task_id].scope
        def mutate(task):
            result = task['execution_result']
            field = 'version' if changed == 'past_version' else changed
            result[field] = (result['version'] - 1 if changed == 'past_version' else
                              {'requester': 'other', 'session': scope.session} if changed == 'scope'
                               else result[changed] + 100 if changed == 'version' else 'another-task')
            result['result_hash'] = canonical_hash({k: v for k, v in result.items() if k != 'result_hash'})
        service._write(scope, h.task_id, h.attempt, mutate)
        with pytest.raises(AgentBoundaryError, match='result_binding_conflict'):
            port.result(h.task_id)


def test_line_reference_uses_search_anchor_and_actual_source_range(tmp_path):
    from test_workspace_tools import run, updates
    lines = ['约束一：保留已有修改。\n'] + ['重复背景。\n'] * 90 + ['约束二：不能重复发送。\n'] + ['尾部。\n'] * 90 + ['约束三：未知费用不能为零。\n']
    (tmp_path / 'material.txt').write_text(''.join(lines))
    with workspace_service(tmp_path) as (_, tools, _, __):
        hit = updates(run(tools, 'search', {'query': '约束二'}))['matches'][0]
        assert hit['line'] == 92
        source = updates(run(tools, 'read_lines', {'path': hit['path'], 'start_line': hit['line'], 'line_count': 1}))
        assert source['start_line'] == source['end_line'] == 92
        assert source['text'] == lines[91] and not source['read_complete']
