"""原始输入到同一AgentLoop的隔离验收；不调用供应商、私用工具或外部发送。"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import threading

import httpx
from openai import OpenAI
import pytest

from shinku.agent_runtime import AgentTurnRequest, AgentRequest, AgentScope, AgentBoundaryError
from shinku.agent_runtime.demo import demo_service, demo_request
from shinku.agent_runtime.materials import TaskMaterialStore
from shinku.agent.task_state import TaskContractError
from shinku.angelos.host import ServiceHost
from test_agent_runtime_boundary import settled, handle_from
from test_angelos_initiative import seed


def observed_contract(name):
    return {'goal': '按原问题交付当前实际读取结果', 'deliverables': ['具体结果与限制'],
        'scope': ['只读，不重跑旧任务'], 'requirements': [
            {'id': 'observed', 'description': '当前请求的真实只读观察',
             'effect': 'tool:' + name, 'targets': [], 'checks': {}}]}


@contextmanager
def turn_env(root, responder, *, materials=False, recovery=False, reply_delivery=False):
    service, tool, base = demo_service(root)
    calls = []
    def runtime(**kw):
        calls.append(kw)
        return responder(kw, len(calls), service, tool)
    base.call_chat_json = runtime
    if materials:
        service.materials = TaskMaterialStore(service.task_store)
    service.recovery_enabled = recovery
    # 本模块保留旧契约宿主的兼容验收；新版默认行为在reply_delivery链路独立覆盖。
    host = ServiceHost(service, requester='owner', reply_delivery=reply_delivery)
    # 如果正式入口仍调用准备器/核对器，这些测试必须失败，不能偷偷多算一次。
    host.prepare = lambda *args: (_ for _ in ()).throw(AssertionError('preparer must not be used'))
    try:
        yield service, tool, host, host.port('one'), calls
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


@pytest.mark.parametrize('text', ['你好', '只给建议，不要执行命令', '解释一下“查一下原因”这句话',
    '怎么理解权限与验收？', 'What does a tool do?'])
def test_plain_reply_one_call_no_preparer_and_no_business_success(tmp_path, text):
    def respond(*args): return {'kind': 'final', 'text': '这是普通解释，不进行操作。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        request = AgentTurnRequest('hello', text)
        handle = port.submit(request)
        view = settled(port, handle.task_id); result = port.result(handle.task_id)
        assert view['status'] == 'completed' and view['verification']['verified'] is False
        assert result['verification']['layers']['policy'] == 'conversation_reply'
        assert not result['verification']['evidence'][-1]['business_goal']
        assert result['contract'] is None and len(calls) == result['budget']['requests'] == 1
        assert tool.calls == result['budget']['tools'] == 0
        repeated = port.submit(request)
        assert (repeated.task_id, repeated.attempt) == (handle.task_id, handle.attempt) and len(calls) == 1
        assert all(c['prompt_cache_key'].startswith('agent-execution:') for c in calls)


@pytest.mark.parametrize('materials', [False, True])
def test_record_read_and_contract_in_final_share_same_loop(tmp_path, materials):
    def respond(kw, count, service, tool):
        if count == 1: return {'tool_call': {'type': 'read_task_records', 'mode': 'detail', 'task_id': 'old'}}
        assert 'synthetic_missing_input' in kw['user_prompt']
        return {'kind': 'final', 'text': '已核对，记录中的读取失败：缺少合成输入 sample.txt。旧任务没有重跑。',
            'task_outcome': 'completed', 'task_contract': observed_contract('read_task_records')}
    with turn_env(tmp_path, respond, materials=materials) as (service, tool, host, port, calls):
        old = seed(service, 'old')
        handle = port.submit(AgentTurnRequest('diagnose', '核实本会话之前失败的具体原因'))
        view = settled(port, handle.task_id); result = port.result(handle.task_id)
        assert view['status'] == 'completed', json.dumps(result, ensure_ascii=False)
        assert result['verification']['accepted'] and result['contract'] == observed_contract('read_task_records')
        assert len(calls) == result['budget']['requests'] == 2 and result['budget']['tools'] == 1 and tool.calls == 0
        assert '缺少合成输入' in result['execution_report']['text']
        assert service.task_store.get('old', requester='owner', session='angelos:one') == old


def test_declaration_then_real_write_then_receipt_gate_no_preparation(tmp_path):
    def respond(kw, count, service, tool):
        contract = demo_request(tool).contract
        if count == 1: return {'tool_call': {'type': 'declare_task_contract', 'task_contract': contract}}
        if count == 2: return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
        return {'kind': 'final', 'text': '合成文件已创建并核验。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        handle = port.submit(AgentTurnRequest('write', '创建合成文件 ' + str(tool.path)))
        view = settled(port, handle.task_id); result = port.result(handle.task_id)
        assert view['status'] == 'completed', json.dumps(result, ensure_ascii=False)
        assert view['verification']['verified'] and result['verification']['satisfied'] == ['file']
        assert tool.calls == 1 and len(calls) == 3 and tool.path.read_text() == 'standalone agent verified'
        stored = service.task_store.get(handle.task_id, requester='owner', session='angelos:one')
        assert stored['contract'] == demo_request(tool).contract


def test_effect_before_contract_is_not_executed_and_same_loop_can_correct(tmp_path):
    def respond(kw, count, service, tool):
        if count in {1, 3}: return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
        if count == 2:
            assert not tool.path.exists() and tool.calls == 0
            assert 'conversation_contract_required_before_effect' in kw['user_prompt']
            return {'tool_call': {'type': 'declare_task_contract', 'task_contract': demo_request(tool).contract}}
        return {'kind': 'final', 'text': '合成文件已创建并核验。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        handle = port.submit(AgentTurnRequest('guard', '创建 ' + str(tool.path)))
        view = settled(port, handle.task_id); result = port.result(handle.task_id)
        assert view['status'] == 'completed', json.dumps(result, ensure_ascii=False)
        assert tool.calls == 1 and len(calls) == 4
        assert result['evidence'][0]['execution_status'] != 'ok'


def test_self_tool_question_is_agent_selected_state_read_not_research_route(tmp_path):
    def respond(kw, count, service, tool):
        if count == 1: return {'tool_call': {'type': 'read_self_state', 'scope': 'tools'}}
        assert 'available_names' in kw['user_prompt']
        return {'kind': 'final', 'text': '当前可以查看本会话记录和执行宿主允许的文件工具；是否需要新增工具应按实际工作判断。',
                'task_outcome': 'completed', 'task_contract': observed_contract('read_self_state')}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        handle = port.submit(AgentTurnRequest('capability', '你目前有什么工具能力？'))
        view = settled(port, handle.task_id); result = port.result(handle.task_id)
        assert view['status'] == 'completed', json.dumps(result, ensure_ascii=False)
        assert len(calls) == 2 and [c['tool'] for c in result['evidence']] == ['read_self_state']
        assert tool.calls == 0 and not result.get('verified_answer')


@pytest.mark.parametrize('case', ['removed', 'target', 'declare_evidence', 'denied'])
def test_invalid_or_weakened_declarations_do_not_execute(tmp_path, case):
    def respond(kw, count, service, tool):
        contract = demo_request(tool).contract
        if case == 'removed' and count == 1:
            return {'tool_call': {'type': 'declare_task_contract', 'task_contract': contract}}
        bad = deepcopy(contract)
        if case == 'removed': bad['requirements'][0]['checks'] = {'contains': 'different'}
        if case == 'target': bad['requirements'][0]['targets'] = [str(tmp_path / 'different.txt')]
        if case == 'declare_evidence': bad = observed_contract('declare_task_contract')
        if count <= (2 if case == 'removed' else 1):
            return {'tool_call': {'type': 'declare_task_contract', 'task_contract': bad}}
        return {'kind': 'final', 'text': '契约条件尚未取得有效校验，操作没有执行。', 'task_outcome': 'failed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        if case == 'denied':
            original = service.authorize
            service.authorize = lambda *args: replace(original(*args), policy=replace(original(*args).policy,
                blocked=frozenset({'declare_task_contract'})))
        handle = port.submit(AgentTurnRequest('invalid', '创建 ' + str(tool.path)))
        view = settled(port, handle.task_id)
        assert view['status'] != 'completed' and tool.calls == 0 and not tool.path.exists()
        task = service.task_store.get(handle.task_id, requester='owner', session='angelos:one')
        if case == 'removed': assert task['contract'] == demo_request(tool).contract


def test_uninitialized_turn_requires_trusted_host_and_legacy_request_stays_strict(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    try:
        with pytest.raises(AgentBoundaryError, match='conversation_host_required'):
            service.bind(AgentScope('owner', 'angelos:one')).submit(AgentTurnRequest('bad', '你好'))
        with pytest.raises(TaskContractError, match='contract_missing'):
            service.bind(AgentScope('owner', 'angelos:one')).submit(AgentRequest('legacy', '你好', None))
        ServiceHost(service, requester='owner', reply_delivery=False)
        with pytest.raises(AgentBoundaryError, match='conversation_host_required'):
            service.bind(AgentScope('other', 'angelos:one')).submit(AgentTurnRequest('bad', '你好'))
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


@pytest.mark.parametrize('finish', ['length', 'content_filter'])
def test_incomplete_first_decision_never_executes_even_if_json_is_parseable(tmp_path, finish):
    from shinku.llm.response_protocol import note_incomplete_response
    def respond(kw, count, service, tool):
        kw['provider_message_sink']({'content': '{"kind":"tool","PARTIAL_RAW":true}', 'reasoning_content': 'PRIVATE_REASONING'})
        note_incomplete_response(finish)
        return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}, 'task_contract': demo_request(tool).contract}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        handle = port.submit(AgentTurnRequest('length', '创建 ' + str(tool.path)))
        view = settled(port, handle.task_id)
        assert view['status'] == 'failed' and view['reason'].startswith('conversation_output_')
        assert len(calls) == 1 and tool.calls == 0
        outputs = host.reasoning_view.outputs(handle.task_id)
        assert len(outputs) == 1 and 'PARTIAL_RAW' in outputs[0]['text']
        assert 'PRIVATE_REASONING' not in json.dumps(outputs) and outputs[0]['diagnostics']['failure']


def test_single_budget_survives_explicit_continuation_without_contract(tmp_path):
    def respond(kw, count, service, tool):
        if count == 1: return {'kind': 'wait', 'text': '请提供要查看的具体对象。', 'reason': 'target_missing'}
        return {'kind': 'final', 'text': '已经收到补充，以下是普通说明。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond, recovery=True) as (service, tool, host, port, calls):
        handle = port.submit(AgentTurnRequest('wait', '这个怎么做？'))
        view = settled(port, handle.task_id)
        assert view['status'] == 'waiting_user' and len(calls) == 1
        resumed = port.resume(handle_from(view), request_key='explicit-continue')
        final = settled(port, resumed.task_id); result = port.result(resumed.task_id)
        assert final['status'] == 'completed', result
        assert result['budget']['requests'] == 2 and len(calls) == 2 and tool.calls == 0


def test_workbench_submits_original_input_without_preparation_model(tmp_path):
    from angelos_fixture import fixture
    from fastapi.testclient import TestClient
    from test_angelos import new, post, wait
    app, bench, old_host = fixture(tmp_path)
    calls = []
    def respond(**kw):
        calls.append(kw)
        return {'kind': 'final', 'text': '普通回复，直接进入原循环。', 'task_outcome': 'completed'}
    old_host.service.executor.profile.runtime.call_chat_json = respond
    host = ServiceHost(old_host.service, requester=old_host.requester, reply_delivery=False)
    host.prepare = lambda *args: (_ for _ in ()).throw(AssertionError('extra preparer'))
    bench.host = host
    with TestClient(app) as client:
        client.headers['x-angelos-token'] = client.get('/angelos/api/bootstrap').json()['token']
        cid = new(client)
        post(client, f'/conversations/{cid}/messages', {'text': '你好', 'request_key': 'one'})
        turn = wait(client, cid)['turns'][-1]
        assert turn['stage'] == 'finished' and turn['preparation_usage']['requests'] == 0
        assert len(calls) == turn['result']['budget']['requests'] == 1
        assert turn['view']['verification']['verified'] is False and turn['model_outputs']


@contextmanager
def native_sdk(responder):
    """真正OpenAI SDK与原生assistant/tool配对，模型响应来自隔离HTTP传输。"""
    from test_expression_runtime import FixtureRuntime
    from shinku.llm.runtime_types import ModelBundle
    payloads = []
    def handle(request):
        payload = json.loads(request.content); payloads.append(payload)
        message, finish = responder(payload, len(payloads))
        return httpx.Response(200, json={'id': 'synthetic-native', 'object': 'chat.completion', 'created': 0,
            'model': 'deepseek-flash', 'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 4, 'total_tokens': 16}})
    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as http:
        with OpenAI(api_key='synthetic-key', base_url='https://offline.example.invalid/v1', http_client=http, max_retries=0) as client:
            client.protocol = 'openai'
            runtime = FixtureRuntime(client.chat.completions.create)
            runtime._swap_lock = threading.RLock()
            runtime.chat = ModelBundle(client, 'deepseek-flash'); runtime.aux = runtime.chat
            runtime._env = lambda key, default='': {'LLM_THINKING_MODE': 'enabled', 'LLM_REASONING_EFFORT': 'high'}.get(key, default)
            runtime.chat_supports_native_tools = lambda: True
            yield runtime, payloads


def test_real_sdk_native_declaration_write_and_result_stay_in_one_history(tmp_path):
    service, tool, _ = demo_service(tmp_path)
    def responder(payload, index):
        if index == 1: name, args = 'declare_task_contract', {'task_contract': demo_request(tool).contract}
        elif index == 2:
            assert any(m['role'] == 'tool' and 'host_validated_contract' in m['content'] for m in payload['messages'])
            name, args = 'demo_file', {'path': str(tool.path)}
        else:
            assert tool.calls == 1
            return {'role': 'assistant', 'content': json.dumps({'kind': 'final', 'text': '合成文件已创建并核验。', 'task_outcome': 'completed'})}, 'stop'
        return {'role': 'assistant', 'content': '', 'reasoning_content': 'SYNTHETIC_PRIVATE_REASONING',
            'tool_calls': [{'id': f'call_{index}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'tool_calls'
    try:
        with native_sdk(responder) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime, runtime_meters_budget=True, native_tool_history=True)
            host = ServiceHost(service, requester='owner', reply_delivery=False); port = host.port('one')
            handle = port.submit(AgentTurnRequest('sdk', '创建 ' + str(tool.path)))
            view = settled(port, handle.task_id); result = port.result(handle.task_id)
            assert view['status'] == 'completed', json.dumps(result, ensure_ascii=False)
            assert len(calls) == result['budget']['requests'] == 3 and result['budget']['reported_tokens'] == 48
            assert all('max_tokens' not in p and p['reasoning_effort'] == 'high' for p in calls)
            for index, payload in enumerate(calls):
                history = [m for m in payload['messages'] if m['role'] in {'assistant', 'tool'}]
                assert len(history) == index * 2
                for a, t in zip(history[::2], history[1::2]):
                    assert a['tool_calls'][0]['id'] == t['tool_call_id'] and a['reasoning_content'] == 'SYNTHETIC_PRIVATE_REASONING'
            assert 'SYNTHETIC_PRIVATE_REASONING' not in json.dumps(result)
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


def test_read_receipt_survives_rejected_premature_final_without_rereading(tmp_path):
    def respond(kw, count, service, tool):
        if count == 1: return {'tool_call': {'type': 'read_task_records', 'mode': 'detail', 'task_id': 'old'}}
        value = {'kind': 'final', 'text': '记录中的具体错误是 synthetic_missing_input。', 'task_outcome': 'completed'}
        if count == 3:
            assert 'task_contract' in kw['user_prompt']
            value['task_contract'] = observed_contract('read_task_records')
        return value
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        seed(service, 'old')
        handle = port.submit(AgentTurnRequest('premature', '查明此前任务失败原因'))
        view = settled(port, handle.task_id); result = port.result(handle.task_id)
        assert view['status'] == 'completed', json.dumps(result, ensure_ascii=False)
        assert len(calls) == 3 and result['budget']['tools'] == 1 and len(result['evidence']) == 1


@pytest.mark.parametrize('boundary', ['revoke', 'cancel', 'approval', 'budget'])
def test_dynamic_contract_never_bypasses_original_execution_boundary(tmp_path, boundary):
    active = [True]; entered = threading.Event(); release = threading.Event()
    def respond(kw, count, service, tool):
        if count == 1:
            if boundary == 'cancel':
                entered.set(); assert release.wait(3)
            return {'tool_call': {'type': 'declare_task_contract', 'task_contract': demo_request(tool).contract}}
        if boundary == 'revoke': active[0] = False
        return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        original = service.authorize
        def authorize(*args):
            grant = original(*args)
            if boundary == 'revoke': return replace(grant, still_authorized=lambda: active[0])
            if boundary == 'approval': return replace(grant, policy=replace(grant.policy,
                risk_by_tool={**grant.policy.risk_by_tool, 'demo_file': 'high'}))
            if boundary == 'budget': return replace(grant, budget_limits={**grant.budget_limits, 'max_requests': 1})
            return grant
        service.authorize = authorize
        handle = port.submit(AgentTurnRequest(boundary, '创建 ' + str(tool.path)))
        if boundary == 'cancel':
            assert entered.wait(3)
            port.cancel(handle_from(port.status(handle.task_id))); release.set()
        view = settled(port, handle.task_id)
        assert view['status'] != 'completed' and tool.calls == 0 and not tool.path.exists()
        if boundary == 'budget':
            assert len(calls) == view['budget']['requests'] == 1 and view['reason'] == 'task_model_budget_exhausted'
        if boundary == 'cancel':
            assert len(calls) == 1 and view['status'] == 'canceled'
            assert port.result(handle.task_id)['contract'] is None
        if boundary == 'revoke': assert view['reason'] == 'agent_permission_changed'
        if boundary == 'approval':
            assert view['status'] == 'failed' and view['reason'] == 'approval_gate_unavailable'
            assert len(calls) == 2  # 原宿主没有审批通道，不能把确认需求变成自动执行。


@pytest.mark.parametrize('initialized', [False, True])
def test_actual_store_restart_preserves_dynamic_contract_and_never_replays_write(tmp_path, initialized):
    """关闭并重开SQLite/协调器，明确恢复；原契约/读取/用量不得重置。"""
    from test_agent_recovery import service_at, close, Runtime
    service, _, tool, runtime = service_at(tmp_path)
    host = ServiceHost(service, requester='owner', reply_delivery=False, reasoning_path=tmp_path / 'outputs.sqlite3'); port = host.port('one')
    if initialized:
        runtime.decisions = [
            {'tool_call': {'type': 'declare_task_contract', 'task_contract': demo_request(tool).contract}},
            {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}}, RuntimeError('synthetic interruption')]
    else:
        seed(service, 'old')
        runtime.decisions = [{'tool_call': {'type': 'read_task_records', 'mode': 'detail', 'task_id': 'old'}},
                             RuntimeError('synthetic interruption')]
    handle = port.submit(AgentTurnRequest('restart', '创建 ' + str(tool.path) if initialized else '核实本会话历史失败'))
    first = settled(port, handle.task_id)
    assert first['status'] == 'failed' and tool.calls == int(initialized)
    old_task = service.task_store.get(handle.task_id, requester='owner', session='angelos:one')
    old_outputs = host.reasoning_view.outputs(handle.task_id)
    assert old_outputs and any(r['diagnostics'].get('call_failure') for r in old_outputs)
    close(service)
    if initialized:
        decisions = [{'tool_call': {'type': 'demo_file', 'path': str(tool.path)}},
                     {'kind': 'final', 'text': '文件已按原回执核验，恢复没有再次写入。', 'task_outcome': 'completed'}]
    else:
        # 跨进程观察凭据不能恢复为当前事实，允许同一循环重新读取；绝不重放旧任务。
        decisions = [{'tool_call': {'type': 'read_task_records', 'mode': 'detail', 'task_id': 'old'}},
                     {'kind': 'final', 'text': '保留记录中的具体错误 synthetic_missing_input，没有重跑旧任务。',
                      'task_outcome': 'completed', 'task_contract': observed_contract('read_task_records')}]
    service, _, fresh_tool, runtime = service_at(tmp_path, Runtime(decisions))
    try:
        host = ServiceHost(service, requester='owner', reply_delivery=False, reasoning_path=tmp_path / 'outputs.sqlite3'); port = host.port('one')
        assert host.reasoning_view.outputs(handle.task_id) == old_outputs
        current = handle_from(port.status(handle.task_id))
        assert port.recovery(current)['status'] == 'continuable'
        port.resume(current, request_key='explicit-recovery')
        view = settled(port, handle.task_id); result = port.result(handle.task_id)
        assert view['status'] == 'completed', json.dumps(result, ensure_ascii=False)
        assert fresh_tool.calls == 0 and tool.calls == int(initialized)
        assert result['budget']['requests'] == first['budget']['requests'] + len(decisions)
        assert result['original_requests'] == old_task['requests']
        assert result['contract'] == (old_task['contract'] if initialized else observed_contract('read_task_records'))
    finally:
        close(service)


def test_missing_required_material_stops_before_any_effect_even_with_valid_contract(tmp_path):
    def respond(kw, count, service, tool):
        return {'tool_call': {'type': 'demo_file', 'path': str(tool.path)}, 'task_contract': demo_request(tool).contract}
    with turn_env(tmp_path, respond, materials=True) as (service, tool, host, port, calls):
        original = service.authorize
        service.authorize = lambda *args: replace(original(*args), material_gaps=({'source_id': 'attachment', 'reason': 'synthetic_attachment_missing', 'required': True},))
        handle = port.submit(AgentTurnRequest('missing', '按附件创建 ' + str(tool.path)))
        view = settled(port, handle.task_id)
        assert view['status'] == 'waiting_user' and tool.calls == 0 and not calls
        assert not port.result(handle.task_id)['verification']['accepted']


def test_ninety_percent_summary_and_native_loop_share_original_task_budget(tmp_path, monkeypatch):
    monkeypatch.setenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS', '30000')
    history = json.dumps({'messages': [{'user': 'a' * 140000, 'assistant': '历史建议只作背景'}]})
    def respond(payload, count):
        system = '\n'.join(m['content'] for m in payload['messages'] if m['role'] == 'system')
        answer = {'summary': '旧建议只作背景，原文仍保留。'} if '只压缩历史' in system else {
            'kind': 'final', 'text': '只讨论当前方案，不执行操作。', 'task_outcome': 'completed'}
        return {'role': 'assistant', 'content': json.dumps(answer)}, 'stop'
    service, tool, _ = demo_service(tmp_path)
    try:
        with native_sdk(respond) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime, runtime_meters_budget=True,
                                               native_tool_history=True)
            host = ServiceHost(service, requester='owner', reply_delivery=False); port = host.port('one')
            handle = port.submit(AgentTurnRequest('summary', '只讨论当前方案', history_context=history))
            view = settled(port, handle.task_id); result = port.result(handle.task_id)
            assert view['status'] == 'completed', json.dumps(result, ensure_ascii=False)
            # 原历史本身超过合成窗口，须两段摘要再执行；三次都在原任务预算内。
            assert len(calls) == result['budget']['requests'] == 3 and result['budget']['reported_tokens'] == 48
            assert '只压缩历史' in calls[0]['messages'][0]['content']
            assert '只压缩历史' in calls[1]['messages'][0]['content']
            assert 'a' * 1000 not in json.dumps(calls[2]) and '当前方案' in json.dumps(calls[2], ensure_ascii=False)
            assert tool.calls == 0 and result['budget']['tools'] == 0
            original = service.task_store.get(handle.task_id, requester='owner', session='angelos:one')
            assert original['history_context'] == history
            assert host.profile.runtime.compactor.status(handle.task_id)['status'] == 'compacted'
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


@pytest.mark.parametrize('case', ['ordinary', 'strict', 'changed'])
def test_read_before_local_contract_rebuilds_real_ledger_and_retains_strict_checks(tmp_path, case):
    from types import SimpleNamespace
    from test_local_analysis_closeout import Reader
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    class LocalReader(Reader):
        def tool_metadata(self):
            return SimpleNamespace(input_schema={'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']})
    source = tmp_path / 'source.txt'; source.write_text('当前入口按职责分离，优化建议需以源码为依据。')
    reader = LocalReader(tmp_path)
    contract = {'goal': '分析源码并提出建议', 'deliverables': ['分析正文'], 'scope': ['只读'], 'requirements': [
        {'id': 'input', 'effect': 'read', 'description': '完整读取源码', 'targets': [str(source)], 'checks': {'unchanged': True}},
        {'id': 'answer', 'effect': 'local_answer', 'description': '分析与依据', 'targets': [],
         'checks': {'required_fields': ['优化方向'], 'min_sources': 1}}]}
    def respond(kw, count, service, tool):
        if count == 1: return {'tool_call': {'type': reader.tool_type, 'path': str(source)}}
        if count == 2:
            if case == 'changed': source.write_text('读取后被外部修改，旧来源失效。')
            return {'kind': 'final', 'text': '源码显示入口按职责分离，建议沿现有职责边界补充接口文档，具体改动前需核实模块依赖。',
                    'task_outcome': 'completed', 'task_contract': contract}
        assert '完成验收未通过' in kw['user_prompt'] or '读取' in kw['user_prompt']
        return {'kind': 'wait', 'text': '当前读取或逐项证据条件未满足，尚不能宣称核验完成。', 'reason': 'evidence_not_verified'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        # 更换为真正的文件读取适配器，原授权仍由同一Service持有。
        handlers = service.executor.tool_host.registry.view()
        service.executor.tool_host = ToolHost(ToolRegistry({**handlers, reader.tool_type: reader}))
        original = service.authorize
        service.authorize = lambda *args: replace(original(*args), allowed_tools=original(*args).allowed_tools | {reader.tool_type},
            policy=replace(original(*args).policy, risk_by_tool={**original(*args).policy.risk_by_tool, reader.tool_type: 'low'}))
        text = ('逐项核验分析依据，精确引用 ' if case == 'strict' else '分析源码并给优化建议 ') + str(source)
        handle = port.submit(AgentTurnRequest('local', text))
        view = settled(port, handle.task_id); result = port.result(handle.task_id)
        assert reader.calls == 1 and tool.calls == 0
        if case == 'ordinary':
            assert view['status'] == 'completed', json.dumps(result, ensure_ascii=False)
            assert result['verification']['layers']['policy'] == 'ordinary_readonly_analysis'
            assert result['verification']['layers']['valid_read_sources'] == 1
            assert result['verification']['layers']['answer_evidence'] == 'unverified'
            assert len(calls) == 2
        else:
            assert view['status'] != 'completed' and not result['verification']['accepted']
            assert result['remaining'] and len(calls) == 3


def test_adapter_original_diagnostics_exclude_reasoning_and_native_objects(tmp_path):
    def respond(kw, *args):
        return {'kind': 'final', 'text': '普通解释。', 'task_outcome': 'completed',
                'reasoning_content': 'PRIVATE_ADAPTER_REASONING', '_native_response': 'PRIVATE_NATIVE'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        handle = port.submit(AgentTurnRequest('safe-original', '你好'))
        assert settled(port, handle.task_id)['status'] == 'completed'
        outputs = host.reasoning_view.outputs(handle.task_id)
        assert outputs[0]['source'] == 'adapter_return_object'
        assert '普通解释' in outputs[0]['text'] and 'PRIVATE_' not in json.dumps(outputs)


def test_fake_action_success_without_receipt_is_never_ordinary_reply(tmp_path):
    def respond(kw, count, service, tool):
        if count == 1:
            return {'kind': 'final', 'text': '文件已创建，命令已执行成功。', 'task_outcome': 'completed'}
        return {'kind': 'final', 'text': '没有实际执行回执，操作未完成。', 'task_outcome': 'failed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        handle = port.submit(AgentTurnRequest('fake', '创建 ' + str(tool.path)))
        view = settled(port, handle.task_id); result = port.result(handle.task_id)
        assert view['status'] == 'failed' and tool.calls == 0
        assert result['verification'].get('layers', {}).get('policy') != 'conversation_reply'
        assert '文件已创建' not in result['execution_report']['text']


def test_invalid_adapter_object_preserves_original_planner_failure(tmp_path):
    from types import SimpleNamespace
    def respond(*args): return SimpleNamespace(invalid='not a model decision')
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        handle = port.submit(AgentTurnRequest('invalid-object', '你好'))
        view = settled(port, handle.task_id)
        assert view['status'] == 'failed' and view['reason'] == 'invalid_decision'
        output = host.reasoning_view.outputs(handle.task_id)[0]
        assert output['diagnostics']['archive_failure'] == 'adapter_output_not_safe_serializable'
        assert len(calls) == 1 and tool.calls == 0
