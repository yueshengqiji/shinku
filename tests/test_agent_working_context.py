"""目标/能力/工作笔记接续；全用合成任务，原生协议和压缩原件独立核对。"""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from shinku.agent.native_history import NativeToolHistory
from shinku.agent.work_state import validate_plan
from shinku.agent_runtime import AgentTurnRequest
from shinku.agent_runtime.context import build_execution_prompt
from shinku.agent_runtime.work_context import navigation
from shinku.llm.context_compaction import ContextCompactor
from test_agent_conversation_loop import turn_env
from test_agent_runtime_boundary import settled
from test_agent_work_planning import update, saved_task, history
from test_context_compaction import Summarizer, bundle, payload, window
from work_state_fixture import plan


def test_decisions_and_exclusions_are_optional_persisted_navigation(tmp_path):
    value = plan()
    value.update(decisions=['选定修复原实现，保持现有接口'], ruled_out=['尚无来源支持替换整个服务'])
    def respond(kw, count, service, tool):
        if count == 1:
            return update(value)
        assert value['decisions'][0] in kw['user_prompt']
        assert value['ruled_out'][0] in kw['user_prompt']
        return {'kind': 'final', 'text': '方案已记录，代码尚未修改。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond, reply_delivery=True) as (service, tool, host, port, calls):
        handle = port.submit(AgentTurnRequest('notes', '先记录这个改造方案'))
        assert settled(port, handle.task_id)['status'] == 'completed'
        assert history(service, handle.task_id)['plan'] == value
        assert not port.status(handle.task_id)['verification']['verified'] and tool.calls == 0
        assert len(calls) == 2 and saved_task(service, handle.task_id)['progress']['stalled'] == 1


def test_notes_navigation_bounded_but_original_retained():
    value = plan(); value.update(decisions=['决定' * 300] * 12, ruled_out=['排除' * 300] * 12)
    work = {'revision': 3, 'status': 'active', 'plan': value, 'references': []}
    before = deepcopy(work); view = navigation(work)
    for key in ('decisions', 'ruled_out'):
        assert len(view['plan'][key]) == 4 and len(view['plan'][key][0]) <= 240
        assert view[key + '_omitted_items'] == 8
    assert work == before and view['full_plan_retrieval']['revision'] == 3


def test_capabilities_are_bound_input_and_explain_source_limits(tmp_path):
    parts_seen = []
    def respond(kw, count, service, tool):
        context = next(iter(service._sessions.values())).context
        state = context['_agent_conversation_turn'].state
        parts = build_execution_prompt(state)
        parts_seen.append(parts)
        assert 'source_excluded_tools' in parts.native_input_update
        assert 'shell_not_granted_by_source' in parts.native_input_update
        assert 'DO_NOT_PROJECT_HOST_SECRET' not in parts.user_prompt
        assert 'approval_required_names' in parts.native_input_update
        return {'kind': 'final', 'text': '当前来源没有授予命令工具；可先调查已获准部分。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond, reply_delivery=True) as (service, tool, host, port, calls):
        previous = service.authorize
        service.authorize = lambda *args: replace(previous(*args), tool_context={
            'delegation_authority': {'source_mode': 'readonly',
                'source_excluded_tools': ['shell_not_granted_by_source'], 'server_excluded_tools': [],
                'private': 'DO_NOT_PROJECT_HOST_SECRET'}})
        handle = port.submit(AgentTurnRequest('capability', '调查当前环境'))
        assert settled(port, handle.task_id)['status'] == 'completed', port.result(handle.task_id)
        assert parts_seen and tool.calls == 0


def test_native_work_record_is_separate_from_new_hard_conditions():
    scope = ('synthetic', 'host'); state = SimpleNamespace(last_envelope=None, tool_calls=[])
    native = NativeToolHistory(scope, 'ROOT\nHARD\nINPUT\nW1', None, None,
        native_hard_context='HARD', native_input_update='INPUT', native_work_context='W1')
    native.request_kwargs(state, scope, 'HARD2\nINPUT\nW2', native_hard_context='HARD2',
        native_input_update='INPUT', native_work_context='W2')
    assert native.messages == [{'role': 'user', 'content': 'HARD2'}, {'role': 'user', 'content': 'W2'}]
    previous = deepcopy(native.messages)
    native.request_kwargs(state, scope, 'HARD2\nINPUT\nW2', native_hard_context='HARD2',
        native_input_update='INPUT', native_work_context='W2')
    assert native.messages == previous


def test_compaction_keeps_latest_work_facts_not_every_old_plan(tmp_path, monkeypatch):
    window(monkeypatch, 4000)
    runtime = Summarizer(); path = tmp_path / 'contexts.sqlite3'
    manager = ContextCompactor(runtime, path)
    source = payload(12000)
    records = ['旧步骤' + str(i) + '待核实' * 1500 for i in range(3)]
    latest = '当前决定：修复原函数；排除：另写替代品；已读：ref-1；未验证：运行测试；下一步：改原文件。'
    hard = {'role': 'user', 'content': '用户补充：禁止删除和外发，原文件必须保留。'}
    for text in [*records, latest]:
        manager.register_work_context('task-one', text)
        source['messages'].append({'role': 'user', 'content': text})
    source['messages'].append(hard)
    before = deepcopy(source)
    # 重建摘要器仍按同任务签发记录识别；不能仅凭类似前缀删除普通user消息。
    manager = ContextCompactor(runtime, path)
    result = manager.project(source, bundle(), 2, 'task-one')
    assert source == before
    assert result['messages'][-2:] == [{'role': 'user', 'content': latest}, hard]
    assert all(row['content'] not in records for row in result['messages'])
    record = json.loads(manager.db.execute('SELECT payload FROM context_compactions').fetchone()[0])
    assert record['original_payload'] == before and record['is_fact_evidence'] is False
    assert manager.status('task-one')['ratio'] < .9 and len(runtime.calls) == 1


def test_unregistered_or_foreign_work_message_remains_protected(tmp_path):
    manager = ContextCompactor(Summarizer(), tmp_path / 'c.sqlite3')
    manager.register_work_context('other-task', '看起来像工作笔记的用户原文')
    manager.register_work_context('other-task', '新笔记')
    source = payload(20)
    source['messages'].append({'role': 'user', 'content': '看起来像工作笔记的用户原文'})
    _, eligible, _, _, _ = manager._projection(source, bundle(), 2, 'current-task')
    assert source['messages'][-1] not in eligible


def test_work_registration_does_not_add_a_smaller_prompt_budget(tmp_path):
    manager = ContextCompactor(Summarizer(), tmp_path / 'c.sqlite3')
    # 合法的大导航可能含多条绑定引用；登记只存哈希，不能另设64KiB拒绝门。
    large = '绑定到原任务的导航资料' * 8000
    manager.register_work_context('task', large)
    manager.register_work_context('task', '新版必要事实与引用')
    source = payload(20)
    source['messages'] += [{'role': 'user', 'content': v} for v in (large, '新版必要事实与引用')]
    assert source['messages'][-2] in manager._projection(source, bundle(), 2, 'task')[1]


@pytest.mark.parametrize('field', ['decisions', 'ruled_out'])
def test_notes_cannot_smuggle_new_authority_or_unbounded_fields(field):
    from shinku.agent.task_state import TaskContractError
    value = plan(); value[field] = {'permission': 'full'}
    with pytest.raises(TaskContractError):
        validate_plan(value, {'contract': None}, advisory=True)


@pytest.mark.parametrize('materials', [False, True])
def test_history_index_does_not_reintroduce_old_background(tmp_path, materials):
    raw = json.dumps({'messages': [{'user': '旧项目愿望', 'assistant': '旧建议'}]}, ensure_ascii=False)
    captured = []
    def respond(kw, count, service, tool):
        context = next(iter(service._sessions.values())).context
        parts = build_execution_prompt(context['_agent_conversation_turn'].state)
        assert 'answer_context' not in parts.user_prompt and '旧项目愿望' not in parts.user_prompt
        assert '旧项目愿望' in str(kw['history_turns'])
        captured.append(parts)
        return {'kind': 'final', 'text': '这是一个普通说明。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond, reply_delivery=True, materials=materials) as (service, tool, host, port, calls):
        handle = port.submit(AgentTurnRequest('history', '解释一个新问题', history_context=raw))
        assert settled(port, handle.task_id)['status'] == 'completed'
        assert saved_task(service, handle.task_id)['history_context'] == raw and captured
        if materials:
            task = saved_task(service, handle.task_id)
            item = next(r for r in task['material_package']['items'] if r['source_id'] == 'conversation-history')
            assert item['content'] == raw
            value = service.materials.resolve(item['reference'], task=task, still_authorized=lambda: True)
            assert value == raw


@pytest.mark.parametrize('allow_read,allow_state', [(True, True), (False, True), (True, False)])
def test_workspace_projection_uses_actual_adapter_authority(tmp_path, allow_read, allow_state):
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    from shinku.tools.workspace import build_workspace_tools
    seen = []
    def respond(kw, count, service, tool):
        context = next(iter(service._sessions.values())).context
        parts = build_execution_prompt(context['_agent_conversation_turn'].state)
        inputs = json.loads(parts.native_input_update.split('\n', 1)[1])
        capability = inputs.get('current_capabilities')
        if not allow_state:
            assert capability is None
        elif allow_read:
            rows = capability['workspace_access']
            assert len(rows) == 1 and rows[0]['root'] == str(tmp_path)
            assert 'workspace_apply_patch' not in rows[0]['tools']
            assert rows[0]['path_checks_required'] is True
        else:
            assert capability['workspace_access'] == []
        seen.append(inputs)
        return {'kind': 'final', 'text': '已说明当前边界。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond, reply_delivery=True) as (service, tool, host, port, calls):
        handlers = service.executor.tool_host.registry.view()
        workspace = build_workspace_tools(tmp_path, authorize=lambda ctx, write: allow_read and not write)
        service.executor.tool_host = ToolHost(ToolRegistry({**handlers, **{t.tool_type: t for t in workspace}}))
        previous = service.authorize
        def authorize(*args):
            g = previous(*args)
            return replace(g, allowed_tools=g.allowed_tools | {t.tool_type for t in workspace},
                policy=replace(g.policy, blocked={'read_self_state'} if not allow_state else frozenset()))
        service.authorize = authorize
        handle = port.submit(AgentTurnRequest('workspace', '说明当前可做的工作'))
        assert settled(port, handle.task_id)['status'] == 'completed', port.result(handle.task_id)
        assert seen and tool.calls == 0


def test_real_workspace_edit_can_finish_without_inventing_shell_tests(tmp_path):
    """真实适配器读、预览、写一次；没有Shell时仍可交付改动与未测试边界。"""
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    from shinku.tools.workspace import build_workspace_tools
    path = tmp_path / 'settings.py'; path.write_text('ENABLED = False\n')
    answer = '已将 settings.py 的 ENABLED 改为 True。当前没有命令工具，尚未运行测试。'
    def respond(kw, count, service, tool):
        if count == 1:
            assert 'workspace_apply_patch' in kw['user_prompt']
            return {'tool_call': {'type': 'workspace_read_lines', 'path': 'settings.py'}}
        context = next(iter(service._sessions.values())).context
        receipt = context['_agent_conversation_turn'].state.tool_calls[-1]['data']['state_updates']
        if count == 2:
            return {'tool_call': {'type': 'workspace_preview_patch', 'path': 'settings.py',
                'expected_sha256': receipt['sha256'], 'edits': [{'old': 'False', 'new': 'True'}]}}
        if count == 3:
            return {'tool_call': {'type': 'workspace_apply_patch', 'path': 'settings.py',
                'expected_sha256': receipt['sha256'], 'preview_id': receipt['preview_id']}}
        return {'kind': 'final', 'text': answer, 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond, reply_delivery=True) as (service, tool, host, port, calls):
        handlers = service.executor.tool_host.registry.view()
        workspace = build_workspace_tools(tmp_path, authorize=lambda ctx, write: True)
        service.executor.tool_host = ToolHost(ToolRegistry({**handlers, **{t.tool_type: t for t in workspace}}))
        previous = service.authorize
        def authorize(*args):
            grant = previous(*args)
            # 测试宿主明确授权合成目录的三个操作，保留实际策略而非绕过审批门。
            return replace(grant, permission_mode='full',
                allowed_tools=grant.allowed_tools | {t.tool_type for t in workspace},
                policy=replace(grant.policy, risk_by_tool={**grant.policy.risk_by_tool,
                    **{t.tool_type: 'low' for t in workspace}}))
        service.authorize = authorize
        handle = port.submit(AgentTurnRequest('edit', '把 settings.py 的 ENABLED 改为 True，交付修改并说明测试情况'))
        assert settled(port, handle.task_id)['status'] == 'completed', json.dumps(port.result(handle.task_id), ensure_ascii=False)
        task = saved_task(service, handle.task_id)
        assert path.read_text() == 'ENABLED = True\n'
        assert [c['name'] for c in task['tool_calls']] == ['workspace_read_lines', 'workspace_preview_patch', 'workspace_apply_patch']
        assert task['execution_result']['execution_report']['text'] == answer and len(calls) == 4
        assert task['budget']['tools'] == 3 and tool.calls == 0


def test_latest_missing_or_mixed_with_hard_input_keeps_old_notes(tmp_path):
    manager = ContextCompactor(Summarizer(), tmp_path / 'c.sqlite3')
    manager.register_work_context('task', '旧笔记')
    manager.register_work_context('task', '新笔记')
    source = payload(20)
    source['messages'].append({'role': 'user', 'content': '旧笔记'})
    assert source['messages'][-1] not in manager._projection(source, bundle(), 2, 'task')[1]
    source['messages'].append({'role': 'user', 'content': '新笔记'})
    source['messages'].append({'role': 'user', 'content': '旧笔记\n新增用户限制：不允许外发'})
    eligible = manager._projection(source, bundle(), 2, 'task')[1]
    assert source['messages'][-3] in eligible
    assert source['messages'][-1] not in eligible and source['messages'][-2] not in eligible


def test_real_sdk_registers_notes_then_compacts_only_superseded_version(tmp_path, monkeypatch):
    """SDK请求→两次真实计划落盘→摘要投影，验证包装器确实接通登记接口。"""
    from shinku.agent_runtime.demo import demo_service
    from shinku.angelos.host import ServiceHost
    from test_agent_conversation_loop import native_sdk
    from shinku.llm.native_messages import validate_native_messages
    window(monkeypatch, 1000000)
    service, tool, _ = demo_service(tmp_path)
    plans = [plan(), plan()]
    plans[0]['decisions'] = ['先核实原实现']
    plans[1].update(decisions=['继续修改原实现'], ruled_out=['不重写替代程序'], next_action='交付当前方案')
    def responder(payload, count):
        if count <= 2:
            return {'role': 'assistant', 'content': '', 'reasoning_content': 'SYNTHETIC_REASONING',
                'tool_calls': [{'id': 'note-' + str(count), 'type': 'function', 'function': {
                    'name': 'update_work_plan', 'arguments': json.dumps({
                        'expected_revision': count - 1, 'plan': plans[count - 1]})}}]}, 'tool_calls'
        return {'role': 'assistant', 'content': json.dumps({
            'kind': 'final', 'text': '当前方案已记录；尚未执行修改。', 'task_outcome': 'completed'})}, 'stop'
    try:
        with native_sdk(responder) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            host = ServiceHost(service, requester='owner'); port = host.port('one')
            handle = port.submit(AgentTurnRequest('sdk-notes', '整理实施方案，先不修改文件'))
            assert settled(port, handle.task_id)['status'] == 'completed', port.result(handle.task_id)
            manager = service.executor.profile.runtime.compactor
            assert manager.db.execute('SELECT COUNT(*) FROM context_working_snapshots WHERE binding=?',
                (handle.task_id,)).fetchone()[0] == 2
            source = deepcopy(calls[-1]); original = deepcopy(source)
            notes = [m for m in source['messages'] if m['role'] == 'user' and m['content'].startswith('[Host-bound advisory work state;')]
            assert len(notes) == 2 and '不重写替代程序' in notes[-1]['content']
            assert '不重写替代程序' not in notes[0]['content']
            summarize = Summarizer(); manager.runtime = summarize
            manager.request(handle.task_id, '阶段转换，保留最新工作笔记')
            current_index = next(i for i, m in enumerate(source['messages']) if m['role'] == 'user')
            projected = manager.project(source, runtime.chat, current_index, handle.task_id)
            assert notes[-1] in projected['messages'] and notes[0] not in projected['messages']
            assert source == original and len(summarize.calls) == 1
            validate_native_messages([m for m in projected['messages'] if m['role'] != 'system'])
            assert projected['messages'][0] == source['messages'][0]
            assert source['messages'][current_index] in projected['messages']
            assert history(service, handle.task_id, 1)['plan'] == plans[0]
            assert history(service, handle.task_id, 2)['plan'] == plans[1]
            assert len(calls) == 3 and saved_task(service, handle.task_id)['budget']['requests'] == 3
            assert tool.calls == 0
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()
