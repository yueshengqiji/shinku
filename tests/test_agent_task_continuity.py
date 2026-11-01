"""两轮真红路由经真实隔离 HTTP 接续同一任务；固定模型响应不冒充真实模型质量。"""
import json
from dataclasses import replace
import threading
import time

import pytest

from shinku.angelos.host import ServiceHost
from shinku.agent_runtime import AgentBoundaryError, AgentScope, AgentSupplement, TaskHandle
from test_agent_delegation import remote, SCOPE
from test_qq_remote_delegation import bridge_for, close, turn, submit


OPTIONS = '已定位现有工具 demo_file。A：只说明；B：沿用此工具创建文件，不另造实现。请选择。'


def waiting_runtime(remote):
    service, tool, runtime, _, _, _ = remote
    ServiceHost(service, requester='owner')
    original = runtime.call_chat_json
    def decide(**kwargs):
        if not runtime.calls:
            runtime.calls.append(kwargs)
            return {'kind': 'wait', 'text': OPTIONS}
        return original(**kwargs)
    runtime.call_chat_json = decide


def task_call(name, task_id, **arguments):
    return {'kind': 'tool', 'tool_call': {'name': 'agent_task_' + name,
            'arguments': {'task_id': task_id, **arguments}}}


def test_two_turn_choice_preserves_goal_options_and_resumes_once(remote, tmp_path):
    waiting_runtime(remote)
    service, tool, runtime, _, client, _ = remote
    bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [submit()])
    try:
        original = '用现有工具创建 ' + str(tool.path) + '，先说明可选方案。'
        first = bridge(turn(original))
        assert first.agent_status == 'waiting_user', first
        row = bridge.delegation._rows(SCOPE)[0]
        task_id = row['task_id']
        old = client.port(SCOPE).status(task_id)
        old_budget = client.port(SCOPE).result(task_id)['budget']
        chat.replies.append(task_call('supplement', task_id, resume=True))
        answer = bridge(turn('  选择 B，开始。\n', message_id='choice'))
        assert answer.agent_status == 'completed', (answer.reason, answer.final_text)
        prompt = chat.calls[-1]['user_prompt']
        assert OPTIONS in prompt and original in prompt
        assert task_id in prompt and 'waiting_user' in prompt
        assert len(chat.calls) == 2  # 每轮只让原聊天模型决策一次，不新增路由裁判。
        task = service.task_store.get(task_id, requester=SCOPE.requester, session=SCOPE.session)
        assert task['execution_attempt'] == old['attempt'] + 1
        assert [r['text'] for r in task['requests']] == [original, '  选择 B，开始。\n']
        assert len(bridge.delegation._rows(SCOPE)) == 1
        assert tool.calls == 1 and tool.path.read_text() == 'standalone agent verified'
        assert task['budget']['requests'] > old_budget['requests']
        assert task['budget']['max_requests'] == old_budget['max_requests']
        inputs = [{k: v for k, v in call.items() if k in {'user_prompt', 'history_turns'}}
                  for call in runtime.calls[1:]]
        assert OPTIONS in json.dumps(inputs, ensure_ascii=False)
        before = len(runtime.calls)
        chat.replies.append(task_call('supplement', task_id, resume=True))
        bridge(turn('  选择 B，开始。\n', message_id='choice'))
        assert tool.calls == 1 and len(runtime.calls) == before
    finally:
        close(store, coordinator)


def test_model_can_select_older_task_without_host_latest_task_rule(remote, tmp_path):
    service, tool, runtime, _, _, _ = remote
    ServiceHost(service, requester='owner')
    original = runtime.call_chat_json
    def decide(**kwargs):
        if len(runtime.calls) < 2:
            runtime.calls.append(kwargs)
            return {'kind': 'wait', 'text': OPTIONS}
        return original(**kwargs)
    runtime.call_chat_json = decide
    bridge, chat, _, store, coordinator = bridge_for(tmp_path, remote, [submit(), submit()])
    try:
        bridge(turn('项目甲：创建原文件，先给方案', message_id='first-project'))
        bridge(turn('项目乙：只讨论，不执行', message_id='second-project'))
        selected = bridge.delegation._rows(SCOPE)[0]['task_id']
        ordinary = chat.call_chat_json
        def choose(**kwargs):
            # 固定模型桩从实际提示读取候选，不能靠测试预先传入task_id绕过路由背景。
            rows = json.loads(kwargs['user_prompt'].split(
                '宿主提供的同会话任务接续背景（模型陈述不是执行证据或授权）：', 1)[1])
            assert len(rows) == 2 and all(r['report']['text'] == OPTIONS for r in rows)
            target = next(r for r in rows if r['requests'][0]['text'].startswith('项目甲'))
            chat.replies.append(task_call('supplement', target['task_id'], resume=True))
            return ordinary(**kwargs)
        chat.call_chat_json = choose
        reply = bridge(turn('按项目甲的 B 方案落实，乙先保持。', message_id='select-first'))
        assert reply.agent_status == 'completed', (reply.reason, reply.final_text)
        tasks = [service._get(SCOPE, row['task_id']) for row in bridge.delegation._rows(SCOPE)]
        assert next(t for t in tasks if t['task_id'] == selected)['execution_attempt'] == 2
        assert next(t for t in tasks if t['task_id'] != selected)['execution_attempt'] == 1
        assert tool.calls == 1 and len(chat.calls) == 3
    finally:
        close(store, coordinator)


def test_concurrent_input_change_rejects_stale_choice_without_losing_either_original(remote, tmp_path):
    waiting_runtime(remote)
    service, tool, _, _, client, _ = remote
    bridge, chat, _, store, coordinator = bridge_for(tmp_path, remote, [submit()])
    try:
        bridge(turn('先给方案'))
        task_id = bridge.delegation._rows(SCOPE)[0]['task_id']
        ordinary = chat.call_chat_json
        def change(**kwargs):
            view = client.port(SCOPE).status(task_id)
            client.port(SCOPE).revise(TaskHandle(task_id, view['attempt'], view['version']),
                AgentSupplement('parallel-input', clarification='并行修改：先停止'))
            chat.replies.append(task_call('supplement', task_id, resume=True))
            return ordinary(**kwargs)
        chat.call_chat_json = change
        answer = bridge(turn('选 B 开始', message_id='old-choice'))
        assert answer.reason == 'delegation_context_changed'
        task = service._get(SCOPE, task_id)
        assert [r['text'] for r in task['requests']] == ['先给方案', '并行修改：先停止']
        assert tool.calls == 0 and task['execution_attempt'] == 1
    finally:
        close(store, coordinator)


def test_permission_denial_after_supplement_keeps_input_and_does_not_resume(remote, tmp_path):
    waiting_runtime(remote)
    service, tool, runtime, gateway, _, base = remote
    bridge, chat, _, store, coordinator = bridge_for(tmp_path, remote, [submit()])
    try:
        bridge(turn('先给方案'))
        task_id = bridge.delegation._rows(SCOPE)[0]['task_id']
        principal = gateway.principals['shinku']
        gateway.principals['shinku'] = replace(principal, authorize=lambda s, t, op:
            replace(base, still_authorized=lambda: op != 'resume'))
        chat.replies.append(task_call('supplement', task_id, resume=True))
        count = len(runtime.calls)
        answer = bridge(turn('按 B 开始', message_id='denied-resume'))
        assert answer.agent_status == 'blocked' and 'authorization' in answer.reason
        task = service._get(SCOPE, task_id)
        assert task['requests'][-1]['text'] == '按 B 开始'
        assert task['execution_attempt'] == 1 and tool.calls == 0 and len(runtime.calls) == count
    finally:
        close(store, coordinator)


def test_long_options_are_not_partially_projected_and_full_original_is_readable(remote, tmp_path):
    service, tool, runtime, _, client, _ = remote
    ServiceHost(service, requester='owner')
    body = '仅适用于隔离样本。' * 800 + 'B：只修改目标文件，不能重启服务。'
    def waiting(**kwargs):
        runtime.calls.append(kwargs)
        return {'kind': 'wait', 'text': body}
    runtime.call_chat_json = waiting
    bridge, _, _, store, coordinator = bridge_for(tmp_path, remote, [submit()])
    try:
        bridge(turn('先给方案'))
        task_id = bridge.delegation._rows(SCOPE)[0]['task_id']
        context = client.port(SCOPE).context(task_id)
        assert context['report']['text'] == '' and context['report']['omitted_chars'] == len(body)
        assert context['retrieval'] == {'tool': 'agent_task_result', 'task_id': task_id, 'view': 'full'}
        assert client.port(SCOPE).details(client.port(SCOPE).handoff(task_id))['execution_report']['text'] == body
        assert len(runtime.calls) == 1 and tool.calls == 0
    finally:
        close(store, coordinator)


def test_work_notes_are_projected_without_becoming_execution_receipts(remote, tmp_path):
    from work_state_fixture import plan
    service, tool, runtime, _, client, _ = remote
    ServiceHost(service, requester='owner')
    notes = plan(); notes['findings'] = ['现有模块已定位，仍需核实接口。']
    def decide(**kwargs):
        runtime.calls.append(kwargs)
        if len(runtime.calls) == 1:
            return {'tool_call': {'type': 'update_work_plan', 'expected_revision': 0, 'plan': notes}}
        return {'kind': 'wait', 'text': OPTIONS}
    runtime.call_chat_json = decide
    bridge, chat, _, store, coordinator = bridge_for(tmp_path, remote, [submit()])
    try:
        bridge(turn('记录检查进度，等我确认'))
        task_id = bridge.delegation._rows(SCOPE)[0]['task_id']
        view = client.port(SCOPE).context(task_id)
        assert view['work']['plan']['findings'] == notes['findings']
        assert view['work']['is_execution_evidence'] is False
        assert view['work']['plan']['steps'][0]['status'] == 'in_progress'
        chat.replies.append({'kind': 'final', 'text': '已保留当前工作记录。'})
        bridge(turn('先看看进度', message_id='notes'))
        assert notes['findings'][0] in chat.calls[-1]['user_prompt']
        assert len(runtime.calls) == 2 and tool.calls == 0
    finally:
        close(store, coordinator)


def test_pending_stop_does_not_start_second_worker_or_lose_supplement(remote, tmp_path):
    service, tool, runtime, gateway, client, _ = remote
    entered, release = threading.Event(), threading.Event()
    def waiting(**kwargs):
        runtime.calls.append(kwargs)
        entered.set()
        assert release.wait(5)
        return {'kind': 'wait', 'text': OPTIONS}
    runtime.call_chat_json = waiting
    bridge, chat, _, store, coordinator = bridge_for(tmp_path, remote, [submit()], asynchronous=True)
    try:
        assert bridge(turn('给方案')).agent_status == 'queued'
        assert entered.wait(3)
        task_id = bridge.delegation._rows(SCOPE)[0]['task_id']
        chat.replies.append(task_call('supplement', task_id, resume=True))
        answer = bridge(turn('改变目标，先只说明', message_id='mid-execution'))
        assert answer.reason == 'delegation_supplement' and '尚未应用或继续' in answer.final_text
        assert any(state == 'pending' for _, _, state in gateway.revisions._rows(task_id))
        assert service._get(SCOPE, task_id)['execution_attempt'] == 1 and len(runtime.calls) == 1
    finally:
        release.set()
        close(store, coordinator)
    deadline = time.monotonic() + 5
    while client.port(SCOPE).status(task_id)['worker_active']:
        assert time.monotonic() < deadline
        time.sleep(.01)
    client.port(SCOPE).messages(task_id)
    assert service._get(SCOPE, task_id)['requests'][-1]['text'] == '改变目标，先只说明'
    assert tool.calls == 0


def test_snapshot_distinguishes_progress_from_changed_input_without_an_arbitrary_deadline():
    from shinku.qq.delegation_context import TaskContextSnapshot, task_context
    row = {'task_id': 'fixture', 'attempt': 1, 'version': 3, 'input_binding': 'input-one'}
    snapshot = TaskContextSnapshot(SCOPE, (row,), 100)
    snapshot.check_selection(SCOPE, {**row, 'version': 10})
    for current in ({**row, 'attempt': 2}, {**row, 'input_binding': 'changed'}):
        with pytest.raises(AgentBoundaryError, match='context_changed'):
            snapshot.check_selection(SCOPE, current)
    with pytest.raises(AgentBoundaryError, match='context_scope_conflict'):
        snapshot.check_selection(AgentScope('other', SCOPE.session), row)
    assert task_context({'_agent_task_context': {'scope': SCOPE, 'rows': [row]}}, SCOPE) is None
    assert task_context({'_agent_task_context': snapshot}, AgentScope('other', SCOPE.session)) is None


@pytest.mark.parametrize('materials,small_store', [(False, False), (True, False), (True, True)])
def test_independent_angelos_keeps_its_question_after_service_recreation(tmp_path, materials, small_store):
    from test_agent_recovery import service_at, Runtime, close as close_service, settle, handle
    from shinku.agent_runtime import AgentTurnRequest
    service, _, _, runtime = service_at(tmp_path, Runtime([{'kind': 'wait', 'text': OPTIONS}]), materials=materials)
    try:
        port = ServiceHost(service, requester='owner').port('continuity')
        h = port.submit(AgentTurnRequest('original', '先给可选方案'))
        first = settle(port, h.task_id)
        assert first['status'] == 'waiting_user'
        original_hash = port.result(h.task_id)['result_hash']
    finally:
        close_service(service)
    # 整个Service和SQLite连接重建，不能依赖内存checkpoint或Bot历史。
    service, _, tool, runtime = service_at(tmp_path, materials=materials)
    if small_store:
        service.materials.policy = replace(service.materials.policy, max_source_bytes=128)
    def respond(**kwargs):
        runtime.calls.append(kwargs)
        assert OPTIONS in kwargs['user_prompt']
        assert 'previous_agent_report' in kwargs['user_prompt'] and original_hash in kwargs['user_prompt']
        return {'kind': 'final', 'text': '原选项仍在，当前只确认方案，未执行写入。'}
    runtime.call_chat_json = respond
    try:
        port = ServiceHost(service, requester='owner').port('continuity')
        port.resume(handle(port.status(h.task_id)), request_key='resume')
        assert settle(port, h.task_id)['status'] == 'completed'
        assert len(runtime.calls) == 1 and tool.calls == 0
        assert port.result(h.task_id)['budget']['requests'] == first['budget']['requests'] + 1
        if small_store:
            task = service._get(AgentScope('owner', 'angelos:continuity'), h.task_id)
            assert any(r['content'].get('text') == OPTIONS for r in task['source_data'])
            assert any(r['reason'] == 'material_source_oversized' for r in task['material_package']['gaps'])
    finally:
        close_service(service)


@pytest.mark.parametrize('value', ['true', 1, {'allowed_tools': ['write']}])
def test_resume_option_cannot_smuggle_authority_or_use_non_boolean(value):
    from shinku.qq.interaction_tools import task_tool_specs, validate_arguments
    spec = next(s['function']['parameters'] for s in task_tool_specs(delegated=True)
                if s['function']['name'] == 'agent_task_supplement')
    assert not validate_arguments({'task_id': 'fixture', 'resume': value}, spec)
    assert not validate_arguments({'task_id': 'fixture', 'resume': True, 'permission': 'ultra'}, spec)
    local = next(s['function']['parameters'] for s in task_tool_specs()
                 if s['function']['name'] == 'agent_task_supplement')
    assert not validate_arguments({'task_id': 'fixture', 'resume': True}, local)


def test_plain_chat_is_not_forced_to_resume_by_choice_keywords(remote, tmp_path):
    waiting_runtime(remote)
    _, tool, runtime, _, _, _ = remote
    bridge, chat, _, store, coordinator = bridge_for(tmp_path, remote, [submit()])
    try:
        bridge(turn('先说明文件工具方案'))
        before = len(runtime.calls)
        chat.replies.append({'kind': 'final', 'text': '继续讨论方案，不执行。'})
        reply = bridge(turn('B 继续是什么意思？先别操作。', message_id='discuss'))
        assert reply.final_text == '继续讨论方案，不执行。'
        assert OPTIONS in chat.calls[-1]['user_prompt']
        assert len(runtime.calls) == before and tool.calls == 0
    finally:
        close(store, coordinator)


def test_context_is_scoped_current_and_does_not_run_models(remote, tmp_path):
    waiting_runtime(remote)
    _, tool, runtime, _, client, _ = remote
    bridge, chat, _, store, coordinator = bridge_for(tmp_path, remote, [submit()])
    try:
        bridge(turn('查看可选方案'))
        task_id = bridge.delegation._rows(SCOPE)[0]['task_id']
        before = len(runtime.calls)
        view = client.port(SCOPE).context(task_id)
        assert view['report']['text'] == OPTIONS and view['report']['current']
        assert view['is_authority'] is False and view['report']['is_fact_evidence'] is False
        handle = TaskHandle(task_id, view['attempt'], view['version'])
        receipt = client.port(SCOPE).revise(handle, AgentSupplement('new-input', clarification='新的约束'))
        assert receipt['state'] == 'applied'
        after = client.port(SCOPE).context(task_id)
        assert after['input_binding'] != view['input_binding']
        assert after['report']['current'] is False and after['report']['text'] == OPTIONS
        assert after['requests'][-1]['text'] == '新的约束'
        assert len(runtime.calls) == before and tool.calls == 0
        with pytest.raises(AgentBoundaryError):
            client.port(AgentScope('stranger', SCOPE.session)).context(task_id)
    finally:
        close(store, coordinator)
