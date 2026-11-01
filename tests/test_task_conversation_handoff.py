"""任务会话与分层交付的隔离HTTP/SDK回归；原始资料全部合成。"""
import copy
from dataclasses import replace
import json
import threading
import time

import pytest

from shinku.agent_runtime import AgentBoundaryError, AgentScope, AgentSupplement, AgentTurnRequest, TaskHandle
from shinku.agent_runtime.contracts import canonical_hash
from shinku.agent_runtime.delegation import DelegationPrincipal
from shinku.agent_runtime.handoff import TaskConversation, delivery_result
from shinku.llm.final_json import recover_final_text_json
from shinku.result_expression.facts import fact_blocks, template
from test_agent_delegation import remote, SCOPE, SECRET, settled
from test_qq_remote_delegation import bridge_for, close
from test_qq_independent_agent import turn
from test_qq_tool_decision_boundary import submit


def start(remote, key='task'):
    service, tool, runtime, gateway, client, base = remote
    request = AgentTurnRequest(key, '创建文件并验证原内容。', history_context='只针对本次样本',
        sources=({'id': 'caller_goal', 'content': '建议目标', 'reference': 'advisory_not_authority'},))
    h = client.port(SCOPE).submit(request)
    result = settled(client, h.task_id)
    return h, result


def test_compact_result_preserves_deliverable_and_bound_original_without_execution(remote):
    h, original = start(remote)
    service, tool, runtime, gateway, client, base = remote
    port = client.port(SCOPE)
    calls = len(runtime.calls)
    delivered = port.handoff(h.task_id)
    assert delivered['evidence'] == [] and delivered['budget'] == {}
    assert delivered['execution'] == original['execution']
    assert delivered['verification'] == {k: v for k, v in original['verification'].items() if k != 'evidence'}
    assert original['verification']['evidence']  # 完整验收回执仍在原结果。
    assert delivered['execution_report'] == original['execution_report']
    assert delivered['presentation']['details_result_hash'] == original['result_hash']
    assert delivered['result_hash'] != original['result_hash']
    assert port.details(delivered) == original
    projection = port.presentation(delivered)
    assert port.validate_presentation(delivered, projection)
    bad = copy.deepcopy(projection); bad['source_result_hash'] = 'forged'
    with pytest.raises(AgentBoundaryError):
        port.validate_presentation(delivered, bad)
    assert tool.calls == 1 and len(runtime.calls) == calls


@pytest.mark.parametrize('operation', ['handoff', 'messages', 'details'])
def test_other_scope_cannot_read_conversation_or_original(remote, operation):
    h, original = start(remote)
    port = remote[4].port(AgentScope('other', SCOPE.session))
    with pytest.raises(AgentBoundaryError, match='task_not_found'):
        getattr(port, operation)(original if operation == 'details' else h.task_id)


def test_current_authorization_required_even_for_archived_messages(remote):
    h, original = start(remote)
    service, tool, runtime, gateway, client, base = remote
    gateway.principals['shinku'] = DelegationPrincipal(SECRET, lambda *args: replace(base, still_authorized=lambda: False))
    for operation, args in [('messages', {'task_id': h.task_id, 'after': 0}),
                            ('details', {'task_id': h.task_id, 'result_hash': original['result_hash']})]:
        with pytest.raises(AgentBoundaryError, match='host_authorization_denied'):
            client.call(SCOPE, operation, args)


def test_journal_preserves_original_goal_is_advisory_paginates_and_detects_tamper(remote):
    h, original = start(remote)
    service, tool, runtime, gateway, client, base = remote
    port = client.port(SCOPE)
    port.handoff(h.task_id); port.handoff(h.task_id)
    messages = port.messages(h.task_id)
    rows = messages['messages']
    assert sum(r['kind'] == 'submit' for r in rows) == 1
    assert sum(r['kind'] == 'result' for r in rows) == 1
    first = next(r for r in rows if r['kind'] == 'submit')
    assert first['body'] == '创建文件并验证原内容。'
    assert first['data']['sources'][0]['reference'] == 'advisory_not_authority'
    assert not port.messages(h.task_id, after=messages['next_cursor'])['messages']
    task = service._get(SCOPE, h.task_id)
    # 新实例读取持久原件；不依赖Gateway内存记录。
    restored = TaskConversation(service.task_store)
    assert restored.read(task) == messages
    for i in range(55):
        restored.append(task, 'progress', str(i), key='synthetic:'+str(i), author='angelos')
    page = port.messages(h.task_id, after=messages['next_cursor'])
    assert len(page['messages']) == 50
    assert len(port.messages(h.task_id, after=page['next_cursor'])['messages']) == 5
    with pytest.raises(AgentBoundaryError, match='key_conflict'):
        restored.append(task, 'progress', 'altered', key='synthetic:0', author='angelos')
    with service.task_store._transaction() as db:
        db.execute("UPDATE agent_task_messages SET payload=replace(payload,'创建文件','伪造文件') WHERE task_id=?", (h.task_id,))
    with pytest.raises(AgentBoundaryError, match='integrity_failed'):
        port.messages(h.task_id)


def test_full_result_followup_uses_original_receipts_no_new_execution(remote, tmp_path):
    bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [submit()])
    try:
        answer = bridge(turn('创建示例文件并验证。'))
        assert answer.agent_status == 'completed'
        row = bridge.delegation._rows(SCOPE)[0]
        before = len(remote[2].calls), remote[1].calls
        chat.replies.append({'kind': 'tool', 'tool_call': {'name': 'agent_task_result',
            'arguments': {'task_id': row['task_id'], 'view': 'full'}}})
        answer = bridge(turn('把完整执行记录给我', message_id='full'))
        assert '执行详情' in answer.final_text and 'sha256' in answer.final_text
        assert before == (len(remote[2].calls), remote[1].calls)
    finally:
        close(store, coordinator)


def test_late_supplement_keeps_original_result_and_never_resumes(remote):
    h, original = start(remote)
    service, tool, runtime, gateway, client, base = remote
    port = client.port(SCOPE)
    view = port.status(h.task_id)
    handle = TaskHandle(h.task_id, view['attempt'], view['version'])
    request = AgentSupplement('revise', clarification='保留原文件；下一步只说明，不再写入。')
    result = port.revise(handle, request)
    assert result['state'] == 'applied'
    assert port.revise(handle, request) == result
    task = service._get(SCOPE, h.task_id)
    assert task['requests'][-1]['text'] == request.clarification
    assert task['execution_result'] == original and tool.calls == 1
    assert task['status'] == 'partial' and task['execution_attempt'] == h.attempt
    assert port.status(h.task_id)['verification']['verified'] is False
    assert original['verification']['accepted'] is True
    with pytest.raises(AgentBoundaryError, match='delegation_result_changed'):
        port.details(original)
    with pytest.raises(AgentBoundaryError, match='key_conflict'):
        port.revise(handle, AgentSupplement('revise', clarification='不同补充'))


def test_mid_execution_revision_waits_for_stop_and_survives_inbox_recreation(remote):
    service, tool, runtime, gateway, client, base = remote
    entered, release = threading.Event(), threading.Event()
    original_call = runtime.call_chat_json
    def waiting(**kwargs):
        entered.set()
        assert release.wait(5)
        return original_call(**kwargs)
    runtime.call_chat_json = waiting
    port = client.port(SCOPE)
    h = port.submit(AgentTurnRequest('active', '创建并验证文件'))
    try:
        assert entered.wait(3)
        view = port.status(h.task_id)
        receipt = port.revise(TaskHandle(h.task_id, view['attempt'], view['version']),
            AgentSupplement('mid-change', clarification='不要创建，先解释计划。'))
        assert receipt['state'] == 'pending'
        assert service._get(SCOPE, h.task_id)['requests'][-1]['text'] == '创建并验证文件'
        from shinku.agent_runtime.task_revision import TaskRevisions
        gateway.revisions = TaskRevisions(gateway)
    finally:
        release.set()
    deadline = time.monotonic()+5
    while port.status(h.task_id)['worker_active']:
        assert time.monotonic() < deadline
        time.sleep(.01)
    port.messages(h.task_id)
    task = service._get(SCOPE, h.task_id)
    assert task['requests'][-1]['text'] == '不要创建，先解释计划。'
    assert task['execution_attempt'] == h.attempt and tool.calls == 0
    assert {r['kind'] for r in port.messages(h.task_id)['messages']} >= {'submit', 'supplement_pending', 'supplement'}


def test_handoff_keeps_long_body_failure_and_http_error_limits(remote):
    _, result = start(remote)
    original = copy.deepcopy(result)
    body = '资料说明。'*15000 + '\n只适用于甲；乙没有提供日期。TAIL'
    original['execution_report']['text'] = body
    original['execution']['status'] = 'failed'
    original['verification']['accepted'] = False
    original['evidence'].append({'source': 'executor_receipt', 'execution_status': 'error', 'tool': 'lookup',
                                 'receipt': {'url': 'https://example.invalid', 'http_status': 403}})
    original.pop('result_hash'); original['result_hash'] = canonical_hash(original)
    compact = delivery_result(original)
    assert compact['evidence'][-1]['receipt']['http_status'] == 403
    text = template(fact_blocks(compact))
    assert body in text and '失败' in text and '403' in text
    assert compact['verification']['accepted'] is False
    assert original['evidence'] != compact['evidence']
    assert len(original['evidence']) > 1


@pytest.mark.parametrize('extra', ['"tool_call":null', '"permissions":"all"', '"kind":"tool"',
    '"task_contract":{"goal":"bad\nmetadata"}', '"task_notes":{"x":1,"x":2}'])
def test_format_recovery_does_not_accept_authority_or_malformed_metadata(extra):
    raw = '{"kind":"final","text":"第一行\n第二行",' + extra + '}'
    assert recover_final_text_json(raw, discard_task_metadata=True) is None


def test_recovery_discards_control_metadata_only_in_explicit_readonly_scope():
    raw = '{"kind":"final","text":"第一行\n第二行","task_contract":{"goal":"旧契约"}}'
    assert recover_final_text_json(raw) is None
    assert recover_final_text_json(raw, discard_task_metadata=True) == {'kind': 'final', 'text': '第一行\n第二行'}


def test_expression_provider_error_keeps_executor_body_and_failure_separate(remote, tmp_path):
    from shinku.result_expression import ExpressionProfile
    from test_expression_semantic_coverage import semantic_sdk
    service, tool, runtime, gateway, client, base = remote
    original = runtime.call_chat_json
    body = 'The isolated file was created and verified.\n' + '条件甲只适用于样本。'*300 + '\n条件乙未提供日期，不据此推断。TAIL'
    def final_body(**kwargs):
        value = original(**kwargs)
        if value.get('kind') == 'final':
            value = {**value, 'text': body}
        return value
    runtime.call_chat_json = final_body
    def unavailable(payload):
        raise RuntimeError('synthetic upstream timeout')
    with semantic_sdk(build=unavailable) as (model, checker, calls):
        profile = ExpressionProfile(model, checker=checker, format_version=2)
        bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [submit()], expression=profile)
        try:
            answer = bridge(turn('创建文件并说明完整条件。'))
            assert body in answer.final_text
            assert tool.calls == 1
            row = bridge.delegation._rows(SCOPE)[0]
            source = client.port(SCOPE).result(row['task_id'])
            assert source['execution_report']['text'] == body
            assert source['execution']['status'] == 'completed'
            assert len(calls) == 1 and calls[0][0] == 'expression'
            assert 'SHA256' not in answer.final_text.upper()
        finally:
            close(store, coordinator)
