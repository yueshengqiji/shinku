"""当前会话真实记录工具与安全边界；全部使用合成资料。

工具观察凭据只证明作用域内的当前读取，不宣称证明真实模型理解质量。
真实SDK测试使用httpx.MockTransport，不访问付费供应商。
"""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from shinku.agent.task_state import new_task
from shinku.agent_runtime import AgentRequest, AgentScope
from shinku.agent_runtime.context import execution_context
from shinku.agent_runtime.demo import demo_service
from shinku.angelos.host import PreparationFailure, ServiceHost
from shinku.angelos.store import WorkbenchStore
from shinku.llm.response_protocol import note_incomplete_response


def contract():
    return {'goal': '读取本会话历史错误并说明原因与证据缺口', 'deliverables': ['具体错误和诊断结果'],
        'scope': ['仅当前会话只读诊断，不重跑或修改旧任务'], 'requirements': [
            {'id': 'records', 'description': '实际读取当前会话任务记录',
             'effect': 'tool:read_task_records', 'targets': [], 'checks': {}}]}


def seed(service, key, *, session='angelos:one', requester='owner', status='failed', error='synthetic_missing_input'):
    task = new_task(task_id=key, requester=requester, session=session,
        request='合成项目诊断', message_id=key + '-input', contract=contract())
    task.update(status=status, execution_attempt=1, stop_reason=error,
        tool_calls=[{'id': 'old-call', 'name': 'synthetic_read', 'status': 'error', 'executed': False,
            'data': {'code': error, 'stderr': '缺少合成输入 sample.txt',
                'reasoning_content': 'DO_NOT_EXPORT_REASONING', 'api_key': 'DO_NOT_EXPORT_KEY'}}])
    return service.task_store.save(task)


@pytest.fixture
def records(tmp_path):
    service, file_tool, runtime = demo_service(tmp_path)
    host = ServiceHost(service, requester='owner')
    old = seed(service, 'old')
    current = seed(service, 'current', status='running')
    scope = AgentScope('owner', 'angelos:one')
    grant = service._grant(scope, current, 'inspect')
    ctx = execution_context(current, grant, attempt=1, work_id='current')
    yield service, host, ctx, old, file_tool
    service.coordinator.shutdown(wait=True)
    service.task_store.close()


def read(host, ctx, **args):
    return host.task_records.execute(call=args, context=ctx)


def test_records_list_detail_pagination_and_originals_unchanged(records):
    service, host, ctx, old, file_tool = records
    for i in range(15):
        seed(service, f'older-{i:02}')
    first = read(host, ctx, mode='list')['state_updates']
    second = read(host, ctx, mode='list', offset=first['next_offset'])['state_updates']
    ids = [r['task_id'] for r in json.loads(first['content']) + json.loads(second['content'])]
    assert len(ids) == len(set(ids)) == 16 and 'current' not in ids and 'old' in ids
    pages, offset = [], 0
    while offset is not None:
        data = read(host, ctx, mode='detail', task_id='old', offset=offset, length=150)['state_updates']
        pages.append(data['content']); offset = data['next_offset']
        assert not data['complete']
    value = json.loads(''.join(pages))
    assert value['stop_reason'] == 'synthetic_missing_input'
    assert value['tool_calls'][0]['data']['stderr'] == '缺少合成输入 sample.txt'
    assert 'DO_NOT_EXPORT' not in ''.join(pages)
    assert host.task_records.agent_verify_result(receipt=data, context=ctx)['result_valid']
    assert service.task_store.get('old', requester='owner', session='angelos:one') == old
    assert file_tool.calls == 0


@pytest.mark.parametrize('other', [('other', 'angelos:one'), ('owner', 'angelos:two')])
def test_no_cross_requester_or_session_read_even_if_task_id_known(records, other):
    service, host, ctx, old, _ = records
    seed(service, 'foreign', requester=other[0], session=other[1], error='FOREIGN_SECRET')
    detail = read(host, ctx, mode='detail', task_id='foreign')
    assert detail['status'] == 'error' and 'FOREIGN_SECRET' not in str(detail)
    assert detail['error'] == 'task_record_not_found_or_expired'
    assert 'foreign' not in read(host, ctx, mode='list')['followup_context']


@pytest.mark.parametrize('mutation', ['receipt', 'query', 'version', 'scope', 'revoked', 'changed', 'expired', 'cancel', 'message', 'authority'])
def test_observation_cannot_be_forged_reused_or_survive_changed_records(records, mutation):
    service, host, ctx, old, _ = records
    data = read(host, ctx, mode='detail', task_id='old')['state_updates']
    assert host.task_records.agent_verify_result(receipt=data, context=ctx)['result_valid']
    if mutation == 'receipt': data['observation']['self_state_receipt']['read_id'] = 'forged'
    if mutation == 'query': data['record_query']['offset'] = 10
    if mutation == 'version': data['record_version'] = 'forged'
    if mutation == 'scope':
        from shinku.agent.request_scope import RequestScope
        ctx['agent_request_scope'] = RequestScope('other', 'angelos:one', 'current-input', (('current-input', '合成项目诊断'),))
    if mutation == 'message':
        from shinku.agent.request_scope import RequestScope
        ctx['agent_request_scope'] = RequestScope('owner', 'angelos:one', 'other-input', (('other-input', '其他请求'),))
    if mutation == 'authority':
        authorize = service.authorize
        service.authorize = lambda *args: replace(authorize(*args), authorization_ref='changed-authority')
    if mutation == 'revoked': service.authorize = lambda *args: None
    if mutation == 'changed': service.task_store.save({**old, 'stop_reason': 'changed'})
    if mutation == 'expired': host.task_records.observations.store.clock = lambda: data['observation']['self_state_receipt']['observed_at'] + 181
    if mutation == 'cancel':
        current = service.task_store.get('current', requester='owner', session='angelos:one')
        service.task_store.save({**current, 'cancel_requested': True})
    assert not host.task_records.agent_verify_result(receipt=data, context=ctx)['result_valid']


def test_expired_diagnostic_read_does_not_prune_old_evidence(records):
    service, host, ctx, old, _ = records
    # 用SQL只设置合成旧记录时间；诊断不调用会清理旧原件的get/candidates。
    with service.task_store._transaction() as db:
        db.execute('UPDATE agent_conversation_tasks SET updated=? WHERE task_id=?',
                   (service.task_store.clock() - service.task_store.ttl_seconds - 1, 'old'))
    value = read(host, ctx, mode='detail', task_id='old')
    assert value['error'] == 'task_record_not_found_or_expired'
    with service.task_store._transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM agent_conversation_tasks WHERE task_id=?', ('old',)).fetchone()[0] == 1


@pytest.mark.parametrize('args', [
    {'mode': 'detail', 'task_id': 'old', 'session': 'angelos:other'},
    {'mode': 'list', 'path': '/arbitrary'}, {'mode': 'list', 'sql': 'SELECT *'},
    {'mode': 'list', 'offset': True}, {'mode': 'detail', 'task_id': 'old', 'length': 4001},
    {'mode': 'input', 'task_id': 'old'}, {'mode': 'list', 'offset': 999999},
    {'mode': []},
    {'mode': 'detail', 'task_id': 'old', 'offset': 100000},
])
def test_query_arguments_never_select_authority_or_unbounded_page(records, args):
    _, host, ctx, _, _ = records
    assert read(host, ctx, **args)['status'] == 'error'


def test_preparation_failures_without_task_id_have_scoped_retrieval(records, tmp_path):
    _, host, ctx, _, _ = records
    store = WorkbenchStore(tmp_path / 'workbench.sqlite3')
    c = store.conversation('default')
    # 指向当前受控会话，只为合成记录建立ID，不给模型选择会话的入口。
    with store.transaction() as db:
        store.put(db, 'conversation', {**c, 'id': 'one', 'turns': [
            {'id': 'bad-input', 'text': '合成原问题', 'stage': 'failed', 'error': 'contract_preparation_rejected',
             'preparation_failure': {'text': 'RAW_SYNTHETIC_DRAFT', 'detail': {'field_path': 'requirements'}}}]})
    host.bind_workbench(store)
    index = read(host, ctx, mode='inputs')['state_updates']
    assert json.loads(index['content'])[0]['id'] == 'bad-input'
    data = read(host, ctx, mode='input', turn_id='bad-input')['state_updates']
    assert 'RAW_SYNTHETIC_DRAFT' in data['content'] and 'requirements' in data['content']
    assert host.task_records.agent_verify_result(receipt=data, context=ctx)['result_valid']
    assert read(host, ctx, mode='input', turn_id='foreign')['status'] == 'error'


@pytest.mark.parametrize('blocked', [True, False])
def test_explicit_block_or_allowlist_is_preserved(tmp_path, blocked):
    service, tool, _ = demo_service(tmp_path)
    original = service.authorize
    def authorize(*args):
        grant = original(*args)
        return replace(grant, policy=replace(grant.policy,
            blocked=frozenset({'read_task_records'}) if blocked else frozenset(),
            allowed=frozenset() if blocked else frozenset({'demo_file'})))
    service.authorize = authorize
    host = ServiceHost(service, requester='owner')
    try:
        current = seed(service, 'current', status='running')
        grant = service._grant(AgentScope('owner', 'angelos:one'), current, 'inspect')
        assert 'read_task_records' not in grant.allowed_tools
        ctx = execution_context(current, grant, attempt=1)
        assert read(host, ctx, mode='list')['status'] == 'error'
        assert grant.allowed_tools == {'demo_file'} and tool.calls == 0
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()


def test_legacy_preparer_reply_remains_single_call_without_route_review(tmp_path, monkeypatch):
    """旧显式prepare API兼容，但移除被拒绝的第二次模型核对。"""
    service, tool, runtime = demo_service(tmp_path)
    calls = []
    def respond(**kw):
        calls.append(kw)
        return {'reply': '合成解释'}
    monkeypatch.setattr(runtime, 'call_chat_json', respond)
    try:
        prepared = ServiceHost(service).prepare('普通解释', [], {})
        assert prepared['reply'] == '合成解释'
        assert len(calls) == prepared['preparation_usage']['requests'] == 1
        assert 'preparation_review' not in prepared and tool.calls == 0
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()
