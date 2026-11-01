"""可控后台摘要：只用合成生成器，验证幂等、预算、过期和撤权，不调用供应商。"""
import copy
from dataclasses import replace
import threading
import time

import pytest

from shinku.agent.task_state import AgentTaskStore
from shinku.agent_runtime import AgentBoundaryError, AgentScope
from shinku.agent_runtime.results import seal_result
from shinku.agent_runtime.summary_worker import SummaryWorker
from shinku.agent_runtime.delegation import DelegationPrincipal
from test_delivery_brief import result_with_brief, BRIEF, REPORT
from test_agent_delegation import remote, SCOPE, SECRET
from test_task_conversation_handoff import start


@pytest.fixture
def setup(tmp_path):
    result = result_with_brief(); result.pop('result_hash'); result.pop('delivery_brief')
    live = {'result': seal_result(result), 'allowed': True}
    store = AgentTaskStore(tmp_path/'summary.sqlite3')
    def load(scope, task_id):
        if not live['allowed']:
            raise AgentBoundaryError('host_authorization_denied')
        return copy.deepcopy(live['result'])
    yield store, load, live
    store.close()


def args(live):
    result = live['result']
    return AgentScope(**result['scope']), result['task_id'], result['result_hash']


def settled(worker, arguments):
    deadline = time.monotonic() + 3
    while True:
        result = worker.read(*arguments)
        if result['status'] != 'pending': return result
        assert time.monotonic() < deadline
        time.sleep(.005)


def test_disabled_worker_does_not_call_model_or_change_task(setup):
    store, load, live = setup
    worker = SummaryWorker(store, load)
    before = copy.deepcopy(live)
    assert worker.request(*args(live))['reason'] == 'summary_not_configured'
    assert worker.read(*args(live))['status'] == 'not_requested' and live == before


def test_background_has_context_is_nonblocking_idempotent_and_persisted(setup):
    store, load, live = setup
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []
    def generate(packet, *, deadline, read_details):
        calls.append(packet); entered.set()
        assert release.wait(3)
        assert read_details()['execution_report']['text'] == REPORT
        finished.set()
        return {'mode': 'summary', 'text': BRIEF}
    worker = SummaryWorker(store, load, generate=generate, admit=lambda *a, **k: True)
    arguments = args(live); before = copy.deepcopy(live)
    assert worker.request(*arguments)['status'] == 'pending'
    assert entered.wait(2)
    try:
        assert worker.request(*arguments)['status'] == 'pending'
        assert calls[0]['original_requests'][-1]['text'] == '只改参数检查，先不部署'
        assert calls[0]['execution']['status'] == 'partial'
        assert calls[0]['execution_report']['text'] == REPORT
        assert 'delivery_brief' not in calls[0]
    finally:
        release.set()
    row = settled(worker, arguments)
    assert row['status'] == 'ready' and row['brief']['text'] == BRIEF
    assert worker.request(*arguments) == row and len(calls) == 1 and live == before
    restored = SummaryWorker(store, load)
    assert restored.read(*arguments) == row
    live['allowed'] = False
    with pytest.raises(AgentBoundaryError, match='authorization_denied'):
        restored.read(*arguments)


@pytest.mark.parametrize('failure', ['invalid', 'exception', 'budget', 'timeout', 'revoked', 'changed'])
def test_failures_do_not_mutate_execution_or_accept_late_summary(setup, failure):
    store, load, live = setup
    now = [100.]; calls = []
    def generate(packet, **kwargs):
        calls.append(packet)
        if failure == 'exception': raise RuntimeError('synthetic')
        if failure == 'timeout': now[0] += 31
        if failure == 'revoked': live['allowed'] = False
        if failure == 'changed':
            changed = copy.deepcopy(live['result']); changed.pop('result_hash'); changed['version'] += 1
            live['result'] = seal_result(changed)
        return {} if failure == 'invalid' else {'mode': 'summary', 'text': BRIEF}
    worker = SummaryWorker(store, load, generate=generate, admit=lambda *a, **k: failure != 'budget', clock=lambda: now[0])
    arguments = args(live)
    worker.request(*arguments)
    # 新状态不可访问旧摘要；仅直接读测试缓存核对后台实际落库结果。
    key = worker._key(*arguments); deadline = time.monotonic() + 3
    while (row := worker._read(key))['status'] == 'pending':
        assert time.monotonic() < deadline
        time.sleep(.005)
    assert row['status'] == 'fallback' and 'brief' not in row
    assert live['result']['execution']['status'] == 'partial'
    assert live['result']['execution_report']['text'] == REPORT
    if failure == 'budget': assert calls == []
    if failure in ('revoked', 'changed'):
        with pytest.raises(AgentBoundaryError): worker.read(*arguments)


def test_scope_oversize_and_completed_same_turn_brief_never_start_generation(setup):
    store, load, live = setup
    calls = []
    worker = SummaryWorker(store, load, generate=lambda *a, **k: calls.append(a),
        admit=lambda *a, **k: True, max_input_chars=10)
    assert worker.request(*args(live))['reason'] == 'summary_context_oversized'
    scope, task, digest = args(live)
    with pytest.raises(AgentBoundaryError, match='scope_mismatch'):
        worker.request(AgentScope('other', scope.session), task, digest)
    live['result'] = result_with_brief()
    result = worker.request(*args(live))
    assert result['status'] == 'ready' and result['source'] == 'execution_model' and not calls
    assert worker.read(*args(live)) == result


def test_corrupt_summary_cache_falls_back_without_regeneration(setup):
    store, load, live = setup
    calls = []
    def generate(*a, **k):
        calls.append(1)
        return {'mode': 'summary', 'text': BRIEF}
    worker = SummaryWorker(store, load, generate=generate, admit=lambda *a, **k: True)
    arguments = args(live)
    worker.request(*arguments)
    assert settled(worker, arguments)['status'] == 'ready'
    with store._transaction() as db:
        db.execute("UPDATE agent_delivery_summaries SET payload=replace(payload, ?, ?)", (BRIEF, '伪造的完成声明'))
    assert worker.read(*arguments)['reason'] == 'summary_cache_invalid'
    assert worker.request(*arguments)['reason'] == 'summary_cache_invalid' and len(calls) == 1


def test_restart_pending_timeout_is_bounded_and_does_not_retry(setup):
    store, load, live = setup
    worker = SummaryWorker(store, load, clock=lambda: 100.)
    arguments = args(live); key = worker._key(*arguments)
    with store._transaction() as db:
        db.execute('INSERT INTO agent_delivery_summaries VALUES (?,?)', (key, worker._dump(
            {'cache_key': key, 'status': 'pending', 'deadline': 99., 'source_result_hash': arguments[-1]})))
    assert worker.read(*arguments)['reason'] == 'summary_timeout'
    assert worker.request(*arguments)['reason'] == 'summary_timeout'


def test_timeout_does_not_spawn_replacements_for_a_generator_that_is_still_running(setup):
    store, load, live = setup
    entered, release = threading.Event(), threading.Event()
    now, calls = [100.], []
    def generate(*a, **k):
        calls.append(1); entered.set()
        assert release.wait(3)
        return {'mode': 'summary', 'text': BRIEF}
    worker = SummaryWorker(store, load, generate=generate, admit=lambda *a, **k: True,
                           max_workers=1, clock=lambda: now[0])
    first = args(live)
    worker.request(*first)
    assert entered.wait(2)
    try:
        now[0] += 31
        assert worker.read(*first)['reason'] == 'summary_timeout'
        assert worker.request(*first)['reason'] == 'summary_timeout'
        changed = copy.deepcopy(live['result']); changed.pop('result_hash'); changed['version'] += 1
        live['result'] = seal_result(changed)
        assert worker.request(*args(live))['reason'] == 'summary_worker_busy'
        assert len(calls) == 1
    finally:
        release.set()
        assert worker.slots.acquire(timeout=2)
        worker.slots.release()
    assert worker._read(worker._key(*first))['reason'] == 'summary_timeout'


def test_http_explicit_request_disabled_by_default_and_scope_checked(remote):
    handle, original = start(remote)
    service, tool, runtime, gateway, client, base = remote
    port = client.port(SCOPE); before = len(runtime.calls), tool.calls
    assert port.request_summary(original)['reason'] == 'summary_not_configured'
    assert port.summary(original)['status'] == 'not_requested'
    assert before == (len(runtime.calls), tool.calls)
    with pytest.raises(AgentBoundaryError):
        client.port(AgentScope('other', SCOPE.session)).request_summary(original)
    gateway.principals['shinku'] = DelegationPrincipal(SECRET,
        lambda *a: replace(base, still_authorized=lambda: False))
    with pytest.raises(AgentBoundaryError, match='authorization_denied'):
        port.summary(original)


def test_http_background_summary_does_not_change_default_handoff_or_run_tools(remote):
    handle, original = start(remote)
    service, tool, runtime, gateway, client, base = remote
    generated = []
    def generate(packet, **kwargs):
        generated.append(packet)
        assert kwargs['read_details']() == original
        return {'mode': 'summary', 'text': '示例文件已创建。'}
    gateway.summaries = SummaryWorker(service.task_store, gateway._summary_original,
        generate=generate, admit=lambda *a, **k: True)
    port = client.port(SCOPE); delivered = port.handoff(handle.task_id)
    before = len(runtime.calls), tool.calls
    port.request_summary(delivered)
    deadline = time.monotonic()+3
    while (result := port.summary(delivered))['status'] == 'pending':
        assert time.monotonic() < deadline
        time.sleep(.005)
    assert result['status'] == 'ready' and len(generated) == 1
    assert port.request_summary(delivered) == result
    assert port.handoff(handle.task_id) == delivered and port.details(delivered) == original
    assert before == (len(runtime.calls), tool.calls)


@pytest.mark.parametrize('change', ['revoke', 'revision', 'material'])
def test_http_worker_rechecks_authority_revision_and_required_source_after_generation(remote, change):
    from shinku.agent_runtime import AgentTurnRequest, AgentSupplement, TaskHandle
    from shinku.agent_runtime.materials import MaterialSource
    from test_agent_delegation import settled as settle_task
    service, tool, runtime, gateway, client, base = remote
    port = client.port(SCOPE)
    handle = port.submit(AgentTurnRequest('summary-bound', '创建示例文件并验证'), materials=(
        MaterialSource('input', 'document', '合成必需资料', {'relation': 'test'}, required=True),))
    original = settle_task(client, handle.task_id)
    entered, release = threading.Event(), threading.Event()
    def generate(packet, **kwargs):
        entered.set()
        assert release.wait(3)
        return {'mode': 'summary', 'text': '不能交付的过期摘要'}
    gateway.summaries = SummaryWorker(service.task_store, gateway._summary_original,
        generate=generate, admit=lambda *a, **k: True)
    worker = gateway.summaries
    port.request_summary(original)
    assert entered.wait(2)
    try:
        if change == 'revoke':
            gateway.principals['shinku'] = DelegationPrincipal(SECRET,
                lambda *a: replace(base, still_authorized=lambda: False))
        elif change == 'revision':
            view = port.status(handle.task_id)
            port.revise(TaskHandle(handle.task_id, view['attempt'], view['version']),
                        AgentSupplement('new-goal', clarification='现在只解释，不要继续写入'))
        else:
            task = service._get(SCOPE, handle.task_id)
            source = next(r for r in task['material_package']['items'] if r['source_id'] == 'input')
            service.materials.revoke(source['reference'], task=task)
    finally:
        release.set()
    key = worker._key(SCOPE, handle.task_id, original['result_hash']); deadline = time.monotonic()+3
    while (row := worker._read(key))['status'] == 'pending':
        assert time.monotonic() < deadline
        time.sleep(.005)
    assert row['status'] == 'fallback' and 'brief' not in row
    with pytest.raises(AgentBoundaryError):
        port.summary(original)
    assert tool.calls == 1
