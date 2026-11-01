"""Complete scoped file reads must satisfy real contracts without weakening identity."""
from dataclasses import replace
import copy
import hashlib
import json
import time

import pytest

from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent_runtime import AgentBoundaryError, AgentScope, ExecutionGrant, ExecutionProfile
from shinku.agent_runtime.demo import demo_request, demo_service
from shinku.agent_runtime.materials import MaterialPolicy, MaterialSource, TaskMaterialSession, TaskMaterialStore
from shinku.tools.execution import ExecutionPolicy

SCOPE = AgentScope('read-fixture', 'private:read-fixture')


@pytest.fixture
def retained(tmp_path):
    clock = [1000.0]
    database = AgentTaskStore(tmp_path / 'tasks.sqlite', clock=lambda: clock[0])
    task = database.save(new_task(task_id='read-task', requester=SCOPE.requester, session=SCOPE.session,
        request='Read the attachment without changing it.', message_id='read-request', contract=None))
    task.update(status='running', execution_attempt=1)
    task = database.save(task)
    materials = TaskMaterialStore(database, approved_roots=(tmp_path,),
        policy=MaterialPolicy(inline_chars=8, page_chars=64))
    source = tmp_path / 'input.txt'
    source.write_text('a' * 80 + 'TAIL: offline only.', encoding='utf-8')
    task['material_package'] = materials.package(task, [MaterialSource('attachment', 'attachment',
        local_path=source)], authorization_ref='read-fixture-grant')
    task = database.save(task)
    session = TaskMaterialSession(materials, task, still_authorized=lambda: True)
    yield database, materials, session, source, clock
    database.close()


def read_all(retained, *, provided=True):
    _, materials, session, _, _ = retained
    item = session.prompt_package()['items'][0]
    result = []
    for offset in range(0, item['content_chars'], 64):
        result.append(materials.read(item['reference'], task=session._bound_task(),
            still_authorized=session.still_authorized, offset=offset, length=64, provided=provided))
    return result


def proof(retained, receipt):
    _, materials, session, _, _ = retained
    return materials.verified_file_read(receipt, task=session._bound_task(),
        still_authorized=session.still_authorized)


def test_paged_file_read_has_exact_current_origin_and_complete_evidence(retained):
    receipt = read_all(retained)[-1]
    value = proof(retained, receipt)
    path = retained[3]
    assert value['verified_effects'] == ['read'] and value['read_complete'] is True
    assert value['result_valid'] is True and value['completion_targets'] == [str(path.resolve())]
    assert value['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert value['bytes'] == len(path.read_bytes()) and value['file_text'] == path.read_text()
    assert value['material_read_ids'] == [r['read_id'] for r in retained[1].read_receipts(retained[2]._bound_task())]
    before = retained[1].limits(retained[2]._bound_task())
    assert proof(retained, receipt) == value
    assert retained[1].limits(retained[2]._bound_task()) == before


def test_unprovided_pages_do_not_attest_complete_read(retained):
    receipts = read_all(retained, provided=False)
    assert not proof(retained, receipts[-1])
    _, materials, session, _, _ = retained
    for receipt in receipts:
        materials.mark_provided(receipt, task=session._bound_task(), still_authorized=session.still_authorized)
    assert proof(retained, receipts[-1])['read_complete'] is True


def test_missing_page_cannot_be_hidden_by_tail_read(retained):
    _, materials, session, _, _ = retained
    reference = session.prompt_package()['items'][0]['reference']
    receipt = materials.read(reference, task=session._bound_task(), still_authorized=session.still_authorized,
                            offset=64, length=64)
    assert not proof(retained, receipt)


@pytest.mark.parametrize('field,value', [('reference', 'tm:' + '0' * 32), ('read_id', '0' * 64),
    ('content_sha256', '0' * 64), ('version', 'stale'), ('offset', 0), ('returned_chars', 100000),
    ('page_sha256', '0' * 64)])
def test_receipt_identity_cannot_be_forged(retained, field, value):
    receipt = read_all(retained)[-1]
    altered = {**receipt, field: value}
    if field == 'reference':
        with pytest.raises(AgentBoundaryError, match='material_reference_not_in_task'):
            proof(retained, altered)
    else:
        assert not proof(retained, altered)


@pytest.mark.parametrize('change', ['mutated', 'revoked', 'expired', 'canceled', 'attempt', 'permission', 'scope'])
def test_current_source_and_authority_are_rechecked(retained, change):
    database, materials, session, path, clock = retained
    receipt = read_all(retained)[-1]
    if change == 'mutated':
        path.write_text('changed', encoding='utf-8')
    elif change == 'revoked':
        materials.revoke(receipt['reference'], task=session._bound_task())
    elif change == 'expired':
        clock[0] += 4000
    elif change in {'canceled', 'attempt'}:
        current = database.get(session.task['task_id'], requester=SCOPE.requester, session=SCOPE.session)
        current['status'] = 'canceled' if change == 'canceled' else 'running'
        if change == 'attempt':
            current['execution_attempt'] += 1
        database.save(current)
    elif change == 'permission':
        session.still_authorized = lambda: False
    else:
        session.task['requester'] = 'different-owner'
    try:
        result = proof(retained, receipt)
    except AgentBoundaryError:
        result = {}
    assert not result


@pytest.mark.parametrize('valid_envelope', [False, True])
def test_stored_page_corruption_cannot_attest_a_read(retained, valid_envelope):
    receipt = read_all(retained)[-1]
    database = retained[0]
    with database._transaction() as db:
        original = json.loads(db.execute('SELECT payload FROM agent_task_material_reads WHERE read_id=?',
                                         (receipt['read_id'],)).fetchone()[0])
        original['content'] = 'corrupted'
        db.execute('UPDATE agent_task_material_reads SET payload=? WHERE read_id=?',
                   (retained[1].encode(original) if valid_envelope else json.dumps(original), receipt['read_id']))
    if valid_envelope:
        assert not proof(retained, receipt)
    else:
        with pytest.raises(AgentBoundaryError, match='material_integrity_failed'):
            proof(retained, receipt)


@pytest.mark.parametrize('origin', ['memory', 'incomplete', 'redacted'])
def test_only_complete_unmodified_local_file_representation_attests_read(retained, origin):
    database, materials, session, source, _ = retained
    if origin == 'redacted':
        source.write_text('api_key=' + 'fixture-secret-' * 8, encoding='utf-8')
    material = (MaterialSource('other', 'document', 'a' * 80) if origin == 'memory' else
                MaterialSource('other', 'attachment', local_path=source, complete=origin != 'incomplete'))
    session.task['material_package'] = materials.package(session._bound_task(), [material], authorization_ref='read-fixture')
    session.task = database.save(session.task)
    receipt = read_all(retained)[-1]
    assert not proof(retained, receipt)


def test_real_task_accepts_full_attachment_read_and_single_verified_write(tmp_path):
    service, writer, _ = demo_service(tmp_path)
    source = tmp_path / 'attachment.txt'
    content = 'a' * 80 + 'TAIL: preserve source, offline only.'
    source.write_text(content, encoding='utf-8')
    service.materials = TaskMaterialStore(service.task_store, approved_roots=(tmp_path,),
        policy=MaterialPolicy(inline_chars=8, page_chars=64))
    grant = ExecutionGrant('read-completion-grant', frozenset({writer.tool_type}),
        policy=ExecutionPolicy(risk_by_tool={writer.tool_type: 'low'}), permission_mode='ultra',
        budget_limits={'max_requests': 16, 'max_tokens': 600000, 'max_tools': 12, 'max_active_seconds': 30},
        still_authorized=lambda: True, resume_safe=True,
        material_sources=(MaterialSource('attachment', 'attachment', local_path=source),))
    service.authorize = lambda scope, task, operation: grant
    request = demo_request(writer, key='read-and-write')
    contract = copy.deepcopy(request.contract)
    contract['requirements'].insert(0, dict(id='read', description='Read all original attachment bytes unchanged',
        effect='read', targets=[str(source)], checks=dict(unchanged=True, contains='TAIL: preserve source')))
    request = replace(request, contract=contract)
    class Runtime:
        calls = 0
        def chat_supports_native_tools(self):
            return True
        def call_chat_json(self, **kwargs):
            self.calls += 1
            data = json.loads(kwargs['user_prompt'].split('\n', 1)[1].split('\n[Task materials', 1)[0])
            package = data['task_materials']
            reference = package['items'][0]['reference']
            receipts = service.materials.read_receipts(dict(task_id=package['task_id'],
                requester=SCOPE.requester, session=SCOPE.session))
            offset = max((r['offset'] + r['returned_chars'] for r in receipts if r.get('provided')), default=0)
            if offset < len(content):
                return {'tool_call': dict(type='task_material_read', reference=reference, offset=offset, length=64)}
            if not writer.calls:
                return {'tool_call': dict(type=writer.tool_type, path=str(writer.path))}
            return dict(kind='final', text='The original attachment was read and preserved; the output was verified.',
                        task_outcome='completed')
    runtime = Runtime()
    service.executor.profile = ExecutionProfile(runtime, max_rounds=8)
    try:
        client = service.bind(SCOPE)
        handle = client.submit(request)
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            view = client.status(handle.task_id)
            if not view['worker_active'] and view['status'] not in {'running', 'queued'}:
                break
            time.sleep(.005)
        result = client.result(handle.task_id)
        assert view['status'] == 'completed' and result['verification']['accepted'] is True, result
        assert writer.calls == 1 and writer.path.read_text() == 'standalone agent verified'
        assert source.read_text() == content and result['information']['required_status'] == 'complete'
        saved = service.task_store.get(handle.task_id, requester=SCOPE.requester, session=SCOPE.session)
        reads = [c for c in saved['tool_calls'] if c['name'] == 'task_material_read'
                 and c['status'] == 'ok' and c.get('executed') is not False]
        assert sum(c['data'].get('state_updates', {}).get('read_complete') is True for c in reads) == 1, reads
        assert len(reads[-1]['data']['state_updates']['material_read_ids']) == 2
        assert all('file_text' not in c['data']['state_updates'] for c in reads)
        before = runtime.calls, writer.calls, service.materials.read_receipts(dict(task_id=handle.task_id,
            requester=SCOPE.requester, session=SCOPE.session))
        assert client.submit(request).task_id == handle.task_id
        client.status(handle.task_id)
        client.result(handle.task_id)
        assert before == (runtime.calls, writer.calls, service.materials.read_receipts(dict(task_id=handle.task_id,
            requester=SCOPE.requester, session=SCOPE.session)))
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()
