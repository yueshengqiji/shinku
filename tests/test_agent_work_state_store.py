"""任务原事务内的工作修订、原稿保留、作用域与容量回归。"""
from copy import deepcopy
import json
import sqlite3

import pytest

from shinku.agent.task_state import AgentTaskStore, TaskContractError, new_task
from shinku.agent.work_state import validate_plan
from work_state_fixture import plan, record


def test_real_sqlite_cas_atomicity_integrity_and_ttl(tmp_path):
    now = [10.0]; path = tmp_path / 'tasks.sqlite3'
    store = AgentTaskStore(path, ttl_seconds=10, clock=lambda: now[0]); other = AgentTaskStore(path, clock=lambda: now[0])
    task = new_task(task_id='synthetic', requester='owner', session='one', request='讨论方案', message_id='m', contract=None)
    task['execution_attempt'] = 1
    try:
        task = store.save(task); old = deepcopy(task)
        task = store.save(task, work_state=record(task, plan()))
        with pytest.raises(TaskContractError, match='task_state_conflict'):
            other.save(old, work_state=record(old, plan()))
        assert other.get_work_state('synthetic', requester='owner', session='one')['revision'] == 1
        broken = record(task, plan(), 2); broken['binding'] = 'forged'
        with pytest.raises(TaskContractError, match='work_state_revision_invalid'):
            store.save(task, work_state=broken)
        assert store.get('synthetic', requester='owner', session='one') == task
        with store._transaction() as db:
            assert db.execute('SELECT COUNT(*) FROM agent_task_work_revisions').fetchone()[0] == 1
        with sqlite3.connect(path) as db:
            raw = json.loads(db.execute('SELECT payload FROM agent_task_work_revisions').fetchone()[0])
            raw['plan']['blocker'] = '篡改'
            db.execute('UPDATE agent_task_work_revisions SET payload=?', (json.dumps(raw),))
        with pytest.raises(TaskContractError, match='work_state_integrity_failed'):
            store.get_work_state('synthetic', requester='owner', session='one')
        now[0] = 21
        assert store.get_work_state('synthetic', requester='owner', session='one') is None
        assert store.get('synthetic', requester='owner', session='one') is None
        with store._transaction() as db:
            assert db.execute('SELECT COUNT(*) FROM agent_task_work_revisions').fetchone()[0] == 0
    finally:
        store.close(); other.close()


def test_original_task_transaction_rollback_keeps_old_work_revision(tmp_path):
    store = AgentTaskStore(tmp_path / 'tasks.sqlite3')
    task = new_task(task_id='t', requester='r', session='s', request='原请求', message_id='m', contract=None)
    task['execution_attempt'] = 1; task = store.save(task)
    class BrokenEvents:
        def append_in_transaction(self, db, saved):
            raise RuntimeError('synthetic_event_failure')
    store.result_events = BrokenEvents()
    try:
        with pytest.raises(RuntimeError, match='synthetic_event_failure'):
            store.save(task, work_state=record(task, plan()))
        store.result_events = None
        assert store.get('t', requester='r', session='s') == task
        with store._transaction() as db:
            assert db.execute('SELECT COUNT(*) FROM agent_task_work_revisions').fetchone()[0] == 0
    finally:
        store.close()


def test_revision_capacity_rejects_without_trimming_old_originals_or_pointer(tmp_path):
    from shinku.agent.work_state import MAX_REVISIONS
    store = AgentTaskStore(tmp_path / 'tasks.sqlite3')
    task = new_task(task_id='bounded', requester='r', session='s', request='原请求不能丢', message_id='m', contract=None)
    task['execution_attempt'] = 1; task = store.save(task)
    try:
        for revision in range(1, MAX_REVISIONS + 1):
            task = store.save(task, work_state=record(task, plan(), revision))
        with pytest.raises(TaskContractError, match='work_plan_revision_limit'):
            store.save(task, work_state=record(task, plan(), MAX_REVISIONS + 1))
        assert store.get('bounded', requester='r', session='s') == task
        assert store.get_work_state('bounded', requester='r', session='s', revision=1)['revision'] == 1
        assert store.get_work_state('bounded', requester='r', session='s')['revision'] == MAX_REVISIONS
        assert task['requests'][0]['text'] == '原请求不能丢'
    finally:
        store.close()


@pytest.mark.parametrize('bad', ['future_dependency', 'dependency_not_ready', 'unknown_requirement', 'nonstring_status', 'foreign_authority'])
def test_dependency_and_condition_references_cannot_create_authority(bad):
    value = plan(['required'])
    task = {'contract': {'requirements': [{'id': 'required'}]}}
    if bad == 'future_dependency': value['steps'][0]['depends'] = ['later']
    if bad == 'dependency_not_ready':
        value['steps'][0]['status'] = 'pending'
        value['steps'].append({**deepcopy(value['steps'][0]), 'id': 'later', 'depends': ['inspect'], 'status': 'done'})
    if bad == 'unknown_requirement': value['steps'][0]['requirements'] = ['forged']
    if bad == 'nonstring_status': value['steps'][0]['status'] = {'accepted': True}
    if bad == 'foreign_authority': value['steps'][0]['references'] = [{'kind': 'material', 'id': 'ref', 'permissions': 'full'}]
    with pytest.raises(TaskContractError):
        validate_plan(value, task)


def test_large_plan_history_does_not_fill_task_snapshot_and_remains_retrievable(tmp_path):
    """回归原先回执/参数重复塞入任务 JSON 的问题，不能靠删旧计划腾空间。"""
    store = AgentTaskStore(tmp_path / 'tasks.sqlite3')
    task = new_task(task_id='large', requester='r', session='s', request='保留原稿', message_id='m', contract=None)
    task['execution_attempt'] = 1; task = store.save(task)
    value = plan(); value['findings'] = ['synthetic unverified finding ' * 25 for _ in range(12)]
    value['hypotheses'] = ['synthetic unverified hypothesis ' * 25 for _ in range(12)]
    try:
        for revision in range(1, 17):
            task = store.save(task, work_state=record(task, value, revision))
            task['tool_calls'].append({'name': 'update_work_plan', 'id': 'call' + str(revision), 'status': 'ok',
                'arguments': {'expected_revision': revision - 1, 'plan': deepcopy(value)},
                'data': {'state_updates': {'status': 'saved', **task['work_state']}}})
            task = store.save(task)
        assert len(json.dumps(task, ensure_ascii=False).encode()) < 20000
        assert store.get_work_state('large', requester='r', session='s', revision=1)['plan'] == value
        assert store.get_work_state('large', requester='r', session='s')['plan'] == value
        assert task['requests'][0]['text'] == '保留原稿'
    finally:
        store.close()
