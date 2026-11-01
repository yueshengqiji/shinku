"""P4a rules: a committed result survives crashes without repeating effects."""
import copy
import sqlite3
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.results import export_task_result
from shinku.agent_runtime.result_events import ResultEventStore

SCOPE = AgentScope("fixture", "private:fixture")


def terminal(store, *, task_id="fixture-task", status="completed", attempt=1):
    task = new_task(task_id=task_id, requester=SCOPE.requester, session=SCOPE.session,
        request="生成并核验隔离报告，不部署", message_id="fixture-message", contract=None)
    task.update(status=status, execution_attempt=attempt, completion_attempt=attempt,
        completion={"accepted": status == "completed"}, revision=0)
    task["execution_result"] = export_task_result({**task, "revision": 1, "updated": store.clock()})
    return task


def test_no_schema_migration_without_explicit_opt_in(tmp_path):
    store = AgentTaskStore(tmp_path / "ledger.db")
    store.save(terminal(store))
    with sqlite3.connect(store.path) as db:
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='agent_result_events'").fetchone()


def test_state_and_event_rollback_together(tmp_path, monkeypatch):
    store = AgentTaskStore(tmp_path / "ledger.db")
    events = ResultEventStore(store)
    original = events.append_in_transaction
    def fail(db, task):
        original(db, task)
        raise RuntimeError("fixture commit failure")
    monkeypatch.setattr(events, "append_in_transaction", fail)
    with pytest.raises(RuntimeError):
        store.save(terminal(store))
    assert store.get("fixture-task", requester=SCOPE.requester, session=SCOPE.session) is None
    assert events.pending(scope=SCOPE) == []


def test_committed_event_reopens_and_is_scoped_immutable_and_idempotent(tmp_path):
    path = tmp_path / "ledger.db"
    first = AgentTaskStore(path)
    events = ResultEventStore(first)
    task = first.save(terminal(first))
    event = events.pending(scope=SCOPE)[0]
    assert event["task_id"] == task["task_id"] and event["version"] == task["revision"]
    assert event["source_hash"] == task["execution_result"]["result_hash"]
    assert event["status"] == "pending"
    events.enqueue(task["execution_result"], scope=SCOPE)
    first.close()
    second = AgentTaskStore(path)
    recovered = ResultEventStore(second)
    assert recovered.pending(scope=SCOPE) == [event]
    assert recovered.get(event["event_id"], scope=AgentScope("other", SCOPE.session)) is None
    changed = copy.deepcopy(task["execution_result"])
    changed["execution"]["status"] = "failed"
    with pytest.raises(ValueError):
        recovered.enqueue(changed, scope=SCOPE)


def test_parallel_claim_is_single_owner_and_lease_fences_old_ack(tmp_path):
    clock = [1000.0]
    store = AgentTaskStore(tmp_path / "ledger.db", clock=lambda: clock[0])
    events = ResultEventStore(store, lease_seconds=10)
    store.save(terminal(store))
    event_id = events.pending(scope=SCOPE)[0]["event_id"]
    peers = [ResultEventStore(AgentTaskStore(store.path, clock=store.clock), lease_seconds=10) for _ in range(8)]
    with ThreadPoolExecutor(8) as pool:
        claims = list(pool.map(lambda peer: peer.claim(event_id, scope=SCOPE), peers))
    owned = [row for row in claims if row is not None]
    assert len(owned) == 1
    old = owned[0]
    clock[0] += 11
    fresh = events.claim(event_id, scope=SCOPE)
    assert fresh and fresh["claim_id"] != old["claim_id"]
    assert not events.finish(old, scope=SCOPE, status="done", reason="")
    assert events.finish(fresh, scope=SCOPE, status="done", reason="")
    assert events.claim(event_id, scope=SCOPE) is None


def test_retry_is_bounded_and_task_failure_is_unchanged(tmp_path):
    store = AgentTaskStore(tmp_path / "ledger.db")
    events = ResultEventStore(store, max_attempts=2)
    task = store.save(terminal(store, status="failed"))
    event_id = events.pending(scope=SCOPE)[0]["event_id"]
    for _ in range(2):
        claim = events.claim(event_id, scope=SCOPE)
        assert events.retry(claim, scope=SCOPE, reason="fixture_consumer_failure")
    assert events.claim(event_id, scope=SCOPE) is None
    event = events.get(event_id, scope=SCOPE)
    assert event["status"] == "failed" and event["attempts"] == 2
    assert store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session) == task


def test_pending_result_pins_task_and_expiry_remains_diagnostic(tmp_path):
    clock = [1000.0]
    store = AgentTaskStore(tmp_path / "ledger.db", ttl_seconds=10, max_entries=1, clock=lambda: clock[0])
    events = ResultEventStore(store)
    task = store.save(terminal(store))
    clock[0] += 1
    another = new_task(task_id="running", requester=SCOPE.requester, session=SCOPE.session,
        request="another", message_id="another", contract=None)
    store.save(another)
    assert store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session) == task
    event_id = events.pending(scope=SCOPE)[0]["event_id"]
    clock[0] += 11
    assert events.pending(scope=SCOPE) == []
    assert events.get(event_id, scope=SCOPE)["status"] == "expired"
    assert events.get(event_id, scope=SCOPE)["reason"] == "result_retention_expired"


@pytest.mark.parametrize("status", ["completed", "failed", "blocked", "partial", "canceled", "unverified"])
def test_every_terminal_result_is_emitted(tmp_path, status):
    store = AgentTaskStore(tmp_path / "ledger.db")
    events = ResultEventStore(store)
    store.save(terminal(store, status=status))
    assert len(events.pending(scope=SCOPE)) == 1


def test_stale_result_not_emitted_and_capacity_rolls_back_save(tmp_path):
    store = AgentTaskStore(tmp_path / "ledger.db")
    events = ResultEventStore(store, max_entries=1)
    saved = store.save(terminal(store))
    saved.update(status="running", execution_attempt=2)
    store.save(saved)  # Old result retained as history, not a new result event.
    assert len(events.pending(scope=SCOPE)) == 1
    with pytest.raises(ValueError, match="result_event_capacity"):
        store.save(terminal(store, task_id="second"))
    assert store.get("second", requester=SCOPE.requester, session=SCOPE.session) is None


def test_snapshot_tampering_is_rejected(tmp_path):
    store = AgentTaskStore(tmp_path / "ledger.db")
    events = ResultEventStore(store)
    store.save(terminal(store))
    event = events.pending(scope=SCOPE)[0]
    with store._transaction() as db:
        db.execute("UPDATE agent_result_events SET payload=? WHERE event_id=?", ('{}', event["event_id"]))
    with pytest.raises(ValueError, match="result_event_integrity_failed"):
        events.get(event["event_id"], scope=SCOPE)


@pytest.mark.parametrize("committed", [False, True])
def test_abrupt_process_exit_preserves_atomic_commit_boundary(tmp_path, committed):
    source = str(Path(__file__).resolve().parents[1] / "src")
    script = '''import os, sys
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent_runtime.result_events import ResultEventStore
from shinku.agent_runtime.results import export_task_result
store = AgentTaskStore(sys.argv[1])
events = ResultEventStore(store)
task = new_task(task_id='fixture-task', requester='fixture', session='private:fixture', request='fixture', message_id='fixture', contract=None)
task.update(status='completed', execution_attempt=1, completion_attempt=1, completion={'accepted':True})
task['execution_result'] = export_task_result({**task, 'revision':1, 'updated':store.clock()})
if sys.argv[2] == 'before':
    original = events.append_in_transaction
    def interrupt(db, task):
        original(db, task)
        os._exit(37)
    events.append_in_transaction = interrupt
store.save(task)
os._exit(37)
'''
    path = tmp_path / "ledger.db"
    result = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script, str(path),
        "after" if committed else "before"], timeout=10, capture_output=True,
        env={**os.environ, "PYTHONPATH": source, "SHINKU_ENV_FILE": str(tmp_path / "missing.env")})
    assert result.returncode == 37, result.stderr
    store = AgentTaskStore(path)
    events = ResultEventStore(store)
    assert bool(store.get("fixture-task", requester=SCOPE.requester, session=SCOPE.session)) is committed
    assert bool(events.pending(scope=SCOPE)) is committed


def test_projection_redacts_credentials_in_source_links(tmp_path):
    store = AgentTaskStore(tmp_path / "ledger.db")
    events = ResultEventStore(store)
    task = terminal(store)
    task["tool_calls"] = [{"id": "fixture-call", "name": "fixture-read", "status": "ok", "data": {
        "state_updates": {"sources": [{"url": "https://fixture-user:FIXTURE_PASSWORD@example.invalid/report?signature=FIXTURE_SIGNATURE&part=2"}]}}}]
    task["execution_result"] = export_task_result({**task, "revision": 1, "updated": store.clock()})
    store.save(task)
    import json
    snapshot = json.dumps(events.pending(scope=SCOPE)[0], ensure_ascii=False)
    assert "FIXTURE_PASSWORD" not in snapshot and "FIXTURE_SIGNATURE" not in snapshot
    assert "part=2" in snapshot
