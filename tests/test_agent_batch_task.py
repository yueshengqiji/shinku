from pathlib import Path
import sqlite3

import pytest

from shinku.agent.batch_task import BatchTaskStore, BatchRunner, BatchBudget, BatchTaskError


SCOPE = {"owner": "owner", "session": "private:test"}


def create(store, count=4, budget=None):
    return store.create(**SCOPE, metadata={"version": "test"},
                        chunks=[(i + 1, f"Original page {i + 1}") for i in range(count)], budget=budget)


def run(store, job, translate=None, **kwargs):
    return BatchRunner(store).run(job["job_id"], **SCOPE,
        translate=translate or (lambda c: "translated " + c["source"]),
        validate=lambda _c, value: value, publish=lambda _j, _c: {"verified": True}, **kwargs)


def test_many_chunks_in_one_runner_and_exact_call_budget(tmp_path):
    store = BatchTaskStore(tmp_path / "tasks.db")
    result = run(store, create(store, 28, BatchBudget(max_calls=28)))
    assert (result["status"], result["calls_used"], result["done_chunks"]) == ("completed", 28, 28)


def test_failed_calls_pause_and_durable_resume_skips_finished(tmp_path):
    store = BatchTaskStore(tmp_path / "tasks.db")
    job = create(store)
    seen = []
    def failed(chunk):
        seen.append(chunk["ordinal"])
        if chunk["ordinal"] == 1:
            raise RuntimeError("api_key=never-store-this")
        return "translated"
    result = run(store, job, failed)
    assert seen == [0, 1, 1]
    assert result["status"] == "paused" and result["calls_used"] == 3 and result["done_chunks"] == 1
    reopened = BatchTaskStore(store.path)
    resumed_seen = []
    result = run(reopened, job, lambda c: resumed_seen.append(c["ordinal"]) or "translated")
    assert resumed_seen == [1, 2, 3]
    assert result["status"] == "completed" and result["calls_used"] == 6
    assert b"never-store-this" not in store.path.read_bytes()


def test_duplicate_start_and_resume_do_not_reset_budget(tmp_path):
    store = BatchTaskStore(tmp_path / "tasks.db")
    job = create(store, budget=BatchBudget(max_calls=2))
    result = run(store, job)
    assert result["status"] == "paused" and result["reason"] == "call_budget_exhausted"
    duplicated = create(store, budget=BatchBudget(max_calls=80))
    assert duplicated["job_id"] == job["job_id"] and duplicated["budget"]["max_calls"] == 2
    assert run(store, duplicated)["calls_used"] == 2


def test_cancel_during_call_does_not_commit_result_or_publish(tmp_path):
    store = BatchTaskStore(tmp_path / "tasks.db")
    job = create(store)
    def cancel(_chunk):
        store.cancel(job["job_id"], **SCOPE)
        return "late result"
    result = run(store, job, cancel)
    assert result["status"] == "canceled" and result["done_chunks"] == 0
    assert not result["artifact"] and run(store, job)["calls_used"] == 1


@pytest.mark.parametrize("scope", [{"owner": "other", "session": SCOPE["session"]},
                                   {"owner": "owner", "session": "group:test"}])
def test_owner_and_typed_session_isolation(tmp_path, scope):
    store = BatchTaskStore(tmp_path / "tasks.db")
    job = create(store)
    for operation in (store.get, store.chunks, store.cancel, store.claim):
        with pytest.raises(BatchTaskError, match="not_found"):
            operation(job["job_id"], **scope)


def test_crashed_inflight_reservation_is_charged_and_old_lease_cannot_commit(tmp_path):
    now = [0.0]
    store = BatchTaskStore(tmp_path / "tasks.db", clock=lambda: now[0])
    job = create(store)
    token = store.claim(job["job_id"], **SCOPE)
    chunk = store.reserve(job["job_id"], **SCOPE, token=token)
    with pytest.raises(BatchTaskError, match="already_running"):
        store.claim(job["job_id"], **SCOPE)
    now[0] = 901
    new = store.claim(job["job_id"], **SCOPE)
    state = store.get(job["job_id"], **SCOPE)
    assert state["active_seconds"] == 900 and state["calls_used"] == 1 and new != token
    with pytest.raises(BatchTaskError, match="lease_lost"):
        store.finish_call(job["job_id"], **SCOPE, token=token, chunk=chunk, translated="old")


def test_active_time_budget_and_clock_zero(tmp_path):
    now = [0.0]
    store = BatchTaskStore(tmp_path / "tasks.db", clock=lambda: now[0])
    job = create(store, budget=BatchBudget(max_seconds=2))
    def translate(_chunk):
        now[0] += 3
        return "translated"
    result = run(store, job, translate)
    assert result["active_seconds"] == 3 and result["calls_used"] == 1
    assert result["status"] == "paused" and result["reason"] == "time_budget_exhausted"


def test_output_limit_is_not_completion(tmp_path):
    store = BatchTaskStore(tmp_path / "tasks.db")
    job = create(store, budget=BatchBudget(max_output_chars=1024))
    result = run(store, job, lambda _c: "x" * 1025)
    assert result["status"] == "paused" and result["reason"] == "output_budget_exhausted"
    assert result["done_chunks"] == 0


def test_guard_interruption_preserves_durable_progress(tmp_path):
    store = BatchTaskStore(tmp_path / "tasks.db")
    job = create(store)
    def guard():
        if store.get(job["job_id"], **SCOPE)["done_chunks"] >= 2:
            raise PermissionError("permissions changed")
    with pytest.raises(PermissionError):
        run(store, job, guard=guard)
    state = store.get(job["job_id"], **SCOPE)
    assert state["status"] == "paused" and state["done_chunks"] == 2
    assert run(BatchTaskStore(store.path), job)["calls_used"] == 4


def test_readonly_existing_store_does_not_recreate_missing_schema(tmp_path):
    path = tmp_path / "tasks.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE unrelated(x)")
    before = path.read_bytes()
    store = BatchTaskStore(path, create=False)
    with pytest.raises(sqlite3.OperationalError):
        store.get("unknown", **SCOPE)
    assert path.read_bytes() == before


@pytest.mark.parametrize("budget", [{"max_calls": True}, {"max_calls": 1000},
                                     {"max_output_chars": 0}, {"max_seconds": True}])
def test_invalid_budget(budget):
    with pytest.raises(BatchTaskError):
        BatchBudget(**budget)
