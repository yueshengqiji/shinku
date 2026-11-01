"""P3: task facts are scoped, sourced and independent of generated reports."""
from copy import deepcopy
from dataclasses import replace
import json
import threading

import pytest

from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.task_view import TaskViewReader
from shinku.conversation import ConversationRuntime, ConversationScope


SCOPE = AgentScope("owner", "private:owner")
CONVERSATION = ConversationScope("private", "owner", "owner")


def task(store, *, task_id="task-fixture", status="completed", attempt=2, accepted=True, now=1000):
    value = new_task(task_id=task_id, requester=SCOPE.requester, session=SCOPE.session,
        request="生成两个报告并核验内容", message_id=task_id + "-message", contract=None)
    value.update(status=status, execution_attempt=attempt,
        completion={"accepted": accepted, "status": "completed" if accepted else "partial",
                    "satisfied": ["第一份报告"], "missing": [] if accepted else ["第二份报告"]},
        completion_attempt=attempt, last_report="已经全部部署并交付。", stop_reason="",
        tool_calls=[{"name": "write_file", "status": "ok"}],
        notes={"remaining": ["模型猜测"], "blocker": "模型猜测"})
    return store.save(value, now=now)


class Coordinator:
    def __init__(self, record=None):
        self.record = record
    def get(self, task_id, *, profile_user_id, session_id):
        return deepcopy(self.record)


def receipt_reader(runtime):
    return lambda scope: runtime.task_receipts(ConversationScope(
        scope.session.split(":", 1)[0], scope.session.split(":", 1)[1], scope.requester))


def test_completed_without_positive_current_verification_stays_unverified():
    store = AgentTaskStore(clock=lambda: 1000)
    row = task(store, accepted=False)
    view = TaskViewReader(store).get(SCOPE, row["task_id"])
    assert view["status"] == "unverified" and not view["verification"]["verified"]
    assert view["verification"]["missing"] == ["第二份报告"]
    assert "已经全部部署" not in json.dumps(view, ensure_ascii=False)
    assert view["done"] == [{"tool": "write_file", "status": "ok", "source": "executor"}]
    assert view["pending"] == ["第二份报告"]
    assert view["field_sources"]["verification"] == "completion_verifier"


@pytest.mark.parametrize("delivery,sent", [("failed", False), ("uncertain", False),
    ("dry_run", False), ("sent", True)])
def test_verified_execution_and_delivery_are_separate(delivery, sent):
    store, runtime = AgentTaskStore(clock=lambda: 1000), ConversationRuntime(clock=lambda: 1000)
    row = task(store)
    runtime.begin(CONVERSATION, "receipt")
    runtime.step("receipt", task_id=row["task_id"], task_attempt=2, task_version=row["revision"])
    runtime.finish("receipt", status="failed" if delivery == "failed" else "completed",
                   delivery_status=delivery, sent=sent)
    view = TaskViewReader(store, delivery_reader=receipt_reader(runtime)).get(SCOPE, row["task_id"])
    assert view["status"] == "completed" and view["verification"]["verified"]
    assert view["delivery"]["status"] == delivery and view["delivery"]["sent"] is sent
    assert view["attempt"] == 2 and view["version"] == row["revision"]


def test_old_attempt_sent_receipt_cannot_mark_current_attempt_delivered():
    store, runtime = AgentTaskStore(clock=lambda: 1000), ConversationRuntime(clock=lambda: 1000)
    row = task(store)
    runtime.begin(CONVERSATION, "old")
    runtime.step("old", task_id=row["task_id"], task_attempt=1, task_version=1)
    runtime.finish("old", status="completed", delivery_status="sent", sent=True)
    view = TaskViewReader(store, delivery_reader=receipt_reader(runtime)).get(SCOPE, row["task_id"])
    assert view["delivery"]["status"] == "not_recorded" and not view["delivery"]["sent"]


def test_stale_verification_and_expected_attempt_are_not_promoted():
    store = AgentTaskStore(clock=lambda: 1000)
    row = task(store)
    row["completion_attempt"] = 1
    store.save(row)
    reader = TaskViewReader(store)
    assert not reader.get(SCOPE, row["task_id"])["verification"]["verified"]
    assert reader.get(SCOPE, row["task_id"], expected_attempt=1)["status"] == "unknown"
    assert reader.get(SCOPE, row["task_id"], expected_version=1)["reason"] == "task_version_conflict"


@pytest.mark.parametrize("scope", [AgentScope("other", "private:owner"), AgentScope("owner", "group:owner")])
def test_cross_scope_does_not_reveal_task(scope):
    store = AgentTaskStore(clock=lambda: 1000)
    row = task(store)
    view = TaskViewReader(store).get(scope, row["task_id"])
    assert not view["available"] and "goal" not in view
    assert not TaskViewReader(store).candidates(scope)["tasks"]


def test_coordinator_attempt_and_ledger_conflicts_do_not_choose_success():
    store = AgentTaskStore(clock=lambda: 1000)
    row = task(store)
    row.update(execution_work_id="work", execution_work_attempt=4)
    store.save(row)
    record = {"task_id": "work", "attempt": 3, "profile_user_id": "owner",
        "session_id": "private:owner", "status": "completed", "worker_active": False,
        "updated_at": 1000, "metadata": {"agent_task_id": row["task_id"]}}
    reader = TaskViewReader(store, Coordinator(record))
    assert reader.get(SCOPE, row["task_id"])["status"] == "conflict"
    record.update(attempt=4, status="failed")
    assert TaskViewReader(store, Coordinator(record)).get(SCOPE, row["task_id"])["status"] == "conflict"


def test_stale_worker_and_expired_conversation_lease_are_unknown():
    store = AgentTaskStore(clock=lambda: 5000)
    row = task(store, status="running", now=1000)
    row.update(execution_work_id="work", execution_work_attempt=1)
    row = store.save(row, now=1000)
    record = {"task_id": "work", "attempt": 1, "profile_user_id": "owner", "session_id": "private:owner",
        "status": "running", "worker_active": True, "updated_at": 1000,
        "metadata": {"agent_task_id": row["task_id"]}}
    assert TaskViewReader(store, Coordinator(record)).get(SCOPE, row["task_id"])["status"] == "unknown"
    runtime = ConversationRuntime(clock=lambda: 5000)
    runtime.begin(CONVERSATION, "expired")
    runtime.step("expired", task_id=row["task_id"], task_attempt=2, task_version=row["revision"])
    runtime.clock = lambda: 8000
    store.clock = lambda: 8000
    row.pop("execution_work_id")
    store.save(row, now=8000)
    view = TaskViewReader(store, delivery_reader=receipt_reader(runtime)).get(SCOPE, row["task_id"])
    assert view["status"] == "unknown" and view["reason"] == "conversation_lease_expired"


def test_storage_failures_are_distinct_from_absent_results():
    store = AgentTaskStore(clock=lambda: 1000)
    row = task(store)
    def broken(*args, **kwargs):
        raise OSError("secret error details")
    assert TaskViewReader(store, delivery_reader=broken).get(SCOPE, row["task_id"])["delivery"]["status"] == "unknown"
    store.get = broken
    view = TaskViewReader(store).get(SCOPE, row["task_id"])
    assert view["reason"] == "task_store_unavailable" and "secret" not in json.dumps(view)


def test_multi_task_binding_never_defaults_to_latest():
    store = AgentTaskStore(clock=lambda: 1000)
    first, second = task(store, task_id="first"), task(store, task_id="second")
    reader = TaskViewReader(store)
    assert reader.select(SCOPE)["reason"] == "task_binding_ambiguous"
    selected = reader.select(SCOPE, message_ids=(first["requests"][0]["message_id"],))
    assert selected["task"]["task_id"] == "first"
    assert reader.select(SCOPE, message_ids=("foreign-message",))["reason"] == "task_binding_not_found"
    for number in range(4):
        task(store, task_id=f"new-{number}")
    assert reader.select(SCOPE, message_ids=(first["requests"][0]["message_id"],))["task"]["task_id"] == "first"


def test_report_reference_binds_attempt_and_old_report_does_not_query_new_attempt():
    store, runtime = AgentTaskStore(clock=lambda: 1000), ConversationRuntime(clock=lambda: 1000)
    row = task(store)
    runtime.begin(CONVERSATION, "report")
    runtime.step("report", task_id=row["task_id"], task_attempt=2, task_version=row["revision"])
    runtime.finish("report", status="completed", delivery_status="sent", sent=True, message_ids=("report-id",))
    reader = TaskViewReader(store, delivery_reader=receipt_reader(runtime))
    assert reader.select(SCOPE, message_ids=("report-id",))["task"]["task_id"] == row["task_id"]
    row["execution_attempt"] = 3
    store.save(row)
    assert reader.select(SCOPE, message_ids=("report-id",))["reason"] == "task_attempt_conflict"


def test_old_finish_cannot_resolve_new_attempt_approval():
    runtime = ConversationRuntime()
    runtime.begin(CONVERSATION, "old")
    runtime.step("old", task_id="task", task_attempt=1)
    runtime.begin(CONVERSATION, "new")
    runtime.step("new", task_id="task", task_attempt=2)
    runtime.finish("new", status="waiting_user")
    runtime.finish("old", status="completed")
    assert next(r for r in runtime.task_receipts(CONVERSATION)["records"] if r["turn_id"] == "new")["status"] == "waiting_user"


def test_concurrent_snapshot_change_is_reported_without_writing():
    store = AgentTaskStore(clock=lambda: 1000)
    row = task(store)
    original = store.get
    reads = []
    def racing(*args, **kwargs):
        result = original(*args, **kwargs)
        reads.append(result)
        if len(reads) == 2:
            result["revision"] += 1
        return result
    store.get = racing
    view = TaskViewReader(store).get(SCOPE, row["task_id"])
    assert view["status"] == "unknown" and view["reason"] == "task_changed_during_read"
    assert original(row["task_id"], requester="owner", session="private:owner")["revision"] == row["revision"]


def test_other_host_cannot_clear_active_turn_and_old_completion_keeps_identity():
    runtime = ConversationRuntime(quiet_seconds=0)
    runtime.begin(CONVERSATION, "old")
    runtime.step("old", task_id="task", task_attempt=1, task_version=2)
    entered = threading.Event()
    def newer():
        runtime.begin(CONVERSATION, "new")
        runtime.step("new", task_id="task", task_attempt=2, task_version=3)
        entered.set()
    worker = threading.Thread(target=newer)
    worker.start()
    worker.join(2)
    assert entered.is_set()
    runtime.finish("old", status="completed", delivery_status="sent", sent=True)
    assert not runtime.proactive_permit(CONVERSATION.channel).allowed
    rows = {row["turn_id"]: row for row in runtime.task_receipts(CONVERSATION)["records"]}
    assert rows["old"]["task_attempt"] == 1 and rows["new"]["task_attempt"] == 2
    assert rows["new"]["phase"] != "finished"


def test_cancel_request_does_not_claim_stopped_while_worker_is_active():
    store = AgentTaskStore(clock=lambda: 1000)
    row = task(store, status="running", accepted=False)
    row.update(execution_work_id="work", execution_work_attempt=1)
    store.save(row)
    record = {"task_id": "work", "attempt": 1, "profile_user_id": "owner", "session_id": "private:owner",
        "status": "running", "worker_active": True, "cancel_requested": True, "updated_at": 1000,
        "metadata": {"agent_task_id": row["task_id"]}}
    view = TaskViewReader(store, Coordinator(record)).get(SCOPE, row["task_id"])
    assert view["status"] == "running" and view["cancel_requested"] and view["worker_active"]
    record["status"] = "canceled"
    assert TaskViewReader(store, Coordinator(record)).get(SCOPE, row["task_id"])["status"] == "unknown"


def test_inconsistent_sent_flag_and_conflicting_delivery_records_stay_unknown():
    store, runtime = AgentTaskStore(clock=lambda: 1000), ConversationRuntime(clock=lambda: 1000)
    row = task(store)
    for identifier, sent in [("first", False), ("second", True)]:
        runtime.begin(CONVERSATION, identifier)
        runtime.step(identifier, task_id=row["task_id"], task_attempt=2, task_version=row["revision"])
        runtime.finish(identifier, status="completed", delivery_status="sent", sent=sent)
        view = TaskViewReader(store, delivery_reader=receipt_reader(runtime)).get(SCOPE, row["task_id"])
        if identifier == "first":
            assert view["delivery"]["status"] == "uncertain" and not view["delivery"]["sent"]
    assert view["status"] == "conflict" and not view["delivery"]["sent"]


def test_existing_operational_payload_without_new_binding_fields_loads(tmp_path):
    import sqlite3
    runtime = ConversationRuntime(tmp_path / "old.db", clock=lambda: 1000)
    runtime.begin(CONVERSATION, "legacy")
    with sqlite3.connect(runtime.path) as conn:
        value = json.loads(conn.execute("SELECT payload FROM conversation_turns").fetchone()[0])
        for key in ("task_attempt", "task_version", "message_ids"):
            value.pop(key)
        conn.execute("UPDATE conversation_turns SET payload=?", (json.dumps(value),))
    row = runtime.task_receipts(CONVERSATION)["records"][0]
    assert row["task_attempt"] == row["task_version"] == 0 and row["message_ids"] == []
