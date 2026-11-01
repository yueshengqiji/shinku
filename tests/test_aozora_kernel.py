"""匿名合成事件检验独立内核：证据归属、持久化、并发、原子性和真实送达边界。"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import json
import sqlite3
import threading

import pytest

from aozora import Decision, Event, Identity, Kernel, KernelError, Store
from aozora.__main__ import emit


def event(event_id="e1", *, scope="room", kind="message", content="合成转述：某人说读数相等。"):
    return Event(event_id, scope, "speaker", kind, content, 1)


def change(key="question", *, refs=None, version=0, status="open", text="比较两组读数"):
    return {"key": key, "text": text, "evidence": refs or ["e1"], "version": version, "status": status}


def decision(*, attention=(), understandings=(), action="reply", reply="合成回复"):
    return Decision.parse({"action": action, "reply": reply,
                           "attention": list(attention), "understandings": list(understandings)})


class Fixed:
    def __init__(self, result=None):
        self.result = result or decision()
        self.contexts = []

    def decide(self, context):
        self.contexts.append(context)
        return self.result


@pytest.fixture
def state(tmp_path):
    return Store(tmp_path / "state.sqlite3", Identity("fixture", "合成个体"))


def test_restart_preserves_owned_state_and_original_sources(state):
    first = Fixed(decision(attention=[change()], understandings=[change("view", status="tentative")]))
    view = Kernel(state, first).handle(event())
    assert view["status"] == "completed" and view["delivery"]["status"] == "prepared"
    assert first.contexts[0]["events"][0]["source"] == "reported"
    reopened = Store(state.path, Identity("fixture", "合成个体"))
    next_decider = Fixed(decision(attention=[change(version=1, status="completed", refs=["e1", "e2"])],
                                  understandings=[change("view", version=1, status="tentative", refs=["e2"], text="测量不支持旧说法")]))
    second = Kernel(reopened, next_decider).handle(event("e2", kind="observation", content="合成观察：A=1，B=2。"))
    assert second["status"] == "completed"
    context = next_decider.contexts[0]
    assert context["attention"][0]["key"] == "question"
    assert context["deliveries"][0]["status"] == "prepared"
    assert context["events"][1]["source"] == "host_observation"
    snapshot = reopened.snapshot("room")
    assert snapshot["attention"] == []
    assert snapshot["understandings"][0]["version"] == 2
    assert snapshot["understandings"][0]["source"] == "model_interpretation"
    assert state.status("e1", "room")["decision"]["understandings"][0]["text"] == "比较两组读数"


def test_duplicate_event_replays_without_model_call_and_conflict_is_rejected(state):
    decider = Fixed()
    kernel = Kernel(state, decider)
    original = kernel.handle(event())
    assert kernel.handle(event()) == original
    assert kernel.handle(event(), retry=True) == original
    assert len(decider.contexts) == 1
    for changed in (replace(event(), content="changed"), replace(event(), scope="other")):
        with pytest.raises(KernelError, match="event_id_conflict"):
            kernel.handle(changed)


def test_cross_scope_has_no_history_state_delivery_or_evidence(state):
    Kernel(state, Fixed(decision(attention=[change()]))).handle(event())
    other = Fixed(decision(understandings=[change("view", status="tentative")]))
    result = Kernel(state, other).handle(event("e2", scope="other"))
    context = other.contexts[0]
    assert context["attention"] == context["understandings"] == context["deliveries"] == []
    assert [row["event_id"] for row in context["events"]] == ["e2"]
    assert result["error"] == "unseen_evidence"
    with pytest.raises(KernelError, match="event_not_found"):
        state.status("e1", "other")
    with pytest.raises(KernelError, match="event_not_found"):
        state.cancel("e1", "other")


def test_identity_cannot_silently_change_and_foreign_sqlite_untouched(state, tmp_path):
    with pytest.raises(KernelError, match="identity_mismatch"):
        Store(state.path, Identity("fixture", "different"))
    assert Store(state.path, Identity("fixture", "合成个体")).snapshot("room")["identity"]["name"] == "合成个体"
    foreign = tmp_path / "foreign.sqlite3"
    with sqlite3.connect(foreign) as db:
        db.execute("CREATE TABLE private_data (value TEXT)")
    original = foreign.read_bytes()
    with pytest.raises(KernelError, match="not_an_aozora_database"):
        Store(foreign, Identity("fixture", "合成个体"))
    assert foreign.read_bytes() == original


def test_in_memory_connection_cannot_silently_lose_persistent_state():
    with pytest.raises(KernelError, match="persistent_database_required"):
        Store(":memory:", Identity("fixture", "合成个体"))


def test_decision_failure_rolls_back_all_changes_and_keeps_event(state):
    bad = Fixed(decision(attention=[change()], understandings=[change("view", refs=["missing"], status="tentative")]))
    result = Kernel(state, bad).handle(event())
    assert result["status"] == "failed" and result["error"] == "unseen_evidence"
    assert result["delivery"] is None
    snapshot = state.snapshot("room")
    assert len(snapshot["events"]) == 1
    assert snapshot["attention"] == snapshot["understandings"] == []


def test_failure_has_no_fallback_or_implicit_retry(state):
    class Broken(Fixed):
        def decide(self, context):
            self.contexts.append(context)
            raise RuntimeError("secret token and private conversation")
    broken = Broken()
    result = Kernel(state, broken).handle(event())
    assert result["error"] == "decider_failed"
    assert "secret" not in json.dumps(result)
    assert Kernel(state, broken).handle(event()) == result
    good = Fixed()
    retry = Kernel(state, good).handle(event(), retry=True)
    assert retry["status"] == "completed" and retry["attempt"] == 2
    with sqlite3.connect(state.path) as db:
        assert db.execute("SELECT status FROM attempts ORDER BY attempt").fetchall() == [("failed",), ("completed",)]


def test_concurrent_duplicate_has_one_provider_call_and_busy_event_not_ingested(state):
    entered, release = threading.Event(), threading.Event()
    class Slow(Fixed):
        def decide(self, context):
            self.contexts.append(context)
            entered.set()
            assert release.wait(5)
            return self.result
    slow = Slow()
    kernel = Kernel(state, slow)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(kernel.handle, event())
        assert entered.wait(5)
        try:
            assert kernel.handle(event())["status"] == "running"
            with pytest.raises(KernelError, match="decision_busy"):
                kernel.handle(event("e2"))
        finally:
            release.set()
        assert future.result(timeout=5)["status"] == "completed"
    assert len(slow.contexts) == 1
    assert len(state.snapshot("room")["events"]) == 1


def test_cancel_then_retry_rejects_late_old_decision(state):
    old, _ = state.begin(event())
    assert state.cancel("e1", "room")["status"] == "cancelled"
    new, _ = state.begin(event(), retry=True)
    assert new["attempt"] == 2
    with pytest.raises(KernelError, match="stale_attempt"):
        state.finish(event(), old["attempt"], decision(attention=[change()]))
    # 旧线程的 KeyboardInterrupt 也不得取消后继尝试。
    state.cancel("e1", "room", expected_attempt=old["attempt"])
    assert state.status("e1", "room")["status"] == "running"
    state.finish(event(), new["attempt"], decision(action="wait", reply=""))
    assert state.snapshot("room")["attention"] == []


def test_restart_running_attempt_requires_explicit_cancel(state):
    state.begin(event())
    reopened = Store(state.path, Identity("fixture", "合成个体"))
    decider = Fixed()
    assert Kernel(reopened, decider).handle(event(), retry=True)["status"] == "running"
    assert decider.contexts == []
    reopened.cancel("e1", "room")
    assert Kernel(reopened, decider).handle(event(), retry=True)["status"] == "completed"


def test_original_sources_survive_recent_window_and_retry_keeps_current_event(state):
    kernel = Kernel(state, Fixed(decision(understandings=[change("view", status="tentative")])))
    kernel.handle(event())
    failed = event("old-failure")
    Kernel(state, Fixed(decision(attention=[change(refs=["missing"])]))).handle(failed)
    for i in range(12):
        Kernel(state, Fixed(decision(action="wait", reply=""))).handle(event(f"later-{i}"))
    snapshot = state.snapshot("room")
    assert "e1" not in {row["event_id"] for row in snapshot["events"]}
    assert snapshot["sources"][0]["event_id"] == "e1"
    retry = Fixed()
    assert Kernel(state, retry).handle(failed, retry=True)["status"] == "completed"
    assert "old-failure" in {row["event_id"] for row in retry.contexts[0]["events"]}


def test_adapter_mutation_cannot_forge_persisted_context(state):
    class Mutating:
        def decide(self, context):
            context["events"].append({"event_id": "invented"})
            context["identity"]["name"] = "hijacked"
            return decision(attention=[change(refs=["invented"])])
    result = Kernel(state, Mutating()).handle(event())
    assert result["error"] == "unseen_evidence"
    assert state.snapshot("room")["identity"]["name"] == "合成个体"


def test_capacity_failure_is_atomic(state):
    for batch in range(2):
        rows = [change(f"q-{batch}-{i}", refs=[f"batch-{batch}"]) for i in range(8)]
        assert Kernel(state, Fixed(decision(attention=rows))).handle(event(f"batch-{batch}"))["status"] == "completed"
    result = Kernel(state, Fixed(decision(attention=[change(refs=["overflow"])]))).handle(event("overflow"))
    assert result["error"] == "state_capacity_exceeded"
    assert len(state.snapshot("room")["attention"]) == 16


@pytest.mark.parametrize("kind,version,status", [("attention", 0, "completed"), ("attention", 2, "open"), ("understandings", 0, "tentative")])
def test_versions_and_closed_state_cannot_be_recreated(state, kind, version, status):
    Kernel(state, Fixed(decision(attention=[change()], understandings=[change("view", status="tentative")]))).handle(event())
    key = "view" if kind == "understandings" else "question"
    result = Kernel(state, Fixed(decision(**{kind: [change(key, version=version, status=status, refs=["e2"])]}))).handle(event("e2"))
    assert result["error"] == "state_version_conflict"


def test_terminal_delivery_requires_claim_and_flush_receipt(state):
    view = Kernel(state, Fixed()).handle(event())
    delivery_id = view["delivery"]["delivery_id"]
    with pytest.raises(KernelError, match="delivery_not_claimed"):
        state.resolve_delivery(delivery_id, "room", delivered=True, receipt="not yet")
    with pytest.raises(KernelError, match="delivery_not_found"):
        state.claim_delivery(delivery_id, "other", "local-terminal")
    stream = io.StringIO()
    emit(state, delivery_id, "room", stream)
    assert stream.getvalue() == "合成回复\n"
    sent = state.status("e1", "room")["delivery"]
    assert sent["status"] == "delivered" and sent["channel"] == "local-terminal"
    state.resolve_delivery(delivery_id, "room", delivered=True, receipt=sent["receipt"])
    with pytest.raises(KernelError, match="delivery_not_prepared"):
        emit(state, delivery_id, "room", stream)


def test_output_failure_stays_unknown_across_restart_and_never_resends(state):
    class BrokenStream:
        def write(self, text):
            raise BrokenPipeError()
    view = Kernel(state, Fixed()).handle(event())
    delivery_id = view["delivery"]["delivery_id"]
    with pytest.raises(BrokenPipeError):
        emit(state, delivery_id, "room", BrokenStream())
    restarted = Store(state.path, Identity("fixture", "合成个体"))
    assert restarted.status("e1", "room")["delivery"]["status"] == "unknown"
    with pytest.raises(KernelError, match="delivery_not_prepared"):
        emit(restarted, delivery_id, "room", io.StringIO())
    restarted.resolve_delivery(delivery_id, "room", delivered=False, receipt="host verified output failed")
    assert restarted.status("e1", "room")["delivery"]["status"] == "failed"


def test_delivery_change_while_deciding_rejects_old_context(state):
    first = Kernel(state, Fixed()).handle(event())
    second = event("e2")
    attempt, _ = state.begin(second)
    emit(state, first["delivery"]["delivery_id"], "room", io.StringIO())
    with pytest.raises(KernelError, match="delivery_context_changed"):
        state.finish(second, attempt["attempt"], decision(attention=[change(refs=["e2"])]))
    assert state.snapshot("room")["attention"] == []
    state.cancel("e2", "room")


def test_keyboard_interrupt_cancels_commit_eligibility(state):
    class Interrupted:
        def decide(self, context):
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        Kernel(state, Interrupted()).handle(event())
    assert state.status("e1", "room")["status"] == "cancelled"


@pytest.mark.parametrize("mutation", [
    {"identity": {}}, {"action": "execute"}, {"action": "wait", "reply": "still replying"},
    {"attention": [change(refs=["e1"], status="observed")]},
    {"understandings": [change(status="confirmed")]},
    {"attention": [{**change(), "source": "firsthand"}]},
    {"attention": [{**change(), "evidence": []}]},
    {"attention": [{**change(), "version": True}]},
    {"attention": [change(), change()]},
    {"action": []}, {"attention": [{**change(), "status": {}}]},
])
def test_model_cannot_extend_authority_or_provenance(mutation):
    with pytest.raises(KernelError):
        Decision.parse({**decision().payload(), **mutation})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "2026", 10**400])
def test_invalid_event_time_rejected_before_write(state, value):
    with pytest.raises(KernelError):
        Kernel(state, Fixed()).handle(replace(event(), occurred_at=value))
    assert state.snapshot("room")["events"] == []
