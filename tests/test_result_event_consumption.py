"""Crash/failure injection around real SQLite expression/event boundaries."""
import copy
import threading

import pytest

from shinku.agent.task_state import AgentTaskStore
from shinku.agent_runtime.contracts import canonical_hash
from shinku.result_expression import ExpressionArchive, ExpressionProfile, ExpressionService
from shinku.result_expression.events import ResultEventConsumer
from shinku.result_expression.facts import identity
from test_result_events import SCOPE, terminal
from test_result_expression import Model


class Crash(BaseException):
    pass


def setup(tmp_path, *, model=None, max_attempts=3):
    clock = [1000.0]
    store = AgentTaskStore(tmp_path / "ledger.db", clock=lambda: clock[0])
    archive = ExpressionArchive(store)
    archive.events.max_attempts = max_attempts
    service = ExpressionService(archive, ExpressionProfile(model or Model(), configuration_id="fixture"))
    task = store.save(terminal(store))
    return clock, store, service, task, archive.events.pending(scope=SCOPE)[0]


def restart(store, model):
    path, clock = store.path, store.clock
    store.close()
    fresh = AgentTaskStore(path, clock=clock)
    service = ExpressionService(ExpressionArchive(fresh), ExpressionProfile(model, configuration_id="fixture"))
    return fresh, service


def publisher(service, sent, *, status="sent", crash=""):
    def publish(source, record, guard):
        if crash == "before_send":
            raise Crash()
        assert service.reserve_publication(record, scope=SCOPE, check_current=guard)
        sent.append(record["text"])
        if crash == "during_send":
            raise Crash()
        receipt = {"status": status, "sent": status == "sent", "message_ids": ["fixture-output"] if status == "sent" else []}
        service.record_delivery(record, scope=SCOPE, receipt=receipt)
        return receipt, "fixture-value"
    return publish


def consume(service, event, publish, guard=lambda: ""):
    return ResultEventConsumer(service).consume(event["event_id"], scope=SCOPE, check_current=guard, publish=publish)


def test_commit_before_consume_survives_reopen(tmp_path):
    clock, store, service, task, event = setup(tmp_path)
    model, sent = service.profile.runtime, []
    fresh, resumed = restart(store, model)
    try:
        outcome = consume(resumed, event, publisher(resumed, sent))
        assert outcome["event"]["status"] == "done" and outcome["receipt"]["sent"]
        assert len(sent) == len(model.calls) == 1
        assert fresh.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session) == task
        assert not consume(resumed, event, publisher(resumed, sent))["claimed"]
        assert len(sent) == len(model.calls) == 1
    finally:
        fresh.close()


def test_crash_during_generation_falls_back_without_second_model_call(tmp_path, monkeypatch):
    clock, store, service, task, event = setup(tmp_path)
    model, sent = service.profile.runtime, []
    monkeypatch.setattr(service.archive, "finish", lambda *a, **kw: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        consume(service, event, publisher(service, sent))
    assert len(model.calls) == 1 and sent == []
    assert not consume(service, event, publisher(service, sent))["claimed"]
    clock[0] += 301
    fresh, resumed = restart(store, model)
    try:
        outcome = consume(resumed, event, publisher(resumed, sent))
        assert outcome["expression"]["status"] == "fallback"
        assert outcome["expression"]["fallback_reason"] == "expression_interrupted"
        assert outcome["event"]["status"] == "done"
        assert len(sent) == len(model.calls) == 1
        assert "原始目标" in sent[0] and "验收通过" in sent[0]
    finally:
        fresh.close()


@pytest.mark.parametrize("crash,expected,send_count", [
    ("before_send", "done", 1), ("during_send", "uncertain", 1)])
def test_crash_around_send_reuses_text_or_retains_uncertainty(tmp_path, crash, expected, send_count):
    clock, store, service, task, event = setup(tmp_path)
    model, sent = service.profile.runtime, []
    with pytest.raises(Crash):
        consume(service, event, publisher(service, sent, crash=crash))
    clock[0] += 301
    fresh, resumed = restart(store, model)
    try:
        outcome = consume(resumed, event, publisher(resumed, sent))
        assert outcome["event"]["status"] == expected
        assert len(sent) == send_count and len(model.calls) == 1
        if expected == "uncertain":
            assert not outcome["receipt"]["sent"]
            assert resumed.archive.events.diagnostics(scope=SCOPE)[0]["status"] == "uncertain"
        consume(resumed, event, publisher(resumed, sent))
        assert len(sent) == send_count and len(model.calls) == 1
    finally:
        fresh.close()


def test_crash_after_receipt_before_event_ack_does_not_resend(tmp_path, monkeypatch):
    clock, store, service, task, event = setup(tmp_path)
    model, sent = service.profile.runtime, []
    monkeypatch.setattr(service.archive.events, "finish", lambda *a, **kw: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        consume(service, event, publisher(service, sent))
    assert len(sent) == 1
    clock[0] += 301
    fresh, resumed = restart(store, model)
    try:
        outcome = consume(resumed, event, lambda *args: pytest.fail("already confirmed send replayed"))
        assert outcome["event"]["status"] == "done" and outcome["receipt"]["sent"]
        assert len(sent) == len(model.calls) == 1
    finally:
        fresh.close()


@pytest.mark.parametrize("receipt,status", [("sent", "done"), ("failed", "failed"),
    ("uncertain", "uncertain"), ("partial", "uncertain"), ("dry_run", "done"), ("suppressed", "suppressed")])
def test_delivery_outcomes_do_not_change_execution_or_retry(tmp_path, receipt, status):
    clock, store, service, task, event = setup(tmp_path)
    model, sent = service.profile.runtime, []
    outcome = consume(service, event, publisher(service, sent, status=receipt))
    assert outcome["event"]["status"] == status
    assert outcome["event"]["delivery"]["sent"] is (receipt == "sent")
    assert store.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session) == task
    assert not consume(service, event, publisher(service, sent))["claimed"]
    assert len(sent) == len(model.calls) == 1


def test_consumer_errors_before_send_are_bounded_and_reuse_expression(tmp_path):
    clock, store, service, task, event = setup(tmp_path, max_attempts=2)
    def fail(*args):
        raise ValueError("fixture unavailable before publication")
    for _ in range(2):
        assert consume(service, event, fail)["error"] == "result_consumer_error"
    assert not consume(service, event, fail)["claimed"]
    assert service.archive.events.get(event["event_id"], scope=SCOPE)["status"] == "failed"
    assert len(service.profile.runtime.calls) == 1


def test_consumer_exception_after_publication_reservation_cannot_resend(tmp_path):
    clock, store, service, task, event = setup(tmp_path)
    sent = []
    def fail(source, record, guard):
        assert service.reserve_publication(record, scope=SCOPE, check_current=guard)
        sent.append(record["text"])
        raise RuntimeError("fixture external send interrupted")
    assert consume(service, event, fail)["error"] == "result_consumer_error"
    assert service.archive.events.get(event["event_id"], scope=SCOPE)["status"] == "uncertain"
    assert not consume(service, event, fail)["claimed"] and len(sent) == 1


@pytest.mark.parametrize("mutation", ["attempt", "version", "authorization"])
def test_stale_events_or_revocation_have_no_model_or_send(tmp_path, mutation):
    clock, store, service, task, event = setup(tmp_path)
    if mutation != "authorization":
        changed = copy.deepcopy(task)
        if mutation == "attempt":
            changed["execution_attempt"] += 1
            changed["status"] = "queued"
        store.save(changed)
    guard = lambda: "authorization_revoked" if mutation == "authorization" else ""
    outcome = consume(service, event, lambda *a: pytest.fail("stale event sent"), guard)
    assert outcome["event"]["status"] == "stale"
    assert len(service.profile.runtime.calls) == 0


def test_cancellation_during_polish_is_fenced_before_send(tmp_path):
    holder = {}
    def mutate(candidate, payload):
        store, task = holder["store"], copy.deepcopy(holder["task"])
        task.update(cancel_requested=True, status="canceled")
        store.save(task)
        return candidate
    model = Model(mutate)
    clock, store, service, task, event = setup(tmp_path, model=model)
    holder.update(store=store, task=task)
    result = consume(service, event, lambda *a: pytest.fail("canceled result sent"))
    assert result["error"] == "stale_result_event"
    assert len(model.calls) == 1


def test_concurrent_consumers_do_not_buy_or_send_twice(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def pause(candidate, payload):
        entered.set()
        assert release.wait(3)
        return candidate
    clock, store, service, task, event = setup(tmp_path, model=Model(pause))
    peer = ExpressionService(ExpressionArchive(AgentTaskStore(store.path, clock=store.clock)), service.profile)
    sent, result = [], []
    thread = threading.Thread(target=lambda: result.append(consume(service, event, publisher(service, sent))))
    thread.start()
    assert entered.wait(3)
    second = consume(peer, event, publisher(peer, sent))
    assert not second["claimed"]
    release.set()
    thread.join(3)
    assert not thread.is_alive() and result[0]["event"]["status"] == "done"
    assert len(sent) == len(service.profile.runtime.calls) == 1


def test_late_generation_cannot_replace_recovered_fallback(tmp_path):
    clock, store, service, task, event = setup(tmp_path)
    source = task["execution_result"]
    key = canonical_hash(identity(source))
    record, owned = service.archive.claim(key, scope=SCOPE, result=source, config_hash=service.profile.config_hash,
        binding_hash=canonical_hash({**identity(source), "result_hash": source["result_hash"]}), processing_seconds=10)
    assert owned
    clock[0] += 11
    fallback = service.archive.recover_interrupted(key, scope=SCOPE)
    late = service.archive.finish(key, scope=SCOPE, status="ready", text="late unverified candidate")
    assert late == fallback and "late unverified" not in late["text"]


def test_event_projection_does_not_store_raw_report_or_credentials(tmp_path):
    clock, store, service, task, event = setup(tmp_path)
    projection = event["canonical_result"]
    assert "execution_report" not in projection
    assert "record_hash" in event and event["projection_hash"] == canonical_hash(projection)


def test_lease_takeover_fences_live_old_generation_and_late_usage(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def paused(candidate, payload):
        entered.set()
        assert release.wait(3)
        return candidate
    clock, store, service, task, event = setup(tmp_path, model=Model(paused))
    old, sent = [], []
    thread = threading.Thread(target=lambda: old.append(consume(service, event, publisher(service, sent))))
    thread.start()
    assert entered.wait(3)
    clock[0] += 301
    fresh = consume(service, event, publisher(service, sent))
    assert fresh["event"]["status"] == "done" and fresh["expression"]["status"] == "fallback"
    release.set()
    thread.join(3)
    assert not thread.is_alive()
    assert len(sent) == len(service.profile.runtime.calls) == 1
    assert old[0]["error"] == "result_event_lease_lost"
    current = service.archive.get(fresh["expression"]["expression_id"], scope=SCOPE)
    assert current["status"] == "fallback" and current["text"] == sent[0]
    assert current["usage"]["expression"]["total_tokens"] == 134


def test_unknown_platform_status_is_unknown_not_suppressed_or_success(tmp_path):
    clock, store, service, task, event = setup(tmp_path)
    sent = []
    outcome = consume(service, event, publisher(service, sent, status="transport_error"))
    assert outcome["event"]["status"] == "uncertain"
    assert outcome["event"]["delivery"]["status"] == "uncertain"
    assert not outcome["event"]["delivery"]["sent"]


def test_ack_write_failure_does_not_downgrade_confirmed_send(tmp_path, monkeypatch):
    clock, store, service, task, event = setup(tmp_path)
    original = service.archive.events.finish
    calls = []
    def flaky(*a, **kw):
        calls.append(kw)
        if len(calls) == 1:
            raise sqlite3.OperationalError("fixture event ack unavailable")
        return original(*a, **kw)
    import sqlite3
    monkeypatch.setattr(service.archive.events, "finish", flaky)
    sent = []
    result = consume(service, event, publisher(service, sent))
    assert result["event"]["status"] == "done" and result["receipt"]["sent"]
    assert result["value"] == "fixture-value" and result["error"] == ""
    assert len(sent) == len(service.profile.runtime.calls) == 1


def test_guard_read_failure_retains_pending_unknown_and_can_retry_without_execution(tmp_path):
    clock, store, service, task, event = setup(tmp_path)
    def unavailable():
        raise RuntimeError("fixture authority temporarily unavailable")
    result = consume(service, event, lambda *a: pytest.fail("unavailable guard sent"), unavailable)
    assert result["error"] == "expression_current_state_unavailable"
    assert result["event"]["status"] == "pending"
    assert len(service.profile.runtime.calls) == 0
    sent = []
    resumed = consume(service, event, publisher(service, sent))
    assert resumed["event"]["status"] == "done"
    assert len(sent) == len(service.profile.runtime.calls) == 1
