"""P4a end-to-end: real QQ, file/verifier, SQLite reopen, expression and send."""
import copy
from dataclasses import replace
import threading

import pytest

from shinku.agent.task_state import AgentTaskStore
from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.task_view import TaskViewReader
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.result_expression import replay_task_expression
from shinku.qq.task_status import reader_for_bridge
from shinku.tasks.coordinator import AgentTaskCoordinator
from test_qq_independent_agent import turn
from test_qq_result_expression import make_host
from test_result_event_consumption import Crash

SCOPE = AgentScope("fixture", "private:fixture")


def reopen(previous, *, store_path, coordinator_path, clock):
    store = AgentTaskStore(store_path, clock=clock)
    coordinator = AgentTaskCoordinator(state_path=coordinator_path, clock=clock, max_workers=1)
    bridge = QQAgentBridge(runtime=previous.runtime, system_prompt="offline persona fixture",
        tool_host=previous.tool_host, task_store=store, task_coordinator=coordinator,
        task_contract_enabled=True, command_state=previous.command_state,
        execution_profile=previous.execution_adapter.executor.profile if previous.execution_adapter else None,
        expression_profile=previous.result_expression.profile, sender=previous.sender,
        task_budget_limits=previous.task_budget_limits)
    return store, coordinator, bridge


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("crash", ["commit", "generation", "before_send", "during_send", "after_receipt"])
def test_crash_reopen_and_original_route_replay_has_one_execution_and_publication(
        tmp_path, monkeypatch, asynchronous, independent, crash):
    clock = [1000.0]
    store_path, work_path = tmp_path / "business.db", tmp_path / "workers.db"
    store = AgentTaskStore(store_path, clock=lambda: clock[0])
    coordinator = AgentTaskCoordinator(state_path=work_path, clock=store.clock, max_workers=1)
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path,
        asynchronous=asynchronous, independent=independent, task_store=store, coordinator=coordinator)
    archives = bridge.result_expression.archive
    model = bridge.result_expression.profile.runtime
    def crash_call(*a, **kw):
        raise Crash()
    if crash == "commit":
        monkeypatch.setattr(archives.events, "claim", crash_call)
    elif crash == "generation":
        monkeypatch.setattr(archives, "finish", crash_call)
    elif crash == "before_send":
        monkeypatch.setattr(bridge.result_expression, "reserve_publication", crash_call)
    elif crash == "during_send":
        original = bridge.sender
        def crashing_sender(message):
            original(message)  # The platform may have accepted the operation.
            raise Crash()
        monkeypatch.setattr(bridge, "sender", crashing_sender)
    else:
        monkeypatch.setattr(archives.events, "finish", crash_call)
    original_turn = turn(request.original_request)
    if asynchronous:
        queued = bridge(original_turn)
        assert queued.agent_status == "waiting_background"
        with pytest.raises(Crash):
            coordinator._records[queued.background_task_id].future.result(timeout=5)
    else:
        with pytest.raises(Crash):
            bridge(original_turn)
    task = store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
    assert task["status"] == "completed" and task["completion"]["accepted"]
    assert tool.calls == 1 and tool.path.is_file()
    events = archives.events.pending(scope=SCOPE)
    event = next(e for e in events if e["version"] == task["revision"])
    assert event["source_hash"] == task["execution_result"]["result_hash"]
    before = (tool.calls, len(execution.calls), len(chat.calls), tool.path.read_bytes(), task["revision"])
    already_sent = len(sent)
    coordinator.shutdown(wait=True)
    store.close()
    clock[0] += 301
    # Restore the actual test transport before bootstrapping a fresh host.
    if crash == "during_send":
        monkeypatch.setattr(bridge, "sender", original)
    fresh, fresh_work, recovered = reopen(bridge, store_path=store_path, coordinator_path=work_path,
        clock=lambda: clock[0])
    try:
        view = reader_for_bridge(recovered).get(SCOPE, task["task_id"])
        assert view["status"] == "completed" and view["verification"]["verified"]
        assert view["result_event"]["status"] in {"pending", "processing"}
        if asynchronous:
            assert view["execution"]["coordinator_status"] == "failed"
            assert view["execution"]["coordinator_recovered_after_restart"]
        reply = replay_task_expression(recovered, original_turn, task_id=task["task_id"],
            context={"agent_authority_revision": recovered.command_state.authority_revision()})
        after = fresh.get(task["task_id"], requester=SCOPE.requester, session=SCOPE.session)
        assert before == (tool.calls, len(execution.calls), len(chat.calls), tool.path.read_bytes(), after["revision"])
        final_event = fresh.result_events.get(event["event_id"], scope=SCOPE)
        if crash == "during_send":
            assert final_event["status"] == "uncertain" and not final_event["delivery"]["sent"]
            assert not reply.sent and len(sent) == already_sent
        else:
            assert final_event["status"] == "done" and final_event["delivery"]["sent"]
            assert (len(sent) > already_sent) is (crash != "after_receipt")
        assert len(model.calls) == 1
        if crash == "generation":
            assert "……结果整理" not in reply.final_text and "原始目标" in reply.final_text
        stable = (len(sent), len(model.calls))
        replay_task_expression(recovered, original_turn, task_id=task["task_id"], context={})
        assert stable == (len(sent), len(model.calls))
        view = reader_for_bridge(recovered).get(SCOPE, task["task_id"])
        assert view["status"] == "completed" and view["delivery"]["source"] == "agent_result_events"
        assert view["delivery"]["sent"] is (crash != "during_send")
    finally:
        fresh_work.shutdown(wait=True)
        fresh.close()


def test_replay_rejects_wrong_route_scope_and_revoked_authority(tmp_path):
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path)
    original = bridge.result_expression.archive.events.claim
    bridge.result_expression.archive.events.claim = lambda *a, **k: (_ for _ in ()).throw(Crash())
    with pytest.raises(Crash):
        bridge(turn(request.original_request))
    bridge.result_expression.archive.events.claim = original
    task = service.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
    before = (len(sent), len(bridge.result_expression.profile.runtime.calls), tool.calls, len(execution.calls))
    try:
        wrong = replay_task_expression(bridge, turn("other", message_id="other"), task_id=task["task_id"], context={})
        assert wrong.reason == "replay_route_unconfirmed"
        cross = replay_task_expression(bridge, turn(request.original_request, user="other"), task_id=task["task_id"], context={})
        assert cross.reason == "replay_result_unavailable"
        revision = bridge.command_state.authority_revision()
        bridge.command_state.set_permission_mode("readonly")
        revoked = replay_task_expression(bridge, turn(request.original_request), task_id=task["task_id"],
            context={"agent_authority_revision": revision})
        assert revoked.reason == "agent_permission_changed" and not revoked.sent
        assert before == (len(sent), len(bridge.result_expression.profile.runtime.calls), tool.calls, len(execution.calls))
    finally:
        service.coordinator.shutdown(wait=True)


@pytest.mark.parametrize("tamper", ["worker_attempt", "real_failure", "result_hash", "cancel", "missing_event"])
def test_only_exact_committed_restart_binding_relaxes_worker_conflict(tmp_path, tamper):
    store = AgentTaskStore(tmp_path / "business.db")
    from shinku.agent_runtime.result_events import ResultEventStore
    from test_result_events import terminal
    events = ResultEventStore(store)
    task = terminal(store)
    task.update(execution_work_id="fixture-work", execution_work_attempt=1)
    task = store.save(task)
    work = {"status": "failed", "worker_active": False, "attempt": 1,
        "profile_user_id": SCOPE.requester, "session_id": SCOPE.session,
        "metadata": {"agent_task_id": task["task_id"], "recovered_after_restart": True}}
    assert events.committed_after_restart(task, work)
    if tamper == "worker_attempt":
        work["attempt"] = 2
    elif tamper == "real_failure":
        work["metadata"].pop("recovered_after_restart")
    elif tamper == "result_hash":
        task["execution_result"]["verification"]["accepted"] = False
    elif tamper == "cancel":
        work["cancel_requested"] = True
    else:
        with store._transaction() as db:
            db.execute("DELETE FROM agent_result_events")
    assert not events.committed_after_restart(task, work)


def test_standalone_queued_cancellation_commits_result_and_event_without_execution(tmp_path):
    from shinku.agent_runtime.demo import demo_service, demo_request
    from shinku.agent_runtime.result_events import ResultEventStore
    service, tool, runtime = demo_service(tmp_path)
    events = ResultEventStore(service.task_store)
    entered, release = threading.Event(), threading.Event()
    def block(event):
        entered.set()
        assert release.wait(5)
    service.coordinator.submit(block, task_id="fixture-blocker")
    assert entered.wait(3)
    client = service.bind(SCOPE)
    try:
        handle = client.submit(demo_request(tool))
        canceled = client.cancel(handle)
        assert canceled["stopped"] and not canceled["files_rolled_back"]
        result = client.result(handle.task_id)
        assert result["execution"]["status"] == "canceled" and not result["verification"]["accepted"]
        assert len(events.pending(scope=SCOPE)) == 1
        assert tool.calls == len(runtime.calls) == 0 and not tool.path.exists()
    finally:
        release.set()
        service.coordinator.shutdown(wait=True)


@pytest.mark.parametrize("independent", [False, True])
def test_canceled_queued_continuation_has_canonical_event_and_explicit_delivery(tmp_path, independent):
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path,
        asynchronous=True, independent=independent)
    if independent:
        bridge.execution_adapter.executor.profile = replace(bridge.execution_adapter.executor.profile, max_rounds=1)
    else:
        bridge.max_rounds = 1
    entered, release = threading.Event(), threading.Event()
    original_tool = tool.execute
    def blocker(event):
        entered.set()
        assert release.wait(5)
    def execute(**kw):
        result = original_tool(**kw)
        service.coordinator.submit(blocker, task_id="fixture-blocker")
        return result
    tool.execute = execute
    try:
        queued = bridge(turn(request.original_request))
        assert entered.wait(3)
        response = service.coordinator.cancel(queued.background_task_id,
            profile_user_id=SCOPE.requester, session_id=SCOPE.session)
        assert response["task"]["status"] == "canceled"
        task = service.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        assert task["execution_result"]["execution"]["status"] == "canceled"
        events = service.task_store.result_events.pending(scope=SCOPE)
        assert len(events) == 1 and events[0]["source_hash"] == task["execution_result"]["result_hash"]
        assert len(sent) == len(bridge.result_expression.profile.runtime.calls) == 0
        assert tool.calls == 1 and tool.path.exists()
        replay = replay_task_expression(bridge, turn(request.original_request), task_id=task["task_id"], context={})
        assert replay.sent and "任务已取消" in replay.final_text and "验收通过" not in replay.final_text
        assert tool.calls == 1 and tool.path.exists()
    finally:
        release.set()
        service.coordinator.shutdown(wait=True)


@pytest.mark.parametrize("independent", [False, True])
def test_terminal_checkpoint_does_not_commit_terminal_status_without_event(tmp_path, monkeypatch, independent):
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path, independent=independent)
    original = service.task_store.result_events.append_in_transaction
    def crash_before_commit(db, task):
        original(db, task)
        if task.get("execution_result"):
            raise Crash()
    monkeypatch.setattr(service.task_store.result_events, "append_in_transaction", crash_before_commit)
    try:
        with pytest.raises(Crash):
            bridge(turn(request.original_request))
        task = service.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        assert task["status"] == "running" and task["result_commit_pending"] == "completed"
        assert task["completion"]["accepted"] and not task.get("execution_result")
        assert service.task_store.result_events.pending(scope=SCOPE) == []
        view = reader_for_bridge(bridge).get(SCOPE, task["task_id"])
        assert view["status"] != "completed" and view["execution"]["finalization_pending"]
        assert tool.calls == 1 and len(sent) == len(bridge.result_expression.profile.runtime.calls) == 0
    finally:
        service.coordinator.shutdown(wait=True)


def test_event_consumer_storage_failure_preserves_accepted_execution_and_pending_event(tmp_path, monkeypatch):
    service, tool, execution, request, bridge, chat, sent = make_host(tmp_path)
    original = service.task_store.result_events.claim
    def unavailable(*args, **kw):
        raise RuntimeError("fixture consumer store unavailable")
    monkeypatch.setattr(service.task_store.result_events, "claim", unavailable)
    try:
        reply = bridge(turn(request.original_request))
        assert reply.agent_status == "completed" and reply.reason == "result_event_store_unavailable"
        task = service.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]
        assert task["status"] == "completed" and task["completion"]["accepted"]
        assert len(service.task_store.result_events.pending(scope=SCOPE)) == 1
        assert tool.calls == 1 and len(sent) == len(bridge.result_expression.profile.runtime.calls) == 0
        monkeypatch.setattr(service.task_store.result_events, "claim", original)
        replay = replay_task_expression(bridge, turn(request.original_request), task_id=task["task_id"], context={})
        assert replay.sent and tool.calls == 1 and len(bridge.result_expression.profile.runtime.calls) == 1
    finally:
        service.coordinator.shutdown(wait=True)
