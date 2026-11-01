"""QQ admission/worker/executor/result/sender boundaries during recovery."""
import copy
from dataclasses import replace
import hashlib
from types import SimpleNamespace

import pytest

from shinku.agent.budget_policy import PROFILES
from shinku.agent_runtime import AgentScope, TaskHandle
from shinku.agent_runtime.recovery import TaskRecoveryGate
from shinku.agent_runtime.results import export_task_result
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.result_expression import replay_task_expression
from shinku.tools.registry import ToolRegistry
from test_qq_independent_agent import turn
import test_qq_result_expression as fixture
from test_agent_recovery import FINAL

SCOPE = AgentScope("fixture", "private:fixture")


def host(tmp_path, monkeypatch, **options):
    original = QQAgentBridge
    monkeypatch.setattr(fixture, "QQAgentBridge", lambda **kw: original(**kw, recovery_enabled=True))
    return fixture.make_host(tmp_path, **options)


def run(bridge, service, text, message_id):
    reply = bridge(turn(text, message_id=message_id))
    if reply.background_task_id:
        fixture.wait_work(service, reply.background_task_id)
    return reply


def task(service):
    return service.task_store.candidates(requester=SCOPE.requester, session=SCOPE.session)[0]


def interrupt_after_actual_write(bridge, service, execution, chat, request, independent):
    if independent:
        def fail(**options):
            execution.calls.append(options)
            raise RuntimeError("fixture execution outage")
        execution.call_chat_json = fail
    else:
        original = chat.call_chat_json
        def fail(**options):
            if chat.calls:
                chat.calls.append(options)
                raise RuntimeError("fixture execution outage")
            return original(**options)
        chat.call_chat_json = fail
    run(bridge, service, request.original_request, "fixture-message")
    saved = task(service)
    assert saved["status"] == "failed" and saved["tool_calls"]
    return saved


def continue_runtime(bridge, execution, chat, tool, saved, *, use_command=False):
    execution.call_chat_json = lambda **kw: (execution.calls.append(kw) or FINAL)
    selected = [False]
    def route(**kw):
        chat.calls.append(kw)
        if not use_command and not selected[0]:
            selected[0] = True
            return {"kind": "tool", "tool_call": {"type": tool.tool_type, "path": str(tool.path)},
                    "task_link": {"mode": "continue", "task_id": saved["task_id"]}}
        return FINAL
    chat.call_chat_json = route


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_qq_continuation_preserves_effect_and_verifies_then_delivers(tmp_path, monkeypatch, independent, asynchronous):
    service, tool, execution, request, bridge, chat, sent = host(tmp_path, monkeypatch,
        independent=independent, asynchronous=asynchronous)
    try:
        before = interrupt_after_actual_write(bridge, service, execution, chat, request, independent)
        assert tool.calls == 1
        continue_runtime(bridge, execution, chat, tool, before)
        sent_before = len(sent)
        run(bridge, service, "继续核验原来的文件", "fixture-resume")
        after = task(service)
        assert after["status"] == "completed", after
        assert after["execution_result"]["verification"]["accepted"]
        assert after["execution_attempt"] == 2 and after["contract"] == before["contract"]
        assert after["requests"][:len(before["requests"])] == before["requests"]
        assert after["budget"]["requests"] >= before["budget"]["requests"]
        assert len(after["recovery_history"]) == 1
        assert after["recovery_history"][0]["original_stop_reason"] == before.get("stop_reason", "")
        assert any(row.get("executed") is False for row in after["tool_calls"])
        event = bridge.task_store.result_events.get(bridge.task_store.result_events.key(after["execution_result"]), scope=SCOPE)
        assert event["status"] == "done" and event["delivery"]["sent"]
        assert tool.calls == 1 and len(sent) - sent_before == len(event["delivery"]["message_ids"])
        assert len(event["delivery"]["message_ids"]) == len(set(event["delivery"]["message_ids"]))
        # Consuming again never executes, polishes, or delivers a second time.
        counts = tool.calls, len(execution.calls), len(chat.calls), len(sent), after["revision"]
        replay_task_expression(bridge, turn("继续", message_id="read-only-event"), task_id=after["task_id"], context={})
        assert counts == (tool.calls, len(execution.calls), len(chat.calls), len(sent), task(service)["revision"])
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("change", ["unknown", "changed"])
def test_qq_recovery_guard_rejects_uncertain_effect_before_selected_tool(tmp_path, monkeypatch, independent, asynchronous, change):
    service, tool, execution, request, bridge, chat, sent = host(tmp_path, monkeypatch,
        independent=independent, asynchronous=asynchronous)
    try:
        before = interrupt_after_actual_write(bridge, service, execution, chat, request, independent)
        if change == "unknown":
            before["inflight"] = [{"name": tool.tool_type, "arguments": {"path": str(tool.path)}, "status": "unknown"}]
            before = service.task_store.save(before)
        else:
            tool.path.write_text("changed by another actor", encoding="utf-8")
        continue_runtime(bridge, execution, chat, tool, before)
        execution_before = len(execution.calls)
        reply = run(bridge, service, "继续原任务", "fixture-resume")
        after = task(service)
        assert after["execution_attempt"] == before["execution_attempt"]
        assert after["revision"] == before["revision"]
        assert after.get("recovery_history", []) == []
        assert tool.calls == 1 and len(execution.calls) == execution_before
        if asynchronous:
            work = service.coordinator.get(reply.background_task_id)
            assert work["status"] == "failed" and not work["worker_active"]
        else:
            assert "recovery_effect" in reply.reason
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize("delivery", ["sent", "uncertain", "failed", "dry_run"])
def test_committed_execution_has_delivery_recovery_only_and_no_execution_resume(tmp_path, monkeypatch, delivery):
    service, tool, execution, request, bridge, chat, sent = host(tmp_path, monkeypatch, delivery=delivery)
    try:
        run(bridge, service, request.original_request, "fixture-message")
        saved = task(service)
        current = TaskHandle(saved["task_id"], saved["execution_attempt"], saved["revision"])
        before = tool.calls, len(execution.calls), len(chat.calls), len(sent), saved["revision"]
        report = bridge.agent_calls.recovery(turn("核对恢复状态"), current)
        assert report["status"] == "delivery_only" and report["delivery"]["status"] == delivery
        assert not report["automatic_send"]
        replay_task_expression(bridge, turn("结果", message_id="fixture-replay"), task_id=saved["task_id"], context={})
        assert before == (tool.calls, len(execution.calls), len(chat.calls), len(sent), task(service)["revision"])
        assert tool.path.read_bytes() == b"standalone agent verified"
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize("unknown", [False, True])
def test_budget_append_is_after_effect_check_and_atomic_with_worker_claim(tmp_path, monkeypatch, unknown):
    service, tool, execution, request, bridge, chat, sent = host(tmp_path, monkeypatch, asynchronous=True)
    bridge.adaptive_budget_enabled = True
    bridge.task_budget_limits = PROFILES["heavy"]
    try:
        before = interrupt_after_actual_write(bridge, service, execution, chat, request, True)
        before["budget"]["reason"] = "task_model_budget_exhausted"
        before["budget"]["requests"] = before["budget"]["max_requests"]
        before["stop_reason"] = before["notes"]["blocker"] = "task_model_budget_exhausted"
        if unknown:
            before["inflight"] = [{"name": tool.tool_type, "arguments": {"path": str(tool.path)}, "status": "unknown"}]
        before = service.task_store.save(before)
        continue_runtime(bridge, execution, chat, tool, before, use_command=True)
        text = "/agent resume " + before["task_id"] + " standard"
        reply = run(bridge, service, text, "fixture-top-up")
        after = task(service)
        if unknown:
            assert "recovery_effect_unknown" in reply.reason
            assert after == before
            assert after.get("budget_allocation", {}).get("grants", []) == []
        else:
            assert after["status"] == "completed"
            assert len(after["budget_allocation"]["grants"]) == 1
            assert after["budget"]["requests"] > before["budget"]["requests"]
            assert after["recovery_history"][0]["budget_stop_reason"] == "task_model_budget_exhausted"
            old_limits = after["budget_allocation"]["limits"]
            run(bridge, service, text, "fixture-top-up")
            assert task(service)["budget_allocation"]["limits"] == old_limits
            assert len(task(service)["budget_allocation"]["grants"]) == 1
        assert tool.calls == 1
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_queue_rejection_never_commits_recovery_or_added_budget(tmp_path, monkeypatch):
    service, tool, execution, request, bridge, chat, sent = host(tmp_path, monkeypatch, asynchronous=True)
    bridge.adaptive_budget_enabled = True
    bridge.task_budget_limits = PROFILES["heavy"]
    try:
        before = interrupt_after_actual_write(bridge, service, execution, chat, request, True)
        before["budget"]["reason"] = before["stop_reason"] = before["notes"]["blocker"] = "task_model_budget_exhausted"
        before = service.task_store.save(before)
        monkeypatch.setattr(service.coordinator, "submit", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("queue full")))
        reply = bridge(turn("/agent resume " + before["task_id"] + " standard", message_id="fixture-grant"))
        assert reply.reason == "background_task_admission_failed"
        assert task(service) == before and tool.calls == 1
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_option_defaults_off_and_requires_contract(tmp_path):
    service, tool, execution, request, bridge, chat, sent = fixture.make_host(tmp_path)
    try:
        assert not bridge.recovery_enabled
        with pytest.raises(ValueError, match="task_recovery_requires_contract_store"):
            QQAgentBridge(runtime=chat, system_prompt="fixture", recovery_enabled=True)
        # The inspect API does not introduce a new table or scheduler.
        import sqlite3
        if service.task_store.path:
            with sqlite3.connect(service.task_store.path) as db:
                assert not db.execute("SELECT name FROM sqlite_master WHERE name LIKE '%recovery%'").fetchall()
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("stale", [False, True])
def test_pending_approval_checks_current_ledger_and_keeps_unsafe_approval(tmp_path, monkeypatch, asynchronous, independent, stale):
    from shinku.agent.approval import ConsentDecision
    service, tool, execution, request, bridge, chat, sent = host(tmp_path, monkeypatch,
        asynchronous=asynchronous, independent=independent)
    bridge.command_state.set_permission_mode("confirm")
    bridge.allowed_tool_names = {tool.tool_type}
    class Consent:
        def resolve(self, **kw):
            return ConsentDecision("approve", 1.0, "fixture approval")
    bridge.consent_resolver = Consent()
    try:
        run(bridge, service, request.original_request, "fixture-message")
        pending = bridge.approval_store.get("private:fixture")
        assert pending is not None and tool.calls == 0
        if stale:
            saved = task(service)
            saved["notes"]["blocker"] = "newer checkpoint"
            service.task_store.save(saved)
        reply = run(bridge, service, "同意执行", "fixture-approval")
        if stale:
            assert reply.reason == "task_version_conflict" and tool.calls == 0
            assert bridge.approval_store.get("private:fixture") is not None
        else:
            assert tool.calls == 1 and task(service)["status"] == "completed"
            assert bridge.approval_store.get("private:fixture") is None
            assert task(service)["execution_result"]["verification"]["accepted"]
            assert task(service)["execution_attempt"] == 2
            assert task(service)["recovery_history"][0]["recovery_reason"] == "explicit_approval"
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize("independent", [False, True])
def test_sync_approval_rejects_another_requester_before_deleting_or_executing(tmp_path, monkeypatch, independent):
    from shinku.agent.approval import ConsentDecision
    service, tool, execution, request, bridge, chat, sent = host(tmp_path, monkeypatch, independent=independent)
    bridge.command_state.set_permission_mode("confirm")
    bridge.allowed_tool_names = {tool.tool_type}
    class Consent:
        def resolve(self, **kw):
            return ConsentDecision("approve", 1.0, "fixture approval")
    bridge.consent_resolver = Consent()
    try:
        run(bridge, service, request.original_request, "fixture-message")
        saved = task(service)
        pending = bridge.approval_store.get("private:fixture")
        assert pending is not None and tool.calls == 0
        reply = bridge(turn("同意执行", user="another-requester", message_id="foreign-approval"))
        assert reply.reason == "recovery_scope_mismatch"
        assert bridge.approval_store.get("private:fixture") == pending
        assert task(service) == saved and tool.calls == 0
        assert not tool.path.exists()
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()
