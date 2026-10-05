"""General background Agent tests: no document adapter, QQ or saved memory."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest

from shinku.agent.loop import AgentDecision, AgentLoop
from shinku.agent.approval import ConsentDecision, make_tool_approval_request
from shinku.agent.pending import JsonPendingApprovalStore, PendingApprovalRecord
from shinku.api.agent_work import build_agent_work_router
from shinku.hosts.tool_host import ToolHost
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.attention import AttentionBatch, AttentionDecision
from shinku.qq.turns import QQTurn, NapCatTurnDispatcher
from shinku.tasks import coordinator as task_module
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.invocation import ToolInvocation
from shinku.tools.registry import ToolRegistry


def wait_for(coordinator, task_id, status, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = coordinator.get(task_id)
        if record and record["status"] == status and not record.get("worker_active"):
            return record
        time.sleep(0.005)
    raise AssertionError(f"{task_id} did not finish with {status}: {coordinator.get(task_id)}")


def turn(text="inspect these files", *, sender="owner", kind="private", conversation=None, message_id=None):
    identifier = message_id or f"m-{time.time_ns()}"
    message = IncomingMessage.from_event({
        "post_type": "message", "message_type": kind, "user_id": sender,
        "group_id": conversation if kind == "group" else None,
        "message_id": identifier, "mentioned_bot": kind == "group",
        "message": [{"type": "text", "data": {"text": text}}],
    })
    return QQTurn(conversation_id=conversation or sender, conversation_type=kind,
                  primary_message_id=identifier, message_ids=(identifier,), combined_text=text,
                  messages=(message,))


class Runtime:
    def chat_supports_native_tools(self):
        return True

    def call_chat_json(self, **kwargs):
        return {"kind": "final", "text": "Inspection complete."}


class BlockingInspector:
    tool_type = "inspect_files"
    agent_permission_class = "read"
    agent_progress_class = "observe"
    description = "Inspect a batch of files."
    input_schema = {"type": "object", "properties": {}}

    def __init__(self, entered, release):
        self.entered, self.release, self.calls = entered, release, 0

    def normalize_call(self, call):
        return dict(call)

    def execute(self, *, call, context):
        self.calls += 1
        self.entered.set()
        if not self.release.wait(3):
            raise TimeoutError("test fixture was not released")
        return {"tool_type": self.tool_type, "followup_context": "three files inspected"}


class RecordingMutator:
    tool_type = "write_fixture"
    agent_permission_class = "change"
    agent_progress_class = "change"
    agent_completion_effects = ("write",)
    agent_completion_target_arguments = ("path",)
    description = "Write one fixture."
    input_schema = {"type": "object", "properties": {"path": {"type": "string"}}}

    def __init__(self):
        self.calls = 0

    def normalize_call(self, call):
        return dict(call)

    def execute(self, **kwargs):
        self.calls += 1
        return {"followup_context": "fixture written"}


class CoordinatorAsyncTests(unittest.TestCase):
    def test_queued_cancel_never_calls_handler_and_processor_can_close(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        called = []
        coordinator.submit(lambda event: called.append(1), task_id="queued", defer_start=True,
                           metadata={"retry_policy": "fresh_continuation_only"})
        self.assertEqual(coordinator.cancel("queued")["task"]["status"], "canceled")
        self.assertFalse(coordinator.dispatch("queued"))
        self.assertIsNone(coordinator._records["queued"].handler)
        coordinator.shutdown()
        with self.assertRaisesRegex(ValueError, "task_processor_closed"):
            coordinator.submit(lambda event: None)
        self.assertEqual(called, [])

    def test_restart_retains_evidence_without_executable_replay(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.sqlite3"
            old = AgentTaskCoordinator(state_path=path)
            old.submit(lambda event: self.fail("must not run"), task_id="interrupted", defer_start=True,
                       metadata={"tool_calls": 2, "agent_task_id": "ledger"})
            new = AgentTaskCoordinator(state_path=path)
            try:
                record = new.get("interrupted")
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["metadata"]["tool_calls"], 2)
                self.assertTrue(record["metadata"]["recovered_after_restart"])
                self.assertFalse(record["worker_active"])
                self.assertEqual(new.retry("interrupted")["reason"], "retry_handler_missing")
            finally:
                new.shutdown()
                old.shutdown()

    def test_history_is_bounded_but_old_admitted_work_survives(self):
        with TemporaryDirectory() as directory:
            coordinator = AgentTaskCoordinator(state_path=Path(directory) / "tasks.sqlite3", max_persisted_records=2)
            try:
                coordinator.submit(lambda event: None, task_id="old-queued", defer_start=True)
                for n in range(4):
                    coordinator.submit(lambda event: "done", task_id=f"terminal-{n}",
                                       metadata={"retry_policy": "fresh_continuation_only"})
                    wait_for(coordinator, f"terminal-{n}", "completed")
                self.assertIsNotNone(coordinator.get("old-queued"))
                self.assertIsNone(coordinator.get("terminal-0"))
                self.assertEqual(len(coordinator.list()), 3)
                self.assertIsNone(coordinator._records["terminal-3"].handler)
            finally:
                coordinator.shutdown()

    def test_local_api_scope_filters_and_rejects_cross_session_cancel(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        coordinator = AgentTaskCoordinator()
        try:
            coordinator.submit(lambda event: None, task_id="private-task", defer_start=True,
                               profile_user_id="owner", session_id="private:42")
            app = FastAPI()
            app.include_router(build_agent_work_router(coordinator))
            with TestClient(app) as client:
                self.assertEqual(client.get("/agent-work/tasks", params={"session_id": "group:42"}).json()["tasks"], [])
                self.assertEqual(client.get("/agent-work/tasks/private-task", params={"profile_user_id": "other"}).status_code, 404)
                self.assertEqual(client.post("/agent-work/tasks/private-task/cancel", json={"session_id": "group:42"}).status_code, 409)
                self.assertEqual(coordinator.get("private-task")["status"], "queued")
        finally:
            coordinator.shutdown()

    def test_structured_failure_is_not_recorded_as_success(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        try:
            coordinator.submit(lambda event: task_module.AgentTaskOutcome(
                status="failed", summary="not complete", error="planner_failed"), task_id="bad")
            record = wait_for(coordinator, "bad", "failed")
            self.assertEqual(record["error"], "planner_failed")
        finally:
            coordinator.shutdown(wait=True)

    def test_cancel_running_is_a_request_until_worker_has_stopped(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release = Event(), Event()
        def work(event):
            entered.set()
            release.wait(3)
            return "done"
        try:
            coordinator.submit(work, task_id="busy", profile_user_id="owner", session_id="private:owner")
            self.assertTrue(entered.wait(1))
            canceled = coordinator.cancel("busy", profile_user_id="owner", session_id="private:owner")
            self.assertTrue(canceled["ok"])
            self.assertEqual(canceled["task"]["status"], "running")
            self.assertTrue(canceled["task"]["worker_active"])
            self.assertTrue(canceled["task"]["cancel_requested"])
            self.assertFalse(coordinator.retry("busy", profile_user_id="owner")["ok"])
            release.set()
            wait_for(coordinator, "busy", "canceled")
        finally:
            release.set()
            coordinator.shutdown(wait=True)

    def test_deferred_dispatch_and_queue_limit(self):
        coordinator = AgentTaskCoordinator(max_workers=1, max_pending_tasks=1)
        called = Event()
        try:
            coordinator.submit(lambda event: called.set(), task_id="deferred", defer_start=True)
            self.assertFalse(called.is_set())
            with self.assertRaisesRegex(ValueError, "task_queue_full"):
                coordinator.submit(lambda event: None, task_id="overflow", defer_start=True)
            self.assertTrue(coordinator.dispatch("deferred"))
            wait_for(coordinator, "deferred", "completed")
            self.assertTrue(called.is_set())
            self.assertFalse(coordinator.dispatch("deferred"))
        finally:
            coordinator.shutdown(wait=True)

    def test_owner_and_typed_session_controls_are_isolated(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        try:
            coordinator.submit(lambda event: None, task_id="scoped", profile_user_id="owner",
                               session_id="private:42", defer_start=True)
            self.assertIsNone(coordinator.get("scoped", profile_user_id="other", session_id="private:42"))
            self.assertIsNone(coordinator.get("scoped", profile_user_id="owner", session_id="group:42"))
            self.assertFalse(coordinator.control("cancel", "scoped", "owner", session_id="group:42")["ok"])
            self.assertEqual(coordinator.list(profile_user_id="owner", session_id="group:42"), [])
            self.assertTrue(coordinator.control("cancel", "scoped", "owner", session_id="private:42")["ok"])
        finally:
            coordinator.shutdown(wait=True)

    def test_general_agent_retry_requires_fresh_continuation_not_replay(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        try:
            coordinator.submit(lambda event: task_module.AgentTaskOutcome(status="failed"), task_id="no-replay",
                               metadata={"retry_policy": "fresh_continuation_only"})
            wait_for(coordinator, "no-replay", "failed")
            result = coordinator.retry("no-replay")
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "fresh_continuation_required")
        finally:
            coordinator.shutdown(wait=True)

    def test_waiting_user_and_fresh_reopen_preserve_attempt_history(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        try:
            coordinator.submit(lambda event: task_module.AgentTaskOutcome(status="waiting_user", summary="approve?"),
                               task_id="ask", profile_user_id="owner", session_id="private:owner")
            wait_for(coordinator, "ask", "waiting_user")
            coordinator.submit(lambda event: "confirmed", task_id="ask", profile_user_id="owner",
                               session_id="private:owner", allow_reopen=True)
            record = wait_for(coordinator, "ask", "completed")
            self.assertEqual(record["attempt"], 2)
            self.assertEqual(record["metadata"]["attempt_history"][0]["status"], "waiting_user")
        finally:
            coordinator.shutdown(wait=True)


class AgentInterruptTests(unittest.TestCase):
    def test_cancel_between_tools_stops_without_starting_second_tool(self):
        event = Event()
        executed = []
        class Handler:
            tool_type = "inspect_files"
            def normalize_call(self, call):
                return dict(call)
            def execute(self, *, call, context):
                executed.append(call["n"])
                event.set()
                return {"followup_context": "done"}
        loop = AgentLoop(planner=lambda **kwargs: AgentDecision.final("done"),
                         handlers={"inspect_files": Handler()}, policy=ExecutionPolicy(default_risk="low"),
                         interruption_check=lambda state: "task_canceled" if event.is_set() else "")
        result = loop.run(first_decision=AgentDecision.tools([
            ToolInvocation("inspect_files", {"n": 1}), ToolInvocation("inspect_files", {"n": 2})]))
        self.assertEqual(result.status, "canceled")
        self.assertEqual(executed, [1])
        self.assertEqual(len(result.tool_envelopes), 1)

    def test_permission_revoked_during_planner_prevents_selected_operation(self):
        revoked, executed = Event(), []
        class Handler:
            def normalize_call(self, call):
                return dict(call)
            def execute(self, *, call, context):
                executed.append(1)
                return {}
        def plan(**kwargs):
            revoked.set()
            return AgentDecision.tool(ToolInvocation("write_files", {}))
        loop = AgentLoop(planner=plan, handlers={"write_files": Handler()},
                         interruption_check=lambda state: "agent_permission_changed" if revoked.is_set() else "")
        result = loop.run()
        self.assertEqual((result.status, result.reason), ("blocked", "agent_permission_changed"))
        self.assertEqual(executed, [])


class QQAsyncProcessorTests(unittest.TestCase):
    def test_read_first_action_or_unknown_intent_is_not_mistaken_for_lightweight_chat(self):
        class Read:
            tool_type = "private_file_read"
            agent_permission_class = "read"
            agent_progress_class = "observe"
            def normalize_call(self, call):
                return dict(call)
            def execute(self, **kwargs):
                return {"followup_context": "42"}
        coordinator = AgentTaskCoordinator()
        try:
            bridge = self.make_bridge(coordinator, Read())
            read = AgentDecision.tool(ToolInvocation("private_file_read", {"path": "fixture.txt"}))
            self.assertFalse(bridge._is_lightweight_perception(read))
            self.assertFalse(bridge._is_lightweight_perception(replace(read, task_link={"mode": "new"})))
            self.assertTrue(bridge._is_lightweight_perception(replace(read, task_link={"mode": "none"})))
        finally:
            coordinator.shutdown()

    def test_small_async_catalog_carries_semantic_intent_in_actual_model_schema(self):
        coordinator = AgentTaskCoordinator()
        entered, release = Event(), Event()
        release.set()
        try:
            bridge = self.make_bridge(coordinator, BlockingInspector(entered, release))
            captured = []
            class CaptureRuntime(Runtime):
                def call_chat_json(self, **kwargs):
                    captured.append(kwargs)
                    return {"kind": "final", "text": "hello"}
            bridge.runtime = CaptureRuntime()
            del bridge._run_chat_decision
            bridge.handle_turn(turn("CHAT"))
            specs = captured[0]["native_tools"]
            self.assertEqual(len(specs), 1)
            parameters = specs[0]["function"]["parameters"]
            self.assertIn("task_link", parameters["required"])
            self.assertIn("none", parameters["properties"]["task_link"]["properties"]["mode"]["enum"])
        finally:
            coordinator.shutdown(wait=True)

    def make_bridge(self, coordinator, handler, sent=None, *, async_enabled=True, task_contract_enabled=False):
        state = QQCommandState()
        state.set_permission_mode("full")
        bridge = QQAgentBridge(runtime=Runtime(), system_prompt="persona",
            tool_host=ToolHost(ToolRegistry({handler.tool_type: handler})), command_state=state,
            master_user_ids=("owner",), owner_gate_required=True,
            completion_gate=lambda state, text: True,
            task_coordinator=coordinator, async_agent_enabled=async_enabled,
            task_contract_enabled=task_contract_enabled,
            sender=(lambda message: sent.append(message) or DeliveryResult(True, "sent", (f"out-{len(sent)}",))) if sent is not None else None)
        def choose(**kwargs):
            request = kwargs["context"]["qq_turn"].combined_text
            if request == "CHAT":
                return AgentDecision.final("I'm here."), {handler.tool_type}
            return AgentDecision.tool(ToolInvocation(handler.tool_type, {})), {handler.tool_type}
        bridge._run_chat_decision = choose
        return bridge

    def test_explicit_none_route_is_rejected_before_sync_or_async_task_admission(self):
        contract = {
            "goal": "写入 fixture",
            "deliverables": ["fixture 写入回执"],
            "scope": ["仅 fixture"],
            "requirements": [{
                "id": "r1", "description": "fixture 必须写入", "effect": "write",
                "targets": ["fixture.txt"], "checks": {},
            }],
        }
        for async_enabled in (False, True):
            with self.subTest(async_enabled=async_enabled):
                coordinator = AgentTaskCoordinator(max_workers=1)
                handler = RecordingMutator()
                class RuntimeSpy:
                    def __init__(self):
                        self.calls = []

                    def call_chat_json(self, **kwargs):
                        self.calls.append(kwargs)
                        raise AssertionError("invalid route must be rejected before metadata calls")

                runtime = RuntimeSpy()
                try:
                    bridge = self.make_bridge(coordinator, handler, async_enabled=async_enabled,
                                              task_contract_enabled=True)
                    bridge.runtime = runtime
                    decision = AgentDecision.tool(ToolInvocation(handler.tool_type, {"path": "fixture.txt"}))
                    decision = decision.with_task_metadata({"task_link": {"mode": "none"}, "task_contract": contract})
                    bridge._run_chat_decision = lambda **kwargs: (decision, {handler.tool_type})

                    reply = bridge.handle_turn(turn("写入 fixture.txt"))

                    self.assertEqual(reply.agent_status, "blocked")
                    self.assertEqual(reply.reason, "task_contract_error:task_handoff_requires_new_or_continue")
                    self.assertIn("没有执行操作", reply.final_text)
                    self.assertEqual(handler.calls, 0)
                    self.assertEqual(runtime.calls, [])
                    self.assertEqual(coordinator.list(), [])
                    requester, session = bridge._task_scope(turn("写入 fixture.txt"))
                    self.assertEqual(bridge.task_store.candidates(requester=requester, session=session), [])
                finally:
                    coordinator.shutdown(wait=True)

    def test_missing_native_metadata_preparation_cannot_return_none_and_enter_repair(self):
        class MetadataRuntime:
            def __init__(self):
                self.calls = []

            def call_chat_json(self, **kwargs):
                self.calls.append(kwargs)
                return {"task_link": {"mode": "none"}, "task_contract": {}}

        coordinator = AgentTaskCoordinator(max_workers=1)
        handler = RecordingMutator()
        runtime = MetadataRuntime()
        try:
            bridge = self.make_bridge(coordinator, handler, async_enabled=False, task_contract_enabled=True)
            bridge.runtime = runtime
            decision = AgentDecision.tool(ToolInvocation(handler.tool_type, {"path": "fixture.txt"}))
            bridge._run_chat_decision = lambda **kwargs: (decision, {handler.tool_type})

            reply = bridge.handle_turn(turn("写入 fixture.txt"))

            self.assertEqual(reply.agent_status, "blocked")
            self.assertEqual(reply.reason, "task_contract_error:task_handoff_requires_new_or_continue")
            self.assertEqual(len(runtime.calls), 1)
            self.assertEqual(handler.calls, 0)
            self.assertEqual(coordinator.list(), [])
            requester, session = bridge._task_scope(turn("写入 fixture.txt"))
            self.assertEqual(bridge.task_store.candidates(requester=requester, session=session), [])
        finally:
            coordinator.shutdown(wait=True)

    def test_background_tool_does_not_block_chat_and_quiet_final_delivers_once(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release, sent = Event(), Event(), []
        handler = BlockingInspector(entered, release)
        try:
            bridge = self.make_bridge(coordinator, handler, sent)
            created = bridge.handle_turn(turn())
            self.assertEqual(created.agent_status, "waiting_background")
            self.assertTrue(created.background_task_id)
            self.assertFalse(created.sent)
            self.assertEqual(sent, [])
            self.assertTrue(entered.wait(1))
            chatted = bridge.handle_turn(turn("CHAT"))
            self.assertEqual(chatted.final_text, "I'm here.")
            self.assertFalse(release.is_set())
            self.assertEqual(len(sent), 1)
            active_trace = next(t for t in bridge.trace_store.snapshot() if t["message_id"] != bridge.trace_store.snapshot()[0]["message_id"])
            self.assertIsNone(active_trace["finished_at"])
            release.set()
            record = wait_for(coordinator, created.background_task_id, "completed")
            self.assertEqual(handler.calls, 1)
            self.assertEqual(len(sent), 2)
            self.assertEqual(record["metadata"]["delivery_status"], "sent")
        finally:
            release.set()
            coordinator.shutdown(wait=True)

    def test_cancel_command_is_responsive_and_does_not_claim_worker_stopped(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release = Event(), Event()
        try:
            bridge = self.make_bridge(coordinator, BlockingInspector(entered, release))
            created = bridge.handle_turn(turn())
            self.assertTrue(entered.wait(1))
            reply = bridge.handle_turn(turn("/agent cancel " + created.background_task_id))
            self.assertIn("取消请求", reply.final_text)
            self.assertTrue(coordinator.get(created.background_task_id)["worker_active"])
            release.set()
            wait_for(coordinator, created.background_task_id, "canceled")
        finally:
            release.set()
            coordinator.shutdown(wait=True)

    def test_status_commands_query_current_session_without_model(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release = Event(), Event()
        try:
            bridge = self.make_bridge(coordinator, BlockingInspector(entered, release))
            created = bridge.handle_turn(turn())
            self.assertTrue(entered.wait(1))
            listing = bridge.handle_turn(turn("/agent tasks"))
            self.assertIn(created.background_task_id, listing.final_text)
            status = bridge.handle_turn(turn("/agent status " + created.background_task_id))
            self.assertIn("running", status.final_text)
            other = bridge.handle_turn(turn("/agent status " + created.background_task_id, sender="owner", kind="group", conversation="owner"))
            self.assertEqual(other.agent_status, "command")
            self.assertNotIn("running", other.final_text)
            self.assertNotIn(created.background_task_id, other.final_text)
        finally:
            release.set()
            coordinator.shutdown(wait=True)

    def test_dispatcher_same_conversation_fifo_is_released_by_handoff(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release, chatted = Event(), Event(), Event()
        bridge = self.make_bridge(coordinator, BlockingInspector(entered, release))
        batches = []
        replies = []
        def consume(t):
            replies.append(bridge.handle_turn(t))
            if t.combined_text == "CHAT":
                chatted.set()
        dispatcher = NapCatTurnDispatcher(handler=consume, flush_batch=lambda: batches.pop(0), async_handler=True)
        def enqueue(text):
            t = turn(text)
            batches.append(AttentionBatch(t.conversation_id, t.messages, (AttentionDecision(True, True, 100, "private"),), 0))
            dispatcher.flush()
        try:
            enqueue("inspect these files")
            self.assertTrue(entered.wait(1))
            enqueue("CHAT")
            self.assertTrue(chatted.wait(1), "same-conversation FIFO still blocked by tool worker")
            self.assertEqual(replies[0].agent_status, "waiting_background")
            self.assertEqual(replies[1].final_text, "I'm here.")
            self.assertFalse(release.is_set())
        finally:
            release.set()
            dispatcher.flush_batch = lambda: None
            dispatcher.close()
            coordinator.shutdown(wait=True)

    def test_quick_authority_off_on_or_mode_flip_does_not_revive_old_worker(self):
        for switch in ("agent", "mode", "responses"):
            with self.subTest(switch=switch):
                coordinator = AgentTaskCoordinator(max_workers=1)
                entered, release = Event(), Event()
                try:
                    bridge = self.make_bridge(coordinator, BlockingInspector(entered, release))
                    created = bridge.handle_turn(turn())
                    self.assertTrue(entered.wait(1))
                    if switch == "agent":
                        bridge.command_state.set_agent_enabled(False)
                        bridge.command_state.set_agent_enabled(True)
                    elif switch == "mode":
                        bridge.command_state.set_permission_mode("readonly")
                        bridge.command_state.set_permission_mode("full")
                    else:
                        bridge.command_state.set_stopped(True)
                        bridge.command_state.set_stopped(False)
                    release.set()
                    record = wait_for(coordinator, created.background_task_id, "blocked")
                    self.assertEqual(record["error"], "agent_authority_revoked")
                finally:
                    release.set()
                    coordinator.shutdown(wait=True)

    def test_completion_gate_rejection_is_not_background_success(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release = Event(), Event()
        release.set()
        try:
            bridge = self.make_bridge(coordinator, BlockingInspector(entered, release))
            bridge.completion_gate = lambda state, text: False
            created = bridge.handle_turn(turn())
            record = wait_for(coordinator, created.background_task_id, "failed")
            self.assertNotEqual(record["metadata"]["agent_status"], "completed")
        finally:
            coordinator.shutdown(wait=True)

    def test_delivery_failure_is_separate_from_successful_execution(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release = Event(), Event()
        release.set()
        try:
            bridge = self.make_bridge(coordinator, BlockingInspector(entered, release))
            bridge.sender = lambda message: DeliveryResult(False, "send_failed")
            created = bridge.handle_turn(turn())
            record = wait_for(coordinator, created.background_task_id, "completed")
            self.assertFalse(record["metadata"]["sent"])
            self.assertEqual(record["metadata"]["delivery_status"], "send_failed")
        finally:
            coordinator.shutdown(wait=True)

    def approval_bridge(self, coordinator, handler):
        bridge = self.make_bridge(coordinator, handler)
        bridge.command_state.set_permission_mode("traditional")
        bridge.allowed_tool_names = (handler.tool_type,)
        bridge.policy = ExecutionPolicy(default_risk="high")
        class Consent:
            def resolve(self, **kwargs):
                return ConsentDecision("approve", 1.0, "explicit fixture approval")
        bridge.consent_resolver = Consent()
        return bridge

    def test_waiting_approval_resumes_exact_call_in_background(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release = Event(), Event()
        handler = BlockingInspector(entered, release)
        try:
            bridge = self.approval_bridge(coordinator, handler)
            created = bridge.handle_turn(turn())
            wait_for(coordinator, created.background_task_id, "waiting_user")
            self.assertFalse(entered.is_set())
            pending = bridge.approval_store.get("private:owner")
            self.assertEqual(pending.request.invocation.name, handler.tool_type)
            resumed = bridge.handle_turn(turn("yes, execute"))
            self.assertEqual(resumed.agent_status, "waiting_background")
            self.assertTrue(entered.wait(1))
            release.set()
            wait_for(coordinator, resumed.background_task_id, "completed")
            self.assertEqual(handler.calls, 1)
            self.assertIsNone(bridge.approval_store.get("private:owner"))
        finally:
            release.set()
            coordinator.shutdown(wait=True)

    def test_second_configured_master_cannot_approve_another_requesters_group_task(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release = Event(), Event()
        try:
            bridge = self.approval_bridge(coordinator, BlockingInspector(entered, release))
            bridge.master_user_ids = frozenset({"owner", "second-master"})
            created = bridge.handle_turn(turn(kind="group", conversation="42"))
            wait_for(coordinator, created.background_task_id, "waiting_user")
            refused = bridge.handle_turn(turn("yes", sender="second-master", kind="group", conversation="42"))
            self.assertEqual(refused.reason, "pending_approval_scope_mismatch")
            self.assertFalse(entered.is_set())
            self.assertIsNotNone(bridge.approval_store.get("group:42"))
            resumed = bridge.handle_turn(turn("yes", kind="group", conversation="42"))
            self.assertEqual(resumed.agent_status, "waiting_background")
            self.assertTrue(entered.wait(1))
            release.set()
            wait_for(coordinator, resumed.background_task_id, "completed")
        finally:
            release.set()
            coordinator.shutdown(wait=True)

    def test_stop_suppresses_late_background_delivery(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release, sent = Event(), Event(), []
        try:
            bridge = self.make_bridge(coordinator, BlockingInspector(entered, release), sent)
            created = bridge.handle_turn(turn())
            self.assertTrue(entered.wait(1))
            bridge.command_state.set_stopped(True)
            release.set()
            record = wait_for(coordinator, created.background_task_id, "blocked")
            self.assertEqual(record["metadata"]["delivery_status"], "suppressed")
            self.assertEqual(sent, [])
        finally:
            release.set()
            coordinator.shutdown(wait=True)

    def test_cooperative_long_tool_observes_cancel_inside_its_units(self):
        class Cooperative(BlockingInspector):
            def execute(self, *, call, context):
                self.calls += 1
                self.entered.set()
                while not self.release.wait(0.005):
                    if context["agent_task_interrupt"]():
                        return {"followup_context": "internal chunk checkpoint stopped"}
                return {"followup_context": "fixture released"}
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release = Event(), Event()
        try:
            bridge = self.make_bridge(coordinator, Cooperative(entered, release))
            created = bridge.handle_turn(turn())
            self.assertTrue(entered.wait(1))
            coordinator.cancel(created.background_task_id)
            wait_for(coordinator, created.background_task_id, "canceled")
            self.assertFalse(release.is_set(), "cancel should exit at an internal checkpoint without external release")
        finally:
            release.set()
            coordinator.shutdown(wait=True)

    def test_cancel_waiting_approval_cannot_be_revived_by_old_yes(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release = Event(), Event()
        try:
            bridge = self.approval_bridge(coordinator, BlockingInspector(entered, release))
            created = bridge.handle_turn(turn())
            wait_for(coordinator, created.background_task_id, "waiting_user")
            self.assertTrue(coordinator.cancel(created.background_task_id)["ok"])
            resumed = bridge.handle_turn(turn("yes, execute"))
            self.assertEqual((resumed.agent_status, resumed.reason), ("canceled", "task_canceled"))
            self.assertFalse(entered.is_set())
        finally:
            release.set()
            coordinator.shutdown(wait=True)

    def test_duplicate_active_continuation_does_not_launch_or_mutate(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release = Event(), Event()
        handler = BlockingInspector(entered, release)
        try:
            bridge = self.make_bridge(coordinator, handler)
            created = bridge.handle_turn(turn())
            self.assertTrue(entered.wait(1))
            active = coordinator.get(created.background_task_id)
            bridge._run_chat_decision = lambda **kwargs: (
                replace(AgentDecision.tool(ToolInvocation(handler.tool_type, {})),
                        task_link={"mode": "continue", "task_id": active["metadata"]["agent_task_id"]}), {handler.tool_type})
            duplicate = bridge.handle_turn(turn("continue"))
            self.assertEqual(duplicate.reason, "background_task_already_active")
            self.assertEqual(len(coordinator.list()), 1)
            self.assertEqual(handler.calls, 1)
            self.assertEqual(coordinator.get(created.background_task_id)["attempt"], 1)
        finally:
            release.set()
            coordinator.shutdown(wait=True)


class PendingConcurrencyTests(unittest.TestCase):
    def test_json_approval_concurrent_put_keeps_all_records_and_reloads_fifo(self):
        with TemporaryDirectory() as directory:
            store = JsonPendingApprovalStore(Path(directory) / "approval.json")
            def put(n):
                request = make_tool_approval_request(task_id=str(n), invocation=ToolInvocation("inspect_files", {"n": n}))
                store.put("private:owner", PendingApprovalRecord(request, str(n), {}, (), time.time()))
            with ThreadPoolExecutor(max_workers=4) as workers:
                list(workers.map(put, range(20)))
            data = json.loads(store.path.read_text(encoding="utf-8"))
            self.assertEqual(len(data["private:owner"]), 20)
            reloaded = JsonPendingApprovalStore(store.path)
            seen = []
            while (record := reloaded.get("private:owner")) is not None:
                seen.append(record.task_id)
                reloaded.delete("private:owner")
            self.assertEqual(seen, [r["task_id"] for r in data["private:owner"]])


if __name__ == "__main__":
    unittest.main()
