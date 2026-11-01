from __future__ import annotations

import threading
import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch

from shinku.agent.loop import AgentDecision, AgentLoop
from shinku.tools.invocation import ToolInvocation
from shinku.tasks.coordinator import AgentTaskCoordinator, AgentTaskContinuation
from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget, current_task_budget, metered_completion
from shinku.agent.approval import ConsentDecision, make_tool_approval_request
from shinku.tools.execution import ExecutionPolicy
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.delivery import DeliveryResult
from shinku.tools.registry import ToolRegistry
from shinku.hosts.tool_host import ToolHost
from test_agent_async_processor import turn, wait_for
from test_contract_c2_5 import _make_runtime, FakeResponse


class Handler:
    tool_type = "read"
    agent_progress_class = "observe"
    agent_permission_class = "read"
    description = "read numbered observation"
    input_schema = {"type": "object", "properties": {"n": {"type": "integer"}}}

    def __init__(self):
        self.calls = []

    def normalize_call(self, call):
        return dict(call)

    def execute(self, *, call, context):
        self.calls.append(call["n"])
        return {"followup_context": f"evidence {call['n']}"}


class AgentSliceTests(unittest.TestCase):
    def test_slices_preserve_state_and_evidence(self):
        handler = Handler()
        def planner(*, state):
            return AgentDecision.tool(ToolInvocation("read", {"n": state.rounds})) if state.rounds <= 9 else AgentDecision.final("verified")
        loop = AgentLoop(planner=planner, handlers={"read": handler}, max_rounds=3, sliced=True,
                         completion_gate=lambda state, text: len(state.tool_calls) == 9)
        result = loop.run(task_id="same")
        states = []
        while result.status == "yielded":
            states.append(result.checkpoint.state)
            result = loop.run(checkpoint=result.checkpoint)
        self.assertEqual((result.status, result.rounds), ("completed", 10))
        self.assertTrue(all(state is states[0] for state in states))
        self.assertEqual(handler.calls, list(range(1, 10)))
        self.assertEqual(len(result.tool_envelopes), 9)

    def test_duplicate_guard_crosses_boundary(self):
        handler = Handler()
        loop = AgentLoop(planner=lambda **kw: AgentDecision.tool(ToolInvocation("read", {"n": 1})),
                         handlers={"read": handler}, max_rounds=1, sliced=True)
        result = loop.run()
        for _ in range(3):
            if result.status != "yielded":
                break
            result = loop.run(checkpoint=result.checkpoint)
        self.assertEqual((result.status, result.reason), ("blocked", "repeated_tool_call"))
        self.assertEqual(handler.calls, [1])

    def test_continuation_is_same_attempt_and_fair(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        order = []
        def first(cancel):
            order.append("first")
            entered.set()
            release.wait(3)
            return AgentTaskContinuation(lambda cancel: order.append("continued") or finished.set())
        try:
            coordinator.submit(first, task_id="long")
            self.assertTrue(entered.wait(2))
            coordinator.submit(lambda cancel: order.append("short"), task_id="short")
            release.set()
            self.assertTrue(finished.wait(3))
            self.assertEqual(order, ["first", "short", "continued"])
            self.assertEqual(coordinator.get("long")["attempt"], 1)
        finally:
            release.set()
            coordinator.shutdown(wait=True)

    def test_no_progress_counter_does_not_reset(self):
        class Unchanged(Handler):
            def execute(self, **kwargs):
                return {"followup_context": "unchanged"}
        loop = AgentLoop(planner=lambda *, state: AgentDecision.tool(ToolInvocation("read", {"n": state.rounds})),
                         handlers={"read": Unchanged()}, max_rounds=1, sliced=True, max_consecutive_observations=1)
        result = loop.run()
        for _ in range(6):
            if result.status != "yielded":
                break
            result = loop.run(checkpoint=result.checkpoint)
        self.assertEqual((result.status, result.reason), ("blocked", "no_progress"))

    def test_approval_after_six_rounds_resumes_full_counters(self):
        reader, writer = Handler(), Handler()
        def plan(*, state):
            count = len(state.tool_calls)
            if count >= 8:
                return AgentDecision.final("verified")
            return AgentDecision.tool(ToolInvocation("write" if count == 7 else "read", {"n": count + 1}))
        def approval(**kwargs):
            return make_tool_approval_request(invocations=kwargs["invocations"], task_id=kwargs["state"].task_id)
        loop = AgentLoop(planner=plan, handlers={"read": reader, "write": writer}, sliced=True, max_rounds=2,
                         policy=ExecutionPolicy(default_risk="low", risk_by_tool={"write": "high"}),
                         approval_gate=approval)
        result = loop.run()
        while result.status == "yielded":
            result = loop.run(checkpoint=result.checkpoint)
        self.assertEqual((result.status, result.rounds, writer.calls), ("waiting_user", 8, []))
        final = loop.run(checkpoint=result.checkpoint, approved_invocations=result.pending_approval.invocations)
        while final.status == "yielded":
            final = loop.run(checkpoint=final.checkpoint)
        self.assertEqual((final.status, final.rounds, writer.calls), ("completed", 10, [8]))
        self.assertEqual(len(final.tool_envelopes), 8)

    def test_research_and_source_discovery_state_are_not_reconfigured(self):
        loop = AgentLoop(planner=lambda *, state: AgentDecision.tool(ToolInvocation("read", {"n": state.rounds})),
                         handlers={"read": Handler()}, max_rounds=1, sliced=True)
        first = loop.run()
        state = first.checkpoint.state
        state.research.sources["source"] = {"url": "https://example.test", "text": "actual retained evidence"}
        state.source_discovery_calls = 3
        state.suppressed_tool_names.add("search")
        with patch.object(state.research, "configure", side_effect=AssertionError("must not reset")):
            second = loop.run(checkpoint=first.checkpoint)
        self.assertIs(second.checkpoint.state, state)
        self.assertEqual(state.research.sources["source"]["text"], "actual retained evidence")
        self.assertEqual(state.source_discovery_calls, 3)
        self.assertIn("search", state.suppressed_tool_names)

    def test_cancel_between_slices_finalizes_without_executing_next_handler(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        entered, release, short_entered, short_release, finalized = [threading.Event() for _ in range(5)]
        calls = []
        def first(cancel):
            entered.set()
            release.wait(2)
            return AgentTaskContinuation(lambda cancel: calls.append("should-not-run"), on_cancel=finalized.set)
        def short(cancel):
            short_entered.set()
            short_release.wait(2)
        try:
            coordinator.submit(first, task_id="long", metadata={"retry_policy": "fresh_continuation_only"})
            self.assertTrue(entered.wait(1))
            coordinator.submit(short, task_id="short")
            release.set()
            self.assertTrue(short_entered.wait(1))
            self.assertEqual(coordinator.get("long")["status"], "queued")
            self.assertTrue(coordinator.cancel("long")["ok"])
            self.assertTrue(finalized.wait(1))
            short_release.set()
            wait_for(coordinator, "short", "completed")
            self.assertEqual(calls, [])
        finally:
            release.set()
            short_release.set()
            coordinator.shutdown(wait=True)


class BudgetTests(unittest.TestCase):
    def test_reported_usage_and_failed_request_reservation(self):
        budget = TaskBudget(max_requests=2)
        payload = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 50}
        with bind_task_budget(budget):
            metered_completion(lambda **kw: {"usage": {"total_tokens": 17}}, payload)
            with self.assertRaises(OSError):
                metered_completion(lambda **kw: (_ for _ in ()).throw(OSError("network")), payload)
            with self.assertRaises(TaskBudgetExceeded):
                metered_completion(lambda **kw: self.fail("must not send"), payload)
        self.assertEqual(budget.requests, 2)
        self.assertEqual(budget.reported_tokens, 17)
        self.assertEqual(budget.estimated_requests, 1)
        self.assertGreater(budget.tokens, 17)
        self.assertEqual(budget.reason, "task_model_budget_exhausted")

    def test_token_budget_denies_before_sending_and_latches(self):
        budget = TaskBudget(max_tokens=10)
        with bind_task_budget(budget), self.assertRaises(TaskBudgetExceeded):
            metered_completion(lambda **kw: self.fail("must not send"), {"messages": [], "max_tokens": 20})
        self.assertEqual(budget.requests, 0)
        self.assertEqual(budget.reason, "task_token_budget_exhausted")

    def test_stream_usage_settles_once_and_close_releases_stream(self):
        closed = []
        def stream():
            try:
                yield {"usage": {"total_tokens": 20}}
                yield {"usage": {"total_tokens": 30}}
            finally:
                closed.append(True)
        budget = TaskBudget()
        with bind_task_budget(budget):
            response = metered_completion(lambda **kw: stream(), {"stream": True, "max_tokens": 100})
            self.assertEqual(len(list(response)), 2)
        self.assertEqual((budget.tokens, budget.reported_tokens, budget.estimated_requests), (30, 30, 0))
        self.assertEqual(closed, [True])

    def test_active_time_excludes_queue_and_nested_scopes(self):
        budget = TaskBudget()
        with patch("shinku.agent.budget.time.monotonic", side_effect=[10, 14, 100, 103]):
            with bind_task_budget(budget):
                with bind_task_budget(budget):
                    self.assertIs(current_task_budget(), budget)
            with bind_task_budget(budget):
                pass
        self.assertEqual(budget.active_seconds, 7)
        self.assertIsNone(current_task_budget())

    def test_tools_and_time_exhaust_cooperatively(self):
        budget = TaskBudget(max_tools=1, max_active_seconds=2)
        budget.use_tool()
        with self.assertRaises(TaskBudgetExceeded):
            budget.use_tool()
        self.assertEqual(budget.reason, "task_tool_budget_exhausted")
        timed = TaskBudget(max_active_seconds=2)
        with patch("shinku.agent.budget.time.monotonic", side_effect=[10, 13, 14]):
            with bind_task_budget(timed):
                self.assertEqual(timed.stop_reason(), "task_time_budget_exhausted")

    def test_explicit_task_continuation_cannot_refresh_prior_budget(self):
        budget = TaskBudget(max_requests=10)
        budget.requests = 2
        old = TaskBudget(max_requests=5)
        old.requests = 4
        budget.absorb_prior(old.snapshot())
        self.assertEqual((budget.max_requests, budget.requests), (5, 6))
        self.assertEqual(budget.reason, "task_model_budget_exhausted")

    def test_sdk_cache_hint_fallback_is_metered_as_second_attempt(self):
        runtime = _make_runtime(scripts=[("raise", TypeError("unexpected prompt_cache_key")), ("response", FakeResponse('{"speech":"ok"}'))])
        budget = TaskBudget(max_requests=1)
        try:
            with bind_task_budget(budget), self.assertRaises(TaskBudgetExceeded):
                runtime._attempt_completion(bundle=runtime.chat, payload={"messages": [], "prompt_cache_key": "key"}, attempt=1)
            self.assertEqual(len(runtime.chat.client.payloads), 1)
            self.assertEqual(budget.requests, 1)
        finally:
            runtime.close()

    def test_sdk_http_retry_cannot_cross_request_budget(self):
        from dataclasses import replace
        runtime = _make_runtime(scripts=[("raise", TimeoutError("timed out"))] * 3)
        runtime.runtime_policy = replace(runtime.runtime_policy, http_attempts=4, retry_backoff_seconds=0)
        budget = TaskBudget(max_requests=2)
        try:
            with bind_task_budget(budget), self.assertRaises(TaskBudgetExceeded):
                runtime._dispatch_completion(bundle=runtime.chat, payload={"messages": []})
            self.assertEqual(len(runtime.chat.client.payloads), 2)
            self.assertEqual(budget.requests, 2)
        finally:
            runtime.close()



class BridgeSliceTests(unittest.TestCase):
    def make_bridge(self, coordinator, max_requests=30, stop=10):
        class Runtime:
            def __init__(self):
                self.calls = 0
            def chat_supports_native_tools(self):
                return True
            def call_chat_json(inner, **kwargs):
                def response(**payload):
                    inner.calls += 1
                    return ({"kind": "tool", "tool_call": {"name": "read", "arguments": {"n": inner.calls}}}
                            if inner.calls < stop else {"kind": "final", "text": "all verified"})
                return metered_completion(response, {"messages": [], "max_tokens": 10})
        runtime, handler, sent = Runtime(), Handler(), []
        state = QQCommandState()
        state.set_permission_mode("full")
        bridge = QQAgentBridge(runtime=runtime, system_prompt="fixture", command_state=state,
            task_coordinator=coordinator, async_agent_enabled=True, max_rounds=3,
            task_budget_limits={"max_requests": max_requests},
            tool_host=ToolHost(ToolRegistry({"read": handler})), completion_gate=lambda state, text: len(handler.calls) == stop - 1,
            sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("receipt",)))
        def route(**kwargs):
            return AgentDecision.from_value(runtime.call_chat_json()), {"read"}
        bridge._run_chat_decision = route
        return bridge, runtime, handler, sent

    def test_bridge_requeues_automatically_and_delivers_only_final(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        try:
            bridge, runtime, handler, sent = self.make_bridge(coordinator)
            created = bridge.handle_turn(turn())
            task = wait_for(coordinator, created.background_task_id, "completed")
            self.assertEqual((task["metadata"]["rounds"], task["metadata"]["slices"]), (10, 3))
            self.assertEqual(handler.calls, list(range(1, 10)))
            self.assertEqual(task["metadata"]["budget"]["requests"], 10)
            self.assertEqual(len(sent), 1)
            self.assertEqual(task["attempt"], 1)
            self.assertFalse(bridge._slice_sessions)
        finally:
            coordinator.shutdown(wait=True)

    def test_global_request_budget_includes_router_and_crosses_slices(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        try:
            bridge, runtime, handler, sent = self.make_bridge(coordinator, max_requests=7)
            created = bridge.handle_turn(turn())
            task = wait_for(coordinator, created.background_task_id, "blocked")
            self.assertEqual(task["error"], "task_model_budget_exhausted")
            self.assertEqual(runtime.calls, 7)
            self.assertEqual(handler.calls, list(range(1, 8)))
            self.assertEqual(task["metadata"]["budget"]["requests"], 7)
            # One final report, shaped through the same sentence boundary as
            # successful/chat replies; no per-slice interim delivery.
            expected = bridge.reply_host.shape(task["result_summary"]).messages
            self.assertEqual(tuple(message.segments[0].text for message in sent), expected)
            ledger = bridge.task_store.get(task["metadata"]["agent_task_id"], requester="owner", session="private:owner")
            self.assertEqual(ledger["budget"]["requests"], 7)
        finally:
            coordinator.shutdown(wait=True)

    def test_approval_reuses_id_full_checkpoint_and_budget(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        try:
            bridge, runtime, handler, sent = self.make_bridge(coordinator, max_requests=2, stop=3)
            bridge.command_state.set_permission_mode("traditional")
            bridge.allowed_tool_names = {"read"}
            bridge.policy = ExecutionPolicy(default_risk="high")
            class Consent:
                def resolve(self, **kwargs):
                    return metered_completion(lambda **kw: ConsentDecision("approve", 1.0), {"messages": [], "max_tokens": 10})
            bridge.consent_resolver = Consent()
            created = bridge.handle_turn(turn())
            waiting = wait_for(coordinator, created.background_task_id, "waiting_user")
            self.assertEqual(waiting["metadata"]["budget"]["requests"], 1)
            self.assertEqual(handler.calls, [])
            old_session = bridge._slice_sessions[created.background_task_id]
            resumed = bridge.handle_turn(turn("yes"))
            self.assertEqual(resumed.background_task_id, created.background_task_id)
            stopped = wait_for(coordinator, resumed.background_task_id, "blocked")
            self.assertEqual(stopped["error"], "task_model_budget_exhausted")
            self.assertEqual(stopped["metadata"]["budget"]["requests"], 2)
            self.assertEqual(handler.calls, [1])
            self.assertEqual(old_session["checkpoint"].state.rounds, 3)
        finally:
            coordinator.shutdown(wait=True)

    def test_restart_missing_executable_checkpoint_cannot_replay_approval(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        try:
            bridge, runtime, handler, sent = self.make_bridge(coordinator)
            bridge.command_state.set_permission_mode("traditional")
            bridge.allowed_tool_names = {"read"}
            bridge.policy = ExecutionPolicy(default_risk="high")
            created = bridge.handle_turn(turn())
            wait_for(coordinator, created.background_task_id, "waiting_user")
            bridge._slice_sessions.clear()
            reply = bridge.handle_turn(turn("yes"))
            self.assertEqual(reply.reason, "task_checkpoint_unavailable")
            self.assertEqual((runtime.calls, handler.calls), (1, []))
        finally:
            coordinator.shutdown(wait=True)

    def test_denied_approval_releases_executable_state(self):
        coordinator = AgentTaskCoordinator(max_workers=1)
        try:
            bridge, runtime, handler, sent = self.make_bridge(coordinator)
            bridge.command_state.set_permission_mode("traditional")
            bridge.allowed_tool_names = {"read"}
            bridge.policy = ExecutionPolicy(default_risk="high")
            class Consent:
                def resolve(self, **kwargs):
                    return ConsentDecision("deny", 1.0)
            bridge.consent_resolver = Consent()
            created = bridge.handle_turn(turn())
            wait_for(coordinator, created.background_task_id, "waiting_user")
            denied = bridge.handle_turn(turn("no"))
            self.assertEqual(denied.reason, "tool_denied")
            self.assertFalse(bridge._slice_sessions)
            self.assertEqual(coordinator.get(created.background_task_id)["status"], "canceled")
            self.assertEqual(handler.calls, [])
        finally:
            coordinator.shutdown(wait=True)
