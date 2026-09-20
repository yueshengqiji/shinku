from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from shinku.agent.approval import LLMConsentResolver, make_tool_approval_request
from shinku.agent.loop import AgentDecision, AgentLoop
from shinku.agent.pending import InMemoryPendingApprovalStore, JsonPendingApprovalStore, PendingApprovalRecord
from shinku.hosts.tool_host import ToolHost
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.invocation import ToolInvocation
from shinku.tools.registry import ToolRegistry


class SearchHandler:
    tool_type = "search"

    def __init__(self):
        self.calls = 0

    def normalize_call(self, call):
        return call if call.get("query") else None

    def execute(self, *, call, context):
        self.calls += 1
        return {"tool_type": "search", "followup_context": f"查到：{call['query']}"}


class ConsentRuntime:
    def __init__(self, *, consent="approve"):
        self.consent = consent
        self.calls = []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if "授权" in kwargs.get("system_prompt", ""):
            return {"decision": self.consent, "confidence": 0.96, "reason": "语义明确"}
        if kwargs.get("native_tools") and len(self.calls) == 1:
            return {"tool_calls": [{"id": "tool-1", "function": {"name": "search", "arguments": '{"query":"Shinku"}'}}]}
        return {"speech": "结果已经拿到了。"}


def _turn(message_id: str, text: str) -> QQTurn:
    message = IncomingMessage.from_event(
        {
            "post_type": "message",
            "message_type": "private",
            "user_id": "u-1",
            "message_id": message_id,
            "message": [{"type": "text", "data": {"text": text}}],
        }
    )
    return QQTurn(
        conversation_id="u-1",
        conversation_type="private",
        primary_message_id=message_id,
        message_ids=(message_id,),
        combined_text=text,
        messages=(message,),
    )


class ToolApprovalTests(unittest.TestCase):
    def test_default_risk_policy_is_conservative_and_environment_can_override(self) -> None:
        default = ExecutionPolicy()
        self.assertEqual(default.risk_of("new_tool"), "medium")
        self.assertTrue(default.requires_approval("new_tool"))
        self.assertFalse(ExecutionPolicy(risk_by_tool={"read_file": "low"}).requires_approval("read_file"))

        configured = ExecutionPolicy.from_environment(
            {
                "SHINKU_TOOL_DEFAULT_RISK": "low",
                "SHINKU_TOOL_APPROVAL_RISKS": "high",
                "SHINKU_TOOL_RISK_OVERRIDES_JSON": '{"read_file":"low","send_message":"high"}',
            }
        )
        self.assertFalse(configured.requires_approval("new_tool"))
        self.assertTrue(configured.requires_approval("send_message"))

    def test_pending_approval_store_can_recover_exact_calls_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pending.json"
            request = make_tool_approval_request(
                invocation=ToolInvocation("search", {"query": "Shinku"}),
                task_id="t-persist",
                risk_by_tool={"search": "high"},
            )
            first = JsonPendingApprovalStore(path)
            first.put(
                "conversation-1",
                PendingApprovalRecord(
                    request=request,
                    task_id="t-persist",
                    context={"qq_turn": object(), "user_images": []},
                    transcript=({"role": "user", "content": "查一下"},),
                    created_at=123.0,
                ),
            )

            recovered = JsonPendingApprovalStore(path).get("conversation-1")
            self.assertIsNotNone(recovered)
            self.assertEqual(recovered.task_id, "t-persist")
            self.assertEqual(recovered.request.invocation.arguments, {"query": "Shinku"})
            self.assertEqual(recovered.request.risk_of("search"), "high")
            self.assertNotIn("qq_turn", recovered.context)

    def test_pending_store_failure_fails_closed_before_agent_execution(self) -> None:
        class FailingStore:
            def get(self, conversation_id):
                raise OSError("store unavailable")

            def put(self, conversation_id, record):
                raise OSError("store unavailable")

            def delete(self, conversation_id):
                raise OSError("store unavailable")

        bridge = QQAgentBridge(
            runtime=ConsentRuntime(),
            system_prompt="p",
            approval_store=FailingStore(),
        )
        result = bridge.handle_turn(_turn("m-store", "查一下"))
        self.assertEqual((result.agent_status, result.reason), ("failed", "approval_store_error:OSError"))

    def test_pending_approval_store_queues_same_conversation_instead_of_overwriting(self) -> None:
        store = InMemoryPendingApprovalStore()
        first = make_tool_approval_request(invocation=ToolInvocation("search", {"query": "first"}))
        second = make_tool_approval_request(invocation=ToolInvocation("search", {"query": "second"}))
        store.put("conversation-queue", PendingApprovalRecord(first, "t-1", {}, (), 1.0))
        store.put("conversation-queue", PendingApprovalRecord(second, "t-2", {}, (), 2.0))

        self.assertEqual(store.get("conversation-queue").request.invocation.arguments["query"], "first")
        store.delete("conversation-queue")
        self.assertEqual(store.get("conversation-queue").request.invocation.arguments["query"], "second")

    def test_low_risk_call_bypasses_gate_but_high_risk_call_does_not(self) -> None:
        handler = SearchHandler()
        steps = iter((AgentDecision.tool(ToolInvocation("search", {"query": "x"})), AgentDecision.final("完成。")))
        gate_calls = []
        result = AgentLoop(
            planner=lambda **_: next(steps),
            handlers={"search": handler},
            policy=ExecutionPolicy(risk_by_tool={"search": "low"}),
            approval_gate=lambda **kwargs: gate_calls.append(kwargs) or make_tool_approval_request(**kwargs),
        ).run(task_id="t-low")
        self.assertEqual((result.status, result.final_text), ("completed", "完成。"))
        self.assertEqual(handler.calls, 1)
        self.assertEqual(gate_calls, [])

        request = make_tool_approval_request(invocation=ToolInvocation("search", {"query": "x"}), task_id="t-high")
        waiting = AgentLoop(
            planner=lambda **_: AgentDecision.tool(request.invocation),
            handlers={"search": SearchHandler()},
            policy=ExecutionPolicy(risk_by_tool={"search": "high"}),
            approval_gate=lambda **_: request,
        ).run(task_id="t-high")
        self.assertEqual(waiting.status, "waiting_user")

    def test_loop_holds_a_valid_tool_call_before_execution(self) -> None:
        handler = SearchHandler()
        request = make_tool_approval_request(invocation=ToolInvocation("search", {"query": "x"}), task_id="t-1")
        result = AgentLoop(
            planner=lambda **_: AgentDecision.tool(request.invocation),
            handlers={"search": handler},
            approval_gate=lambda **_: request,
        ).run(task_id="t-1")

        self.assertEqual((result.status, result.reason), ("waiting_user", "tool_approval_required"))
        self.assertIsNotNone(result.pending_approval)
        self.assertEqual(handler.calls, 0)

    def test_approved_invocation_resumes_exact_tool_then_plans(self) -> None:
        handler = SearchHandler()
        result = AgentLoop(
            planner=lambda **_: AgentDecision.final("完成。"),
            handlers={"search": handler},
        ).run(
            task_id="t-1",
            approved_invocation=ToolInvocation("search", {"query": "x"}),
        )

        self.assertEqual((result.status, result.final_text), ("completed", "完成。"))
        self.assertEqual(result.tool_envelopes[0].status, "ok")

    def test_one_decision_can_execute_multiple_tools_in_order(self) -> None:
        first = SearchHandler()
        second = SearchHandler()
        result = AgentLoop(
            planner=lambda **_: AgentDecision.tools(
                (
                    ToolInvocation("first", {"query": "a"}),
                    ToolInvocation("second", {"query": "b"}),
                )
            ),
            handlers={"first": first, "second": second},
            max_rounds=1,
        ).run(task_id="t-batch")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "max_rounds")
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertEqual([item.data["tool_type"] for item in result.tool_envelopes], ["search", "search"])

    def test_batch_approval_contains_all_calls_and_resumes_all(self) -> None:
        first = SearchHandler()
        second = SearchHandler()
        captured = []

        def gate(**kwargs):
            captured.append(kwargs)
            return make_tool_approval_request(invocations=kwargs["invocations"], task_id="t-batch")

        waiting = AgentLoop(
            planner=lambda **_: AgentDecision.tools(
                (ToolInvocation("first", {"query": "a"}), ToolInvocation("second", {"query": "b"}))
            ),
            handlers={"first": first, "second": second},
            approval_gate=gate,
        ).run(task_id="t-batch")

        self.assertEqual(waiting.status, "waiting_user")
        self.assertEqual(len(waiting.pending_approval.invocations), 2)
        self.assertEqual(len(captured[0]["invocations"]), 2)

        resumed = AgentLoop(
            planner=lambda **_: AgentDecision.final("完成。"),
            handlers={"first": first, "second": second},
        ).run(
            task_id="t-batch",
            approved_invocations=waiting.pending_approval.invocations,
        )
        self.assertEqual(resumed.final_text, "完成。")
        self.assertEqual((first.calls, second.calls), (1, 1))

    def test_consent_resolver_is_semantic_and_has_no_tools(self) -> None:
        runtime = ConsentRuntime()
        request = make_tool_approval_request(invocation=ToolInvocation("search", {"query": "x"}))
        decision = LLMConsentResolver(runtime).resolve(request=request, user_message="行，你按刚才说的查吧")

        self.assertTrue(decision.approved)
        self.assertNotIn("native_tools", runtime.calls[0])

    def test_qq_bridge_waits_then_resumes_after_natural_language_approval(self) -> None:
        runtime = ConsentRuntime()
        handler = SearchHandler()
        sent = []
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            tool_host=ToolHost(ToolRegistry({"search": handler})),
            sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("out",)),
        )

        first = bridge.handle_turn(_turn("m-1", "帮我查一下"))
        second = bridge.handle_turn(_turn("m-2", "可以，你按刚才的方案做吧"))

        self.assertEqual(first.agent_status, "waiting_user")
        self.assertIn("准备调用", first.final_text)
        self.assertEqual((second.agent_status, second.final_text, second.sent), ("completed", "结果已经拿到了。", True))
        self.assertEqual(len(sent), 2)
        self.assertEqual(len(runtime.calls), 3)  # planner -> consent judge -> resumed planner

    def test_unclear_consent_does_not_execute(self) -> None:
        runtime = ConsentRuntime(consent="unclear")
        handler = SearchHandler()
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="p",
            tool_host=ToolHost(ToolRegistry({"search": handler})),
        )
        bridge.handle_turn(_turn("m-1", "查一下"))
        result = bridge.handle_turn(_turn("m-2", "我先问个别的"))

        self.assertEqual(result.agent_status, "waiting_user")
        self.assertEqual(handler.calls, 0)


if __name__ == "__main__":
    unittest.main()
