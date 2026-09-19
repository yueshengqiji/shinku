from __future__ import annotations

import unittest

from shinku.agent.loop import AgentDecision, AgentLoop
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.invocation import ToolInvocation


class Handler:
    tool_type = "search"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def normalize_call(self, call):
        return call if call.get("query") else None

    def execute(self, *, call, context):
        self.calls += 1
        if self.fail:
            raise TimeoutError("private detail")
        return {"tool_type": "search", "followup_context": f"result for {call['query']}", "state_updates": {"found": True}}


class LoopTests(unittest.TestCase):
    def test_tool_result_is_fed_back_before_final_answer(self) -> None:
        steps = iter([
            AgentDecision.tool(ToolInvocation("search", {"query": "x"})),
            AgentDecision.final("完成了。"),
        ])
        seen = []

        def planner(*, state):
            seen.append((state.rounds, state.last_envelope.model_feedback if state.last_envelope else ""))
            return next(steps)

        handler = Handler()
        result = AgentLoop(planner=planner, handlers={"search": handler}).run(task_id="t-1")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.rounds, 2)
        self.assertEqual(handler.calls, 1)
        self.assertIn("result for x", seen[1][1])

    def test_mapping_decision_is_normalized(self) -> None:
        steps = iter([
            {"action": "tool_call", "tool_call": {"type": "search", "query": "x"}},
            {"kind": "done", "speech": "好。"},
        ])
        result = AgentLoop(planner=lambda *, state: next(steps), handlers={"search": Handler()}).run()
        self.assertEqual((result.status, result.final_text), ("completed", "好。"))

    def test_wait_is_a_normal_terminal_status(self) -> None:
        result = AgentLoop(planner=lambda *, state: AgentDecision.wait("还缺一个文件。", reason="missing_input"), handlers={}).run()
        self.assertEqual(result.status, "waiting_user")
        self.assertEqual(result.reason, "missing_input")

    def test_tool_failure_is_returned_to_planner_for_recovery(self) -> None:
        steps = iter([
            AgentDecision.tool(ToolInvocation("search", {"query": "x"})),
            AgentDecision.final("工具失败了，我把原因说明给你。"),
        ])
        result = AgentLoop(planner=lambda *, state: next(steps), handlers={"search": Handler(fail=True)}).run()
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_envelopes[0].data["code"], "handler_error")
        self.assertNotIn("private detail", result.tool_envelopes[0].model_feedback)

    def test_unknown_tool_can_be_recovered(self) -> None:
        steps = iter([
            AgentDecision.tool(ToolInvocation("missing", {"x": 1})),
            AgentDecision.final("这个工具当前不可用。"),
        ])
        result = AgentLoop(planner=lambda *, state: next(steps), handlers={}).run()
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_envelopes[0].data["code"], "unknown_tool")

    def test_duplicate_calls_get_one_recovery_turn_then_block(self) -> None:
        def planner(*, state):
            return AgentDecision.tool(ToolInvocation("search", {"query": "same"}))

        result = AgentLoop(planner=planner, handlers={"search": Handler()}, max_rounds=6).run()
        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.reason, "repeated_tool_call")
        self.assertEqual(len(result.tool_envelopes), 3)

    def test_max_rounds_prevents_unbounded_planning(self) -> None:
        calls = []

        def planner(*, state):
            calls.append(state.rounds)
            return AgentDecision.tool(ToolInvocation("search", {"query": str(state.rounds)}))

        result = AgentLoop(planner=planner, handlers={"search": Handler()}, max_rounds=2).run()
        self.assertEqual((result.status, result.reason, result.rounds), ("failed", "max_rounds", 2))
        self.assertEqual(calls, [1, 2])

    def test_completion_gate_rejects_first_premature_answer(self) -> None:
        steps = iter([AgentDecision.final("好了。"), AgentDecision.final("已验证并完成。")])
        result = AgentLoop(
            planner=lambda *, state: next(steps),
            handlers={},
            completion_gate=lambda state, text: text.startswith("已验证"),
        ).run()
        self.assertEqual(result.final_text, "已验证并完成。")
        self.assertEqual(result.rounds, 2)
        self.assertIn("完成验收未通过", result.transcript[0]["content"])

    def test_policy_is_applied_before_handler_execution(self) -> None:
        handler = Handler()
        steps = iter([AgentDecision.tool(ToolInvocation("search", {"query": "x"})), AgentDecision.final("被阻止。")])
        result = AgentLoop(
            planner=lambda *, state: next(steps),
            handlers={"search": handler},
            policy=ExecutionPolicy(blocked=frozenset({"search"})),
        ).run()
        self.assertEqual(result.tool_envelopes[0].data["code"], "tool_blocked")
        self.assertEqual(handler.calls, 0)


if __name__ == "__main__":
    unittest.main()
