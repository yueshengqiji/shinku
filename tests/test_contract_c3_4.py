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


class ObservationHandler(Handler):
    agent_progress_class = "observe"


class ChangeHandler(Handler):
    agent_progress_class = "change"

    def execute(self, *, call, context):
        result = super().execute(call=call, context=context)
        result["state_updates"]["changed"] = True
        return result


class RecoverOnceHandler(Handler):
    agent_progress_class = "observe"

    def execute(self, *, call, context):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("temporary")
        return {"tool_type": "search", "followup_context": "recovered"}


class LoopTests(unittest.TestCase):
    def test_first_decision_is_reused_before_planner_continues(self) -> None:
        handler = Handler()
        planner_calls = []

        def planner(*, state):
            planner_calls.append(state.rounds)
            return AgentDecision.final("查到了。")

        result = AgentLoop(planner=planner, handlers={"search": handler}).run(
            task_id="t-seeded",
            first_decision=AgentDecision.tool(ToolInvocation("search", {"query": "x"})),
        )

        self.assertEqual((result.status, result.final_text), ("completed", "查到了。"))
        self.assertEqual(result.rounds, 2)
        self.assertEqual(handler.calls, 1)
        self.assertEqual(planner_calls, [2])

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

    def test_failed_call_can_retry_same_arguments_once_it_recovers(self) -> None:
        handler = RecoverOnceHandler()
        steps = iter(
            [
                AgentDecision.tool(ToolInvocation("search", {"query": "same"})),
                AgentDecision.tool(ToolInvocation("search", {"query": "same"})),
                AgentDecision.final("现在拿到结果了。"),
            ]
        )
        result = AgentLoop(planner=lambda **_: next(steps), handlers={"search": handler}).run()
        self.assertEqual(result.status, "completed")
        self.assertEqual(handler.calls, 2)
        self.assertEqual([item.status for item in result.tool_envelopes], ["error", "ok"])

    def test_read_can_repeat_after_a_successful_change_for_verification(self) -> None:
        observer = ObservationHandler()
        changer = ChangeHandler()
        steps = iter(
            [
                AgentDecision.tool(ToolInvocation("observe", {"query": "same"})),
                AgentDecision.tool(ToolInvocation("change", {"query": "write"})),
                AgentDecision.tool(ToolInvocation("observe", {"query": "same"})),
                AgentDecision.final("已经核对修改结果。"),
            ]
        )
        result = AgentLoop(
            planner=lambda **_: next(steps),
            handlers={"observe": observer, "change": changer},
        ).run()
        self.assertEqual(result.status, "completed")
        self.assertEqual((observer.calls, changer.calls), (2, 1))

    def test_unclassified_tools_are_bounded_as_observations(self) -> None:
        handler = Handler()
        result = AgentLoop(
            planner=lambda *, state: AgentDecision.tool(ToolInvocation("search", {"query": str(state.rounds)})),
            handlers={"search": handler},
            max_rounds=5,
            max_consecutive_observations=1,
        ).run()
        self.assertEqual((result.status, result.reason), ("failed", "max_rounds"))
        self.assertEqual(handler.calls, 5)

    def test_max_rounds_prevents_unbounded_planning(self) -> None:
        calls = []

        def planner(*, state):
            calls.append(state.rounds)
            return AgentDecision.tool(ToolInvocation("search", {"query": str(state.rounds)}))

        result = AgentLoop(planner=planner, handlers={"search": Handler()}, max_rounds=2).run()
        self.assertEqual((result.status, result.reason, result.rounds), ("failed", "max_rounds", 2))
        self.assertEqual(calls, [1, 2])
        self.assertTrue(result.final_text)

    def test_repeated_read_only_observation_is_bounded_and_stops_with_a_reply(self) -> None:
        handler = ObservationHandler()

        original_execute = handler.execute
        def unchanged(*, call, context):
            result = original_execute(call=call, context=context)
            result["followup_context"] = "unchanged evidence"
            return result
        handler.execute = unchanged

        def planner(*, state):
            return AgentDecision.tool(ToolInvocation("observe", {"query": str(state.rounds)}))

        result = AgentLoop(
            planner=planner,
            handlers={"observe": handler},
            max_rounds=8,
        ).run()

        self.assertEqual((result.status, result.reason, result.rounds), ("blocked", "no_progress", 7))
        self.assertEqual(handler.calls, 7)
        self.assertEqual((result.observation_calls, result.observation_budget_rejections), (7, 1))
        self.assertIn("没有完成", result.final_text)
        self.assertEqual(result.tool_envelopes[-1].status, "ok")

    def test_non_observation_tool_resets_the_read_only_streak(self) -> None:
        observations = ObservationHandler()
        change = ChangeHandler()
        steps = iter(
            [
                AgentDecision.tool(ToolInvocation("observe", {"query": "a"})),
                AgentDecision.tool(ToolInvocation("observe", {"query": "b"})),
                AgentDecision.tool(ToolInvocation("change", {"query": "c"})),
                AgentDecision.tool(ToolInvocation("observe", {"query": "d"})),
                AgentDecision.tool(ToolInvocation("observe", {"query": "e"})),
                AgentDecision.final("已经处理并核实。"),
            ]
        )
        result = AgentLoop(
            planner=lambda *, state: next(steps),
            handlers={"observe": observations, "change": change},
            max_rounds=6,
            max_consecutive_observations=2,
        ).run()

        self.assertEqual((result.status, result.final_text), ("completed", "已经处理并核实。"))
        self.assertEqual((observations.calls, change.calls, result.observation_calls), (4, 1, 4))

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

    def test_completion_gate_exception_fails_closed(self) -> None:
        def broken_gate(*_):
            raise RuntimeError("broken")

        result = AgentLoop(
            planner=lambda **_: AgentDecision.final("已经完成。"),
            handlers={},
            completion_gate=broken_gate,
        ).run()
        self.assertEqual((result.status, result.reason), ("failed", "completion_gate_error:RuntimeError"))
        self.assertIn("不能确认", result.final_text)

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

    def test_required_approval_without_gate_fails_closed(self) -> None:
        handler = Handler()
        result = AgentLoop(
            planner=lambda **_: AgentDecision.tool(ToolInvocation("search", {"query": "x"})),
            handlers={"search": handler},
            policy=ExecutionPolicy(risk_by_tool={"search": "high"}),
        ).run()
        self.assertEqual((result.status, result.reason), ("failed", "approval_gate_unavailable"))
        self.assertEqual(handler.calls, 0)

    def test_approval_gate_failure_is_returned_as_agent_failure(self) -> None:
        handler = Handler()

        def broken_gate(**_kwargs):
            raise RuntimeError("approval backend unavailable")

        result = AgentLoop(
            planner=lambda **_: AgentDecision.tool(ToolInvocation("search", {"query": "x"})),
            handlers={"search": handler},
            approval_gate=broken_gate,
        ).run()
        self.assertEqual((result.status, result.reason), ("failed", "approval_gate_error:RuntimeError"))
        self.assertEqual(handler.calls, 0)


if __name__ == "__main__":
    unittest.main()
