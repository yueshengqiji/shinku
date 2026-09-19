from __future__ import annotations

import unittest

from shinku.agent.loop import AgentState
from shinku.agent.planner import LLMPlanner


class FakeRuntime:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class LLMPlannerTests(unittest.TestCase):
    def test_final_reply_is_parsed_and_structured_tools_are_forwarded(self) -> None:
        runtime = FakeRuntime({"speech": "完成了。"})
        planner = LLMPlanner(runtime=runtime, system_prompt="persona", tool_specs=[{"type": "function"}], prompt_cache_key="agent:v1")
        decision = planner(state=AgentState(transcript=[{"role": "user", "content": "做它"}]))
        self.assertEqual((decision.kind, decision.text), ("final", "完成了。"))
        self.assertEqual(runtime.calls[0]["native_tools"], [{"type": "function"}])
        self.assertEqual(runtime.calls[0]["native_tool_choice"], "auto")
        self.assertEqual(runtime.calls[0]["prompt_cache_key"], "agent:v1")

    def test_legacy_tool_call_with_json_arguments_is_parsed(self) -> None:
        runtime = FakeRuntime({"tool_call": {"type": "search", "arguments": '{"query":"x"}', "id": "call-1"}})
        decision = LLMPlanner(runtime=runtime, system_prompt="p")(state=AgentState())
        assert decision is not None
        self.assertEqual((decision.kind, decision.invocation.name, decision.invocation.arguments), ("tool", "search", {"query": "x"}))

    def test_native_openai_tool_call_is_parsed(self) -> None:
        runtime = FakeRuntime({"tool_calls": [{"id": "call-2", "function": {"name": "search", "arguments": '{"query":"y"}'}}]})
        decision = LLMPlanner(runtime=runtime, system_prompt="p")(state=AgentState())
        assert decision is not None
        self.assertEqual(decision.invocation.id, "call-2")
        self.assertEqual(decision.invocation.arguments, {"query": "y"})

    def test_custom_prompt_builder_and_history_are_passed(self) -> None:
        runtime = FakeRuntime({"kind": "wait", "text": "缺文件"})
        planner = LLMPlanner(runtime=runtime, system_prompt="p", build_user_prompt=lambda state: f"round={state.rounds}", tool_specs=[])
        decision = planner(state=AgentState(rounds=3, transcript=[{"role": "tool", "content": "ok"}]))
        self.assertEqual(decision.kind, "wait")
        self.assertEqual(runtime.calls[0]["user_prompt"], "round=3")
        self.assertEqual(runtime.calls[0]["history_turns"], [{"role": "tool", "content": "ok"}])
        self.assertIsNone(runtime.calls[0]["native_tools"])

    def test_invalid_model_shape_is_left_for_loop_to_handle(self) -> None:
        planner = LLMPlanner(runtime=FakeRuntime({"unexpected": True}), system_prompt="p")
        self.assertIsNone(planner(state=AgentState()))


if __name__ == "__main__":
    unittest.main()
