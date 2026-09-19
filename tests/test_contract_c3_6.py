from __future__ import annotations

import unittest
from types import SimpleNamespace

from shinku.agent.loop import AgentLoop
from shinku.agent.planner import LLMPlanner
from shinku.tools.registry import ToolRegistry


class SearchHandler:
    tool_type = "search"

    def tool_metadata(self):
        return SimpleNamespace(input_schema={"type": "object", "properties": {"query": {"type": "string"}}})

    def build_prompt_instruction(self):
        return "查找公开资料。"

    def normalize_call(self, call):
        return call if call.get("query") else None

    def execute(self, *, call, context):
        return {"tool_type": "search", "followup_context": f"已找到：{call['query']}"}


class ScriptedRuntime:
    def __init__(self):
        self.calls = []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {"tool_calls": [{"id": "native-1", "function": {"name": "search", "arguments": '{"query":"Shinku"}'}}]}
        return {"speech": "找到了。"}


class RegistryAndJointRegressionTests(unittest.TestCase):
    def test_registry_rejects_bad_or_duplicate_handlers(self) -> None:
        registry = ToolRegistry()
        handler = SearchHandler()
        self.assertEqual(registry.register(handler), "search")
        with self.assertRaises(ValueError):
            registry.register(handler)
        with self.assertRaises(ValueError):
            registry.register(object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            registry.register(object(), name="bad")  # type: ignore[arg-type]

    def test_registry_view_is_read_only(self) -> None:
        registry = ToolRegistry({"search": SearchHandler()})
        view = registry.view()
        self.assertEqual(registry.names(), ("search",))
        with self.assertRaises(TypeError):
            view["other"] = SearchHandler()  # type: ignore[index]

    def test_native_specs_are_generated_from_registered_handlers(self) -> None:
        registry = ToolRegistry({"search": SearchHandler()})
        specs = registry.native_specs()
        self.assertEqual(specs[0]["function"]["name"], "search")
        self.assertEqual(specs[0]["function"]["parameters"]["type"], "object")
        self.assertEqual(registry.native_specs(allowed_tool_names={"missing"}), [])

    def test_planner_loop_registry_runtime_joint_path(self) -> None:
        registry = ToolRegistry({"search": SearchHandler()})
        runtime = ScriptedRuntime()
        planner = LLMPlanner(
            runtime=runtime,
            system_prompt="你是任务执行器。",
            tool_specs=registry.native_specs(),
            prompt_cache_key="agent:joint:v1",
        )
        result = AgentLoop(planner=planner, handlers=registry.view(), max_rounds=3).run(
            task_id="task::joint",
            initial_transcript=[{"role": "user", "content": "帮我查 Shinku"}],
        )
        self.assertEqual((result.status, result.final_text, result.rounds), ("completed", "找到了。", 2))
        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual(runtime.calls[0]["native_tools"][0]["function"]["name"], "search")
        self.assertIn("已找到：Shinku", result.transcript[-1]["content"])


if __name__ == "__main__":
    unittest.main()
