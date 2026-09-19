from __future__ import annotations

import unittest
from types import SimpleNamespace

from shinku.hosts.tool_host import ToolHost
from shinku.llm.runtime import LLMRuntime
from shinku.tools.registry import ToolRegistry


class SearchHandler:
    tool_type = "search"

    def tool_metadata(self):
        return SimpleNamespace(input_schema={"type": "object", "properties": {"query": {"type": "string"}}})

    def normalize_call(self, call):
        return call if call.get("query") else None

    def execute(self, *, call, context):
        return {"tool_type": "search", "followup_context": f"runtime:{call['query']}"}


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **payload):
        self.calls.append(payload)
        if len(self.calls) == 1:
            message = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="runtime-call-1",
                        function=SimpleNamespace(name="search", arguments='{"query":"Shinku"}'),
                    )
                ],
            )
        else:
            message = SimpleNamespace(content='{"speech":"真实 runtime 链路完成。"}', tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class FakeOpenAIClient:
    base_url = "https://api.deepseek.com/v1"
    protocol = "openai"

    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


class RuntimeNativeToolRegressionTests(unittest.TestCase):
    def test_real_runtime_shape_reaches_agent_handler(self) -> None:
        runtime = LLMRuntime()
        fake_client = FakeOpenAIClient()
        runtime.chat.client = fake_client
        runtime.chat.model = "deepseek-v4-flash"
        host = ToolHost(ToolRegistry({"search": SearchHandler()}), host_id="runtime-gray")

        health = host.runtime_health(runtime)
        result = host.run(
            runtime=runtime,
            system_prompt="你是一个工具执行 Agent。",
            task_id="runtime-gray-task",
            initial_transcript=[{"role": "user", "content": "查一下 Shinku"}],
        )

        self.assertTrue(health.ready)
        self.assertTrue(health.native_tools_supported)
        self.assertEqual((result.status, result.final_text, result.rounds), ("completed", "真实 runtime 链路完成。", 2))
        self.assertEqual(result.tool_envelopes[0].data["tool_type"], "search")
        self.assertEqual(len(fake_client.chat.completions.calls), 2)
        self.assertEqual(fake_client.chat.completions.calls[0]["tools"][0]["function"]["name"], "search")


if __name__ == "__main__":
    unittest.main()
