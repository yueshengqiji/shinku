from __future__ import annotations

import unittest
from types import SimpleNamespace

from shinku.hosts.tool_host import ToolHost
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry


class HostSearch:
    tool_type = "search"

    def tool_metadata(self):
        return SimpleNamespace(input_schema={"type": "object", "properties": {"query": {"type": "string"}}})

    def normalize_call(self, call):
        return call if call.get("query") else None

    def execute(self, *, call, context):
        return {"tool_type": "search", "followup_context": f"host:{call['query']}"}


class GrayRuntime:
    def __init__(self, *, blocked: bool = False):
        self.calls = []
        self.blocked = blocked

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {"tool_calls": [{"id": "gray-1", "function": {"name": "search", "arguments": '{"query":"ok"}'}}]}
        return {"speech": "宿主链路完成。"}


class UnsupportedNativeRuntime:
    def __init__(self):
        self.calls = []

    def chat_supports_native_tools(self):
        return False

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {"speech": "当前模型不支持工具。"}


class ToolHostGrayTests(unittest.TestCase):
    def test_health_reports_empty_host_as_not_ready(self) -> None:
        health = ToolHost(ToolRegistry(), host_id="gray").health()
        self.assertEqual(health.as_dict(), {"host_id": "gray", "ready": False, "tool_names": [], "reason": "no_tools_registered"})

    def test_run_connects_runtime_registry_and_agent_loop(self) -> None:
        host = ToolHost(ToolRegistry({"search": HostSearch()}), host_id="gray")
        runtime = GrayRuntime()
        result = host.run(
            runtime=runtime,
            system_prompt="agent",
            task_id="gray-task",
            initial_transcript=[{"role": "user", "content": "查一下"}],
            prompt_cache_key="gray:v1",
        )
        self.assertEqual((result.status, result.final_text), ("completed", "宿主链路完成。"))
        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual(runtime.calls[0]["native_tools"][0]["function"]["name"], "search")

    def test_allowed_tool_names_limit_schema_without_mutating_registry(self) -> None:
        registry = ToolRegistry({"search": HostSearch()})
        host = ToolHost(registry)
        runtime = GrayRuntime()
        result = host.run(runtime=runtime, system_prompt="agent", allowed_tool_names={"missing"})
        self.assertEqual(result.status, "completed")
        self.assertEqual(runtime.calls[0]["native_tools"], None)
        self.assertEqual(result.tool_envelopes[0].data["code"], "unknown_tool")
        self.assertEqual(registry.names(), ("search",))

    def test_policy_reaches_real_host_handler_boundary(self) -> None:
        host = ToolHost(ToolRegistry({"search": HostSearch()}))
        runtime = GrayRuntime()
        result = host.run(
            runtime=runtime,
            system_prompt="agent",
            policy=ExecutionPolicy(blocked=frozenset({"search"})),
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_envelopes[0].data["code"], "tool_blocked")

    def test_runtime_capability_gate_prevents_false_tool_advertising(self) -> None:
        host = ToolHost(ToolRegistry({"search": HostSearch()}), host_id="gray")
        runtime = UnsupportedNativeRuntime()
        health = host.runtime_health(runtime)
        self.assertEqual(
            health.as_dict(),
            {
                "host_id": "gray",
                "ready": False,
                "native_tools_supported": False,
                "tool_names": ["search"],
                "reason": "runtime_native_tools_unsupported",
            },
        )
        result = host.run(runtime=runtime, system_prompt="agent")
        self.assertEqual((result.status, result.final_text), ("completed", "当前模型不支持工具。"))
        self.assertIsNone(runtime.calls[0]["native_tools"])
        self.assertEqual(result.tool_envelopes, ())


if __name__ == "__main__":
    unittest.main()
