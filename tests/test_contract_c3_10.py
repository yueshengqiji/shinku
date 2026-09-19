from __future__ import annotations

import unittest
from types import SimpleNamespace

from shinku.hosts.external import ExternalToolBridge, ExternalToolDescriptor
from shinku.hosts.tool_host import ToolHost


class RecordingTransport:
    def __init__(self):
        self.calls = []

    def call(self, *, tool_name, arguments, context=None):
        self.calls.append((tool_name, arguments, context))
        return {"tool_type": tool_name, "followup_context": f"外部宿主已执行：{arguments['query']}"}


class ExternalRuntime:
    def __init__(self):
        self.calls = []

    def chat_supports_native_tools(self):
        return True

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {"tool_calls": [{"id": "external-1", "function": {"name": "browser.search", "arguments": '{"query":"Shinku"}'}}]}
        return {"speech": "外部宿主链路完成。"}


class ExternalHostContractTests(unittest.TestCase):
    def test_descriptor_and_health_are_transport_neutral(self) -> None:
        transport = RecordingTransport()
        bridge = ExternalToolBridge(
            host_id="browser",
            transport=transport,
            descriptors=[
                ExternalToolDescriptor(
                    name="browser.search",
                    description="搜索公开网页。",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                )
            ],
        )
        self.assertEqual(
            bridge.health(),
            {"host_id": "browser", "ready": True, "tool_names": ["browser.search"], "reason": ""},
        )
        self.assertEqual(bridge.registry().names(), ("browser.search",))

    def test_external_bridge_runs_through_the_same_agent_host_boundary(self) -> None:
        transport = RecordingTransport()
        bridge = ExternalToolBridge(
            host_id="browser",
            transport=transport,
            descriptors=[
                ExternalToolDescriptor(
                    name="browser.search",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                )
            ],
        )
        runtime = ExternalRuntime()
        result = ToolHost(bridge.registry(), host_id="browser").run(
            runtime=runtime,
            system_prompt="你是外部工具 Agent。",
            context={"source": "gray"},
            initial_transcript=[{"role": "user", "content": "搜索 Shinku"}],
        )
        self.assertEqual((result.status, result.final_text, result.rounds), ("completed", "外部宿主链路完成。", 2))
        self.assertEqual(transport.calls, [("browser.search", {"query": "Shinku"}, {"source": "gray"})])

    def test_invalid_descriptor_is_rejected_before_transport_use(self) -> None:
        with self.assertRaises(ValueError):
            ExternalToolDescriptor(name="browser search")


if __name__ == "__main__":
    unittest.main()
