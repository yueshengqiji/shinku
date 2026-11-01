"""The initial offer and registry follow-up must use one audited definition."""
from types import SimpleNamespace

import pytest

from shinku.qq.interaction_tools import AuditedReadHandler, ChatReadCapability, InteractionRuntime
from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.tools.registry import ToolRegistry


@pytest.mark.parametrize("route", ["perception", "public_query"])
def test_registry_followup_is_accepted_by_same_strict_catalog_as_initial_offer(route):
    schema = {"type": "object", "properties": {"scope": {"type": "string", "enum": ["current"],
        "maxLength": 16}}, "required": ["scope"], "additionalProperties": False}
    cap = ChatReadCapability("scoped_read", "Read the selected public fixture.", schema,
        route=route, max_calls=2, max_result_bytes=1024, max_seconds=2.5)
    handler = SimpleNamespace(tool_type=cap.name,
        tool_metadata=lambda: SimpleNamespace(description="Unreviewed handler prose", input_schema=schema))
    audited = AuditedReadHandler(handler, cap)
    registry = ToolRegistry({cap.name: audited})
    initial = [cap.spec()]
    followup = registry.native_specs(allowed_tool_names={cap.name})
    assert initial == followup
    calls = []
    runtime = InteractionRuntime(SimpleNamespace(call_chat_json=lambda **kw: calls.append(kw) or
        {"kind": "final", "text": "Read accepted."}), registry, {cap.name: cap})
    for specs in (initial, followup):
        assert runtime.call_chat_json(native_tools=specs, system_prompt="Read selected scope.")["kind"] == "final"
    assert len(calls) == 2
    # Mutating a serialized copy never expands the owned capability schema.
    followup[0]["function"]["parameters"]["properties"]["scope"]["enum"].append("other")
    with pytest.raises(AgentBoundaryError, match="chat_execution_tools_rejected"):
        runtime.call_chat_json(native_tools=followup, system_prompt="Read selected scope.")
    assert len(calls) == 2 and cap.parameters["properties"]["scope"]["enum"] == ["current"]
