"""An enum-valid route must also be admissible for its actual capability."""

import json

import pytest

from shinku.qq.agent_bridge import QQAgentBridge, QQAgentReply
from test_qq_public_query_fast_track import (
    Runtime, SourceTool, bridge, call, final, turn,
    close_coordinators,  # noqa: F401 -- isolated coordinator lifecycle
)


def setup_probe(*responses):
    probe = SourceTool("network_probe")
    probe.agent_fast_perception = True
    shell = SourceTool("arbitrary_shell")
    shell.agent_fast_perception = False
    shell.agent_permission_class = "change"
    shell.agent_progress_class = "change"
    runtime = Runtime(*responses)
    host, _, _ = bridge(runtime, extra={probe.tool_type: probe, shell.tool_type: shell})
    return host, runtime, probe, shell


def stages(host):
    return host.trace_store.snapshot()[0]["stages"]


def test_observation_wrong_track_is_corrected_without_extra_model_call():
    host, runtime, probe, shell = setup_probe(
        call("network_probe", {}, track="agent"), final("检查结果仅包含本次探测。"))
    result = host.handle_turn(turn("检查网络状态"))
    assert result.agent_status == "completed"
    assert result.reason == "lightweight_perception_fast_track"
    assert len(probe.calls) == 1 and not shell.calls
    assert len(runtime.calls) == 2 and not host.task_coordinator.list()
    assert any(s["name"] == "host_task_route" for s in stages(host))


def test_shell_none_becomes_tracked_task_without_executing_or_replanning():
    host, runtime, probe, shell = setup_probe(call("arbitrary_shell", {"command": "fixture-only"}, track="perception"))
    t = turn("检查网络状态")
    decision, _ = host._run_chat_decision(task_id="fixture", context={"qq_turn": t,
        "audience_context": {"role": "master", "is_master": True}}, initial_transcript=host._transcript(t))
    assert decision.task_link == {"mode": "new"}
    assert decision.invocation.name == "arbitrary_shell"
    assert decision.invocation.arguments == {"command": "fixture-only"}
    assert len(runtime.calls) == 1 and not probe.calls and not shell.calls


def test_missing_route_still_requires_contract_before_execution():
    value = call("arbitrary_shell", {"command": "fixture-only"})
    del value["tool_call"]["arguments"]["task_link"]
    host, runtime, probe, shell = setup_probe(value, {}, {})
    host._prepare_agent_task = QQAgentBridge._prepare_agent_task.__get__(host, QQAgentBridge)
    result = host.handle_turn(turn("检查网络状态"))
    assert result.agent_status != "completed"
    assert not probe.calls and not shell.calls and not host.task_coordinator.list()
    assert len(runtime.calls) == 3


def test_admissible_observation_does_not_add_a_repair_call():
    host, runtime, probe, _ = setup_probe(call("network_probe", {}, track="perception"), final("观察结果。"))
    result = host.handle_turn(turn("检查网络状态"))
    assert result.agent_status == "completed" and len(probe.calls) == 1
    assert len(runtime.calls) == 2


@pytest.mark.parametrize("mode", ["new", "continue"])
def test_explicit_task_routes_are_not_reclassified(mode):
    value = call("arbitrary_shell", {}, mode=mode)
    if mode == "continue":
        value["tool_call"]["arguments"]["task_link"]["task_id"] = "existing"
    host, runtime, _, shell = setup_probe(value)
    t = turn("继续排查网络")
    decision, _ = host._run_chat_decision(task_id="fixture", context={"qq_turn": t,
        "audience_context": {"role": "master", "is_master": True, "conversation_type": "private"}},
        initial_transcript=host._transcript(t))
    assert decision.task_link["mode"] == mode
    assert len(runtime.calls) == 1 and not shell.calls


def test_corrected_new_task_still_prepares_contract_before_enqueue():
    writer = SourceTool("fixture_write")
    writer.agent_permission_class = writer.agent_progress_class = "change"
    writer.agent_fast_public_query = False
    writer.agent_completion_effects = ("write",)
    contract = {"goal": "写入 fixture.txt", "deliverables": ["fixture.txt"], "scope": ["fixture"],
        "requirements": [{"id": "r1", "description": "文件写入", "effect": "write", "targets": ["fixture.txt"], "checks": {}}]}
    runtime = Runtime(call("fixture_write", {"path": "fixture.txt"}),
        {"task_link": {"mode": "new"}, "task_contract": contract})
    host, _, _ = bridge(runtime, extra={writer.tool_type: writer})
    host._prepare_agent_task = QQAgentBridge._prepare_agent_task.__get__(host, QQAgentBridge)
    admitted = []
    host._enqueue_agent_task = lambda *args, **kwargs: admitted.append(kwargs) or QQAgentReply("queued", reason="fixture-admitted")
    result = host.handle_turn(turn("写入 fixture.txt"))
    assert result.reason == "fixture-admitted" and len(admitted) == 1
    assert not writer.calls and len(runtime.calls) == 2
    assert not runtime.calls[1].get("native_tools")
    payload = json.loads(runtime.calls[1]["user_prompt"])
    assert payload["selected_tools"] == ["fixture_write"]
    assert admitted[0]["prepared_task"].task["contract"] == contract
