"""Initial schema repair must preserve public-query versus research intent."""
import pytest
from shinku.qq.agent_bridge import QQAgentBridge
from test_qq_public_query_fast_track import Runtime, bridge, final, turn, malformed_native, close_coordinators


@pytest.mark.parametrize("initial", [malformed_native("private_web_search"),
    {"kind": "tool", "tool_call": {"name": "__shinku_tool_dispatch__", "arguments": {
        "tool_name": "private_web_search", "arguments": "INVALID_JSON"}}}])
def test_initial_repair_preserves_lightweight_route_without_heavy_execution(initial):
    repaired = {"kind": "tool", "tool_call": {"name": "private_web_search", "arguments": {"query": "拉格朗日点"}},
                "task_link": {"mode": "none", "track": "public_query"}}
    runtime = Runtime(initial, repaired, final("未取得正文，无法完成查证。", "failed"))
    host, search, _ = bridge(runtime)
    reply = host.handle_turn(turn("什么是拉格朗日点？"))
    assert reply.reason == "lightweight_public_query_fast_track"
    assert len(search.calls) == 1 and not reply.background_task_id
    if "_native_tool_call" in initial:
        assert runtime.calls[1]["native_tools"][0]["function"]["name"] != "__shinku_tool_dispatch__"
        assert runtime.calls[1]["native_tool_choice"] == "auto"
    else:
        assert runtime.calls[1]["native_tools"][0]["function"]["name"] == "__shinku_tool_dispatch__"
        assert runtime.calls[1]["native_tool_choice"] == "auto"


def test_missing_route_after_one_repair_does_not_execute_or_start_task():
    raw = {"kind": "tool", "tool_call": {"name": "private_web_search", "arguments": {"query": "拉格朗日点"}}}
    runtime = Runtime(malformed_native(), raw, {}, {})
    host, search, reader = bridge(runtime)
    host._prepare_agent_task = QQAgentBridge._prepare_agent_task.__get__(host, QQAgentBridge)
    reply = host.handle_turn(turn("什么是拉格朗日点？"))
    assert reply.agent_status != "completed"
    assert len(runtime.calls) == 2 and not search.calls and not reader.calls
    assert not reply.background_task_id


def test_repaired_research_intent_is_not_coerced_to_lightweight():
    raw = {"kind": "tool", "tool_call": {"name": "private_web_search", "arguments": {"query": "拉格朗日点"}},
           "task_link": {"mode": "new", "track": "agent"}}
    runtime = Runtime(malformed_native(), raw)
    host, search, _ = bridge(runtime)
    context = {"qq_turn": turn("深入研究并比较五个来源。"), "audience_context": {"is_master": True, "role": "master"}}
    decision, _ = host._run_chat_decision(task_id="probe", context=context,
        initial_transcript=[{"role":"user", "content":"深入研究并比较五个来源。"}])
    assert decision.task_link["mode"] == "new"
    assert not host._is_lightweight_public_query(decision, context=context)
    assert not search.calls


def test_protocol_repair_cannot_silently_skip_lookup_required_by_original_request():
    runtime = Runtime(malformed_native(), final("我已经联网检索并读取了网页，查证完成。", "completed"), {"route": "defer"})
    host, search, reader = bridge(runtime)
    reply = host.handle_turn(turn("什么是拉格朗日点？"))
    assert reply.agent_status == "failed" and reply.reason == "direct_reply_unverified"
    assert "我已经联网检索" not in reply.final_text
    assert len(runtime.calls) == 2 and not search.calls and not reader.calls
    assert not reply.background_task_id
