"""Protocol recovery is not proof that a tool is required or has executed."""

import json

import pytest
from test_qq_at_failure_regression import turn as addressed_turn
from test_qq_public_query_fast_track import (
    Runtime,
    bridge,
    call,
    close_coordinators,  # noqa: F401 -- shared pytest lifecycle fixture
    final,
    malformed_native,
    turn,
)

from shinku.conversation import AnswerEvidence, ConversationScope
from shinku.qq.commands import QQCommandState
from shinku.qq.delivery import DeliveryResult

CHAT_REVIEW = {"route": "chat", "relation": "unrelated"}
INVALID_DISPATCH = call(args="[]")
INVALID_INTENT = {"kind": "tool", "tool_call": {
    "name": "__shinku_tool_dispatch__", "arguments": {
        "tool_name": "private_web_search", "arguments": {"query": "fixture"}, "request_kind": "invalid",
    },
}}


@pytest.mark.parametrize("initial,reason", [
    (malformed_native(), "malformed_native_tool_call"),
    (INVALID_DISPATCH, "invalid_dispatch"),
    (INVALID_INTENT, "invalid_dispatch"),
])
@pytest.mark.parametrize("kind", ["private", "group"])
def test_ordinary_chat_can_recover_from_tool_shaped_protocol_failure(initial, reason, kind):
    runtime = Runtime(initial, final("我在。想聊什么？"), CHAT_REVIEW)
    host, search, reader = bridge(runtime)
    result = host.handle_turn(addressed_turn(kind, "老婆你说句话", "master"))
    assert result.agent_status == "completed"
    assert result.final_text == "我在。想聊什么？"
    assert not result.background_task_id and not host.task_coordinator.list()
    assert not search.calls and not reader.calls
    assert len(runtime.calls) == 2
    assert runtime.calls[1]["native_tool_choice"] == "auto"
    if reason == "malformed_native_tool_call":
        assert runtime.calls[1]["native_tools"][0]["function"]["name"] != "__shinku_tool_dispatch__"
    stages = host.trace_store.snapshot()[0]["stages"]
    retry = next(row for row in stages if row["name"] == "chat_tool_route" and row.get("status") == "schema_retry")
    assert retry["reason"] == reason
    assert not hasattr(host, "_review_direct_reply_route")


def test_recovered_chat_reaches_voice_delivery_without_executing_tools():
    runtime = Runtime(malformed_native(), final("我在这里。"), CHAT_REVIEW)
    host, search, reader = bridge(runtime, state=QQCommandState(permission_mode="full", reply_mode="voice"))
    rendered, sent = [], []
    host.voice_renderer = lambda text: rendered.append(text) or "C:/fixture-voice.wav"
    host.sender = lambda message: sent.append(message.segments[0].kind) or DeliveryResult(True, "sent", ("fixture",))
    result = host.handle_turn(turn("老婆你说句话"))
    assert result.agent_status == "completed" and result.sent
    assert rendered == ["我在这里。"] and sent == ["text", "audio"]
    assert not search.calls and not reader.calls and not host.task_coordinator.list()


@pytest.mark.parametrize("question,status,reason", [
    ("什么是拉格朗日点？", "failed", "direct_reply_unverified"),
    ("帮我联网查询今天的温度", "blocked", "direct_chat_reply"),
    ("核对以前那件事的原因", "failed", "direct_reply_unverified"),
    ("删除 C:/fixture/target.zip", "failed", "direct_reply_unverified"),
    ("继续完成刚才的研究任务", "failed", "direct_reply_unverified"),
])
def test_repaired_prose_does_not_discharge_real_evidence_or_action_obligation(question, status, reason):
    runtime = Runtime(malformed_native(), final("已经查证并完成了。"), {"route": "defer"})
    host, search, reader = bridge(runtime)
    result = host.handle_turn(turn(question))
    assert result.agent_status == status and result.reason == reason
    assert "已经查证并完成了" not in result.final_text
    assert not search.calls and not reader.calls and not host.task_coordinator.list()
    assert len(runtime.calls) == 2
    if status == "blocked":
        # 明确查询已在直复入口被拦截；检查具体原因，不能仅把 failed 放宽为任意非成功状态。
        assert any(row["name"] == "chat_query_grounding" and row.get("reason") == "chat_query_not_executed"
                   for row in host.trace_store.snapshot()[0]["stages"])


@pytest.mark.parametrize("review", [
    {}, {"route": "chat"}, {"route": "chat", "relation": "historical_fact"},
    {"route": "history", "tool_name": "unknown_write", "arguments": {}},
    RuntimeError("isolated reviewer unavailable"),
])
def test_unavailable_or_invalid_exit_review_cannot_promote_repair_to_success(review):
    runtime = Runtime(malformed_native(), final("我在这里。"), review)
    host, search, reader = bridge(runtime)
    result = host.handle_turn(turn("老婆你说句话"))
    assert result.agent_status == "completed" and len(runtime.calls) == 2
    assert result.final_text == "我在这里。"
    assert not search.calls and not reader.calls and not host.task_coordinator.list()


def test_wrong_chat_verdict_still_cannot_verify_unexecuted_mutation():
    runtime = Runtime(malformed_native(), final("已删除 C:/fixture/target.zip。"), CHAT_REVIEW)
    host, search, reader = bridge(runtime)
    result = host.handle_turn(turn("删除 C:/fixture/target.zip"))
    assert result.agent_status == "failed" and result.reason == "direct_reply_unverified"
    assert "已删除" not in result.final_text
    assert not search.calls and not reader.calls


def test_repair_review_also_owns_existing_history_exit_gate_without_duplicate_call():
    runtime = Runtime(malformed_native(), final("我在这里。"), CHAT_REVIEW)
    host, search, reader = bridge(runtime)
    scope = ConversationScope("private", "master", "master")
    host.conversation_runtime.begin(scope, "old-unconfirmed")
    host.conversation_runtime.step("old-unconfirmed", route="history", evidence=AnswerEvidence("history", "unverified"))
    host.conversation_runtime.finish("old-unconfirmed", status="failed")
    result = host.handle_turn(turn("换个话题，老婆你说句话"))
    assert result.agent_status == "completed" and len(runtime.calls) == 2
    snapshot = host.conversation_runtime.snapshot(scope)
    assert any(row["turn_id"] == "old-unconfirmed" and not row["verified"]
               for row in snapshot["recent_verifications"])
    assert not search.calls and not reader.calls


def test_explicit_failure_after_repair_is_not_promoted_or_re_reviewed():
    runtime = Runtime(malformed_native(), final("这次还没有完成。", "failed"))
    host, search, reader = bridge(runtime)
    result = host.handle_turn(turn("老婆你说句话"))
    assert result.agent_status == "failed" and len(runtime.calls) == 2
    assert not search.calls and not reader.calls


def test_normal_chat_does_not_add_a_repair_review():
    runtime = Runtime(final("我在这里。"))
    host, _, _ = bridge(runtime)
    result = host.handle_turn(turn("老婆你说句话"))
    assert result.agent_status == "completed" and len(runtime.calls) == 1
