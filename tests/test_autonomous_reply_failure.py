"""Autonomous delivery needs a real model decision, never a local error template.

All SDK responses, memories, approvals and senders are isolated fixtures.
"""
import copy
import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.tasks.coordinator import AgentTaskCoordinator
from test_contract_c2_5 import FakeResponse, _make_runtime


FALLBACK = "……刚才那句我没接好。你再说一遍？"


def turn(*, kind="group", mode="autonomous", mentioned=False):
    parts = [{"type": "text", "data": {"text": "大家觉得今天怎么样？"}}]
    if mentioned:
        parts.insert(0, {"type": "at", "data": {"qq": "10001"}})
    message = IncomingMessage.from_event({"post_type": "message", "message_type": kind,
        "message_id": "fixture-auto", "user_id": "guest", "group_id": "42",
        "self_id": "10001", "message": parts})
    item = QQTurn(conversation_id=message.conversation_id, conversation_type=kind,
        primary_message_id=message.message_id, message_ids=(message.message_id,),
        combined_text=message.text, messages=(message,), attention_mode=mode)
    return item


class FailedRuntime:
    calls = 0

    def call_chat_json(self, **kwargs):
        self.calls += 1
        return copy.deepcopy(kwargs["fallback"])


def bot(runtime, sent, **kwargs):
    return QQAgentBridge(runtime=runtime, system_prompt="isolated fixture", bot_ids=("10001",),
        sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("receipt",)),
        **kwargs)


def assert_suppressed(bridge, result, sent):
    assert result.agent_status == "failed"
    assert result.delivery_status == "suppressed"
    assert result.reason == "chat_generation_failed"
    assert not result.sent and not result.final_text and sent == []
    row = bridge.trace_store.snapshot()[0]
    assert row["status"] == "failed" and not row["sent"] and row["reply_chars"] == 0
    assert any(s["name"] == "autonomous_delivery" and s["status"] == "suppressed"
               for s in row["stages"])
    assert not any(s["name"] == "delivery_prepare" for s in row["stages"])


def test_transport_fallback_cannot_speak_or_record_an_assistant_reply():
    runtime, sent = FailedRuntime(), []
    bridge = bot(runtime, sent)
    bridge._record_assistant_reply = Mock(side_effect=AssertionError("no fake assistant memory"))
    bridge.voice_renderer = Mock(side_effect=AssertionError("no voice generation"))
    bridge._finish_chat_flow = Mock(side_effect=AssertionError("no phantom conversation reply"))
    result = bridge.handle_turn(turn())
    assert_suppressed(bridge, result, sent)
    assert runtime.calls == 1


@pytest.mark.parametrize("content", ["", "{}", '{"kind":"unknown"}',
    '{"kind":"final","text":""}', '{"kind":"wait","text":""}',
    '{"emotion":"normal",'])
def test_actual_runtime_empty_invalid_and_metadata_only_output_is_quiet(content):
    runtime, sent = _make_runtime(scripts=[("response", FakeResponse(content))]), []
    try:
        bridge = bot(runtime, sent)
        assert_suppressed(bridge, bridge.handle_turn(turn()), sent)
    finally:
        runtime.close()


@pytest.mark.parametrize("item", [turn(kind="private"), turn(mode="immediate"),
    turn(mentioned=True), turn(kind="private", mentioned=True)])
def test_explicit_turns_keep_failure_feedback(item):
    sent = []
    result = bot(FailedRuntime(), sent).handle_turn(item)
    assert result.agent_status == "failed" and result.reason == "chat_generation_failed"
    assert result.sent and sent and result.final_text == FALLBACK


@pytest.mark.parametrize("content", [
    "今天还不错。你呢？",
    json.dumps({"kind": "final", "text": "今天还不错。你呢？"}, ensure_ascii=False),
    json.dumps({"kind": "final", "text": "这次没有完成。", "task_outcome": "failed"}, ensure_ascii=False),
    json.dumps({"kind": "final", "text": FALLBACK, "task_outcome": "failed"}, ensure_ascii=False),
    '{"speech":"今天还不错。",',
])
def test_real_model_text_including_honest_failure_is_not_blanket_suppressed(content):
    runtime, sent = _make_runtime(scripts=[("response", FakeResponse(content)),
        ("response", FakeResponse(json.dumps({"verdict": "allow", "relation": "join", "task_claim": "none", "task_id": ""})))],
        dialogue_reviews=True), []
    try:
        result = bot(runtime, sent).handle_turn(turn())
        assert result.sent and sent and result.delivery_status == "sent", result.reason
        assert result.final_text
        if "speech" in content:
            assert result.final_text == "今天还不错。"
        if content.startswith('{"kind"') and FALLBACK in content:
            assert result.reason == "direct_chat_reply"
    finally:
        runtime.close()


def test_model_selected_silence_remains_distinct_from_generation_failure():
    runtime = _make_runtime(scripts=[("response", FakeResponse('{"speech":"<SILENCE>"}'))])
    sent = []
    try:
        result = bot(runtime, sent).handle_turn(turn())
        assert (result.agent_status, result.delivery_status, result.reason) == (
            "silent", "suppressed", "attention_silence")
        assert not sent
    finally:
        runtime.close()


@pytest.mark.parametrize("async_enabled", [False, True])
def test_pending_approval_is_not_read_consumed_or_rejected_by_autonomous_chatter(async_enabled):
    runtime = _make_runtime(scripts=[("response", FakeResponse('{"speech":"<SILENCE>"}'))])
    sent = []
    approvals = Mock()
    approvals.get.side_effect = AssertionError("autonomous chatter must not consume an approval")
    coordinator = AgentTaskCoordinator() if async_enabled else None
    try:
        bridge = bot(runtime, sent, approval_store=approvals,
            master_user_ids=("owner",), owner_gate_required=True,
            async_agent_enabled=async_enabled, task_coordinator=coordinator)
        result = bridge.handle_turn(turn())
        assert result.reason == "attention_silence" and not sent
        approvals.get.assert_not_called()
        approvals.delete.assert_not_called()
    finally:
        if coordinator is not None:
            coordinator.shutdown(wait=True)
        runtime.close()


def test_delivery_boundary_rejects_unselected_local_error_before_memory_or_shaping():
    sent = []
    bridge = bot(FailedRuntime(), sent)
    bridge._record_assistant_reply = Mock(side_effect=AssertionError("no fake memory"))
    bridge.reply_host.shape = Mock(side_effect=AssertionError("no shaping"))
    result = bridge._deliver_text(turn(), "固定失败提示", agent_status="blocked", reason="fixture_error")
    assert result.delivery_status == "suppressed" and result.reason == "fixture_error"
    assert not result.sent and not result.final_text and not sent


def test_mixed_batch_with_an_explicit_mention_keeps_failure_feedback():
    item = turn()
    addressed = turn(mentioned=True).messages[0]
    item = replace(item, messages=(*item.messages, addressed))
    sent = []
    result = bot(FailedRuntime(), sent).handle_turn(item)
    assert result.sent and sent and result.final_text == FALLBACK


@pytest.mark.parametrize("kind", ["group", "private"])
def test_budget_denial_before_sdk_is_quiet_only_for_autonomous_group(kind):
    runtime = _make_runtime(scripts=[])
    coordinator, sent = AgentTaskCoordinator(), []
    try:
        bridge = bot(runtime, sent, async_agent_enabled=True, task_coordinator=coordinator,
            task_budget_limits={"max_tokens": 1})
        result = bridge.handle_turn(turn(kind=kind))
        assert result.reason == "task_token_budget_exhausted"
        assert result.delivery_status == ("suppressed" if kind == "group" else "sent")
        assert bool(sent) == (kind == "private")
        assert not runtime.chat.client.payloads
    finally:
        coordinator.shutdown(wait=True)
        runtime.close()


@pytest.mark.parametrize("status", ["failed", "blocked"])
def test_recovered_partial_model_report_preserves_its_explicit_status(status):
    content = '{"speech":"这次没有完成。","task_outcome":"' + status + '",'
    runtime, sent = _make_runtime(scripts=[("response", FakeResponse(content)),
        ("response", FakeResponse(json.dumps({"verdict": "allow", "relation": "join", "task_claim": "none"})))]), []
    try:
        result = bot(runtime, sent).handle_turn(turn())
        assert result.sent and result.final_text == "这次没有完成。"
        assert result.agent_status == status
    finally:
        runtime.close()


def test_recovered_partial_status_does_not_bypass_tool_evidence_verification():
    runtime = _make_runtime(scripts=[("response", FakeResponse(
        '{"speech":"部分完成了。","task_outcome":"partial",')),
        ("response", FakeResponse(json.dumps({"verdict": "allow", "relation": "join", "task_claim": "none"})))],
        dialogue_reviews=True)
    sent = []
    try:
        result = bot(runtime, sent).handle_turn(turn())
        assert result.agent_status == "failed" and result.reason == "direct_reply_unverified"
        assert result.delivery_status == "suppressed" and not sent
    finally:
        runtime.close()


def test_raised_sdk_failure_is_also_quiet_without_usage_receipt():
    runtime = _make_runtime(scripts=[("raise", RuntimeError("isolated upstream error"))])
    sent = []
    try:
        bridge = bot(runtime, sent)
        assert_suppressed(bridge, bridge.handle_turn(turn()), sent)
        assert runtime.chat.client.payloads
        assert not any(s["name"] == "llm_usage" for s in bridge.trace_store.snapshot()[0]["stages"])
    finally:
        runtime.close()
