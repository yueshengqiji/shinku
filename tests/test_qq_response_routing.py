from dataclasses import replace

import pytest

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult, deliver
from shinku.qq.message import IncomingMessage
from shinku.qq.napcat import NapCatActionTransport
from test_qq_outgoing_mentions import _group_turn, _private_turn, _Runtime


def bridge_and_sent(**kwargs):
    sent = []
    def sender(message):
        sent.append(message)
        return DeliveryResult(True, "sent", ("out-" + str(len(sent)),))
    return QQAgentBridge(runtime=_Runtime(), system_prompt="test", sender=sender,
                         bot_ids=("90000",), **kwargs), sent


def test_plain_text_binds_source_not_its_incoming_quote_or_batch_last():
    bridge, sent = bridge_and_sent()
    base = _group_turn()
    first = IncomingMessage.from_event({**base.messages[0].raw_event, "message": [
        {"type": "reply", "data": {"id": "old-quoted-id"}},
        {"type": "text", "data": {"text": "你觉得呢"}}]})
    other = IncomingMessage.from_event({**base.messages[0].raw_event, "message_id": "other",
        "user_id": "20002", "sender": {"user_id": "20002"}, "message": [{"type": "text", "data": {"text": "旁观"}}]})
    turn = replace(base, messages=(first, other), message_ids=("m-1", "other"))
    bridge._deliver_text(turn, "第一句。第二句。", record_memory=False, track_chat_flow=False)
    assert [m.reply_to_message_id for m in sent] == ["m-1", ""]
    assert [[s.target_id for s in m.segments if s.kind == "at"] for m in sent] == [[], []]
    payload = NapCatActionTransport._build_action(sent[0])[1]["message"]
    assert [s["type"] for s in payload] == ["reply", "text"]


def test_structured_model_choice_is_validated_and_encoded_without_text_marker():
    bridge, sent = bridge_and_sent()
    base = _group_turn()
    other = IncomingMessage.from_event({**base.messages[0].raw_event, "message_id": "other",
        "user_id": "20002", "sender": {"user_id": "20002"}})
    turn = replace(base, messages=(base.messages[0], other), message_ids=("m-1", "other"))
    routed = bridge._response_turn(turn, {"message_id": "other", "mention_user_ids": ["10001", "20002"]})
    bridge._deliver_text(routed, "你们说的两个点都看到了。", record_memory=False, track_chat_flow=False)
    assert not sent[0].reply_to_message_id
    assert [s.target_id for s in sent[0].segments if s.kind == "at"] == ["10001", "20002"]


@pytest.mark.parametrize("routing", [
    {"message_id": "missing", "mention_user_ids": ["10001"]},
    {"message_id": "m-1", "mention_user_ids": ["all"]},
    {"message_id": "m-1", "mention_user_ids": ["90000"]},
    {"message_id": "m-1", "mention_user_ids": ["999999"]},
    {"message_id": "m-1", "mention_user_ids": ["10001", "20002", "30003"]},
])
def test_invalid_selection_cannot_misdirect_reply(routing):
    bridge, sent = bridge_and_sent()
    turn = bridge._response_turn(_group_turn(), routing)
    assert turn.response_message_id == ""
    bridge._deliver_text(turn, "收到", record_memory=False, track_chat_flow=False)
    assert sent[0].reply_to_message_id == "m-1"
    assert not any(s.kind == "at" for s in sent[0].segments)


def test_cross_conversation_source_and_private_mode_never_get_group_routing():
    bridge, sent = bridge_and_sent()
    group = _group_turn()
    wrong = replace(group.messages[0], conversation_id="another-group")
    turn = replace(group, messages=(wrong,))
    bridge._deliver_text(turn, "收到", record_memory=False, track_chat_flow=False)
    assert not sent[0].reply_to_message_id
    assert all(s.kind != "at" for s in sent[0].segments)
    sent.clear()
    bridge._deliver_text(_private_turn(), "收到", record_memory=False, track_chat_flow=False)
    assert not sent[0].reply_to_message_id
    assert all(s.kind != "at" for s in sent[0].segments)


@pytest.mark.parametrize("mode", ["voice", "both"])
def test_voice_and_mixed_delivery_only_first_bubble_has_routing(mode):
    bridge, sent = bridge_and_sent(voice_renderer=lambda _: "C:/synthetic.wav")
    bridge.command_state.set_reply_mode(mode)
    result = bridge._deliver_text(_group_turn(), "第一句。第二句。", record_memory=False, track_chat_flow=False)
    assert result.sent
    assert sent[0].reply_to_message_id == "m-1"
    assert all(not m.reply_to_message_id for m in sent[1:])
    assert sum(s.kind == "at" for m in sent for s in m.segments) == 0
    assert sent[-1].segments[-1].kind == "audio"


def test_known_voice_receipt_prevents_text_replay_even_on_partial_failure():
    bridge, sent = bridge_and_sent(voice_renderer=lambda _: "C:/synthetic.wav")
    bridge.command_state.set_reply_mode("voice")
    def partial(message):
        sent.append(message)
        return DeliveryResult(False, "partial", ("known-sent",), "post_send_error")
    bridge.sender = partial
    result = bridge._deliver_text(_group_turn(), "收到", record_memory=False, track_chat_flow=False)
    assert len(sent) == 1
    assert not result.sent and result.delivery_status == "partial"
    assert result.delivery.message_ids == ("known-sent",)


def test_full_chat_structured_selection_reaches_transport():
    bridge, sent = bridge_and_sent()
    bridge.runtime = type("Runtime", (), {"call_chat_json": lambda self, **kw: {
        "kind": "final", "text": "收到", "reply_routing": {"message_id": "m-1", "mention_user_ids": []}}})()
    result = bridge.handle_turn(_group_turn())
    assert result.sent and sent[0].reply_to_message_id == "m-1"
    assert all(s.kind != "at" for s in sent[0].segments)


@pytest.mark.parametrize("mode", ["text", "voice", "both"])
@pytest.mark.parametrize("selection", ["default", "structured", "marker", "explicit_empty"])
def test_quote_and_at_are_mutually_exclusive_across_entire_reply(mode, selection):
    bridge, sent = bridge_and_sent(voice_renderer=lambda _: "C:/synthetic.wav")
    bridge.command_state.set_reply_mode(mode)
    turn = _group_turn()
    text = "第一句。第二句。"
    if selection == "structured":
        turn = bridge._response_turn(turn, {"message_id": "m-1", "mention_user_ids": ["20002"]})
    elif selection == "marker":
        text = "第一句。[[SHINKU_AT_3]]第二句。"
    elif selection == "explicit_empty":
        turn = bridge._response_turn(turn, {"message_id": "m-1", "mention_user_ids": []})
    result = bridge._deliver_text(turn, text, record_memory=False, track_chat_flow=False)
    assert result.sent
    payloads = [NapCatActionTransport._build_action(message)[1]["message"] for message in sent]
    quotes = [part for payload in payloads for part in payload if part["type"] == "reply"]
    mentions = [part for payload in payloads for part in payload if part["type"] == "at"]
    assert not (quotes and mentions)
    assert len(quotes) == (1 if selection in {"default", "explicit_empty"} else 0)
    assert [part["data"]["qq"] for part in mentions] == (["20002"] if selection in {"structured", "marker"} else [])
    assert all(part["type"] not in {"reply", "at"} for payload in payloads[1:] for part in payload)


def test_missing_source_message_id_falls_back_to_at_without_fake_quote():
    bridge, sent = bridge_and_sent()
    base = _group_turn()
    turn = replace(base, primary_message_id="", messages=(replace(base.messages[0], message_id=""),))
    bridge._deliver_text(turn, "收到", record_memory=False, track_chat_flow=False)
    assert not sent[0].reply_to_message_id
    assert [part.target_id for part in sent[0].segments if part.kind == "at"] == ["10001"]


def test_full_chat_explicit_at_selection_does_not_also_quote_source():
    bridge, sent = bridge_and_sent()
    bridge.runtime = type("Runtime", (), {"call_chat_json": lambda self, **kw: {
        "kind": "final", "text": "请青空看看", "reply_routing": {"message_id": "m-1", "mention_user_ids": ["20002"]}}})()
    result = bridge.handle_turn(_group_turn())
    assert result.sent and not sent[0].reply_to_message_id
    assert [part.target_id for part in sent[0].segments if part.kind == "at"] == ["20002"]


def test_only_actual_selected_recipient_affects_continuation_tracking():
    observations = []
    bridge, sent = bridge_and_sent(group_delivery_observer=lambda *args: observations.append(args))
    base = _group_turn()
    other = IncomingMessage.from_event({**base.messages[0].raw_event, "message_id": "other",
        "user_id": "20002", "sender": {"user_id": "20002"}})
    turn = replace(base, messages=(base.messages[0], other), message_ids=("m-1", "other"))
    routed = bridge._response_turn(turn, {"message_id": "other", "mention_user_ids": []})
    bridge._deliver_text(routed, "收到", record_memory=False, track_chat_flow=False)
    assert observations[0][1] == ("20002",)
    assert "10001" not in bridge._group_delivered_replies["g-1"][0][1]


def test_self_authored_source_is_not_a_quote_or_reply_recipient():
    bridge, sent = bridge_and_sent()
    base = _group_turn()
    self_message = replace(base.messages[0], sender_id="90000")
    bridge._deliver_text(replace(base, messages=(self_message,)), "收到", record_memory=False, track_chat_flow=False)
    assert not sent[0].reply_to_message_id
    assert not any(s.kind == "at" for s in sent[0].segments)


def test_known_partial_text_receipt_is_preserved_and_never_repeated_or_renotified():
    bridge, sent = bridge_and_sent()
    def sender(message):
        sent.append(message)
        return DeliveryResult(False, "partial", ("known",), "after_send") if len(sent) == 1 else DeliveryResult(True, "sent", ("second",))
    bridge.sender = sender
    result = bridge._deliver_text(_group_turn(), "第一句。第二句。", record_memory=False, track_chat_flow=False)
    assert not result.sent and result.delivery_status == "partial" and result.delivery.message_ids == ("known", "second")
    assert len(sent) == 2 and not sent[1].reply_to_message_id
    assert not any(s.kind == "at" for s in sent[1].segments)


def test_uncertain_first_send_does_not_repeat_notification_or_replay_the_bubble():
    bridge, sent = bridge_and_sent()
    def sender(message):
        sent.append(message)
        if len(sent) == 1:
            raise TimeoutError("may have sent")
        return DeliveryResult(True, "sent", ("second",))
    bridge.sender = sender
    bridge._deliver_text(_group_turn(), "第一句。第二句。", record_memory=False, track_chat_flow=False)
    assert len(sent) == 2 and not sent[1].reply_to_message_id
    assert not any(s.kind == "at" for s in sent[1].segments)


@pytest.mark.parametrize("kind", ["raises", "result_error", "invalid_result"])
def test_uncertain_voice_send_never_falls_back_to_duplicate_text(kind):
    bridge, sent = bridge_and_sent(voice_renderer=lambda _: "C:/synthetic.wav")
    bridge.command_state.set_reply_mode("voice")
    def sender(message):
        sent.append(message)
        if kind == "raises":
            raise TimeoutError("may have sent")
        return DeliveryResult(False, "transport_error", error="timeout") if kind == "result_error" else None
    bridge.sender = sender
    result = bridge._deliver_text(_group_turn(), "收到", record_memory=False, track_chat_flow=False)
    assert not result.sent and len(sent) == 1


def test_napcat_partial_error_preserves_known_id_and_is_not_definitive_rejection():
    bridge, sent = bridge_and_sent()
    bridge._deliver_text(_group_turn(), "收到", record_memory=False, track_chat_flow=False)
    transport = NapCatActionTransport(lambda **kw: {"status": "failed", "retcode": 1,
        "data": {"message_id": "known-id"}, "message": "post-send failure"})
    result = deliver(transport, sent[0])
    assert result.message_ids == ("known-id",) and result.status == "partial" and not result.definitive_not_sent
