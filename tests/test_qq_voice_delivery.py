from __future__ import annotations

from dataclasses import replace
import pytest

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState, QQCommandInterpreter
from shinku.qq.delivery import DeliveryResult, OutgoingMessage, OutgoingSegment
from shinku.qq.message import IncomingMessage
from shinku.qq.napcat import NapCatActionTransport
from shinku.qq.turns import QQTurn


class _Runtime:
    def call_chat_json(self, **kwargs):
        return {"speech": "unused"}


def _turn():
    return QQTurn(
        conversation_id="12345",
        conversation_type="private",
        primary_message_id="1",
        message_ids=("1",),
        combined_text="你好",
        messages=(),
    )


def test_napcat_audio_uses_onebot_record_segment():
    action, params = NapCatActionTransport._build_action(OutgoingMessage(
        conversation_id="12345",
        conversation_type="private",
        segments=(OutgoingSegment(kind="audio", local_path="C:/voice.wav"),),
    ))
    assert action == "send_private_msg"
    assert params["message"] == [{"type": "record", "data": {"file": "C:/voice.wav"}}]


def test_both_mode_sends_text_then_voice():
    sent = []
    state = QQCommandState(reply_mode="both")

    def sender(message):
        sent.append(message.segments[0].kind)
        return DeliveryResult(True, "sent")

    bridge = QQAgentBridge(
        runtime=_Runtime(), system_prompt="test", sender=sender,
        command_state=state, voice_renderer=lambda text: "C:/voice.wav",
    )
    result = bridge._deliver_text(_turn(), "你好。", record_memory=False, track_chat_flow=False)
    assert result.sent
    assert sent == ["text", "audio"]


def test_voice_rejection_keeps_sent_text_without_replay():
    sent = []
    state = QQCommandState(reply_mode="voice")

    def sender(message):
        kind = message.segments[0].kind
        sent.append(kind)
        return DeliveryResult(kind == "text", "sent" if kind == "text" else "rejected", definitive_not_sent=kind != "text")

    bridge = QQAgentBridge(
        runtime=_Runtime(), system_prompt="test", sender=sender,
        command_state=state, voice_renderer=lambda text: "C:/voice.wav",
    )
    result = bridge._deliver_text(_turn(), "你好。", record_memory=False, track_chat_flow=False)
    assert result.delivery_status == "partial"
    assert sent == ["text", "audio"]


def test_voice_render_failure_keeps_text_reply():
    sent = []

    def renderer(_text):
        raise TimeoutError("local TTS unavailable")

    def sender(message):
        sent.append(message.segments[0].kind)
        return DeliveryResult(True, "sent")

    bridge = QQAgentBridge(
        runtime=_Runtime(), system_prompt="test", sender=sender,
        command_state=QQCommandState(reply_mode="voice"), voice_renderer=renderer,
    )
    result = bridge._deliver_text(_turn(), "你好。", record_memory=False, track_chat_flow=False)
    assert result.sent
    assert sent == ["text"]


@pytest.mark.parametrize("mode", ["voice", "both"])
@pytest.mark.parametrize("mention", [False, True])
def test_group_quote_or_mention_stays_on_text_and_audio_is_standalone(mode, mention):
    message = IncomingMessage.from_event({
        "post_type":"message", "message_type":"group", "self_id":"999",
        "group_id":"12345", "user_id":"67890", "message_id":"1",
        "message":[{"type":"text","data":{"text":"你好"}}],
    })
    turn = replace(_turn(), conversation_type="group", messages=(message,),
                   response_mention_ids=("67890",) if mention else None)
    actions, order = [], []
    def sender(outgoing):
        action, payload = NapCatActionTransport._build_action(outgoing)
        actions.append((action,payload))
        order.append("audio" if outgoing.segments[-1].kind=="audio" else "text")
        return DeliveryResult(True,"sent",(f"receipt-{len(actions)}",))
    def renderer(text):
        order.append("render")
        assert len(actions)==2  # Both text bubbles arrived before TTS started.
        return "C:/voice.wav"
    bridge = QQAgentBridge(runtime=_Runtime(),system_prompt="test",sender=sender,
                          command_state=QQCommandState(reply_mode=mode),voice_renderer=renderer,
                          bot_ids=("999",))
    result = bridge._deliver_text(turn,"第一句。第二句。",record_memory=False,track_chat_flow=False)
    assert result.sent and order == ["text","text","render","audio"]
    assert all(action=="send_group_msg" for action,_ in actions)
    first = actions[0][1]["message"]
    assert [segment["type"] for segment in first] == (["at","text"] if mention else ["reply","text"])
    if not mention:
        assert str(first[0]["data"]["id"]) == "1"
    assert [segment["type"] for segment in actions[1][1]["message"]] == ["text"]
    assert actions[-1][1]["message"] == [{"type":"record","data":{"file":"C:/voice.wav"}}]


@pytest.mark.parametrize("failure", ["rejected","uncertain","exception"])
def test_text_failure_does_not_render_or_send_followup_voice(failure):
    sent, rendered = [], []
    def sender(message):
        sent.append(message)
        if failure == "exception":
            raise TimeoutError("fixture")
        return DeliveryResult(False,failure,definitive_not_sent=failure=="rejected")
    bridge = QQAgentBridge(runtime=_Runtime(),system_prompt="test",sender=sender,
        command_state=QQCommandState(reply_mode="voice"),voice_renderer=lambda text: rendered.append(text))
    result = bridge._deliver_text(_turn(),"你好。",record_memory=False,track_chat_flow=False)
    assert not result.sent and not rendered
    assert [message.segments[0].kind for message in sent] == ["text"]


def test_uncertain_audio_delivery_never_replays_text():
    sent=[]
    def sender(message):
        kind=message.segments[0].kind;sent.append(kind)
        if kind=="audio":raise TimeoutError("fixture")
        return DeliveryResult(True,"sent",("text-receipt",))
    bridge = QQAgentBridge(runtime=_Runtime(),system_prompt="test",sender=sender,
        command_state=QQCommandState(reply_mode="voice"),voice_renderer=lambda text:"C:/voice.wav")
    result=bridge._deliver_text(_turn(),"你好。",record_memory=False,track_chat_flow=False)
    assert sent==["text","audio"] and result.delivery_status=="partial"
    assert result.delivery.message_ids==("text-receipt",)


def test_voice_command_describes_text_then_voice():
    state=QQCommandState()
    outcome=QQCommandInterpreter(state).execute("/reply mode voice")
    assert outcome.status=="reply_mode_changed" and state.reply_mode()=="voice"
    assert "先发文字" in outcome.reply and "再单独发送语音" in outcome.reply
