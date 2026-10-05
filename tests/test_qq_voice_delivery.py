from __future__ import annotations

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.delivery import DeliveryResult, OutgoingMessage, OutgoingSegment
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


def test_voice_mode_definite_rejection_falls_back_to_text():
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
    assert result.sent
    assert sent == ["audio", "text"]


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
