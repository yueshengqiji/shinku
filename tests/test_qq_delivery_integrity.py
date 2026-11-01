"""Reply delivery must not invent conversational evidence or ignore /stop."""
from types import SimpleNamespace
import socket

import pytest

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.reply_host import ReplyHost
from shinku.qq.turns import QQTurn


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def deny(*args, **kwargs):
        raise AssertionError("delivery-integrity fixtures cannot use the network")
    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket.socket, "connect_ex", deny)


@pytest.fixture
def memory(tmp_path):
    return MemoryService(store=MemoryStore(tmp_path / "fixture.sqlite3"), router=MemoryRouter(),
                         policy=MemoryPolicy(master_user_ids=("owner",)), clock=lambda: 1_800_000_000)


def turn(text="两个杯子怎么选", mid="request", *, group=False):
    message = IncomingMessage.from_event({
        "post_type": "message", "message_type": "group" if group else "private",
        "user_id": "owner", "self_id": "bot", "group_id": "g-1" if group else None,
        "message_id": mid, "sender": {"nickname": "主人"},
        "message": [{"type": "text", "data": {"text": text}}],
    })
    return QQTurn(conversation_id="g-1" if group else "owner",
                  conversation_type="group" if group else "private", primary_message_id=mid,
                  message_ids=(mid,), combined_text=text, messages=(message,))


def bridge(sender, memory=None, **kwargs):
    return QQAgentBridge(runtime=SimpleNamespace(call_chat_json=lambda **kw: {"speech": "unused"}),
                         system_prompt="fixture", sender=sender, memory_service=memory, **kwargs)


def history(memory, scope="private:owner"):
    return memory.store.recent_assistant_events(scope=scope, now=int(memory.clock()))


@pytest.mark.parametrize("failure", ["rejected", "exception", "invalid"])
def test_total_failure_creates_no_assistant_history_or_previous_exchange(memory, failure):
    def sender(message):
        if failure == "exception":
            raise TimeoutError("fixture")
        return None if failure == "invalid" else DeliveryResult(False, "rejected")
    subject = bridge(sender, memory)
    result = subject._deliver_text(turn(), "大的给你。小的给我。")
    assert not result.sent
    assert not history(memory)
    assert not history(memory, "private:master")
    assert subject.chat_flow.begin_turn(turn("为什么这么选", "followup")).previous_exchange == ""
    assert not memory.store.search(scope="private:owner", query="大的给你", now=int(memory.clock()), limit=8)


def test_partial_delivery_remembers_only_delivered_fragments_and_receipt(memory):
    attempted = []
    def sender(message):
        attempted.append(message.segments[0].text)
        return DeliveryResult(len(attempted) != 2, "sent" if len(attempted) != 2 else "rejected",
                              (f"out-{len(attempted)}",) if len(attempted) != 2 else ())
    subject = bridge(sender, memory)
    result = subject._deliver_text(turn(), "第一句。第二句。第三句。")
    assert result.delivery_status == "partial" and not result.sent
    previous = subject.chat_flow.begin_turn(turn("为什么这样", "followup")).previous_exchange
    assert "第一句" in previous and "第三句" in previous and "第二句" not in previous
    records = history(memory)
    assert len(records) == 1
    assert records[0].text == "第一句。\n第三句。"
    assert records[0].metadata["delivery_status"] == "partial"
    assert records[0].metadata["delivery_content_complete"] is False
    assert records[0].metadata["delivery_message_ids"] == ["out-1", "out-3"]
    assert result.final_text == "第一句。第二句。第三句。"  # Diagnostic result, not injected history.


def test_assistant_history_is_written_once_after_delivery_not_before(memory):
    counts = []
    def sender(message):
        counts.append(len(history(memory)))
        return DeliveryResult(True, "sent")
    subject = bridge(sender, memory)
    result = subject._deliver_text(turn(), "大的给你。小的给我。")
    assert result.sent and counts == [0, 0]
    assert len(history(memory)) == 1
    assert history(memory)[0].text == "大的给你。小的给我。"
    assert history(memory)[0].metadata["delivery_status"] == "sent"


def test_stop_during_model_generation_suppresses_reply_and_memory(memory):
    state = QQCommandState()
    sent = []
    def generate(**kwargs):
        state.set_stopped(True)
        return {"speech": "大的给你。小的给我。"}
    subject = bridge(lambda m: sent.append(m) or DeliveryResult(True, "sent"), memory, command_state=state)
    subject.runtime = SimpleNamespace(call_chat_json=generate)
    result = subject.handle_turn(turn())
    assert not result.sent and result.delivery_status == "suppressed" and not sent
    assert not history(memory)
    assert subject.chat_flow.begin_turn(turn("为什么这么选", "followup")).previous_exchange == ""


def test_stop_after_first_bubble_keeps_only_first_delivery(memory):
    state = QQCommandState()
    sent = []
    def sender(message):
        sent.append(message.segments[0].text)
        state.set_stopped(True)
        return DeliveryResult(True, "sent", ("out-1",))
    subject = bridge(sender, memory, command_state=state)
    result = subject._deliver_text(turn(), "第一句。第二句。第三句。")
    assert sent == ["第一句。"] and result.delivery_status == "partial"
    assert history(memory)[0].text == "第一句。"
    assert "第二句" not in subject.chat_flow.begin_turn(turn("然后呢", "followup")).previous_exchange


def test_stop_during_segment_delay_is_rechecked_before_send(memory, monkeypatch):
    state = QQCommandState()
    sent = []
    monkeypatch.setattr("shinku.qq.agent_bridge.time.sleep", lambda _: state.set_stopped(True))
    subject = bridge(lambda m: sent.append(m.segments[0].text) or DeliveryResult(True, "sent"),
                     memory, command_state=state, reply_segment_delay_seconds=1)
    result = subject._deliver_text(turn(), "第一句。第二句。")
    assert sent == ["第一句。"] and result.delivery_status == "partial"


def test_stop_during_voice_render_preserves_prior_text_and_prevents_audio(memory):
    state = QQCommandState(reply_mode="voice")
    sent = []
    def renderer(text):
        state.set_stopped(True)
        return "C:/fixture.wav"
    subject = bridge(lambda m: sent.append(m) or DeliveryResult(True, "sent"), memory,
                     command_state=state, voice_renderer=renderer)
    result = subject._deliver_text(turn(), "第一句。第二句。")
    assert [message.segments[0].kind for message in sent] == ["text", "text"]
    assert not result.sent and result.delivery_status == "partial"
    assert len(history(memory)) == 1 and history(memory)[0].text == "第一句。第二句。"


def test_stop_after_text_in_both_mode_prevents_voice(memory):
    state = QQCommandState(reply_mode="both")
    sent = []
    def sender(message):
        sent.append(message.segments[0].kind)
        state.set_stopped(True)
        return DeliveryResult(True, "sent")
    subject = bridge(sender, memory, command_state=state, voice_renderer=lambda _: "C:/fixture.wav")
    result = subject._deliver_text(turn(), "第一句。")
    assert sent == ["text"]
    assert result.delivery_status == "partial"
    assert history(memory)[0].text == "第一句。"


def test_stop_command_confirmation_remains_sendable_without_memory(memory):
    state = QQCommandState(stopped=True)
    sent = []
    subject = bridge(lambda m: sent.append(m) or DeliveryResult(True, "sent"), memory, command_state=state)
    result = subject.handle_turn(turn("/stop"))
    assert result.agent_status == "command" and result.sent and len(sent) == 1
    assert not history(memory)


def test_dry_run_supports_simulated_continuity_without_persistent_delivery(memory):
    subject = bridge(None, memory)
    result = subject._deliver_text(turn(), "大的给你。小的给我。")
    assert result.delivery_status == "dry_run" and not result.sent
    assert not history(memory) and not history(memory, "private:master")
    assert "大的给你" in subject.chat_flow.begin_turn(turn("为什么这么选", "followup")).previous_exchange


def test_voice_delivery_records_complete_original_once(memory):
    subject = bridge(lambda m: DeliveryResult(True, "sent", ("audio-1",)), memory,
                     command_state=QQCommandState(reply_mode="voice"), voice_renderer=lambda _: "C:/fixture.wav")
    result = subject._deliver_text(turn(), "大的给你。小的给我。")
    assert result.sent and len(history(memory)) == 1
    assert history(memory)[0].text == "大的给你。小的给我。"
    assert history(memory)[0].metadata["delivery_content_complete"] is True


def test_voice_mode_stop_during_text_prevents_remaining_text_and_audio(memory):
    state = QQCommandState(reply_mode="voice")
    attempted = []
    def sender(message):
        attempted.append((message.segments[0].kind, message.segments[0].text))
        if message.segments[0].kind == "audio":
            return DeliveryResult(False, "rejected", definitive_not_sent=True)
        state.set_stopped(True)
        return DeliveryResult(True, "sent")
    subject = bridge(sender, memory, command_state=state, voice_renderer=lambda _: "C:/fixture.wav")
    result = subject._deliver_text(turn(), "第一句。第二句。")
    assert attempted == [("text", "第一句。")]
    assert result.delivery_status == "partial" and history(memory)[0].text == "第一句。"


@pytest.mark.parametrize("partial", [False, True])
def test_group_receipt_attaches_to_actual_source_without_unsent_content(memory, partial):
    sent = []
    def sender(message):
        sent.append(message.segments[0].text)
        ok = not partial or len(sent) != 2
        return DeliveryResult(ok, "sent" if ok else "rejected", (f"out-{len(sent)}",) if ok else ())
    subject = bridge(sender, memory, bot_ids=("bot",))
    result = subject._deliver_text(turn(group=True), "第一句。第二句。")
    records = history(memory, "group:g-1")
    assert len(records) == 1
    assert records[0].text == ("第一句。" if partial else "第一句。第二句。")
    assert records[0].metadata["delivery_status"] == ("partial" if partial else "sent")
    assert records[0].metadata["delivery_proof"] == "qq_transport"
    assert records[0].metadata["delivery_content_complete"] is (not partial)
    assert result.delivery_status == ("partial" if partial else "sent")


def test_memory_error_after_send_does_not_reclassify_success():
    memory = SimpleNamespace(record_turn=lambda **kw: (_ for _ in ()).throw(RuntimeError("fixture")))
    result = bridge(lambda _: DeliveryResult(True, "sent"), memory)._deliver_text(turn(), "收到。")
    assert result.sent


def test_single_message_adapter_uses_turn_identity_without_an_unsent_assistant(memory):
    message = SimpleNamespace(text="请记住：我喜欢安静的环境", has_media=False)
    incoming = QQTurn(conversation_id="g-1", conversation_type="group", primary_message_id="original-user",
                      message_ids=("original-user",), combined_text=message.text, messages=(message,))
    subject = bridge(None, memory)
    result = subject.handle_turn(incoming)
    records = memory.store.search(scope="group:g-1", query="安静的环境", now=int(memory.clock()), limit=8)
    assert result.delivery_status == "dry_run"
    assert any(record.source_id == "turn:g-1:original-user" for record in records)
    assert not history(memory, "group:g-1")
