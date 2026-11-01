"""真实发送后的无引用接话机会：不冒充明确呼叫，也不绕过群聊审查。"""
import json
from unittest.mock import patch

import pytest

from shinku.qq.attention import AttentionBatch
from shinku.agent.reply_body import AUTONOMOUS_SILENCE_MARKER
from shinku.qq.reply_host import ReplyHost
from shinku.qq.turns import NapCatTurnDispatcher
from test_group_participation import ALLOW, final, host, turn
from test_turn_state_attention import adapter, event


def delivered(ingress, *, at=100, group="g", recipient="alice"):
    ingress.note_group_reply(group, recipient_ids=(recipient,), message_ids=("sent",),
                             autonomous=True, at=at)


def test_unquoted_recipient_gets_one_short_burst_without_confirmed_addressing():
    ingress = adapter()
    delivered(ingress)
    follow = ingress.ingest(event("follow", "alice", "那这个选项要怎么填？"), at=131)
    assert follow.scheduled and not follow.attention.immediate
    assert follow.attention.reason == "possible_bot_followup"
    assert not follow.message.raw_event["_shinku_group_context"]["current"]["bot_exchange_confirmed"]
    assert ingress.ingest(event("more", "alice", "我试过另外一个了"), at=132).scheduled
    # 候选本身不能续期、重置退让计数或让同一人持续绕过冷却。
    assert ingress.ingest(event("again", "alice", "再补一句"), at=134).attention.reason == "autonomous_cooldown"
    assert ingress.ingest(event("other", "bob", "大家聊另一件事"), at=135).attention.reason == "autonomous_cooldown"


def test_unquoted_window_expires_even_while_backoff_remains():
    ingress = adapter()
    delivered(ingress, at=100)
    delivered(ingress, at=200)
    assert ingress.ingest(event("late", "alice", "那然后呢？"), at=321).attention.reason == "autonomous_cooldown"
    assert ingress.ingest(event("elsewhere", "alice", "那然后呢？", group="g2"), at=322).attention.reason == "group_open_question"


def test_candidate_budget_still_applies_without_consuming_denied_opportunity():
    ingress = adapter()
    for stamp in (50, 53, 56):
        assert ingress.ingest(event(str(stamp), "alice", f"这是第 {stamp} 项的说明"), at=stamp).scheduled
    delivered(ingress)
    assert ingress.ingest(event("full", "alice", "那这个怎么做？"), at=101).attention.reason == "autonomous_admission_budget"
    # 最早的候选额度到期后仍可用接话机会，预算拒绝没有冒充模型已处理。
    assert ingress.ingest(event("available", "alice", "刚才那个选项呢？"), at=111).attention.reason == "possible_bot_followup"


@pytest.mark.parametrize("direction", ["at", "reply"])
def test_explicit_other_recipient_still_wins(direction):
    ingress = adapter()
    ingress.ingest(event("their-message", "bob", "我来处理"), at=90)
    delivered(ingress)
    result = ingress.ingest(event("other", "alice", "那就这样", **{
        direction: "bob" if direction == "at" else "their-message"}), at=101)
    assert not result.scheduled


def test_other_focus_and_operator_switch_are_not_overridden():
    from shinku.qq.commands import QQCommandState
    ingress = adapter()
    ingress.ingest(event("other", "bob", "你看一下", at="alice"), at=99)
    delivered(ingress)
    assert not ingress.ingest(event("follow", "alice", "可以"), at=101).scheduled
    ingress = adapter()
    delivered(ingress)
    state = QQCommandState(autonomous_reply_enabled=False)
    ingress.set_command_state(state)
    assert ingress.ingest(event("disabled", "alice", "然后呢？"), at=101).attention.reason == "autonomous_reply_disabled"


@pytest.mark.parametrize("sent", [False, True])
def test_actual_transport_receipt_controls_followup(host, sent):
    from shinku.qq.delivery import DeliveryResult
    bridge, runtime, sends = host
    ingress = adapter()
    bridge.group_delivery_observer = lambda group, recipients, mids, autonomous, at: ingress.note_group_reply(
        group, recipient_ids=recipients, message_ids=mids, autonomous=autonomous, at=at)
    if not sent:
        bridge.sender = lambda _: DeliveryResult(False, "failed")
    first = ingress.ingest(event("first", "alice", "选项太多了"), at=99)
    dispatcher = NapCatTurnDispatcher(handler=bridge, flush_batch=lambda: None, async_handler=False)
    try:
        batch = AttentionBatch("g", (first.message,), (first.attention,), 0)
        with dispatcher._lock:
            turn = dispatcher._build_turn_locked(batch, now=99)
        runtime.script.extend([final("先看你最关心的那一项。"), ALLOW])
        with patch("shinku.qq.agent_bridge.time.time", return_value=100):
            result = bridge(turn)
        assert result.sent is sent
        follow = ingress.ingest(event("follow", "alice", "那这个要怎么填？"), at=101)
        assert (follow.attention.reason == "possible_bot_followup") is sent
        if sent:
            with dispatcher._lock:
                next_turn = dispatcher._build_turn_locked(
                    AttentionBatch("g", (follow.message,), (follow.attention,), 0), now=101)
            runtime.script.extend([final("可以先保留默认值。"), ALLOW])
            with patch("shinku.qq.agent_bridge.time.time", return_value=102):
                result = bridge(next_turn)
            assert result.sent
            review = json.loads(runtime.calls[-1]["user_prompt"])
            assert not review["confirmed_bot_exchange"]
            assert review["recent_delivered_group_replies"][0]["text"] == "先看你最关心的那一项。"
    finally:
        dispatcher.close()


@pytest.mark.parametrize("silence_at", ["generation", "review"])
def test_unquoted_farewell_candidate_still_allows_silence(host, silence_at):
    bridge, runtime, sends = host
    ingress = adapter()
    bridge.group_delivery_observer = lambda group, recipients, mids, autonomous, at: ingress.note_group_reply(
        group, recipient_ids=recipients, message_ids=mids, autonomous=autonomous, at=at)
    runtime.script.extend([final("晚安。"), ALLOW])
    with patch("shinku.qq.agent_bridge.time.time", return_value=100):
        assert bridge(turn("大家晚安。", mid="first", mode="autonomous")).sent
    follow = ingress.ingest(event("thanks", "alice", "你也是。"), at=101)
    assert follow.attention.reason == "possible_bot_followup"
    dispatcher = NapCatTurnDispatcher(handler=bridge, flush_batch=lambda: None, async_handler=False)
    try:
        with dispatcher._lock:
            next_turn = dispatcher._build_turn_locked(
                AttentionBatch("g", (follow.message,), (follow.attention,), 0), now=101)
        runtime.script.extend([final(AUTONOMOUS_SILENCE_MARKER)] if silence_at == "generation" else [
            final("那就做个好梦。"), {**ALLOW, "verdict": "silence"}])
        before = len(sends)
        with patch("shinku.qq.agent_bridge.time.time", return_value=102):
            result = bridge(next_turn)
        assert result.agent_status == "silent" and not result.sent
        assert len(sends) == before and not runtime.script
        # 沉默没有新送达，不会重新获得接话机会或无限延长当前窗口。
        assert ingress.ingest(event("again", "alice", "明天见。"), at=105).attention.reason == "autonomous_cooldown"
    finally:
        dispatcher.close()


def test_long_pause_reaches_napcat_as_two_complete_bubbles(host):
    bridge, runtime, sends = host
    text = "原本只打算整理其中几个目录……现在连旁边的测试数据也一起搬过来了。"
    runtime.script.extend([final(text), ALLOW])
    result = bridge(turn("整理到哪里了？", mode="autonomous"))
    assert result.sent and len(sends) == 2 and not runtime.script
    bubbles = ["".join(segment["data"]["text"] for segment in params["message"]
                        if segment["type"] == "text") for _, params in sends]
    assert bubbles == ["原本只打算整理其中几个目录……", "现在连旁边的测试数据也一起搬过来了。"]
    assert "".join(bubbles) == text
    assert result.delivery.message_ids == ("out-1", "out-2")


def test_long_ellipsis_join_is_shaped_without_rewording():
    text = "原本只打算整理其中几个目录……现在连旁边的测试数据也一起搬过来了。"
    shaped = ReplyHost().shape_text(text)
    assert shaped == ["原本只打算整理其中几个目录……", "现在连旁边的测试数据也一起搬过来了。"]
    assert "".join(shaped) == text


@pytest.mark.parametrize("text", [
    "嗯……那就先按你说的办法试试，看看能不能解决。",
    "这句很短……先这样吧。",
    "这份说明里写着“原本只打算整理其中几个目录……现在连旁边的测试数据也一起搬过来了”",
    "运行 `print('原本只打算整理其中几个目录……现在连旁边的测试数据也一起搬过来了')`",
    "https://example.test/原本只打算整理其中几个目录……现在连旁边的测试数据也一起搬过来了",
    "```text\n原本只打算整理其中几个目录……现在连旁边的测试数据也一起搬过来了\n```",
])
def test_short_hesitation_and_quoted_or_atomic_material_stay_whole(text):
    assert ReplyHost().shape_text(text) == [text]
