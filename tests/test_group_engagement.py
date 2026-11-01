"""群聊持续参与与快聊回归；模型结论是脚本，送达使用真实 OneBot 适配器的本地回执。"""
import json
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from shinku.qq.attention import AttentionBatch
from shinku.qq.delivery import DeliveryResult
from shinku.qq.turns import NapCatTurnDispatcher
from test_group_participation import ALLOW, final, host, turn
from test_turn_state_attention import adapter, event


def continuation(options):
    data = json.loads(options["user_prompt"])
    return {**ALLOW, "input_id": data.get("input_id", ""),
            "continuation_receipt_id": data["recent_delivered_group_replies"][-1]["receipt_id"]}


def connect(bridge, ingress):
    bridge.group_delivery_observer = lambda group, recipients, mids, autonomous, at: ingress.note_group_reply(
        group, recipient_ids=recipients, message_ids=mids, autonomous=autonomous, at=at)
    bridge.group_continuation_observer = getattr(ingress, "note_group_continuation", lambda *_: None)


def seed(host, ingress):
    bridge, runtime, _ = host
    connect(bridge, ingress)
    runtime.script.extend([final("我也觉得这段旋律挺有意思。"), ALLOW])
    with patch("shinku.qq.agent_bridge.time.time", return_value=100):
        assert bridge(turn("这段旋律挺好听。", mid="seed", mode="autonomous")).sent


def process(host, ingress, mid, *, at, review=continuation):
    bridge, runtime, _ = host
    incoming = ingress.ingest(event(mid, "alice", "后面那一段也很好听。"), at=at)
    assert incoming.scheduled
    dispatcher = NapCatTurnDispatcher(handler=bridge, flush_batch=lambda: None)
    item = dispatcher._build_turn_locked(AttentionBatch("g", (incoming.message,), (incoming.attention,), 0), now=at)
    runtime.script.extend([final("尤其是后面节奏变化的地方。"), review])
    with patch("shinku.qq.agent_bridge.time.time", return_value=at + .5):
        result = bridge(item)
    return incoming, result


def test_natural_exchange_survives_four_batches_without_direct_address(host):
    ingress = adapter()
    seed(host, ingress)
    reasons = []
    for index, stamp in enumerate((110, 120, 130, 140)):
        incoming, result = process(host, ingress, str(index), at=stamp)
        assert result.sent
        reasons.append(incoming.attention.reason)
        assert not incoming.message.raw_event["_shinku_group_context"]["current"]["bot_exchange_confirmed"]
    assert reasons == ["possible_bot_followup", *["participating_followup"] * 3]
    bridge, runtime, sends = host
    assert len(sends) == 5 and len(runtime.calls) == 10 and not runtime.script
    assert not bridge.task_store.candidates(requester="alice", session="group:g")
    assert ingress.autonomous_tracker._groups["g"]["unanswered"] == 0


@pytest.mark.parametrize("bad", ["missing", "wrong_id", "wrong_input", "foreign_sender", "expired", "changed", "partial_parent"])
def test_unbound_review_cannot_create_participation(host, bad):
    ingress = adapter()
    seed(host, ingress)
    bridge, runtime, _ = host
    if bad == "foreign_sender":
        bridge._group_delivered_replies["g"][-1] = bridge._group_delivered_replies["g"][-1]._replace(recipient_ids=("bob",))
    if bad == "partial_parent":
        bridge._group_delivered_replies["g"][-1] = bridge._group_delivered_replies["g"][-1]._replace(content_complete=False)

    def review(options):
        response = continuation(options)
        if bad == "missing":
            response.pop("continuation_receipt_id")
        elif bad == "wrong_id":
            response["continuation_receipt_id"] = "foreign-receipt"
        elif bad == "wrong_input":
            response["input_id"] = "foreign-input"
        elif bad == "expired":
            bridge._group_delivered_replies["g"][-1] = bridge._group_delivered_replies["g"][-1]._replace(at=-1000)
        elif bad == "changed":
            bridge._group_delivered_replies["g"][-1] = bridge._group_delivered_replies["g"][-1]._replace(text="changed")
        return response
    _, result = process(host, ingress, "follow", at=110, review=review)
    assert result.sent and not runtime.script
    assert "participants" not in ingress.autonomous_tracker._groups["g"]
    assert ingress.ingest(event("extra", "alice", "再补充一句"), at=115).attention.reason == "possible_bot_followup"


@pytest.mark.parametrize("send_kind", ["failed", "dry_run", "partial"])
def test_only_complete_delivery_renews_participation(host, send_kind):
    ingress = adapter()
    seed(host, ingress)
    bridge, runtime, _ = host
    if send_kind == "dry_run":
        bridge.sender = None
    elif send_kind == "failed":
        bridge.sender = lambda _: DeliveryResult(False, "failed")
    else:
        count = 0
        def sender(_):
            nonlocal count
            count += 1
            return DeliveryResult(count == 1, "sent" if count == 1 else "failed", ("partial",) if count == 1 else ())
        bridge.sender = sender
    runtime.script.extend([final("后面那段也不错。\n尤其是节奏变化的地方。"), continuation])
    with patch("shinku.qq.agent_bridge.time.time", return_value=110):
        result = bridge(turn("后面那段呢？", mid="follow", mode="autonomous"))
    assert not result.sent and not runtime.script
    assert "participants" not in ingress.autonomous_tracker._groups["g"]


def test_participation_keeps_scope_addressing_expiry_and_budget(host):
    ingress = adapter()
    seed(host, ingress)
    process(host, ingress, "first", at=110)
    other = ingress.ingest(event("other-person", "bob", "我来谈另一件事"), at=114)
    assert other.attention.reason == "group_open_turn"
    assert not other.message.raw_event["_shinku_group_context"]["current"]["bot_exchange_confirmed"]
    assert not ingress.autonomous_tracker.signals(other.message, now=114).participating_followup
    assert ingress.ingest(event("elsewhere", "alice", "还是那个话题", group="other"), at=115).attention.reason == "group_open_turn"
    for index in range(6):
        assert ingress.ingest(event(f"fast{index}", "alice", "还有一点补充"), at=120+index*3).scheduled
    assert ingress.ingest(event("budget", "alice", "继续"), at=145).attention.reason == "autonomous_admission_budget"
    assert ingress.ingest(event("direct", "alice", "你看呢", at="bot"), at=146).attention.immediate
    # 无新送达不续期；不靠单纯入站或候选把参与状态永久延长。
    assert not ingress.autonomous_tracker.signals(ingress.ingest(event("late", "alice", "过了一会儿"), at=240).message, now=240).participating_followup


class NoTimer:
    def __init__(self, *args):
        pass
    def start(self):
        pass
    def cancel(self):
        pass


def run_with_arrivals(host, *, during="generation", allow=True, explicit=False, echo=True, unreadable=""):
    bridge, runtime, sends = host
    ingress = adapter()
    receipts = []
    dispatcher = NapCatTurnDispatcher(handler=lambda item: receipts.append(bridge(item)), flush_batch=lambda: None,
        timer_factory=NoTimer, async_handler=True, max_workers=1)
    original = ingress.ingest(event("original", "alice", "这段旋律不错。"), at=100)
    batch = AttentionBatch("g", (original.message,), (original.attention,), 0)

    def arrive(mid, text):
        raw = event(mid, "alice", text, at="bot" if explicit else "")
        if unreadable == "long":
            raw["message"][-1]["data"]["text"] = "合成补充" * 501 + "最后更正前面的说法。"
        elif unreadable == "image":
            raw["message"].append({"type": "image", "data": {"file": "synthetic.png"}})
        item = ingress.ingest(raw, at=101)
        dispatcher.submit(SimpleNamespace(ingress=item, scheduled=item.scheduled, visual=None))

    def generate(_):
        if during == "generation":
            arrive("later", "更正，刚才说错了。" if not allow else "后面还有一段。")
        return final("中间的节奏变化也不错。")

    def review(options):
        data = json.loads(options["user_prompt"])
        if during == "generation":
            assert [r["message_id"] for r in data["later_messages"]] == ["later"]
        if during == "review":
            arrive("during-review", "等等，刚才那句话不对。")
        return {**ALLOW, "verdict": "allow" if allow else "silence",
                "input_id": data["input_id"] if echo else "old-input"}

    runtime.script.extend([generate] + ([] if explicit or unreadable else [review]))
    if during == "delivery":
        original_ground = bridge._ground_delivery_text
        def ground(*args):
            arrive("during-delivery", "刚才的说法有误，先别发。")
            return original_ground(*args)
        bridge._ground_delivery_text = ground
    try:
        with dispatcher._lock:
            assert dispatcher._dispatch_locked(batch) == 1
            future = dispatcher._conversation_futures[("group", "g")]
        future.result(timeout=5)
    finally:
        dispatcher.close()
    assert not runtime.script
    return receipts[0], runtime, sends


def test_running_reply_survives_related_new_message_with_existing_review(host):
    result, runtime, sends = run_with_arrivals(host)
    assert result.sent and len(sends) == 1 and len(runtime.calls) == 2


@pytest.mark.parametrize("case", ["correction", "review_race", "delivery_race", "wrong_binding", "explicit"])
def test_invalid_or_unreviewed_updates_never_publish(host, case):
    result, runtime, sends = run_with_arrivals(host,
        during={"review_race": "review", "delivery_race": "delivery"}.get(case, "generation"),
        allow=case != "correction", explicit=case == "explicit", echo=case != "wrong_binding")
    assert not result.sent and not sends
    assert len(runtime.calls) == (1 if case == "explicit" else 2)


@pytest.mark.parametrize("unreadable", ["long", "image"])
def test_unread_late_attachment_or_truncated_correction_cannot_be_acknowledged(host, unreadable):
    result, runtime, sends = run_with_arrivals(host, unreadable=unreadable)
    assert not result.sent and not sends and len(runtime.calls) == 1
    assert result.reason == "autonomous_updates_unreadable"


def test_queued_turns_still_coalesce_and_running_window_is_bounded():
    ingress = adapter()
    clock = [100.0]
    release, started = threading.Event(), threading.Event()
    turns = []
    queue_reasons = []
    def handler(item):
        turns.append(item)
        if len(turns) == 1:
            assert item.suppression_reason("queue") == ""
            started.set()
            assert release.wait(3)
        else:
            # 处理器入口才是排队有效性的检查点；完成回调会清理 revision，不能在清理后冒充排队检查。
            queue_reasons.append(item.suppression_reason("queue"))
    dispatcher = NapCatTurnDispatcher(handler=handler, flush_batch=lambda: None, async_handler=True,
        max_workers=1, clock=lambda: clock[0])
    def batch(mid):
        item = ingress.ingest(event(mid, "alice", "关于音乐的话题"), at=clock[0])
        return AttentionBatch("g", (item.message,), (item.attention,), 0)
    try:
        with dispatcher._lock:
            dispatcher._dispatch_locked(batch("first"))
        assert started.wait(1)
        with dispatcher._lock:
            dispatcher._dispatch_locked(batch("second"))
        assert turns[0].suppression_reason("reply") == ""
        clock[0] = 221
        assert turns[0].suppression_reason("reply") == "autonomous_reply_expired"
    finally:
        release.set()
        dispatcher.close()
    assert queue_reasons == ["autonomous_queue_expired"]


@pytest.mark.parametrize("mutation", ["group", "input_id", "text", "sender", "receipt", "revoke", "route", "recipient"])
def test_continuation_is_bound_to_original_inputs_receipt_and_delivery_recipient(mutation):
    from shinku.agent.loop import AgentDecision
    from shinku.qq.group_participation import bind_continuation, continuation_recipient
    original = turn("后面还有一段。", mode="autonomous")
    row = {"receipt_id": "receipt", "recipient_ids": ["alice"], "text": "刚才那段旋律不错。", "content_complete": True}
    context = {"qq_turn": original}
    bind_continuation(original, AgentDecision.final("我听到了。"), context,
                      {"continuation_receipt_id": "receipt"}, [row])
    bound = context["qq_turn"]
    assert continuation_recipient(bound, [row], ("alice",)) == "alice"
    recipients, rows = ("alice",), [row]
    if mutation == "group":
        bound = replace(bound, conversation_id="other")
    elif mutation == "input_id":
        bound = replace(bound, message_ids=("other",))
    elif mutation == "text":
        changed = replace(bound.messages[0], segments=tuple(replace(segment, text="changed")
            if segment.kind == "text" else segment for segment in bound.messages[0].segments))
        bound = replace(bound, messages=(changed,))
    elif mutation == "sender":
        bound = replace(bound, messages=(replace(bound.messages[0], sender_id="changed"),))
    elif mutation == "receipt":
        rows = [{**row, "text": "changed"}]
    elif mutation == "revoke":
        rows = []
    elif mutation == "route":
        bound = replace(bound, response_message_id="other")
    else:
        recipients = ("bob",)
    assert continuation_recipient(bound, rows, recipients) == ""


def test_late_message_window_does_not_retain_raw_payload_or_unbounded_text():
    ingress = adapter()
    dispatcher = NapCatTurnDispatcher(handler=lambda _: None, flush_batch=lambda: None,
        timer_factory=NoTimer, clock=lambda: 0)
    original = dispatcher._bind_ambient_window(turn("原始话题", mode="autonomous"), ("group", "g"), 0, 0)
    raw = event("later", "alice", "合成正文" * 2000)
    raw["private_test_payload"] = "synthetic" * 10000
    incoming = ingress.ingest(raw, at=100)
    dispatcher.submit(SimpleNamespace(ingress=incoming, scheduled=incoming.scheduled, visual=None))
    try:
        snapshot = dispatcher._ambient_arrivals[("group", "g")][0][1]
        assert len(snapshot.text) == 2001 and not hasattr(snapshot, "raw_event")
        assert not hasattr(snapshot, "segments")
        assert original.suppression_reason("reply") == "autonomous_updates_unreadable"
    finally:
        dispatcher.close()


@pytest.mark.parametrize("chat_draft", [False, True])
def test_social_publication_guard_does_not_cancel_independent_result_delivery(host, chat_draft):
    bridge, runtime, sends = host
    ingress = adapter()
    dispatcher = NapCatTurnDispatcher(handler=lambda _: None, flush_batch=lambda: None,
        timer_factory=NoTimer, clock=lambda: 0)
    original = dispatcher._bind_ambient_window(turn("原始话题", mode="autonomous"), ("group", "g"), 0, 0)
    incoming = ingress.ingest(event("later", "bob", "另一个话题的补充。"), at=100)
    dispatcher.submit(SimpleNamespace(ingress=incoming, scheduled=incoming.scheduled, visual=None))
    try:
        publication_turn = replace(original, ambient_reply_reviewed=chat_draft)
        result = bridge._deliver_text(publication_turn, "这是隔离测试的合成正文。",
            reason="direct_chat_reply" if chat_draft else "agent_result", autonomous_authorized=True)
        assert result.sent is not chat_draft
        assert len(sends) == (0 if chat_draft else 1) and not runtime.calls
        if chat_draft:
            assert result.reason == "autonomous_updates_unreviewed"
    finally:
        dispatcher.close()
