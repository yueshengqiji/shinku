import pytest

from shinku.qq.attention import AttentionPolicy
from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.routing import ReplyRoutePolicy
from test_turn_state_attention import adapter, event
import threading
from dataclasses import replace
from shinku.qq.attention import AttentionBatch, AttentionDecision
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import NapCatTurnDispatcher


@pytest.mark.parametrize("text", ["真红", "真红，你在吗", "真红你在吗", "真红你怎么看", "真红帮我看看", "真红在吗"])
def test_natural_calls_take_explicit_priority(text):
    a = QQIngressAdapter(route_policy=ReplyRoutePolicy(bot_names=("真红",)),
                         attention_policy=AttentionPolicy(autonomous_enabled=False, mode="turn_state"))
    result = a.ingest(event("m", "guest", text), at=100.0)
    assert result.scheduled and result.attention.immediate


@pytest.mark.parametrize("text", ["真红的衣服很好看", "我喜欢真红", "真红色的外套", "真红这个角色很好看"])
def test_talking_about_name_is_not_direct_call(text):
    result = QQIngressAdapter(route_policy=ReplyRoutePolicy(bot_names=("真红",))).ingest(event("m", "guest", text), at=100.0)
    assert not result.route.should_reply


def test_busy_group_candidate_and_explicit_call_bypass_admission_budget():
    a = adapter()
    for n in range(7):
        result = a.ingest(event(str(n), "guest" + str(n), "今天真开心"), at=100 + n * .2)
    assert result.scheduled
    assert "busy_group" in result.attention.signals
    for stamp in (110, 120):
        assert a.ingest(event(str(stamp), "guest", "分享一下"), at=stamp).scheduled
    blocked = a.ingest(event("limited", "guest", "还有一件事"), at=130)
    assert blocked.attention.reason == "autonomous_admission_budget"
    explicit = a.ingest(event("direct", "guest", "你在吗", at="bot"), at=131)
    assert explicit.scheduled and explicit.attention.immediate


def test_unbounded_stream_has_finite_autonomous_admissions_and_resets():
    a = adapter()
    results = [a.ingest(event(str(n), "guest", "今天真开心"), at=100 + n * .01) for n in range(1000)]
    assert sum(r.scheduled for r in results) <= 24
    assert a.ingest(event("later", "guest", "好久不见"), at=200).scheduled


def batch(number, *, direct=False, group="g"):
    message = IncomingMessage.from_event(event(str(number), "guest", "分享一下"))
    message = replace(message, conversation_id=group)
    return AttentionBatch(group, (message,), (AttentionDecision(True, direct, 100 if direct else 20,
        "mentioned_bot" if direct else "group_open_turn", signals=("busy_group",)),), 0)


def test_slow_worker_caps_passive_queue_but_keeps_explicit_call():
    released, started = threading.Event(), threading.Event()
    handled = []
    running = []
    def handler(turn):
        running.append(turn)
        started.set()
        assert released.wait(3)
        handled.append(turn)
    dispatcher = NapCatTurnDispatcher(handler=handler, flush_batch=lambda: None, async_handler=True, max_workers=1)
    try:
        with dispatcher._lock:
            assert dispatcher._dispatch_locked(batch(1)) == 1
        assert started.wait(1)
        with dispatcher._lock:
            assert dispatcher._dispatch_locked(batch(2)) == 1
            assert dispatcher._dispatch_locked(batch(3)) == 1
            assert dispatcher.last_error == "autonomous_queued_turn_coalesced"
            assert dispatcher._pending_turn_counts[("group", "g")] == 2
            assert dispatcher._dispatch_locked(batch(4, direct=True)) == 1
            assert running[0].suppression_reason("reply") == "autonomous_turn_superseded"
            assert dispatcher._pending_turn_counts[("group", "g")] == 3
    finally:
        released.set()
        dispatcher.close()
    # The explicit call precedes queued ambient work; the running turn is
    # allowed to finish. LB01 previously asserted the obsolete FIFO contract.
    assert [t.primary_message_id for t in handled] == ["1", "4", "3"]
    assert handled[1].attention_mode == "immediate" and "busy_group" in handled[1].attention_signals
    assert not dispatcher._pending_turn_counts


def test_many_slow_groups_cannot_create_unbounded_passive_executor_queue():
    released = threading.Event()
    handled = []
    def handler(turn):
        assert released.wait(3)
        handled.append(turn)
    dispatcher = NapCatTurnDispatcher(handler=handler, flush_batch=lambda: None, async_handler=True, max_workers=1)
    try:
        with dispatcher._lock:
            assert all(dispatcher._dispatch_locked(batch(n, group="g" + str(n))) == 1 for n in range(24))
            assert dispatcher._dispatch_locked(batch(24, group="extra")) == 0
            assert sum(dispatcher._pending_turn_counts.values()) == 24
            assert dispatcher._dispatch_locked(batch(25, direct=True, group="explicit")) == 1
    finally:
        released.set()
        dispatcher.close()
    assert len(handled) == 25 and not dispatcher._pending_turn_counts
