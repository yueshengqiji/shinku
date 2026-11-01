"""LA01: bounded ambient admission with explicit ordering and cleanup."""
import threading

from shinku.qq.turns import NapCatTurnDispatcher
from test_qq_natural_attention import batch


def test_explicit_calls_keep_order_ahead_of_queued_passive_work_and_failures_cleanup():
    started, released = threading.Event(), threading.Event()
    handled = []
    def handle(turn):
        started.set()
        assert released.wait(3)
        handled.append(turn.primary_message_id)
        if turn.primary_message_id == "1":
            raise ValueError("controlled failure")
    dispatcher = NapCatTurnDispatcher(handler=handle, flush_batch=lambda: None, async_handler=True, max_workers=1)
    try:
        with dispatcher._lock:
            assert dispatcher._dispatch_locked(batch(1)) == 1
        assert started.wait(1)
        with dispatcher._lock:
            assert dispatcher._dispatch_locked(batch(2)) == 1
            assert dispatcher._dispatch_locked(batch(3)) == 1
            assert dispatcher._dispatch_locked(batch(4, direct=True)) == 1
            assert dispatcher._dispatch_locked(batch(5, direct=True)) == 1
            assert dispatcher._dispatch_locked(batch(6)) == 0
            assert dispatcher._pending_turn_counts[("group", "g")] == 4
    finally:
        released.set()
        dispatcher.close()
    assert handled == ["1", "4", "5", "3"]
    assert not dispatcher._pending_turn_counts and not dispatcher._running_conversations
    assert not dispatcher._conversation_queues and not dispatcher._conversation_futures


def test_cancelling_not_started_conversation_releases_all_counts():
    started, released = threading.Event(), threading.Event()
    handled = []
    def handle(turn):
        started.set()
        assert released.wait(3)
        handled.append(turn.conversation_id)
    dispatcher = NapCatTurnDispatcher(handler=handle, flush_batch=lambda: None, async_handler=True, max_workers=1)
    try:
        with dispatcher._lock:
            dispatcher._dispatch_locked(batch(1, group="running"))
        assert started.wait(1)
        with dispatcher._lock:
            dispatcher._dispatch_locked(batch(2, group="cancelled"))
            dispatcher._dispatch_locked(batch(3, group="cancelled"))
            assert dispatcher._conversation_futures[("group", "cancelled")].cancel()
            assert ("group", "cancelled") not in dispatcher._pending_turn_counts
            assert ("group", "cancelled") not in dispatcher._conversation_queues
            assert ("group", "cancelled") not in dispatcher._running_conversations
            assert dispatcher._dispatch_locked(batch(4, group="cancelled")) == 1
    finally:
        released.set()
        dispatcher.close()
    assert handled == ["running", "cancelled"]
    assert not dispatcher._pending_turn_counts


def test_executor_closed_admission_does_not_leak_queue_count():
    dispatcher = NapCatTurnDispatcher(handler=lambda _: None, flush_batch=lambda: None, async_handler=True)
    dispatcher.close()
    with dispatcher._lock:
        assert dispatcher._dispatch_locked(batch(1)) == 0
    assert dispatcher.last_error == "turn_executor_closed"
    assert not dispatcher._pending_turn_counts and not dispatcher._conversation_queues
    assert not dispatcher._running_conversations
