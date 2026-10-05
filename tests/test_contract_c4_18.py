from __future__ import annotations

import unittest
import threading

from shinku.qq.host import NapCatHost
from shinku.qq.turns import NapCatTurnDispatcher


class _FakeTimer:
    instances: list["_FakeTimer"] = []

    def __init__(self, delay: float, callback):
        self.delay = delay
        self.callback = callback
        self.cancelled = False
        self.started = False
        self.instances.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


class NapCatTurnDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeTimer.instances.clear()

    @staticmethod
    def _event(message_id: str, text: str, *, image: bool = False) -> dict:
        message = [{"type": "text", "data": {"text": text}}]
        if image:
            message.append({"type": "image", "data": {"url": "data:image/png;base64,aGVsbG8="}})
        return {
            "post_type": "message",
            "message_type": "private",
            "user_id": "u-18",
            "message_id": message_id,
            "message": message,
        }

    def test_host_dispatches_closed_batch_and_keeps_images_by_message(self) -> None:
        turns = []
        dispatcher = NapCatTurnDispatcher(
            handler=turns.append,
            flush_batch=lambda: host.ingress.flush(),
            timer_factory=_FakeTimer,
        )
        host = NapCatHost(turn_dispatcher=dispatcher)

        host.handle_event(self._event("m-1", "第一句", image=True), at=1.0)
        host.handle_event(self._event("m-2", "第二句"), at=1.4)
        receipt = dispatcher.flush()

        self.assertEqual(receipt.dispatched, 1)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].combined_text, "第一句\n第二句")
        self.assertEqual(turns[0].visuals_for("m-1")[0].reference, "data:image/png;base64,aGVsbG8=")
        self.assertEqual(turns[0].visuals_for("m-2"), ())
        self.assertTrue(_FakeTimer.instances[-1].started)

    def test_handler_failure_does_not_make_submit_fail(self) -> None:
        dispatcher = NapCatTurnDispatcher(
            handler=lambda turn: (_ for _ in ()).throw(RuntimeError("boom")),
            flush_batch=lambda: host.ingress.flush(),
            timer_factory=_FakeTimer,
        )
        host = NapCatHost(turn_dispatcher=dispatcher)
        host.handle_event(self._event("m-3", "测试"), at=1.0)
        receipt = dispatcher.flush()
        self.assertTrue(receipt.accepted)
        self.assertEqual(receipt.dispatched, 1)
        self.assertEqual(receipt.error, "handler_error:RuntimeError")

    def test_recent_visual_is_inherited_by_explicit_followup_only(self) -> None:
        turns = []
        now = [0.0]
        dispatcher = NapCatTurnDispatcher(
            handler=turns.append,
            flush_batch=lambda: host.ingress.flush(),
            timer_factory=_FakeTimer,
            clock=lambda: now[0],
            visual_followup_turns=3,
        )
        host = NapCatHost(turn_dispatcher=dispatcher)

        host.handle_event(self._event("m-img", "", image=True), at=1.0)
        dispatcher.flush()
        now[0] = 1.5
        host.handle_event(self._event("m-q1", "你看这是谁"), at=2.0)
        dispatcher.flush()

        self.assertTrue(turns[1].visual_context_inherited)
        self.assertEqual(turns[1].visual_context_source_message_id, "m-img")
        self.assertEqual(turns[1].visuals_for("m-q1"), ())
        self.assertEqual(turns[1].visuals_for("m-img")[0].reference, "data:image/png;base64,aGVsbG8=")

        now[0] = 2.0
        host.handle_event(self._event("m-unrelated", "天气怎么样"), at=3.0)
        dispatcher.flush()
        self.assertFalse(turns[2].visual_context_inherited)
        self.assertEqual(turns[2].visuals_for("m-unrelated"), ())

    def test_slow_handler_does_not_block_ingress_and_preserves_conversation_order(self) -> None:
        started = threading.Event()
        release = threading.Event()
        second_done = threading.Event()
        handled = []

        def handle(turn):
            if turn.primary_message_id == "m-1":
                started.set()
                self.assertTrue(release.wait(3))
            handled.append(turn.primary_message_id)
            if turn.primary_message_id == "m-2":
                second_done.set()

        dispatcher = NapCatTurnDispatcher(
            handler=handle,
            flush_batch=lambda: host.ingress.flush(),
            timer_factory=_FakeTimer,
            async_handler=True,
            max_workers=2,
        )
        host = NapCatHost(turn_dispatcher=dispatcher)
        try:
            host.handle_event(self._event("m-1", "第一句"), at=1.0)
            dispatcher.flush()
            self.assertTrue(started.wait(2))
            accepted = threading.Event()

            def receive_next():
                host.handle_event(self._event("m-2", "第二句"), at=3.0)
                dispatcher.flush()
                accepted.set()

            receiver = threading.Thread(target=receive_next, daemon=True)
            receiver.start()
            self.assertTrue(accepted.wait(1), "QQ webhook must not wait for the model")
            self.assertFalse(second_done.is_set(), "same-conversation turns must stay ordered")
            release.set()
            self.assertTrue(second_done.wait(2))
            self.assertEqual(handled, ["m-1", "m-2"])
        finally:
            release.set()
            dispatcher.close()


if __name__ == "__main__":
    unittest.main()
