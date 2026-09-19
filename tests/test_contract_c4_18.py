from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
