from __future__ import annotations

import unittest

from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.attention import AttentionBatcher, AttentionPolicy
from shinku.qq.routing import ReplyRoutePolicy


def _event(message_id: str, segments, *, conversation_id: str = "g-1"):
    return {
        "message_id": message_id,
        "conversation_id": conversation_id,
        "conversation_type": "group",
        "segments": segments,
    }


class QQIngressAdapterTests(unittest.TestCase):
    def _adapter(self, *, autonomous=False):
        return QQIngressAdapter(
            route_policy=ReplyRoutePolicy(bot_ids=frozenset({"bot-1"})),
            attention_policy=AttentionPolicy(
                autonomous_enabled=autonomous,
                autonomous_probability=1.0 if autonomous else 0.0,
                random_source=lambda: 0.0,
            ),
            batcher=AttentionBatcher(window_seconds=2.0),
        )

    def test_unaddressed_group_event_is_filtered_before_agent(self) -> None:
        result = self._adapter().ingest(
            _event("m-1", [{"type": "text", "data": {"text": "你们继续"}}]),
            at=1.0,
        )
        self.assertFalse(result.scheduled)
        self.assertIsNone(result.closed_batch)

    def test_at_event_is_scheduled_and_keeps_image_segment(self) -> None:
        result = self._adapter().ingest(
            _event(
                "m-2",
                [
                    {"type": "at", "data": {"qq": "bot-1"}},
                    {"type": "image", "data": {"file": "img-1", "url": "https://img.test/x.png"}},
                ],
            ),
            at=2.0,
        )
        self.assertTrue(result.scheduled)
        self.assertEqual(result.route.reason, "mentioned_bot")
        self.assertEqual(result.message.attachments[0]["media_id"], "img-1")

    def test_two_addressed_events_are_emitted_as_one_closed_batch(self) -> None:
        adapter = self._adapter()
        first = adapter.ingest(
            _event("m-1", [{"type": "text", "data": {"text": "真红，先看这个"}}]),
            at=1.0,
        )
        second = adapter.ingest(
            _event("m-2", [{"type": "text", "data": {"text": "真红，再看下一条"}}]),
            at=2.0,
        )
        self.assertIsNone(first.closed_batch)
        self.assertIsNone(second.closed_batch)
        batch = adapter.flush()
        assert batch is not None
        self.assertEqual(batch.as_dict()["message_ids"], ["m-1", "m-2"])

    def test_explicit_autonomous_mode_schedules_unaddressed_event(self) -> None:
        result = self._adapter(autonomous=True).ingest(
            _event("m-3", [{"type": "text", "data": {"text": "刚才想到一件事"}}]),
            at=3.0,
        )
        self.assertEqual((result.scheduled, result.attention.reason), (True, "autonomous_probability_hit"))

    def test_conversation_change_returns_previous_batch_without_mixing(self) -> None:
        adapter = self._adapter()
        adapter.ingest(
            _event("m-1", [{"type": "text", "data": {"text": "真红，群一"}}], conversation_id="g-1"),
            at=1.0,
        )
        result = adapter.ingest(
            _event("m-2", [{"type": "text", "data": {"text": "真红，群二"}}], conversation_id="g-2"),
            at=1.2,
        )
        assert result.closed_batch is not None
        self.assertEqual(result.closed_batch.conversation_id, "g-1")
        self.assertEqual(adapter.flush().conversation_id, "g-2")


if __name__ == "__main__":
    unittest.main()
