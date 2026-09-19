from __future__ import annotations

import unittest

from shinku.qq.attention import AttentionBatcher, AttentionPolicy
from shinku.qq.message import IncomingMessage
from shinku.qq.routing import ReplyRoutePolicy


def _message(message_id: str, text: str, *, conversation_id: str = "g-1", private: bool = False):
    return IncomingMessage.from_event(
        {
            "message_id": message_id,
            "conversation_id": conversation_id,
            "conversation_type": "private" if private else "group",
            "segments": [{"type": "text", "data": {"text": text}}],
        }
    )


class AttentionPolicyTests(unittest.TestCase):
    def test_addressed_message_is_immediate_without_probability_sampling(self) -> None:
        policy = AttentionPolicy(autonomous_enabled=False, random_source=lambda: 1.0)
        message = _message("m-1", "真红，回答一下")
        route = ReplyRoutePolicy().decide(message)
        decision = policy.evaluate(message, route)
        self.assertEqual((decision.should_schedule, decision.immediate, decision.reason), (True, True, "direct_name"))

    def test_passive_messages_do_not_call_model_when_disabled(self) -> None:
        policy = AttentionPolicy(autonomous_enabled=False)
        message = _message("m-2", "你们继续聊")
        decision = policy.evaluate(message, ReplyRoutePolicy().decide(message))
        self.assertEqual((decision.should_schedule, decision.reason), (False, "autonomous_reply_disabled"))

    def test_autonomous_probability_is_injectable_and_bounded(self) -> None:
        hit = AttentionPolicy(autonomous_enabled=True, autonomous_probability=0.5, random_source=lambda: 0.2)
        miss = AttentionPolicy(autonomous_enabled=True, autonomous_probability=0.5, random_source=lambda: 0.8)
        message = _message("m-3", "有人在吗")
        route = ReplyRoutePolicy().decide(message)
        self.assertEqual(hit.evaluate(message, route).reason, "autonomous_probability_hit")
        self.assertEqual(miss.evaluate(message, route).reason, "autonomous_probability_miss")


class AttentionBatcherTests(unittest.TestCase):
    def test_same_conversation_messages_merge_and_keep_addressed_message_primary(self) -> None:
        policy = AttentionPolicy()
        router = ReplyRoutePolicy()
        first = _message("m-1", "真红，先看这个")
        second = _message("m-2", "真红，再帮我查一下")
        batcher = AttentionBatcher(window_seconds=2.0)
        self.assertIsNone(batcher.push(first, policy.evaluate(first, router.decide(first)), at=1.0))
        self.assertIsNone(batcher.push(second, policy.evaluate(second, router.decide(second)), at=2.0))
        batch = batcher.flush()
        assert batch is not None
        self.assertEqual(batch.primary_message.message_id, "m-1")
        self.assertEqual(batch.combined_text, "真红，先看这个\n真红，再帮我查一下")

    def test_timeout_or_conversation_change_closes_previous_batch(self) -> None:
        policy = AttentionPolicy()
        router = ReplyRoutePolicy()
        batcher = AttentionBatcher(window_seconds=1.0)
        first = _message("m-1", "真红，回答", conversation_id="g-1")
        second = _message("m-2", "真红，另一个群", conversation_id="g-2")
        decision1 = policy.evaluate(first, router.decide(first))
        decision2 = policy.evaluate(second, router.decide(second))
        batcher.push(first, decision1, at=1.0)
        closed = batcher.push(second, decision2, at=1.5)
        self.assertIsNotNone(closed)
        self.assertEqual(closed.conversation_id, "g-1")
        self.assertEqual(batcher.flush().conversation_id, "g-2")

    def test_lower_priority_autonomous_message_does_not_replace_direct_address(self) -> None:
        router = ReplyRoutePolicy()
        immediate = _message("m-1", "真红，先回答")
        passive = _message("m-2", "顺便说一句")
        passive_decision = AttentionPolicy(
            autonomous_enabled=True,
            autonomous_probability=1.0,
            random_source=lambda: 0.0,
        ).evaluate(passive, router.decide(passive))
        batcher = AttentionBatcher(window_seconds=2.0)
        batcher.push(immediate, AttentionPolicy().evaluate(immediate, router.decide(immediate)), at=1.0)
        batcher.push(passive, passive_decision, at=1.5)
        batch = batcher.flush()
        assert batch is not None
        self.assertEqual(batch.primary_message.message_id, "m-1")


if __name__ == "__main__":
    unittest.main()
