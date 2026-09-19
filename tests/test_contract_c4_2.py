from __future__ import annotations

import unittest

from shinku.qq.message import IncomingMessage
from shinku.qq.routing import ReplyRoutePolicy


def _message(*segments, **event):
    event.setdefault("conversation_type", "group")
    event["segments"] = list(segments)
    return IncomingMessage.from_event(event)


class ReplyRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ReplyRoutePolicy(bot_ids=frozenset({"bot-1"}))

    def test_private_message_always_enters_agent(self) -> None:
        decision = self.policy.decide(_message({"type": "text", "data": {"text": "你好"}}, conversation_type="private"))
        self.assertEqual((decision.should_reply, decision.reason, decision.priority), (True, "private_message", 100))

    def test_group_at_bot_enters_agent(self) -> None:
        decision = self.policy.decide(_message(
            {"type": "at", "data": {"qq": "bot-1"}},
            {"type": "text", "data": {"text": "帮我看看"}},
        ))
        self.assertEqual((decision.should_reply, decision.reason), (True, "mentioned_bot"))

    def test_direct_name_requires_name_at_the_start(self) -> None:
        direct = self.policy.decide(_message({"type": "text", "data": {"text": "真红，过来一下"}}))
        incidental = self.policy.decide(_message({"type": "text", "data": {"text": "我刚才看见真红了"}}))
        self.assertEqual(direct.reason, "direct_name")
        self.assertEqual(incidental.reason, "unaddressed_group_message")

    def test_unaddressed_group_text_does_not_enter_agent(self) -> None:
        decision = self.policy.decide(_message({"type": "text", "data": {"text": "你们继续聊"}}))
        self.assertFalse(decision.should_reply)
        self.assertEqual(decision.reason, "unaddressed_group_message")

    def test_unaddressed_image_does_not_enter_agent(self) -> None:
        decision = self.policy.decide(_message({"type": "image", "data": {"file": "img-1"}}))
        self.assertEqual((decision.should_reply, decision.reason, decision.image_only), (False, "unaddressed_image", True))

    def test_reply_to_bot_allows_image_only_reply(self) -> None:
        decision = self.policy.decide(_message(
            {"type": "reply", "data": {"id": "m-1", "target": "bot-1"}},
            {"type": "image", "data": {"file": "img-2"}},
        ))
        self.assertEqual((decision.should_reply, decision.reason, decision.reply_to_message_id), (True, "reply_to_bot", "m-1"))

    def test_at_other_user_is_not_an_at_to_bot(self) -> None:
        decision = self.policy.decide(_message({"type": "at", "data": {"qq": "user-2"}}))
        self.assertFalse(decision.should_reply)


if __name__ == "__main__":
    unittest.main()
