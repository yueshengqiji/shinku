from __future__ import annotations

import unittest

from shinku.qq.delivery import build_reply, deliver
from shinku.qq.napcat import NapCatActionTransport, NapCatEventAdapter


class NapCatAdapterTests(unittest.TestCase):
    def test_group_message_becomes_onebot_group_action_with_reply_and_image(self) -> None:
        calls = []

        def action_call(**kwargs):
            calls.append(kwargs)
            return {"status": "ok", "retcode": 0, "data": {"message_id": "n-1"}}

        message = build_reply(
            "group-1",
            "我看到了。",
            conversation_type="group",
            attachments=[{"kind": "image", "url": "https://img.test/a.png"}],
            reply_to_message_id="m-1",
        )
        result = deliver(NapCatActionTransport(action_call), message)
        self.assertTrue(result.sent)
        self.assertEqual(calls[0]["action"], "send_group_msg")
        self.assertEqual(calls[0]["params"]["group_id"], "group-1")
        self.assertEqual([item["type"] for item in calls[0]["params"]["message"]], ["reply", "text", "image"])

    def test_private_message_uses_user_target(self) -> None:
        calls = []
        transport = NapCatActionTransport(lambda **kwargs: calls.append(kwargs) or {"retcode": 0})
        result = deliver(transport, build_reply("user-1", "早", conversation_type="private"))
        self.assertTrue(result.sent)
        self.assertEqual((calls[0]["action"], calls[0]["params"]["user_id"]), ("send_private_msg", "user-1"))

    def test_failed_onebot_response_is_not_sent(self) -> None:
        transport = NapCatActionTransport(lambda **kwargs: {"retcode": 100, "wording": "bad target"})
        result = deliver(transport, build_reply("group-1", "尝试", conversation_type="group"))
        self.assertEqual((result.sent, result.status, result.error), (False, "rejected", "bad target"))

    def test_non_message_event_is_rejected_before_normalization(self) -> None:
        with self.assertRaises(ValueError):
            NapCatEventAdapter().normalize({"post_type": "notice"})

    def test_message_event_keeps_original_image_segment(self) -> None:
        message = NapCatEventAdapter().normalize(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "message": [{"type": "image", "data": {"file": "img-1", "url": "https://img.test/a"}}],
            }
        )
        self.assertEqual(message.attachments[0]["media_id"], "img-1")
        self.assertEqual(message.attachments[0]["url"], "https://img.test/a")


if __name__ == "__main__":
    unittest.main()
