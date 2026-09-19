from __future__ import annotations

import unittest

from shinku.qq.message import IncomingMessage, MessageSegment


class MessageBoundaryTests(unittest.TestCase):
    def test_image_and_text_segments_are_not_collapsed(self) -> None:
        message = IncomingMessage.from_event(
            {
                "message_id": "m-1",
                "user_id": "u-1",
                "group_id": "g-1",
                "message": [
                    {"type": "text", "data": {"text": "看看这个"}},
                    {"type": "image", "data": {"file": "img-7", "url": "https://img.test/a.png"}},
                ],
            }
        )
        self.assertEqual(message.text, "看看这个")
        self.assertTrue(message.has_media)
        self.assertEqual(message.attachments[0]["media_id"], "img-7")
        self.assertEqual(message.attachments[0]["url"], "https://img.test/a.png")
        self.assertNotIn("[图片]", message.text)
        self.assertEqual([item.kind for item in message.segments], ["text", "image"])

    def test_quote_and_local_file_fields_survive_normalization(self) -> None:
        message = IncomingMessage.from_event(
            {
                "messageId": "m-2",
                "sender": {"user_id": "u-2"},
                "conversation_type": "private",
                "conversation_id": "c-2",
                "segments": [
                    {"type": "reply", "data": {"id": "quoted-1"}},
                    {"type": "file", "data": {"file_id": "f-1", "path": "C:/tmp/a.txt"}},
                ],
            }
        )
        self.assertEqual((message.message_id, message.sender_id, message.conversation_id), ("m-2", "u-2", "c-2"))
        self.assertTrue(message.has_reply)
        self.assertEqual(message.attachments[0]["local_path"], "C:/tmp/a.txt")
        self.assertEqual(message.as_dict()["segments"][0]["payload"]["data"]["id"], "quoted-1")

    def test_plain_text_event_has_no_fake_attachment(self) -> None:
        message = IncomingMessage.from_event({"raw_message": "你好", "time": "12.5"})
        self.assertEqual(message.text, "你好")
        self.assertFalse(message.has_media)
        self.assertEqual(message.timestamp, 12.5)
        self.assertEqual(message.segments, (MessageSegment(kind="text", text="你好"),))


if __name__ == "__main__":
    unittest.main()
