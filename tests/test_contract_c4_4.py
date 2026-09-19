from __future__ import annotations

import unittest

from shinku.qq.delivery import build_reply, deliver


class RecordingDelivery:
    def __init__(self, response=None, error=False):
        self.messages = []
        self.response = response if response is not None else {"ok": True, "message_ids": ["sent-1"]}
        self.error = error

    def send(self, message):
        self.messages.append(message)
        if self.error:
            raise TimeoutError("delivery timeout")
        return self.response


class DeliveryContractTests(unittest.TestCase):
    def test_text_image_and_quote_are_one_outgoing_request(self) -> None:
        message = build_reply(
            "group-1",
            "我看到了。",
            attachments=[{"kind": "image", "url": "https://img.test/a.png", "media_id": "img-1"}],
            reply_to_message_id="m-1",
        )
        self.assertEqual([segment.kind for segment in message.segments], ["text", "image"])
        self.assertEqual(message.reply_to_message_id, "m-1")
        transport = RecordingDelivery()
        result = deliver(transport, message)
        self.assertTrue(result.sent)
        self.assertEqual(len(transport.messages), 1)
        self.assertEqual(transport.messages[0].as_dict()["segments"][1]["url"], "https://img.test/a.png")

    def test_attachment_only_reply_is_supported_without_fake_text(self) -> None:
        message = build_reply("private-1", attachments=[{"kind": "image", "local_path": "C:/tmp/a.png"}])
        self.assertEqual([segment.kind for segment in message.segments], ["image"])
        self.assertNotIn("看不到", str(message.as_dict()))

    def test_empty_reply_is_rejected_before_transport(self) -> None:
        with self.assertRaises(ValueError):
            build_reply("group-1")

    def test_transport_failure_is_structured_and_has_no_fallback_text(self) -> None:
        message = build_reply("group-1", "已准备发送。")
        result = deliver(RecordingDelivery(error=True), message)
        self.assertEqual(result.as_dict(), {"sent": False, "status": "transport_error", "message_ids": [], "error": "TimeoutError"})

    def test_rejected_transport_is_not_reported_as_sent(self) -> None:
        message = build_reply("group-1", "尝试发送。")
        result = deliver(RecordingDelivery(response={"ok": False, "error": "attachment_missing"}), message)
        self.assertEqual((result.sent, result.status, result.error), (False, "rejected", "attachment_missing"))

    def test_invalid_transport_does_not_raise_into_agent(self) -> None:
        message = build_reply("group-1", "尝试发送。")
        result = deliver(object(), message)
        self.assertEqual((result.sent, result.status), (False, "transport_invalid"))


if __name__ == "__main__":
    unittest.main()
