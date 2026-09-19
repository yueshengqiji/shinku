from __future__ import annotations

import json
import unittest

from shinku.qq.napcat import NapCatEventDecoder, NapCatVisualInputBridge


class NapCatEventAndVisualBridgeTests(unittest.TestCase):
    def _event(self) -> dict:
        return {
            "post_type": "message",
            "message_type": "group",
            "group_id": "g-1",
            "message_id": "m-1",
            "message": [
                {"type": "text", "data": {"text": "看一下"}},
                {"type": "image", "data": {"file": "img-1", "url": "https://img.test/a.png"}},
            ],
        }

    def test_json_body_decoder_keeps_image_reference(self) -> None:
        message = NapCatEventDecoder().decode(json.dumps(self._event()).encode("utf-8"))
        self.assertEqual(message.text, "看一下")
        self.assertEqual(message.attachments[0]["media_id"], "img-1")
        self.assertEqual(message.attachments[0]["url"], "https://img.test/a.png")

    def test_unresolved_image_is_pending_instead_of_becoming_plain_text(self) -> None:
        message = NapCatEventDecoder().decode(self._event())
        batch = NapCatVisualInputBridge().build(message)
        self.assertTrue(batch.pending)
        self.assertFalse(batch.ready)
        self.assertEqual(batch.model_images(), [])

    def test_injected_materializer_makes_image_model_ready(self) -> None:
        message = NapCatEventDecoder().decode(self._event())
        calls = []

        def materialize(attachment):
            calls.append(dict(attachment))
            return b"png-bytes"

        batch = NapCatVisualInputBridge(materialize=materialize).build(message)
        self.assertTrue(batch.ready)
        self.assertFalse(batch.pending)
        self.assertEqual(len(calls), 1)
        self.assertEqual(batch.model_images()[0]["data_url"], "data:image/png;base64,cG5nLWJ5dGVz")

    def test_multiple_images_keep_order_and_are_not_deduplicated(self) -> None:
        event = self._event()
        event["message"].append({"type": "image", "data": {"file": "img-2"}})
        message = NapCatEventDecoder().decode(event)
        batch = NapCatVisualInputBridge(lambda attachment: b"same").build(message)
        self.assertEqual([item.reference for item in batch.inputs], ["https://img.test/a.png", "img-2"])
        self.assertEqual(len(batch.model_images()), 2)

    def test_decoder_rejects_invalid_json_and_non_object_json(self) -> None:
        decoder = NapCatEventDecoder()
        with self.assertRaises(ValueError):
            decoder.decode(b"not-json")
        with self.assertRaises(ValueError):
            decoder.decode("[]")


if __name__ == "__main__":
    unittest.main()
