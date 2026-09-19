from __future__ import annotations

import base64
from email.message import Message
from pathlib import Path
import tempfile
import unittest

from shinku.qq.napcat import NapCatImageMaterializer, NapCatVisualInputBridge
from shinku.qq.napcat import NapCatEventDecoder


class _ImageResponse:
    def __init__(self, body: bytes, mime: str = "image/jpeg") -> None:
        self.body = body
        self.headers = Message()
        self.headers["Content-Type"] = mime

    def read(self, size: int) -> bytes:
        return self.body[:size]


class NapCatImageMaterializerTests(unittest.TestCase):
    def _message(self, attachment: dict) -> object:
        event = {
            "post_type": "message",
            "message_type": "group",
            "group_id": "g-1",
            "message": [{"type": "image", "data": attachment}],
        }
        return NapCatEventDecoder().decode(event)

    def test_remote_url_is_downloaded_only_through_explicit_materializer(self) -> None:
        calls = []

        def opener(request, *, timeout):
            calls.append((request.full_url, timeout))
            return _ImageResponse(b"remote-image")

        materializer = NapCatImageMaterializer(url_opener=opener, timeout=2.0)
        attachment = {"kind": "image", "url": "https://img.test/a.jpg"}
        data_url = materializer(attachment)
        self.assertEqual(data_url, "data:image/jpeg;base64," + base64.b64encode(b"remote-image").decode())
        self.assertEqual(calls, [("https://img.test/a.jpg", 2.0)])

    def test_local_file_url_respects_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "photo.png"
            path.write_bytes(b"local-image")
            materializer = NapCatImageMaterializer(allowed_roots=(temp_dir,))
            result = materializer({"kind": "image", "local_path": path.as_uri()})
            self.assertTrue(result.startswith("data:image/png;base64,"))

    def test_disallowed_local_file_stays_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as allowed_dir:
            path = Path(temp_dir) / "photo.png"
            path.write_bytes(b"local-image")
            materializer = NapCatImageMaterializer(allowed_roots=(allowed_dir,))
            self.assertIsNone(materializer({"kind": "image", "local_path": str(path)}))

    def test_media_id_uses_injected_loader(self) -> None:
        calls = []

        def loader(media_id: str) -> bytes:
            calls.append(media_id)
            return b"media-image"

        materializer = NapCatImageMaterializer(media_loader=loader)
        result = materializer({"kind": "image", "media_id": "img-9", "mime_type": "image/webp"})
        self.assertTrue(result.startswith("data:image/webp;base64,"))
        self.assertEqual(calls, ["img-9"])

    def test_oversized_payload_is_not_sent_to_visual_bridge(self) -> None:
        materializer = NapCatImageMaterializer(max_bytes=4, file_reader=lambda path: b"12345")
        message = self._message({"file": "img-1", "path": "C:/photo.png"})
        bridge = NapCatVisualInputBridge(materialize=materializer)
        batch = bridge.build(message)
        self.assertTrue(batch.pending)
        self.assertEqual(batch.model_images(), [])


if __name__ == "__main__":
    unittest.main()
