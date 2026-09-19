from __future__ import annotations

import json
import unittest

from shinku.qq.delivery import build_reply
from shinku.qq.host import NapCatHost
from shinku.qq.napcat import NapCatVisualInputBridge


class _Response:
    def read(self) -> bytes:
        return b'{"status":"ok","retcode":0,"data":{"message_id":"sent-1"}}'


class NapCatHostTests(unittest.TestCase):
    def _private_event(self) -> dict:
        return {
            "post_type": "message",
            "message_type": "private",
            "user_id": "u-1",
            "message_id": "m-1",
            "message": [
                {"type": "text", "data": {"text": "看这张"}},
                {"type": "image", "data": {"file": "img-1", "url": "https://img.test/a.png"}},
            ],
        }

    def test_host_composes_decode_ingress_and_visual_preparation(self) -> None:
        host = NapCatHost(
            visual_bridge=NapCatVisualInputBridge(materialize=lambda attachment: b"image-bytes"),
            clock=lambda: 12.0,
        )
        result = host.handle_event(json.dumps(self._private_event()).encode("utf-8"))
        self.assertTrue(result.scheduled)
        self.assertTrue(result.visual.ready)
        self.assertEqual(result.visual.model_images()[0]["data_url"], "data:image/png;base64,aW1hZ2UtYnl0ZXM=")

    def test_http_host_does_not_call_network_until_send(self) -> None:
        calls = []

        def opener(request, *, timeout):
            calls.append((request.full_url, timeout))
            return _Response()

        host = NapCatHost.from_http("http://napcat.test/api", opener=opener)
        host.handle_event(self._private_event(), at=1.0)
        self.assertEqual(calls, [])
        result = host.send(build_reply("u-1", "收到了", conversation_type="private"))
        self.assertTrue(result.sent)
        self.assertEqual(calls[0][0], "http://napcat.test/api/send_private_msg")

    def test_host_without_transport_returns_explicit_unconfigured_result(self) -> None:
        host = NapCatHost()
        result = host.send(build_reply("u-1", "暂时不发", conversation_type="private"))
        self.assertEqual((result.sent, result.status, result.error), (False, "transport_unconfigured", "napcat_transport_unconfigured"))

    def test_host_flushes_scheduled_messages_through_existing_batcher(self) -> None:
        host = NapCatHost(clock=lambda: 10.0)
        first = host.handle_event({**self._private_event(), "message_id": "m-1"})
        second = host.handle_event({**self._private_event(), "message_id": "m-2"}, at=10.5)
        self.assertIsNone(first.ingress.closed_batch)
        self.assertIsNone(second.ingress.closed_batch)
        batch = host.flush()
        self.assertEqual([message.message_id for message in batch.messages], ["m-1", "m-2"])


if __name__ == "__main__":
    unittest.main()
