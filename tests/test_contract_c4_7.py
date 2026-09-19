from __future__ import annotations

import json
import unittest
from urllib.error import URLError

from shinku.qq.delivery import build_reply, deliver
from shinku.qq.napcat import NapCatActionTransport, NapCatHttpActionCaller


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload


class NapCatHttpCallerTests(unittest.TestCase):
    def test_http_caller_posts_action_json_and_bearer_token(self) -> None:
        seen = {}

        def opener(request, *, timeout):
            seen["request"] = request
            seen["timeout"] = timeout
            return _FakeResponse(b'{"status":"ok","retcode":0,"data":{"message_id":"m-7"}}')

        caller = NapCatHttpActionCaller(
            "http://napcat.test/api",
            access_token="secret-token",
            timeout=3.5,
            opener=opener,
        )
        result = NapCatActionTransport(caller).send(build_reply("g-1", "收到", conversation_type="group"))

        self.assertEqual(result["message_ids"], ["m-7"])
        request = seen["request"]
        self.assertEqual(request.full_url, "http://napcat.test/api/send_group_msg")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")
        self.assertEqual(json.loads(request.data.decode("utf-8"))["group_id"], "g-1")
        self.assertEqual(seen["timeout"], 3.5)

    def test_http_caller_never_sends_on_construction(self) -> None:
        calls = []

        def opener(*args, **kwargs):
            calls.append((args, kwargs))
            return _FakeResponse(b'{"retcode":0}')

        NapCatHttpActionCaller("http://napcat.test", opener=opener)
        self.assertEqual(calls, [])

    def test_http_connection_error_becomes_delivery_transport_error(self) -> None:
        def opener(*args, **kwargs):
            raise URLError("offline")

        transport = NapCatActionTransport(NapCatHttpActionCaller("http://napcat.test", opener=opener))
        result = deliver(transport, build_reply("u-1", "稍等", conversation_type="private"))
        self.assertEqual((result.sent, result.status), (False, "transport_error"))

    def test_http_caller_rejects_path_traversal_and_non_http_urls(self) -> None:
        with self.assertRaises(ValueError):
            NapCatHttpActionCaller("ws://napcat.test")
        caller = NapCatHttpActionCaller("http://napcat.test")
        with self.assertRaises(ValueError):
            caller(action="../send_group_msg", params={})


if __name__ == "__main__":
    unittest.main()
