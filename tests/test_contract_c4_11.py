from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from shinku.qq.config import NapCatConnectionConfig
from shinku.qq.delivery import build_reply
from shinku.qq.host import NapCatHost


class _Response:
    def read(self) -> bytes:
        return b'{"status":"ok","retcode":0,"data":{"message_id":"m-11"}}'


class NapCatConfigAndPreflightTests(unittest.TestCase):
    def test_environment_config_has_safe_diagnostics(self) -> None:
        config = NapCatConnectionConfig.from_env(
            {
                "SHINKU_NAPCAT_BASE_URL": "http://127.0.0.1:3000/api",
                "SHINKU_NAPCAT_ACCESS_TOKEN": "do-not-display",
                "SHINKU_NAPCAT_TIMEOUT": "4.5",
            }
        )
        diagnostics = config.diagnostics()
        self.assertEqual((diagnostics["configured"], diagnostics["valid"], diagnostics["timeout"]), (True, True, 4.5))
        self.assertEqual(diagnostics["endpoint"], "http://127.0.0.1:3000/api")
        self.assertTrue(diagnostics["token_present"])
        self.assertNotIn("do-not-display", str(diagnostics))

    def test_missing_or_unsafe_configuration_is_rejected_before_host_creation(self) -> None:
        missing = NapCatConnectionConfig()
        self.assertIn("base_url_missing", missing.validate())
        unsafe = NapCatConnectionConfig("ftp://user:password@napcat.test", timeout=0)
        self.assertEqual(
            set(unsafe.validate()),
            {"base_url_scheme_invalid", "base_url_userinfo_forbidden", "timeout_invalid"},
        )
        with self.assertRaises(ValueError):
            NapCatHost.from_config(missing)

    def test_configured_host_does_not_call_opener_until_send(self) -> None:
        calls = []

        def opener(request, *, timeout):
            calls.append((request.full_url, request.get_header("Authorization"), timeout))
            return _Response()

        host = NapCatHost.from_config(
            NapCatConnectionConfig("http://napcat.test", access_token="secret", timeout=2.0),
            opener=opener,
        )
        self.assertEqual(calls, [])
        result = host.send(build_reply("u-1", "收到", conversation_type="private"))
        self.assertTrue(result.sent)
        self.assertEqual(calls, [("http://napcat.test/send_private_msg", "Bearer secret", 2.0)])

    def test_token_file_is_read_only_when_configuring_explicit_http_host(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            token_path = Path(temp_dir) / "token.txt"
            token_path.write_text("file-secret\n", encoding="utf-8")
            config = NapCatConnectionConfig("http://napcat.test", access_token_file=str(token_path))
            self.assertEqual(config.token_value(), "file-secret")
            self.assertTrue(config.diagnostics()["token_present"])


if __name__ == "__main__":
    unittest.main()
