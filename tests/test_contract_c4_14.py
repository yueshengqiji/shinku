from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from shinku import names
from shinku.cli import main


def isolated_env(root: Path) -> dict[str, str]:
    return {
        names.DATA_ROOT_ENV: str(root / "data"),
        names.CONFIG_ROOT_ENV: str(root / "config"),
        names.LOG_ROOT_ENV: str(root / "logs"),
    }


class NapCatLauncherIntegrationTests(unittest.TestCase):
    def _event(self) -> dict:
        return {
            "post_type": "message",
            "message_type": "private",
            "user_id": "u-14",
            "message_id": "m-14",
            "message": [{"type": "text", "data": {"text": "真红"}}],
        }

    def test_doctor_reports_disabled_without_echoing_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = io.StringIO()
            env = {**isolated_env(Path(temp_dir)), "SHINKU_NAPCAT_ACCESS_TOKEN": "hidden"}
            with mock.patch.dict(os.environ, env, clear=True), redirect_stdout(output):
                self.assertEqual(main(["doctor"]), 0)
            text = output.getvalue()
            self.assertIn("napcat webhook    : disabled", text)
            self.assertIn("action_token=set", text)
            self.assertNotIn("hidden", text)

    def test_enabled_invalid_config_refuses_to_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            error = io.StringIO()
            env = {**isolated_env(Path(temp_dir)), "SHINKU_NAPCAT_WEBHOOK_ENABLED": "true"}
            with mock.patch.dict(os.environ, env, clear=True), redirect_stderr(error), mock.patch("uvicorn.run") as run:
                code = main(["serve", "--service", "backend"])
            self.assertEqual(code, 2)
            self.assertFalse(run.called)
            self.assertIn("base_url_missing", error.getvalue())

    def test_enabled_valid_config_mounts_webhook_without_starting_it_in_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env = {
                **isolated_env(Path(temp_dir)),
                "SHINKU_NAPCAT_BASE_URL": "http://127.0.0.1:3000",
                "SHINKU_NAPCAT_WEBHOOK_ENABLED": "true",
                "SHINKU_NAPCAT_WEBHOOK_PATH": "/hooks/napcat",
                "SHINKU_NAPCAT_EVENT_TOKEN": "hidden-event-token",
            }
            with mock.patch.dict(os.environ, env, clear=True), mock.patch("uvicorn.run") as run:
                self.assertEqual(main(["serve", "--service", "backend"]), 0)
            app = run.call_args.args[0]
            response = TestClient(app).post(
                "/hooks/napcat",
                json=self._event(),
                headers={"Authorization": "Bearer hidden-event-token"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["scheduled"])


if __name__ == "__main__":
    unittest.main()
