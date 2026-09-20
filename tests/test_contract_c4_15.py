from __future__ import annotations

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from shinku import names
from shinku.cli import main
from shinku.config import load_project_env


class ProjectEnvLoaderTests(unittest.TestCase):
    def test_loader_accepts_only_shinku_keys_and_preserves_shell_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "SHINKU_BACKEND_PORT=18081\n"
                "SHINKU_QUOTED=\"hello world\"\n"
                "COMPANION_PORT=1234\n"
                "BAD-KEY=ignored\n",
                encoding="utf-8",
            )
            environ = {"SHINKU_BACKEND_PORT": "18082"}
            loaded = load_project_env(environ, path=env_file)
            self.assertEqual(loaded, env_file)
            self.assertEqual(environ["SHINKU_BACKEND_PORT"], "18082")
            self.assertEqual(environ["SHINKU_QUOTED"], "hello world")
            self.assertNotIn("COMPANION_PORT", environ)
            self.assertNotIn("BAD-KEY", environ)

    def test_missing_file_is_a_noop(self) -> None:
        environ = {}
        self.assertIsNone(load_project_env(environ, path="missing.env"))
        self.assertEqual(environ, {})

    def test_empty_shell_value_does_not_mask_env_file_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "SHINKU_CHAT_API_KEY=project-key\n",
                encoding="utf-8",
            )
            environ = {"SHINKU_CHAT_API_KEY": ""}
            load_project_env(environ, path=env_file)
            self.assertEqual(environ["SHINKU_CHAT_API_KEY"], "project-key")

    def test_cli_doctor_reads_explicit_env_file_without_printing_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_file = root / ".env"
            env_file.write_text(
                f"{names.DATA_ROOT_ENV}={root / 'data'}\n"
                f"{names.CONFIG_ROOT_ENV}={root / 'config'}\n"
                f"{names.LOG_ROOT_ENV}={root / 'logs'}\n"
                f"{names.env_key('backend')}=18083\n"
                "SHINKU_NAPCAT_ACCESS_TOKEN=hidden-token\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with mock.patch.dict(os.environ, {"SHINKU_ENV_FILE": str(env_file)}, clear=True):
                with redirect_stdout(output):
                    self.assertEqual(main(["doctor"]), 0)
            text = output.getvalue()
            self.assertIn("18083", text)
            self.assertIn("action_token=set", text)
            self.assertNotIn("hidden-token", text)


if __name__ == "__main__":
    unittest.main()
