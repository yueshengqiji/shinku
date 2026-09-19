from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

from shinku import names
from shinku.cli import main
from shinku.persona import PersonaLoadError, load_persona
from shinku.qq.assembly import (
    QQAgentConfig,
    QQAgentConfigError,
    assemble_qq_agent,
    load_qq_agent_config,
)


def _isolated_env(root: Path) -> dict[str, str]:
    return {
        "SHINKU_ENV_FILE": str(root / "missing.env"),
        "SHINKU_DATA_ROOT": str(root / "data"),
        "SHINKU_CONFIG_ROOT": str(root / "config"),
        "SHINKU_LOG_ROOT": str(root / "logs"),
        names.env_key("backend"): "18098",
    }


class PersonaLoaderTests(unittest.TestCase):
    def test_loads_utf8_document_and_exposes_only_safe_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "persona.md"
            path.write_text("你是二阶堂真红。", encoding="utf-8")
            document = load_persona(path)
        self.assertEqual(document.text, "你是二阶堂真红。")
        self.assertTrue(document.sha256)
        self.assertEqual(document.diagnostics()["path"], str(path.resolve()))

    def test_missing_file_does_not_fallback(self) -> None:
        with self.assertRaisesRegex(PersonaLoadError, "persona_file_not_found"):
            load_persona("C:/not-a-shinku-persona.md")


class QQAgentAssemblyTests(unittest.TestCase):
    def test_environment_is_explicit_and_disabled_by_default(self) -> None:
        config = load_qq_agent_config({})
        self.assertFalse(config.enabled)
        self.assertFalse(config.send_enabled)
        self.assertEqual(config.validation_errors(), ())

    def test_enabled_config_requires_persona_and_model_boundary(self) -> None:
        config = QQAgentConfig(enabled=True)
        self.assertIn("persona_file_missing", config.validation_errors())
        self.assertIn("chat_base_url_missing", config.validation_errors())
        with self.assertRaisesRegex(QQAgentConfigError, "persona_file_missing"):
            assemble_qq_agent(config)

    def test_assembly_keeps_sender_dry_run_until_send_flag_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            persona_path = Path(tmp) / "persona.md"
            persona_path.write_text("独立主人设", encoding="utf-8")
            config = QQAgentConfig(
                enabled=True,
                send_enabled=False,
                persona_file=str(persona_path),
                chat_api_key="test-key",
                chat_base_url="https://example.invalid/v1",
                chat_model_name="test-model",
            )
            fake_runtime = mock.Mock()
            assembly = assemble_qq_agent(config, runtime=fake_runtime, sender=lambda message: None)
        self.assertIsNone(assembly.bridge.sender)
        self.assertEqual(assembly.persona.text, "独立主人设")

    def test_cli_wires_agent_dispatcher_without_starting_or_sending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            persona_path = root / "persona.md"
            persona_path.write_text("独立主人设", encoding="utf-8")
            env = {
                **_isolated_env(root),
                "SHINKU_NAPCAT_WEBHOOK_ENABLED": "true",
                "SHINKU_NAPCAT_BASE_URL": "http://127.0.0.1:3000",
                "SHINKU_QQ_AGENT_ENABLED": "true",
                "SHINKU_QQ_AGENT_SEND_ENABLED": "false",
                "SHINKU_PERSONA_FILE": str(persona_path),
                "SHINKU_CHAT_API_KEY": "test-key",
                "SHINKU_CHAT_BASE_URL": "https://example.invalid/v1",
                "SHINKU_CHAT_MODEL_NAME": "test-model",
            }
            fake_runtime = mock.Mock()
            fake_runtime.call_chat_json.return_value = {"speech": "灰测"}
            with (
                mock.patch.dict(os.environ, env, clear=True),
                mock.patch("uvicorn.run") as run,
                mock.patch("shinku.qq.assembly.LLMRuntime", return_value=fake_runtime),
            ):
                self.assertEqual(main(["serve", "--service", "backend"]), 0)
            app = run.call_args.args[0]
            host = app.state.napcat_host
            self.assertIsNotNone(host.turn_dispatcher)
            self.assertIsNone(host.turn_dispatcher.handler.sender)


if __name__ == "__main__":
    unittest.main()
