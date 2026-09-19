from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shinku.persona import PersonaLoadError, load_persona
from shinku.qq.assembly import (
    QQAgentConfig,
    QQAgentConfigError,
    assemble_qq_agent,
    load_qq_agent_config,
)


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


if __name__ == "__main__":
    unittest.main()
