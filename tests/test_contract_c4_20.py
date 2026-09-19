from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path
from unittest import mock

from shinku import names
from shinku.cli import main
from shinku.persona import PersonaLoadError, load_persona
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.attention import AttentionBatcher
from shinku.qq.host import NapCatHost
from shinku.qq.napcat import NapCatVisualInputBridge
from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.routing import ReplyRoutePolicy
from shinku.qq.turns import NapCatTurnDispatcher
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

    def test_local_gray_path_merges_ingress_preserves_image_and_returns_one_dry_run_reply(self) -> None:
        class Runtime:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def call_chat_json(self, **kwargs):
                self.calls.append(kwargs)
                return {"speech": "看到了，先把这两句合在一起处理。"}

        runtime = Runtime()
        bridge = QQAgentBridge(runtime=runtime, system_prompt="独立主人设")
        replies = []
        ingress = QQIngressAdapter(
            route_policy=ReplyRoutePolicy(bot_ids=frozenset({"bot-1"})),
            batcher=AttentionBatcher(window_seconds=2.0),
        )
        host = NapCatHost(
            ingress=ingress,
            visual_bridge=NapCatVisualInputBridge(),
            clock=lambda: 1.0,
        )
        dispatcher = NapCatTurnDispatcher(
            handler=lambda turn: replies.append(bridge.handle_turn(turn)),
            flush_batch=host.flush,
            timer_factory=lambda delay, callback: mock.Mock(),
        )
        host.set_turn_dispatcher(dispatcher)
        event = {
            "post_type": "message",
            "message_type": "group",
            "group_id": "g-20",
            "user_id": "u-20",
            "message": [
                {"type": "at", "data": {"qq": "bot-1"}},
                {"type": "text", "data": {"text": "先看这张"}},
                {"type": "image", "data": {"url": "data:image/png;base64,aGVsbG8="}},
            ],
        }
        host.handle_event({**event, "message_id": "m-1"}, at=1.0)
        host.handle_event({**event, "message_id": "m-2", "message": [{"type": "at", "data": {"qq": "bot-1"}}, {"type": "text", "data": {"text": "再补一句"}}]}, at=1.5)
        receipt = dispatcher.flush()

        self.assertEqual((receipt.dispatched, len(replies)), (1, 1))
        self.assertEqual((replies[0].delivery_status, replies[0].sent), ("dry_run", False))
        self.assertIn("两句", replies[0].final_text)
        self.assertEqual(runtime.calls[0]["user_images"], [{"data_url": "data:image/png;base64,aGVsbG8="}])
        self.assertIn("先看这张", runtime.calls[0]["history_turns"][0]["content"])
        self.assertIn("再补一句", runtime.calls[0]["history_turns"][1]["content"])


if __name__ == "__main__":
    unittest.main()
