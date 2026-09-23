"""独立版只读控制中枢的契约测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from shinku import names
from shinku.api import create_app
from shinku.api.model_runtime import RuntimeModelServiceConfig, RuntimeModelServiceController
from shinku.config import load_settings
from shinku.providers.catalog import ModelServiceConfigStore
from shinku.qq.commands import QQCommandState
from shinku.qq.trace import TurnTraceStore


def _settings(root: Path):
    return load_settings(
        {
            names.DATA_ROOT_ENV: str(root / "data"),
            names.CONFIG_ROOT_ENV: str(root / "config"),
            names.LOG_ROOT_ENV: str(root / "logs"),
            names.env_key("backend"): "19998",
        }
    )


class DebugHubTests(unittest.TestCase):
    def test_hub_is_local_read_only_and_redacts_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _settings(root)
            settings.log_root.mkdir(parents=True)
            (settings.log_root / "backend.log").write_text(
                "INFO request ok\napi_key=not-for-display\n", encoding="utf-8"
            )

            class Runtime:
                def snapshot_metrics(self):
                    return {"requests": 2}

                def snapshot_last_error(self):
                    return {}

            class Store:
                path = root / "data" / "memory.sqlite3"

                def stats(self):
                    return {"turns": 3, "active_events": 1, "active_facts": 2, "forgotten": 0}

            command_state = QQCommandState()
            trace_store = TurnTraceStore()
            trace_id = trace_store.begin(
                SimpleNamespace(
                    conversation_type="private",
                    conversation_id="user-1",
                    primary_message_id="message-1",
                    combined_text="不应出现在控制中枢的原文",
                )
            )
            trace_store.finish(trace_id, agent_status="command", reason="test")
            assembly = SimpleNamespace(
                diagnostics=lambda: {
                    "config": {"chat_model_name": "test-model", "send_enabled": False},
                    "persona": {"path": "C:/persona.md"},
                },
                bridge=SimpleNamespace(
                    runtime=Runtime(),
                    memory_service=SimpleNamespace(store=Store()),
                    sender=None,
                    allowed_tool_names={"look"},
                    max_rounds=4,
                    history_turns=8,
                    recent_context_messages=5,
                    command_state=command_state,
                    trace_store=trace_store,
                ),
            )
            config = SimpleNamespace(
                diagnostics=lambda: {"webhook_enabled": True, "webhook_path": "/events/napcat"}
            )
            client = TestClient(
                create_app(settings, qq_agent_assembly=assembly, napcat_config=config)
            )

            page = client.get("/debug/hub")
            self.assertEqual(page.status_code, 200)
            self.assertIn("本机控制", page.text)
            self.assertIn("运行指标", page.text)
            self.assertIn("运行配置", page.text)
            self.assertIn("立即刷新", page.text)
            self.assertIn("自动刷新：开", page.text)
            self.assertIn("QQ 自检", page.text)
            self.assertIn("工具目录", page.text)
            self.assertIn("待授权工具", page.text)
            self.assertIn("记忆检视", page.text)
            self.assertEqual(page.headers["x-shinku-debug-mode"], "control")
            css = client.get("/debug/hub/assets/app.css")
            self.assertEqual(css.status_code, 200)
            self.assertIn("--bg", css.text)
            javascript = client.get("/debug/hub/assets/app.js")
            self.assertEqual(javascript.status_code, 200)
            self.assertIn("/debug/hub/state", javascript.text)
            self.assertIn("/metrics/recent?days=7", javascript.text)
            self.assertIn("/debug/hub/traces?limit=30", javascript.text)
            self.assertIn("/api/qq/self-check", javascript.text)
            self.assertIn("/capabilities", javascript.text)
            self.assertIn("/agent-work/tasks?limit=20", javascript.text)
            self.assertIn("/capabilities/approvals/", javascript.text)
            self.assertIn("/agent-work/tasks/${encodeURIComponent(taskId)}", javascript.text)
            self.assertIn("async function fetchOptional", javascript.text)
            self.assertIn("refreshInFlight", javascript.text)
            self.assertIn("const providerDrafts = new Map()", javascript.text)
            self.assertIn("captureProviderDrafts", javascript.text)
            self.assertIn("function renderRuntimeConfig", javascript.text)

            state = client.get("/debug/hub/state")
            self.assertEqual(state.status_code, 200)
            body = state.json()
            self.assertFalse(body["read_only"])
            self.assertEqual(body["agent"]["memory"]["stats"]["turns"], 3)
            self.assertEqual(body["agent"]["runtime"]["metrics"]["requests"], 2)
            lines = "\n".join(body["logs"][0]["lines"])
            self.assertNotIn("not-for-display", lines)
            self.assertIn("<redacted>", lines)

            traces = client.get("/debug/hub/traces?limit=10")
            self.assertEqual(traces.status_code, 200)
            self.assertEqual(traces.json()["traces"][0]["trace_id"], trace_id)
            self.assertNotIn("不应出现在控制中枢的原文", traces.text)

            self.assertEqual(client.post("/debug/hub/state").status_code, 405)

    def test_agent_control_and_model_services_are_local_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _settings(root)
            command_state = QQCommandState()
            assembly = SimpleNamespace(
                diagnostics=lambda: {"config": {"chat_model_name": "test-model"}},
                bridge=SimpleNamespace(
                    runtime=SimpleNamespace(chat=SimpleNamespace(model="test-model"), aux=SimpleNamespace(model="test-model")),
                    memory_service=None,
                    sender=None,
                    allowed_tool_names=set(),
                    max_rounds=4,
                    history_turns=8,
                    recent_context_messages=5,
                    command_state=command_state,
                ),
            )
            config = SimpleNamespace(
                diagnostics=lambda: {"webhook_enabled": False, "webhook_path": "/events/napcat"}
            )
            agent_config = SimpleNamespace(
                chat_api_key="",
                chat_base_url="",
                chat_model_name="test-model",
                chat_api_protocol="openai",
            )
            controller = RuntimeModelServiceController(
                store=ModelServiceConfigStore(root / "data" / "model_services.json"),
                config_module=RuntimeModelServiceConfig.from_agent_config(agent_config),
                runtime=None,
            )
            client = TestClient(
                create_app(
                    settings,
                    qq_agent_assembly=assembly,
                    napcat_config=config,
                    model_services_controller=controller,
                )
            )

            model_page = client.get("/control-center/model-services")
            self.assertEqual(model_page.status_code, 200)
            self.assertIn("providers", model_page.json())

            stopped = client.post("/control-center/agent/control", json={"action": "stop"})
            self.assertEqual(stopped.status_code, 200)
            self.assertTrue(stopped.json()["agent"]["controls"]["stopped"])



if __name__ == "__main__":
    unittest.main()
