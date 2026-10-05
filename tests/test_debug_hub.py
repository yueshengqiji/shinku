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
from shinku.qq.attention_trace import AttentionTraceStore
from shinku.qq.commands import QQCommandState
from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.host import NapCatHost
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

                def recent_routing_audits(self, *, scope=None, limit=20):
                    return [{
                        "audit_id": "audit-1",
                        "conversation_id": "g-1",
                        "scope": "group:g-1",
                        "created_at": 100,
                        "history_mode": "recent",
                        "need_retrieval": False,
                        "used_model": False,
                        "confidence": 0.92,
                        "reason": "local_recent_topic",
                        "retrieved_count": 2,
                        "injected_chars": 84,
                        "decision": {
                            "query": "这句检索内容不应被前端读取",
                            "local_signals": ["reply_chain", "topic_continuation"],
                        },
                    }]

            command_state = QQCommandState()
            trace_store = TurnTraceStore()
            trace_id = trace_store.begin(
                SimpleNamespace(
                    conversation_type="private",
                    conversation_id="user-1",
                    primary_message_id="message-1",
                    message_ids=["message-1"],
                    combined_text="不应出现在控制中枢的原文",
                )
            )
            trace_store.stage(
                trace_id,
                "llm_usage",
                channel="chat",
                calls=1,
                input_tokens=21,
                output_tokens=5,
                total_tokens=26,
                cache_read_tokens=12,
            )
            trace_store.finish(trace_id, agent_status="command", reason="test")
            attention_store = AttentionTraceStore()
            attention_store.record(
                observed_at=100.0,
                conversation_id="user-1",
                conversation_type="private",
                message_id="message-1",
                sender_id="user-1",
                text_chars=12,
                has_media=False,
                is_poke=False,
                admission={"accepted": True, "reason": "accepted"},
                route={"should_reply": True, "reason": "private_message"},
                attention={"should_schedule": True, "immediate": True, "score": 100, "reason": "private_message"},
                queued=True,
                topic_state={"reply_message_count": 0, "group_context_count": 0},
            )
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
                create_app(
                    settings,
                    qq_agent_assembly=assembly,
                    napcat_host=NapCatHost(
                        ingress=QQIngressAdapter(attention_trace_store=attention_store),
                    ),
                    napcat_config=config,
                )
            )

            page = client.get("/debug/hub")
            self.assertEqual(page.status_code, 200)
            self.assertIn("本机控制", page.text)
            self.assertIn("运行指标", page.text)
            self.assertIn("最近注意力决策", page.text)
            self.assertIn("最近记忆路由", page.text)
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
            self.assertIn("/debug/hub/attention?limit=80", javascript.text)
            self.assertIn("/debug/hub/memory-routing?limit=60", javascript.text)
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

            attention = client.get("/debug/hub/attention?limit=10")
            self.assertEqual(attention.status_code, 200)
            row = attention.json()["items"][0]
            self.assertEqual(row["turn"]["status"], "finished")
            self.assertEqual(row["turn"]["llm_usage"]["total_tokens"], 26)
            self.assertEqual(row["turn"]["llm_usage"]["cache_read_tokens"], 12)
            self.assertNotIn("不应出现在控制中枢的原文", attention.text)

            memory_routing = client.get("/debug/hub/memory-routing?limit=10")
            self.assertEqual(memory_routing.status_code, 200)
            memory_row = memory_routing.json()["items"][0]
            self.assertEqual(memory_row["history_mode"], "recent")
            self.assertEqual(memory_row["retrieved_count"], 2)
            self.assertEqual(memory_row["injected_chars"], 84)
            self.assertEqual(memory_row["local_signals"], ["reply_chain", "topic_continuation"])
            self.assertNotIn("这句检索内容不应被前端读取", memory_routing.text)

            self.assertEqual(client.post("/debug/hub/state").status_code, 405)

    def test_agent_control_and_model_services_are_local_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _settings(root)
            command_state = QQCommandState()
            permission_audits = []

            class PermissionAudit:
                def audit(self, operation, **payload):
                    permission_audits.append((operation, payload))

            class Registry:
                def names(self):
                    return ("private_file_write",)

                def view(self):
                    return {"private_file_write": SimpleNamespace(config=PermissionAudit())}

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
                    tool_host=SimpleNamespace(registry=Registry()),
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

            missing_ack = client.post(
                "/control-center/agent/control",
                json={"action": "permission", "mode": "full"},
            )
            self.assertEqual(missing_ack.status_code, 400)
            self.assertEqual(command_state.permission_mode(), "traditional")

            malformed_ack = client.post(
                "/control-center/agent/control",
                json={"action": "permission", "mode": "ultra", "acknowledge_risk": "false"},
            )
            self.assertEqual(malformed_ack.status_code, 400)
            self.assertEqual(command_state.permission_mode(), "traditional")

            read_only = client.post(
                "/control-center/agent/control",
                json={"action": "permission", "mode": "readonly"},
            )
            self.assertEqual(read_only.status_code, 200)
            self.assertEqual(command_state.permission_mode(), "readonly")

            full_access = client.post(
                "/control-center/agent/control",
                json={"action": "permission", "mode": "full", "acknowledge_risk": True},
            )
            self.assertEqual(full_access.status_code, 200)
            self.assertEqual(command_state.permission_mode(), "full")
            self.assertEqual(permission_audits[-1][0], "agent_permission_mode_changed")
            self.assertEqual(permission_audits[-1][1]["permission_mode"], "full")

    def test_desktop_pet_control_center_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _settings(root)
            trace_store = TurnTraceStore()
            trace_id = trace_store.begin(
                SimpleNamespace(
                    conversation_type="desktop",
                    conversation_id="desktop_pet",
                    primary_message_id="pet-msg-1",
                    message_ids=["pet-msg-1"],
                    combined_text="真红早上好",
                )
            )
            trace_store.stage(
                trace_id,
                "memory",
                memory_status="available",
                history_mode="none",
                retrieved_count=0,
                injected_chars=0,
            )
            trace_store.stage(
                trace_id,
                "llm_usage",
                channel="chat",
                calls=1,
                input_tokens=150,
                output_tokens=30,
                total_tokens=180,
            )
            trace_store.finish(
                trace_id,
                agent_status="completed",
                delivery_status="sent",
                sent=True,
                reason="desktop_direct_reply",
                reply_chars=12,
            )
            assembly = SimpleNamespace(
                diagnostics=lambda: {},
                bridge=SimpleNamespace(
                    runtime=None,
                    memory_service=None,
                    sender=None,
                    allowed_tool_names=set(),
                    max_rounds=4,
                    history_turns=8,
                    recent_context_messages=5,
                    command_state=QQCommandState(),
                    trace_store=trace_store,
                ),
            )
            client = TestClient(create_app(settings, qq_agent_assembly=assembly))

            # 1. State includes pet snapshot
            state_res = client.get("/debug/hub/state")
            self.assertEqual(state_res.status_code, 200)
            state_data = state_res.json()
            self.assertIn("pet", state_data)
            self.assertEqual(state_data["pet"]["character_name"], "二阶堂真红")
            self.assertIn("风之滨", state_data["pet"]["worldview"])
            self.assertIsNotNone(state_data["pet"]["last_turn"])
            self.assertEqual(state_data["pet"]["last_turn"]["trace_id"], trace_id)

            # 2. GET /control-center/pet endpoint
            pet_res = client.get("/control-center/pet")
            self.assertEqual(pet_res.status_code, 200)
            self.assertTrue(pet_res.json()["ok"])
            self.assertEqual(pet_res.json()["pet"]["window"]["width"], 340)

            # 3. POST /control-center/pet/control endpoint
            status_ctrl = client.post("/control-center/pet/control", json={"action": "status"})
            self.assertEqual(status_ctrl.status_code, 200)
            self.assertEqual(status_ctrl.json()["status"], "status")

            invalid_ctrl = client.post("/control-center/pet/control", json={"action": "invalid_xyz"})
            self.assertEqual(invalid_ctrl.status_code, 400)

            # 4. Traces endpoint contains desktop pet trace
            traces_res = client.get("/debug/hub/traces")
            self.assertEqual(traces_res.status_code, 200)
            traces = traces_res.json()["traces"]
            self.assertTrue(any(t.get("conversation_type") == "desktop" for t in traces))


if __name__ == "__main__":
    unittest.main()
