import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from shinku import names
from shinku.api import create_app
from shinku.config import load_settings
from shinku.hosts.tool_host import ToolHost
from shinku.memory import MemoryRouter, MemoryService, MemoryStore
from shinku.agent.self_state import SelfStateTool
from shinku.qq.host import NapCatHost
from shinku.qq.commands import QQCommandState
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry


def _settings(root: Path):
    return load_settings(
        {
            names.DATA_ROOT_ENV: str(root / "data"),
            names.CONFIG_ROOT_ENV: str(root / "config"),
            names.LOG_ROOT_ENV: str(root / "logs"),
            names.env_key("backend"): "19998",
        }
    )


class P2ManagementApiTests(unittest.TestCase):
    def test_observability_and_qq_diagnostics_are_local_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            class Runtime:
                def snapshot_metrics(self):
                    return {"chat_json_calls": 2}

                def metrics_persistence_metadata(self):
                    return {"persistent": False}

                def snapshot_last_error(self):
                    return {}

                def recent_metric_days(self, days):
                    return [{"day": "today", "calls": days}]

            registry = ToolRegistry()
            registry.register(SelfStateTool(lambda: {"runtime": {"model": "test"}}))
            bridge = SimpleNamespace(
                runtime=Runtime(),
                command_state=QQCommandState(),
                tool_host=ToolHost(registry),
                memory_service=None,
                sender=None,
                allowed_tool_names=set(),
                policy=ExecutionPolicy(risk_by_tool={"read_self_state": "low"}),
                approval_store=None,
            )
            assembly = SimpleNamespace(bridge=bridge, task_coordinator=None)
            host = NapCatHost(quote_fetcher=lambda message_id: None)
            client = TestClient(
                create_app(
                    _settings(root),
                    qq_agent_assembly=assembly,
                    napcat_host=host,
                    napcat_config=SimpleNamespace(diagnostics=lambda: {"valid": True}, validate=lambda: ()),
                )
            )
            self.assertEqual(client.get("/health/live").json()["status"], "ok")
            self.assertEqual(client.get("/metrics").json()["runtime"]["counters"]["chat_json_calls"], 2)
            self.assertEqual(client.get("/metrics/recent?days=3").json()["days"][0]["calls"], 3)
            self.assertTrue(client.get("/api/qq/napcat/status").json()["configured"])
            self.assertTrue(client.get("/capabilities").json()["capabilities"])

    def test_memory_timeline_facts_rollup_and_explicit_forget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = MemoryService(
                store=MemoryStore(root / "memory.sqlite3"),
                router=MemoryRouter(),
                clock=lambda: 1_800_000_000,
            )
            service.record_turn(
                conversation_id="g-1",
                conversation_type="group",
                message_id="m-1",
                text="请记住：我喜欢蓝色。",
            )
            service.observe_message(
                SimpleNamespace(
                    conversation_id="g-1",
                    conversation_type="group",
                    message_id="m-2",
                    sender_id="u-1",
                    text="今天讨论了图片问题",
                    has_media=False,
                    timestamp=1_800_000_000,
                    raw_event={"sender": {"nickname": "甲"}},
                    segments=[],
                )
            )
            bridge = SimpleNamespace(
                runtime=SimpleNamespace(snapshot_metrics=lambda: {}, snapshot_last_error=lambda: {}),
                command_state=QQCommandState(),
                tool_host=ToolHost(ToolRegistry()),
                memory_service=service,
                sender=None,
                allowed_tool_names=set(),
                policy=ExecutionPolicy(),
                approval_store=None,
            )
            assembly = SimpleNamespace(bridge=bridge, task_coordinator=None)
            client = TestClient(create_app(_settings(root), qq_agent_assembly=assembly))
            timeline = client.get("/memory/timeline", params={"conversation_id": "g-1", "conversation_type": "group"})
            self.assertEqual(timeline.status_code, 200)
            self.assertTrue(timeline.json()["records"])
            facts = client.get("/memory/facts", params={"conversation_id": "g-1", "conversation_type": "group"})
            self.assertEqual(facts.status_code, 200)
            self.assertTrue(facts.json()["facts"])
            rollup = client.post("/memory/rollup", json={"scope": "group:g-1"})
            self.assertEqual(rollup.status_code, 200)
            self.assertEqual(client.post("/memory/forget", json={"conversation_id": "g-1", "conversation_type": "group", "query": "蓝色"}).status_code, 400)
            forgotten = client.post(
                "/memory/forget",
                json={"conversation_id": "g-1", "conversation_type": "group", "query": "蓝色", "confirm": True},
            )
            self.assertEqual(forgotten.status_code, 200)
            self.assertGreaterEqual(forgotten.json()["forgotten"], 1)


if __name__ == "__main__":
    unittest.main()
