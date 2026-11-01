"""桌宠专属 API 的契约测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from shinku import names
from shinku.api import create_app
from shinku.config import load_settings
from shinku.qq.agent_bridge import QQAgentReply


def _settings(root: Path):
    return load_settings(
        {
            names.DATA_ROOT_ENV: str(root / "data"),
            names.CONFIG_ROOT_ENV: str(root / "config"),
            names.LOG_ROOT_ENV: str(root / "logs"),
            names.env_key("backend"): "19998",
        }
    )


class PetApiTests(unittest.TestCase):
    def test_pet_status_and_live2d(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _settings(root)

            app = create_app(settings)
            client = TestClient(app)

            # 1. 状态接口
            res = client.get("/api/pet/status")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertTrue(data.get("ok"))
            self.assertEqual(data.get("status"), "online")
            self.assertIn("character", data)
            self.assertEqual(data["character"]["name"], "真红")
            self.assertIn("local_click_lines", data["character"]["dialogue"])
            self.assertIn("live2d", data["character"])

            # 2. Live2D 资产端点保护测试
            live2d_res = client.get("/api/pet/live2d/non_existent.model3.json")
            self.assertEqual(live2d_res.status_code, 404)

    def test_pet_chat_with_mock_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = _settings(root)

            class MockBridge:
                master_user_ids = frozenset(["master_123"])

                def handle_turn(self, turn):
                    return QQAgentReply("completed", final_text="嗯，我在呢。风之滨今天天气不错。")

            mock_assembly = SimpleNamespace(bridge=MockBridge())

            app = create_app(settings, qq_agent_assembly=mock_assembly)
            client = TestClient(app)

            # 1. 空内容拒绝
            empty_res = client.post("/api/pet/chat", json={"text": ""})
            self.assertEqual(empty_res.status_code, 400)

            # 2. 正常聊天交互
            chat_res = client.post("/api/pet/chat", json={"text": "你好真红", "conversation_id": "desktop_pet"})
            self.assertEqual(chat_res.status_code, 200)
            data = chat_res.json()
            self.assertTrue(data.get("ok"))
            self.assertEqual(data.get("reply"), "嗯，我在呢。风之滨今天天气不错。")
            self.assertIn("expression", data)
            self.assertIn("motion", data)
            self.assertIn("emotion", data)
            self.assertTrue(data.get("trace_id", "").startswith("desktop_"))


if __name__ == "__main__":
    unittest.main()
