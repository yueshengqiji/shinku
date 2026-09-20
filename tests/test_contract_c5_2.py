from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shinku.memory import MemoryRouter, MemoryService, MemoryStore
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.turns import QQTurn


class _Message:
    def __init__(self, text: str) -> None:
        self.text = text
        self.has_media = False


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {"kind": "final", "text": "收到。"}


class IndependentMemoryGrayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.memory = MemoryService(
            store=MemoryStore(Path(self.tmp.name) / "memory.sqlite3"),
            router=MemoryRouter(),
            clock=lambda: 1_800_000_000,
        )
        self.runtime = _Runtime()
        self.bridge = QQAgentBridge(
            runtime=self.runtime,
            system_prompt="独立主人设",
            memory_service=self.memory,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def turn(message_id: str, text: str) -> QQTurn:
        message = _Message(text)
        return QQTurn(
            conversation_id="g-gray",
            conversation_type="group",
            primary_message_id=message_id,
            message_ids=(message_id,),
            combined_text=text,
            messages=(message,),
        )

    def test_bridge_memory_lifecycle_is_on_demand_and_scoped(self) -> None:
        self.bridge.handle_turn(self.turn("m-1", "请记住：我喜欢安静的环境"))
        self.bridge.handle_turn(self.turn("m-2", "今天继续做项目。"))
        ordinary_prompt = str(self.runtime.calls[-1]["user_prompt"])
        self.assertNotIn("我喜欢安静的环境", ordinary_prompt)

        self.bridge.handle_turn(self.turn("m-3", "你还记得我喜欢什么吗？"))
        recall_prompt = str(self.runtime.calls[-1]["user_prompt"])
        self.assertIn("我喜欢安静的环境", recall_prompt)
        self.assertIn("source=turn:g-gray:m-1", recall_prompt)

        forget = self.memory.record_turn(
            conversation_id="g-gray",
            conversation_type="group",
            message_id="m-4",
            text="忘记我喜欢安静的环境",
        )
        self.assertIsNotNone(forget)
        self.bridge.handle_turn(self.turn("m-5", "你还记得我喜欢什么吗？"))
        forgotten_prompt = str(self.runtime.calls[-1]["user_prompt"])
        self.assertNotIn("我喜欢安静的环境", forgotten_prompt)

        other_scope = self.memory.prepare_turn(
            conversation_id="g-other",
            conversation_type="group",
            user_message="你还记得我喜欢什么吗？",
        )
        self.assertEqual(other_scope.records, ())


if __name__ == "__main__":
    unittest.main()
