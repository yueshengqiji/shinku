"""QQ Agent 回合追踪的脱敏与容量契约。"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from shinku.qq.agent_bridge import QQAgentReply
from shinku.qq.trace import TurnTraceStore


class QQTraceTests(unittest.TestCase):
    def test_ring_buffer_and_sensitive_values_are_not_persisted(self) -> None:
        store = TurnTraceStore(max_items=1)
        turn = SimpleNamespace(
            conversation_type="group",
            conversation_id="group-1",
            primary_message_id="message-1",
            combined_text="用户原始秘密内容",
        )

        trace_id = store.begin(turn)
        store.stage(
            trace_id,
            "memory",
            history_mode="recent",
            retrieved_count=2,
            injected_chars=42,
            prompt="不要保存这段提示词",
            content="不要保存这段原文",
        )
        store.finish(
            trace_id,
            QQAgentReply("completed", final_text="最终回复", sent=False, reason="dry_run"),
        )

        row = store.snapshot()[0]
        serialized = json.dumps(row, ensure_ascii=False)
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["delivery_status"], "not_sent")
        self.assertEqual(row["input_chars"], len(turn.combined_text))
        self.assertNotIn("用户原始秘密内容", serialized)
        self.assertNotIn("不要保存这段提示词", serialized)
        self.assertNotIn("不要保存这段原文", serialized)
        self.assertNotIn("prompt", row["stages"][0])
        self.assertNotIn("content", row["stages"][0])

    def test_snapshot_is_newest_first_and_bounded(self) -> None:
        store = TurnTraceStore(max_items=2)
        for index in range(3):
            trace_id = store.begin(SimpleNamespace(
                conversation_type="private",
                conversation_id="user-1",
                primary_message_id=f"message-{index}",
                combined_text=str(index),
            ))
            store.finish(trace_id, agent_status="command", reason=str(index))

        rows = store.snapshot(limit=10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["reason"], "2")
        self.assertEqual(rows[1]["reason"], "1")


if __name__ == "__main__":
    unittest.main()
