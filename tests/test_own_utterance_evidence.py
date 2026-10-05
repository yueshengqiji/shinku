from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn


class OwnUtteranceEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.now = [1_800_000_000]
        self.service = MemoryService(
            store=MemoryStore(Path(self.tmp.name) / "memory.sqlite3"),
            router=MemoryRouter(),
            policy=MemoryPolicy(master_user_ids=("owner",)),
            clock=lambda: self.now[0],
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _assistant(self, mid: str, text: str, *, seconds_ago: int) -> None:
        self.now[0] = 1_800_000_000 - seconds_ago
        self.service.record_turn(
            conversation_id="g-1", conversation_type="group", message_id=mid,
            text=text, role="assistant", metadata={
                "sender_id": "bot", "requester_id": "owner", "shinku_involved": True,
            },
        )
        self.now[0] = 1_800_000_000

    def _claim(self, text: str) -> None:
        self.service.store.record_event(
            conversation_id="g-1", scope="group:g-1", text=text,
            source_id="claim", layer="observation", created_at=self.now[0] - 1,
            metadata={"sender_id": "other", "sender_name": "塞西莉亚"},
        )

    def _context(self):
        return self.service.prepare_turn(
            conversation_id="g-1", conversation_type="group",
            user_message="既然有人买单，那就这样吧",
            recent_context="真红刚才说过什么？",
            message_meta={"sender_id": "other", "topic_continuation": True},
        )

    def test_pancake_claim_recovers_original_and_not_unrelated_voice(self) -> None:
        self._assistant("pancake", "@曰生岂几 我要一份松饼。既然是你买单，那我就不跟你客气了。", seconds_ago=11 * 60)
        self._assistant("unrelated", "……真拿你没办法，今天可真够吵的。", seconds_ago=2 * 60)
        self._claim("啊，还有真红小姐刚才点名要的一份松饼，主人可千万别漏掉啦~")

        context = self._context()
        self.assertEqual(context.decision.history_mode, "recent")
        self.assertEqual(context.audit["own_utterance_count"], 1)
        self.assertIn("我要一份松饼", context.rendered)
        self.assertIn("【真红本人近期发言原文核对】", context.rendered)
        self.assertNotIn("真拿你没办法", context.rendered)
        self.assertEqual(context.injected_chars, len(context.rendered))

    def test_false_third_party_attribution_does_not_invent_own_statement(self) -> None:
        self._assistant("pancake", "我要一份松饼。", seconds_ago=11 * 60)
        self._claim("真红小姐刚才点了一份蛋糕，还说自己要买单。")

        context = self._context()
        self.assertEqual(context.audit["own_utterance_count"], 0)
        self.assertNotIn("【真红本人近期发言原文核对】", context.rendered)
        self.assertIn("转述只是待核实的说法", context.rendered)

    def test_bridge_records_bot_as_speaker_but_owner_still_eligible(self) -> None:
        event = {
            "post_type": "message", "message_type": "group", "group_id": "g-1",
            "user_id": "owner", "self_id": "bot", "message_id": "request-1",
            "sender": {"nickname": "主人"},
            "message": [{"type": "text", "data": {"text": "我要一份松饼"}}],
        }
        message = IncomingMessage.from_event(event)
        turn = QQTurn(
            conversation_id=message.conversation_id, conversation_type="group",
            primary_message_id=message.message_id, message_ids=(message.message_id,),
            combined_text=message.text, messages=(message,),
        )
        class Runtime:
            def call_chat_json(self, **kwargs):
                return {"text": "test"}

        bridge = QQAgentBridge(runtime=Runtime(), system_prompt="test", memory_service=self.service)
        bridge._record_assistant_reply(turn, "我要一份松饼")

        records = self.service.store.recent_assistant_events(scope="group:g-1", now=self.now[0])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].metadata["sender_id"], "bot")
        self.assertEqual(records[0].metadata["requester_id"], "owner")
        owner = self.service.store.recent_assistant_events(scope="private:master", now=self.now[0])
        self.assertEqual(len(owner), 1)
        self.assertTrue(owner[0].source_id.startswith("owner:"))
        owner_handoff = self.service.recent_owner_handoff(requester_id="owner")
        self.assertTrue(any("我要一份松饼" in item.text for item in owner_handoff))


if __name__ == "__main__":
    unittest.main()
