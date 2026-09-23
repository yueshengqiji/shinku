from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore


class OwnerMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.service = MemoryService(
            store=MemoryStore(Path(self.tmp.name) / "memory.sqlite3"),
            router=MemoryRouter(),
            policy=MemoryPolicy(master_user_ids=("u-master",)),
            clock=lambda: 1_800_000_000,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _record(self, *, conversation_id: str, conversation_type: str, message_id: str, text: str, sender_id: str, shinku: bool = True) -> None:
        self.service.record_turn(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            message_id=message_id,
            text=text,
            metadata={
                "sender_id": sender_id,
                "shinku_involved": shinku,
                "at_targets": ["bot-1"] if shinku else [],
            },
        )

    def test_owner_memory_is_shared_between_private_and_group(self) -> None:
        self._record(
            conversation_id="u-master",
            conversation_type="private",
            message_id="private-1",
            text="昨晚我们私聊了模型缓存命中率和上下文压缩。",
            sender_id="u-master",
        )
        self._record(
            conversation_id="g-1",
            conversation_type="group",
            message_id="group-1",
            text="今天群里一起讨论了本地向量检索。",
            sender_id="u-master",
        )

        from_group = self.service.prepare_turn(
            conversation_id="g-1",
            conversation_type="group",
            user_message="你还记得昨晚我们私聊了什么吗？",
            message_meta={"sender_id": "u-master"},
        )
        self.assertTrue(any("模型缓存命中率" in item.text for item in from_group.records))
        self.assertIn("private:master", from_group.audit["retrieval_scopes"])

        from_private = self.service.prepare_turn(
            conversation_id="u-master",
            conversation_type="private",
            user_message="今天群里我们聊了什么？",
            message_meta={"sender_id": "u-master"},
        )
        self.assertTrue(any("本地向量检索" in item.text for item in from_private.records))

    def test_other_group_member_cannot_read_owner_memory(self) -> None:
        self._record(
            conversation_id="u-master",
            conversation_type="private",
            message_id="private-2",
            text="这是只属于我和真红的私聊内容。",
            sender_id="u-master",
        )
        context = self.service.prepare_turn(
            conversation_id="g-1",
            conversation_type="group",
            user_message="你还记得私聊内容吗？",
            message_meta={"sender_id": "u-other"},
        )
        self.assertNotIn("private:master", context.audit.get("retrieval_scopes", []))
        self.assertFalse(any("只属于我和真红" in item.text for item in context.records))

    def test_unrelated_group_chatter_is_not_promoted(self) -> None:
        self._record(
            conversation_id="g-1",
            conversation_type="group",
            message_id="group-2",
            text="我在群里和别人讨论，但真红没有参与。",
            sender_id="u-master",
            shinku=False,
        )
        self.assertFalse(self.service.store.search(query="别人讨论", scope="private:master"))

    def test_recent_owner_handoff_is_owner_only_and_uses_latest_source_conversation(self) -> None:
        current = [1_800_000_000]
        self.service.clock = lambda: current[0]
        self._record(
            conversation_id="u-master",
            conversation_type="private",
            message_id="old-private",
            text="前一个话题：上下文压缩。",
            sender_id="u-master",
        )
        current[0] += 60
        self._record(
            conversation_id="g-1",
            conversation_type="group",
            message_id="recent-group",
            text="刚才和真红讨论了向量检索的结果。",
            sender_id="u-master",
        )
        current[0] += 1
        self.service.record_turn(
            conversation_id="g-1",
            conversation_type="group",
            message_id="recent-reply",
            text="真红补充了本地模型的索引状态。",
            role="assistant",
            metadata={"sender_id": "u-master", "shinku_involved": True},
        )

        handoff = self.service.recent_owner_handoff(requester_id="u-master")
        texts = [item.text for item in handoff]

        self.assertTrue(any("向量检索" in text for text in texts))
        self.assertTrue(any("索引状态" in text for text in texts))
        self.assertFalse(any("上下文压缩" in text for text in texts))
        self.assertEqual(self.service.recent_owner_handoff(requester_id="u-other"), ())

    def test_recent_owner_handoff_excludes_uninvolved_group_chatter(self) -> None:
        self._record(
            conversation_id="g-1",
            conversation_type="group",
            message_id="observer-only",
            text="主人和别人聊了但真红没有参与的内容。",
            sender_id="u-master",
            shinku=False,
        )

        self.assertEqual(self.service.recent_owner_handoff(requester_id="u-master"), ())

    def test_explicit_owner_fact_is_available_cross_context(self) -> None:
        self._record(
            conversation_id="u-master",
            conversation_type="private",
            message_id="private-fact",
            text="请记住：我更喜欢安静的环境。",
            sender_id="u-master",
        )
        context = self.service.prepare_turn(
            conversation_id="g-1",
            conversation_type="group",
            user_message="你记得我的环境偏好吗？",
            message_meta={"sender_id": "u-master"},
        )
        self.assertTrue(any("安静的环境" in item.text for item in context.records))


if __name__ == "__main__":
    unittest.main()
