from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shinku.memory import MemoryRouter, MemoryService, MemoryStore
from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.host import NapCatHost


class _Message:
    def __init__(self, message_id: str, text: str, *, timestamp: float = 1_800_000_000.0, reply_to: str = "") -> None:
        self.message_id = message_id
        self.sender_id = "u-1"
        self.conversation_id = "g-1"
        self.conversation_type = "group"
        self.text = text
        self.has_media = False
        self.timestamp = timestamp
        self.raw_event = {"sender": {"nickname": "测试成员"}}
        self.segments = []
        if reply_to:
            self.segments.append(type("Segment", (), {"kind": "reply", "payload": {"data": {"id": reply_to}}})())


class MemoryContextModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.service = MemoryService(
            store=MemoryStore(Path(self.tmp.name) / "memory.sqlite3"),
            router=MemoryRouter(),
            clock=lambda: 1_800_000_000,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_unaddressed_group_message_is_observed_and_continuation_is_recent(self) -> None:
        first = _Message("m-1", "图片加载失败了")
        second = _Message("m-2", "图片还是加载失败", reply_to="m-1")
        self.assertTrue(self.service.observe_message(first))
        self.assertTrue(self.service.observe_message(second))

        records = self.service.store.search(query="图片加载", scope="group:g-1")
        self.assertTrue(records)
        self.assertTrue(any(item.metadata["reply_to_message_id"] == "m-1" for item in records))
        self.assertTrue(any(item.metadata["sender_name"] == "测试成员" for item in records))

        context = self.service.prepare_turn(
            conversation_id="g-1",
            conversation_type="group",
            user_message="这个问题继续看一下",
            recent_context="图片加载失败了\n图片还是加载失败",
            message_meta={"message_id": "m-2"},
        )
        self.assertEqual(context.decision.history_mode, "recent")
        self.assertFalse(context.decision.used_model)
        self.assertGreaterEqual(context.retrieved_count, 2)
        self.assertIn("当前话题的最近消息", context.rendered)
        audit = self.service.store.recent_routing_audits(scope="group:g-1", limit=1)[0]
        self.assertEqual(audit["history_mode"], "recent")
        self.assertEqual(audit["retrieved_count"], context.retrieved_count)
        self.assertEqual(audit["injected_chars"], context.injected_chars)

    def test_plain_message_does_not_call_model_judge(self) -> None:
        calls: list[dict[str, object]] = []

        def judge(**kwargs):
            calls.append(kwargs)
            return {"need_memory": True, "history_mode": "retrieve", "confidence": 0.9}

        router = MemoryRouter(judge=judge)
        decision = router.decide(user_message="今天继续做项目。", conversation_type="group")
        self.assertEqual(decision.history_mode, "none")
        self.assertEqual(calls, [])

    def test_summary_mode_rolls_up_completed_observation_day(self) -> None:
        old = 1_800_000_000 - 86400
        self.service.store.record_event(
            conversation_id="g-1",
            scope="group:g-1",
            text="最后决定采用方案B",
            source_id="turn:g-1:old-1",
            layer="observation",
            created_at=old,
            metadata={"message_id": "old-1", "sender_name": "测试成员", "sender_id": "u-1"},
        )
        context = self.service.prepare_turn(
            conversation_id="g-1",
            conversation_type="group",
            user_message="上周群里发生了什么？",
        )
        self.assertEqual(context.decision.history_mode, "summary")
        self.assertTrue(context.records)
        self.assertEqual(context.records[0].layer, "summary")
        self.assertIn("方案B", context.rendered)

    def test_recent_context_does_not_reinject_generated_assistant_voice(self) -> None:
        self.service.store.record_event(
            conversation_id="g-1",
            scope="group:g-1",
            text="用户刚才提出的真实问题",
            source_id="recent-user-1",
            layer="episodic",
            created_at=1_799_999_998,
            metadata={"role": "user", "source": "qq_turn"},
        )
        self.service.store.record_event(
            conversation_id="g-1",
            scope="group:g-1",
            text="一条不该被当成口吻范本的冷淡回复",
            source_id="recent-assistant-1",
            layer="episodic",
            created_at=1_799_999_999,
            metadata={"role": "assistant", "source": "qq_turn"},
        )

        context = self.service.prepare_turn(
            conversation_id="g-1",
            conversation_type="group",
            user_message="这个问题继续看一下",
            recent_context="用户刚才提出的真实问题",
            message_meta={"topic_continuation": True},
        )

        self.assertEqual(context.decision.history_mode, "recent")
        self.assertIn("用户刚才提出的真实问题", context.rendered)
        self.assertNotIn("冷淡回复", context.rendered)

        # Explicit retrieval remains able to include assistant history when
        # the user actually asks what Shinku said before.
        all_recent = self.service.store.recent_events(
            scope="group:g-1",
            now=1_800_000_000,
            include_assistant=True,
        )
        self.assertTrue(any(item.metadata.get("role") == "assistant" for item in all_recent))

    def test_host_observes_before_reply_route(self) -> None:
        order: list[str] = []

        class OrderedIngress(QQIngressAdapter):
            def ingest_message(self, message, *, at):
                order.append("route")
                return super().ingest_message(message, at=at)

        host = NapCatHost(
            ingress=OrderedIngress(),
            passive_observer=lambda message: order.append("observe"),
            clock=lambda: 1.0,
        )
        host.handle_event(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "user_id": "u-1",
                "message_id": "m-1",
                "raw_message": "未@真红的群消息",
            },
            at=1.0,
        )
        self.assertEqual(order, ["observe", "route"])


if __name__ == "__main__":
    unittest.main()
