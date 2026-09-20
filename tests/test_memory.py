from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore


class MemoryLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.service = MemoryService(
            store=MemoryStore(Path(self.tmp.name) / "memory.sqlite3"),
            router=MemoryRouter(),
            clock=lambda: 1_800_000_000,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_normal_turn_does_not_retrieve_history(self) -> None:
        self.service.record_turn(
            conversation_id="g-1",
            conversation_type="group",
            message_id="m-1",
            text="请记住：我喜欢安静的环境",
        )
        context = self.service.prepare_turn(
            conversation_id="g-1",
            conversation_type="group",
            user_message="今天继续做项目。",
        )
        self.assertFalse(context.decision.need_retrieval)
        self.assertEqual(context.records, ())

    def test_explicit_recall_retrieves_only_current_scope(self) -> None:
        self.service.record_turn(
            conversation_id="g-1",
            conversation_type="group",
            message_id="m-1",
            text="请记住：我喜欢安静的环境",
        )
        context = self.service.prepare_turn(
            conversation_id="g-1",
            conversation_type="group",
            user_message="你还记得我喜欢什么吗？",
        )
        self.assertTrue(context.decision.need_retrieval)
        self.assertTrue(any("安静" in item.text for item in context.records))

    def test_explicit_forget_removes_records_from_future_retrieval(self) -> None:
        self.service.record_turn(
            conversation_id="g-1",
            conversation_type="group",
            message_id="m-1",
            text="请记住：我喜欢安静的环境",
        )
        report = self.service.record_turn(
            conversation_id="g-1",
            conversation_type="group",
            message_id="m-2",
            text="忘记我喜欢安静的环境",
        )
        self.assertIsNotNone(report)
        self.assertEqual(report.forgotten_count, 1)
        context = self.service.prepare_turn(
            conversation_id="g-1",
            conversation_type="group",
            user_message="你还记得我喜欢什么吗？",
        )
        self.assertEqual(context.records, ())

    def test_ambiguous_turn_can_use_model_judge(self) -> None:
        calls: list[dict[str, str]] = []

        def judge(**kwargs):
            calls.append(kwargs)
            return {
                "need_memory": True,
                "memory_type": "event",
                "scope": "group",
                "query": "上次的图片问题",
                "confidence": 0.9,
            }

        router = MemoryRouter(judge=judge)
        decision = router.decide(
            user_message="那个问题后来怎么样了？",
            recent_context="我们刚才讨论过图片加载。",
            conversation_type="group",
        )
        self.assertTrue(decision.need_retrieval)
        self.assertTrue(decision.used_model)
        self.assertEqual(len(calls), 1)

    def test_memory_policy_can_change_limits_and_ttl_without_code_changes(self) -> None:
        policy = MemoryPolicy.from_environment(
            {
                "SHINKU_MEMORY_EVENT_TTL_SECONDS": "60",
                "SHINKU_MEMORY_RETRIEVAL_LIMIT": "2",
                "SHINKU_MEMORY_MAX_QUERY_TERMS": "4",
                "SHINKU_MEMORY_DETERMINISTIC_GUARDS": "false",
            }
        )
        self.assertEqual(policy.event_ttl_seconds, 60)
        self.assertEqual(policy.retrieval_limit, 2)
        self.assertEqual(policy.max_query_terms, 4)
        self.assertFalse(policy.deterministic_guards)

    def test_deterministic_memory_shortcuts_can_be_disabled(self) -> None:
        calls: list[dict[str, str]] = []

        def judge(**kwargs):
            calls.append(kwargs)
            return {"need_memory": False, "confidence": 0.9}

        router = MemoryRouter(judge=judge, deterministic_guards=False)
        decision = router.decide(
            user_message="你还记得昨天发生了什么吗？",
            conversation_type="private",
        )
        self.assertFalse(decision.need_retrieval)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
