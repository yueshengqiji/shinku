import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from shinku.memory import MemoryRouter, MemoryService, MemoryStore
from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.attention import AttentionBatcher
from shinku.qq.group_context import GroupContextTracker
from shinku.qq.host import NapCatHost
from shinku.qq.ingress_guard import IngressPolicy


class P1IngressAndContextTests(unittest.TestCase):
    def test_duplicate_stale_and_self_events_are_rejected_before_batching(self) -> None:
        adapter = QQIngressAdapter(
            batcher=AttentionBatcher(window_seconds=0),
            ingress_policy=IngressPolicy(event_max_age_seconds=10),
        )
        event = {
            "post_type": "message",
            "message_type": "private",
            "user_id": "u-1",
            "message_id": "m-1",
            "message": [{"type": "text", "data": {"text": "你好"}}],
        }
        first = adapter.ingest(event, at=100)
        duplicate = adapter.ingest(event, at=101)
        stale = adapter.ingest({**event, "message_id": "m-old", "time": 1}, at=100)
        self_event = adapter.ingest(
            {**event, "message_id": "m-self", "self_id": "u-1", "time": 100},
            at=100,
        )
        self.assertTrue(first.admitted)
        self.assertEqual(duplicate.admission.reason, "duplicate_event")
        self.assertEqual(stale.admission.reason, "stale_event")
        self.assertEqual(self_event.admission.reason, "self_event")
        self.assertIsNone(duplicate.closed_batch)

    def test_concurrent_duplicate_webhooks_are_admitted_only_once(self) -> None:
        observed = []

        class Dispatcher:
            def __init__(self) -> None:
                self.scheduled = 0
                self.lock = threading.Lock()

            def submit(self, result) -> None:
                if result.scheduled:
                    with self.lock:
                        self.scheduled += 1

        dispatcher = Dispatcher()
        host = NapCatHost(
            ingress=QQIngressAdapter(),
            passive_observer=observed.append,
            turn_dispatcher=dispatcher,
        )
        event = {
            "post_type": "message",
            "message_type": "private",
            "user_id": "u-1",
            "message_id": "m-concurrent-1",
            "time": 100,
            "message": [{"type": "text", "data": {"text": "同一条 webhook 重放"}}],
        }
        workers = 16
        barrier = threading.Barrier(workers)

        def ingest_once(_index: int) -> str:
            barrier.wait(timeout=5)
            return host.handle_event(event, at=100).ingress.admission.reason

        with ThreadPoolExecutor(max_workers=workers) as pool:
            reasons = list(pool.map(ingest_once, range(workers)))

        self.assertEqual(reasons.count("accepted"), 1)
        self.assertEqual(reasons.count("duplicate_event"), workers - 1)
        self.assertEqual(len(observed), 1)
        self.assertEqual(dispatcher.scheduled, 1)

    def test_poke_can_survive_old_notice_timestamp(self) -> None:
        adapter = QQIngressAdapter(
            ingress_policy=IngressPolicy(event_max_age_seconds=1),
        )
        result = adapter.ingest(
            {
                "post_type": "notice",
                "notice_type": "group_poke",
                "group_id": "g-1",
                "user_id": "u-2",
                "target_id": "bot-1",
                "self_id": "bot-1",
                "time": 1,
            },
            at=100,
        )
        self.assertTrue(result.admitted)
        self.assertTrue(result.route.should_reply)

    def test_unaddressed_messages_feed_local_group_snapshot_without_reply(self) -> None:
        adapter = QQIngressAdapter(
            batcher=AttentionBatcher(window_seconds=0),
            group_context=GroupContextTracker(ttl_seconds=120),
        )
        first = adapter.ingest(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "user_id": "u-a",
                "message_id": "g-1",
                "sender": {"user_id": "u-a", "card": "甲"},
                "message": [{"type": "text", "data": {"text": "我们先看图片问题"}}],
            },
            at=10,
        )
        second = adapter.ingest(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "user_id": "u-b",
                "self_id": "bot-1",
                "message_id": "g-2",
                "sender": {"user_id": "u-b", "nickname": "乙"},
                "message": [
                    {"type": "at", "data": {"qq": "bot-1"}},
                    {"type": "text", "data": {"text": "刚才那个怎么处理？"}},
                ],
            },
            at=11,
        )
        self.assertFalse(first.route.should_reply)
        self.assertTrue(second.route.should_reply)
        context = second.message.raw_event.get("_shinku_group_context")
        self.assertEqual(context["items"][-1]["sender_label"], "甲")
        self.assertIn("我们先看图片问题", context["items"][-1]["text"])

    def test_host_fetches_reply_before_observation_and_visual_bridge(self) -> None:
        observed = []

        def fetch(message_id: str):
            self.assertEqual(message_id, "quoted-1")
            return {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "user_id": "u-a",
                "message_id": "quoted-1",
                "sender": {"user_id": "u-a", "nickname": "甲"},
                "message": [
                    {"type": "text", "data": {"text": "原图"}},
                    {"type": "image", "data": {"url": "data:image/png;base64,aGVsbG8="}},
                ],
            }

        host = NapCatHost(passive_observer=observed.append, quote_fetcher=fetch)
        result = host.handle_event(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "user_id": "u-b",
                "self_id": "bot-1",
                "message_id": "m-2",
                "message": [
                    {"type": "reply", "data": {"id": "quoted-1"}},
                    {"type": "text", "data": {"text": "看这个"}},
                ],
            },
            at=10,
        )
        self.assertTrue(result.quote_fetched)
        self.assertEqual(result.ingress.message.quoted_text, "原图")
        self.assertEqual(len(result.visual.inputs), 1)
        self.assertTrue(result.visual.ready)
        self.assertEqual(observed[0].quoted_message["message_id"], "quoted-1")

    def test_unaddressed_group_message_is_persisted_and_later_addressed_turn_recalls_it(self) -> None:
        base_time = 1_800_000_000
        with tempfile.TemporaryDirectory() as root:
            service = MemoryService(
                store=MemoryStore(Path(root) / "memory.sqlite3"),
                router=MemoryRouter(),
                clock=lambda: base_time + 1,
            )
            ingress = QQIngressAdapter(batcher=AttentionBatcher(window_seconds=0))
            host = NapCatHost(
                ingress=ingress,
                passive_observer=service.observe_message,
                clock=lambda: base_time,
            )

            unaddressed = host.handle_event(
                {
                    "post_type": "message",
                    "message_type": "group",
                    "group_id": "g-1",
                    "user_id": "u-a",
                    "self_id": "bot-1",
                    "message_id": "m-1",
                    "time": base_time,
                    "sender": {"user_id": "u-a", "nickname": "甲"},
                    "message": [{"type": "text", "data": {"text": "图片一直加载失败，我已经试过重启"}}],
                },
                at=base_time,
            )

            self.assertFalse(unaddressed.ingress.route.should_reply)
            self.assertFalse(unaddressed.ingress.attention.should_schedule)
            saved = service.store.observation_for_message(scope="group:g-1", message_id="m-1")
            self.assertIsNotNone(saved)
            self.assertIn("图片一直加载失败", saved.text)

            addressed = host.handle_event(
                {
                    "post_type": "message",
                    "message_type": "group",
                    "group_id": "g-1",
                    "user_id": "u-b",
                    "self_id": "bot-1",
                    "message_id": "m-2",
                    "time": base_time + 1,
                    "sender": {"user_id": "u-b", "nickname": "乙"},
                    "message": [
                        {"type": "reply", "data": {"id": "m-1"}},
                        {"type": "at", "data": {"qq": "bot-1"}},
                        {"type": "text", "data": {"text": "刚才那个怎么处理？"}},
                    ],
                },
                at=base_time + 1,
            )

            self.assertTrue(addressed.ingress.route.should_reply)
            self.assertTrue(addressed.ingress.attention.should_schedule)
            group_snapshot = addressed.ingress.message.raw_event["_shinku_group_context"]
            self.assertIn("图片一直加载失败", group_snapshot["items"][0]["text"])

            memory_context = service.prepare_turn(
                conversation_id="g-1",
                conversation_type="group",
                user_message=addressed.ingress.message.text,
                # Match the bridge's actual short transcript: it contains
                # this turn, while the historical evidence comes from memory.
                recent_context="乙：刚才那个怎么处理？",
                message_meta={"message_id": "m-2", "sender_id": "u-b"},
            )
            self.assertEqual(memory_context.decision.history_mode, "recent")
            self.assertFalse(memory_context.decision.used_model)
            self.assertIn("图片一直加载失败", memory_context.rendered)


if __name__ == "__main__":
    unittest.main()
