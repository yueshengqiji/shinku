from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.attention import AttentionBatch, AttentionBatcher, AttentionDecision, AttentionPolicy
from shinku.qq.delivery import DeliveryResult
from shinku.qq.group_context import GroupContextTracker
from shinku.qq.message import IncomingMessage
from shinku.qq.routing import ReplyRoutePolicy
from shinku.qq.turns import QQTurn
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore


def _group_message(
    *,
    message_id: str,
    sender_id: str,
    sender_name: str,
    target_id: str,
    target_name: str,
    text: str,
) -> IncomingMessage:
    return IncomingMessage.from_event(
        {
            "post_type": "message",
            "message_type": "group",
            "group_id": "g-1",
            "user_id": sender_id,
            "self_id": "bot-1",
            "message_id": message_id,
            "sender": {"user_id": sender_id, "nickname": sender_name},
            "message": [
                {"type": "at", "data": {"qq": target_id, "name": target_name}},
                {"type": "text", "data": {"text": text}},
            ],
        }
    )


class QQMessageAttributionTests(unittest.TestCase):
    def test_owner_at_then_other_to_other_keeps_authors_through_batch_prompt_and_memory(self) -> None:
        owner = _group_message(
            message_id="owner-1", sender_id="owner-id", sender_name="主人",
            target_id="bot-1", target_name="真红", text="你想吃什么",
        )
        bystander = _group_message(
            message_id="other-1", sender_id="blue-id", sender_name="blue",
            target_id="cecilia-id", target_name="塞西莉亚", text="是啊，吃什么",
        )
        owner.raw_event["_shinku_group_context"] = {
            "current": {"sender_id": "owner-id", "sender_label": "主人"}, "items": [],
        }
        bystander.raw_event["_shinku_group_context"] = {
            "current": {"sender_id": "blue-id", "sender_label": "blue"}, "items": [],
        }
        batch = AttentionBatch(
            conversation_id="g-1", messages=(owner, bystander),
            decisions=(AttentionDecision(True, True, 90, "mentioned_bot"),
                       AttentionDecision(True, False, 20, "autonomous_probability_hit")),
            primary_index=0,
        )
        self.assertIn("主人（QQ owner-id）：", batch.combined_text)
        self.assertIn("blue（QQ blue-id）：", batch.combined_text)
        turn = QQTurn(
            conversation_id="g-1", conversation_type="group", primary_message_id="owner-1",
            message_ids=("owner-1", "other-1"), combined_text=batch.combined_text,
            messages=batch.messages, attention_mode="immediate",
        )

        transcript = QQAgentBridge._transcript(turn, master_user_ids=("owner-id",))
        self.assertIn("blue → @塞西莉亚：是啊，吃什么", transcript[0]["content"])
        self.assertIn("主人（主人） → @真红：你想吃什么", transcript[-1]["content"])

        class Runtime:
            def __init__(self) -> None:
                self.calls = []

            def call_chat_json(self, **kwargs):
                self.calls.append(kwargs)
                return {"kind": "final", "text": "松饼"}

        with tempfile.TemporaryDirectory() as directory:
            service = MemoryService(
                store=MemoryStore(Path(directory) / "memory.sqlite3"),
                router=MemoryRouter(),
                policy=MemoryPolicy(master_user_ids=("owner-id",)),
            )
            runtime = Runtime()
            bridge = QQAgentBridge(
                runtime=runtime, system_prompt="真红", memory_service=service,
                master_user_ids=("owner-id",), bot_ids=("bot-1",),
            )
            self.assertIn("当前发言者：主人（QQ owner-id）", bridge._group_context(turn))
            self.assertNotIn("当前发言者：blue（QQ blue-id）", bridge._group_context(turn))
            bridge.handle_turn(turn)
            self.assertTrue(runtime.calls)
            self.assertTrue(runtime.calls[-1]["user_prompt"].rstrip().endswith(transcript[-1]["content"]))
            with service.store._connect() as conn:
                rows = conn.execute(
                    "SELECT source_id, content, metadata_json FROM memory_turns WHERE scope = ? AND role = ? ORDER BY source_id",
                    ("group:g-1", "user"),
                ).fetchall()
            self.assertEqual(len(rows), 2)
            by_source = {row["source_id"]: row for row in rows}
            self.assertEqual(by_source["turn:g-1:owner-1"]["content"], owner.text)
            self.assertEqual(json.loads(by_source["turn:g-1:owner-1"]["metadata_json"])["sender_id"], "owner-id")
            self.assertEqual(by_source["turn:g-1:other-1"]["content"], bystander.text)
            self.assertEqual(json.loads(by_source["turn:g-1:other-1"]["metadata_json"])["sender_id"], "blue-id")
            with service.store._connect() as conn:
                owner_rows = conn.execute(
                    "SELECT content FROM memory_turns WHERE scope = ? AND role = ?",
                    ("private:master", "user"),
                ).fetchall()
            self.assertEqual([row["content"] for row in owner_rows], [owner.text])

    def test_transcript_preserves_explicit_recipient_and_image_sender(self) -> None:
        directed = _group_message(
            message_id="m-1",
            sender_id="blue-id",
            sender_name="blue",
            target_id="cecilia-id",
            target_name="塞西莉亚",
            text="那骂我吧",
        )
        image = IncomingMessage.from_event(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "user_id": "cecilia-id",
                "self_id": "bot-1",
                "message_id": "m-2",
                "sender": {"user_id": "cecilia-id", "nickname": "塞西莉亚"},
                "message": [{"type": "image", "data": {"file": "image-1"}}],
            }
        )
        turn = QQTurn(
            conversation_id="g-1",
            conversation_type="group",
            primary_message_id="m-2",
            message_ids=("m-1", "m-2"),
            combined_text="那骂我吧",
            messages=(directed, image),
        )

        transcript = QQAgentBridge._transcript(turn)

        self.assertEqual(transcript[0]["content"], "blue → @塞西莉亚：那骂我吧")
        self.assertIn("塞西莉亚：", transcript[1]["content"])
        self.assertIn("图片附件已收到", transcript[1]["content"])
        self.assertNotIn("那骂我吧", transcript[1]["content"])

    def test_target_id_is_kept_when_onebot_does_not_supply_a_display_name(self) -> None:
        message = IncomingMessage.from_event(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "user_id": "blue-id",
                "self_id": "bot-1",
                "message_id": "m-1",
                "sender": {"user_id": "blue-id", "nickname": "blue"},
                "message": [
                    {"type": "at", "data": {"qq": "cecilia-id"}},
                    {"type": "text", "data": {"text": "那骂我吧"}},
                ],
            }
        )

        self.assertEqual(message.at_target_labels, ("QQ cecilia-id",))
        self.assertEqual(message.text_without_at_mentions, "那骂我吧")

    def test_context_snapshot_keeps_at_recipient(self) -> None:
        tracker = GroupContextTracker(ttl_seconds=120)
        routes = ReplyRoutePolicy()
        directed = _group_message(
            message_id="m-1",
            sender_id="blue-id",
            sender_name="blue",
            target_id="cecilia-id",
            target_name="塞西莉亚",
            text="那骂我吧",
        )
        tracker.enrich(directed, routes.decide(directed), now=10)
        addressed = IncomingMessage.from_event(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "user_id": "owner-id",
                "self_id": "bot-1",
                "message_id": "m-2",
                "sender": {"user_id": "owner-id", "nickname": "主人"},
                "message": [
                    {"type": "at", "data": {"qq": "bot-1", "name": "真红"}},
                    {"type": "text", "data": {"text": "刚才她怎么了"}},
                ],
            }
        )
        tracker.enrich(addressed, routes.decide(addressed), now=11)
        turn = QQTurn(
            conversation_id="g-1",
            conversation_type="group",
            primary_message_id="m-2",
            message_ids=("m-2",),
            combined_text="刚才她怎么了",
            messages=(addressed,),
        )

        context = QQAgentBridge._group_context(turn)

        self.assertIn("blue（QQ blue-id） → @塞西莉亚（QQ cecilia-id）：那骂我吧", context)

    def test_autonomous_sampling_skips_messages_atting_another_member(self) -> None:
        message = _group_message(
            message_id="m-1",
            sender_id="blue-id",
            sender_name="blue",
            target_id="cecilia-id",
            target_name="塞西莉亚",
            text="那骂我吧",
        )
        route = ReplyRoutePolicy().decide(message)
        draws: list[bool] = []
        policy = AttentionPolicy(
            autonomous_enabled=True,
            autonomous_probability=1.0,
            random_source=lambda: draws.append(True) or 0.0,
        )

        decision = policy.evaluate(message, route)

        self.assertFalse(route.should_reply)
        self.assertFalse(decision.should_schedule)
        self.assertEqual(decision.reason, "autonomous_at_other_member")
        self.assertEqual(draws, [])

    def test_direct_at_to_shinku_still_schedules_immediately(self) -> None:
        message = _group_message(
            message_id="m-1",
            sender_id="blue-id",
            sender_name="blue",
            target_id="bot-1",
            target_name="真红",
            text="你怎么看",
        )
        route = ReplyRoutePolicy().decide(message)
        policy = AttentionPolicy(
            autonomous_enabled=True,
            autonomous_probability=0.0,
            random_source=lambda: 1.0,
        )

        decision = policy.evaluate(message, route)

        self.assertTrue(route.should_reply)
        self.assertEqual(route.reason, "mentioned_bot")
        self.assertTrue(decision.should_schedule)
        self.assertTrue(decision.immediate)

    def test_autonomous_welcome_followups_keep_speaker_and_recipient_context(self) -> None:
        adapter = QQIngressAdapter(
            route_policy=ReplyRoutePolicy(bot_ids=frozenset({"shinku-id"})),
            attention_policy=AttentionPolicy(
                autonomous_enabled=True,
                autonomous_probability=1.0,
                random_source=lambda: 0.0,
            ),
            batcher=AttentionBatcher(window_seconds=0),
            group_context=GroupContextTracker(ttl_seconds=120),
        )
        newcomer = _group_message(
            message_id="newcomer-1",
            sender_id="newcomer-id",
            sender_name="浊酒劝梦",
            target_id="cecilia-id",
            target_name="塞西莉亚",
            text="你好啊",
        )
        first = adapter.ingest_message(newcomer, at=100)
        self.assertFalse(first.scheduled)
        self.assertEqual(first.attention.reason, "autonomous_at_other_member")

        cecilia_reply = IncomingMessage.from_event(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "user_id": "cecilia-id",
                "self_id": "shinku-id",
                "message_id": "cecilia-1",
                "sender": {"user_id": "cecilia-id", "nickname": "塞西莉亚"},
                "message": [
                    {"type": "reply", "data": {"id": "newcomer-1"}},
                    {"type": "text", "data": {"text": "欢迎来到群里"}},
                ],
            }
        )
        second = adapter.ingest_message(cecilia_reply, at=105)
        self.assertTrue(second.scheduled, "welcome replies may still reach Shinku")
        second_context = second.message.raw_event["_shinku_group_context"]
        self.assertEqual(second_context["current"]["sender_label"], "塞西莉亚")
        self.assertEqual(second_context["current"]["reply_parents"][0]["sender_label"], "浊酒劝梦")
        self.assertEqual(second_context["current"]["reply_parents"][0]["at_target_labels"], ["塞西莉亚"])

        cecilia_followup = IncomingMessage.from_event(
            {
                "post_type": "message",
                "message_type": "group",
                "group_id": "g-1",
                "user_id": "cecilia-id",
                "self_id": "shinku-id",
                "message_id": "cecilia-2",
                "sender": {"user_id": "cecilia-id", "nickname": "塞西莉亚"},
                "message": [{"type": "text", "data": {"text": "叫我塞西莉亚就好啦"}}],
            }
        )
        third = adapter.ingest_message(cecilia_followup, at=110)
        self.assertTrue(third.scheduled)
        third_context = third.message.raw_event["_shinku_group_context"]
        self.assertEqual(third_context["reason"], "active_addressed_thread")
        self.assertEqual(third_context["current"]["sender_label"], "塞西莉亚")
        self.assertEqual(
            [item["message_id"] for item in third_context["items"]],
            ["newcomer-1", "cecilia-1"],
        )

        turn = QQTurn(
            conversation_id="g-1",
            conversation_type="group",
            primary_message_id="cecilia-2",
            message_ids=("cecilia-2",),
            combined_text="叫我塞西莉亚就好啦",
            messages=(third.message,),
        )
        prompt_context = QQAgentBridge._group_context(turn)
        self.assertIn("当前发言者：塞西莉亚（QQ cecilia-id）", prompt_context)
        self.assertIn("这是自主回复候选，不是明确发给真红的消息。", prompt_context)
        self.assertIn(
            "短时对话起点：浊酒劝梦（QQ newcomer-id） → @塞西莉亚（QQ cecilia-id）",
            prompt_context,
        )
        self.assertIn("浊酒劝梦（QQ newcomer-id） → @塞西莉亚（QQ cecilia-id）：你好啊", prompt_context)

        class Runtime:
            def __init__(self) -> None:
                self.calls = []

            def call_chat_json(self, **kwargs):
                self.calls.append(kwargs)
                return {"speech": "刚进群吧？欢迎。"}

        runtime = Runtime()
        sent = []
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            sender=lambda outgoing: sent.append(outgoing) or DeliveryResult(True, "sent", ("out-1",)),
        )
        reply = bridge.handle_turn(turn)
        system_context = "\n".join(runtime.calls[0]["system_extra_blocks"])
        self.assertIn("可以自然加入或欢迎新人", system_context)
        self.assertIn("不要把别人的自我介绍当成真红自己的身份", system_context)
        self.assertTrue(reply.sent)
        self.assertTrue(sent, "attribution guidance must not silence a valid welcome reply")


if __name__ == "__main__":
    unittest.main()
