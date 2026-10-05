from __future__ import annotations

import unittest

from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult, OutgoingMessage, OutgoingSegment
from shinku.qq.message import IncomingMessage
from shinku.qq.napcat import NapCatActionTransport
from shinku.qq.turns import QQTurn


class _Runtime:
    def call_chat_json(self, **kwargs):
        return {"speech": "unused"}


def _group_turn() -> QQTurn:
    message = IncomingMessage.from_event(
        {
            "post_type": "message",
            "message_type": "group",
            "group_id": "g-1",
            "user_id": "10001",
            "self_id": "90000",
            "message_id": "m-1",
            "sender": {"user_id": "10001", "nickname": "主人"},
            "message": [
                {"type": "at", "data": {"qq": "90000", "name": "真红"}},
                {"type": "at", "data": {"qq": "20002", "name": "青空"}},
                {"type": "at", "data": {"qq": "30003", "name": "蓝"}},
                {"type": "text", "data": {"text": "你们看看这个"}},
            ],
        }
    )
    return QQTurn(
        conversation_id="g-1",
        conversation_type="group",
        primary_message_id="m-1",
        message_ids=("m-1",),
        combined_text="你们看看这个",
        messages=(message,),
    )


def _private_turn() -> QQTurn:
    message = IncomingMessage.from_event(
        {
            "post_type": "message",
            "message_type": "private",
            "user_id": "10001",
            "message_id": "p-1",
            "sender": {"user_id": "10001", "nickname": "主人"},
            "message": [{"type": "text", "data": {"text": "继续"}}],
        }
    )
    return QQTurn(
        conversation_id="10001",
        conversation_type="private",
        primary_message_id="p-1",
        message_ids=("p-1",),
        combined_text="继续",
        messages=(message,),
    )


class QQOutgoingMentionTests(unittest.TestCase):
    def test_group_prompt_exposes_only_contextual_members_and_internal_markers(self) -> None:
        turn = _group_turn()

        context = QQAgentBridge._group_context(turn, bot_ids=("90000",))

        self.assertIn("[[SHINKU_AT_1]] = @主人", context)
        self.assertIn("[[SHINKU_AT_2]] = @蓝", context)
        self.assertIn("[[SHINKU_AT_3]] = @青空", context)
        self.assertNotIn("90000", context)
        self.assertIn("不要 @全体", context)

    def test_valid_group_marker_becomes_a_real_onebot_at_segment(self) -> None:
        sent: list[OutgoingMessage] = []

        def sender(message: OutgoingMessage) -> DeliveryResult:
            sent.append(message)
            return DeliveryResult(True, "sent", ("out-1",))

        bridge = QQAgentBridge(runtime=_Runtime(), system_prompt="test", sender=sender, bot_ids=("90000",))
        result = bridge._deliver_text(
            _group_turn(),
            "[[SHINKU_AT_3]]你来看看这个。",
            record_memory=False,
            track_chat_flow=False,
        )

        self.assertTrue(result.sent)
        self.assertEqual(result.final_text, "@青空你来看看这个。")
        self.assertEqual(
            [(segment.kind, segment.target_id, segment.text) for segment in sent[0].segments],
            [("at", "20002", ""), ("text", "", "你来看看这个。")],
        )
        action, params = NapCatActionTransport._build_action(sent[0])
        self.assertEqual(action, "send_group_msg")
        self.assertEqual(
            params["message"],
            [
                {"type": "at", "data": {"qq": "20002"}},
                {"type": "text", "data": {"text": "你来看看这个。"}},
            ],
        )

    def test_unknown_malformed_and_excessive_markers_do_not_leak(self) -> None:
        sent: list[OutgoingMessage] = []

        def sender(message: OutgoingMessage) -> DeliveryResult:
            sent.append(message)
            return DeliveryResult(True, "sent")

        bridge = QQAgentBridge(runtime=_Runtime(), system_prompt="test", sender=sender, bot_ids=("90000",))
        result = bridge._deliver_text(
            _group_turn(),
            "收到 [[SHINKU_AT_99]] [[SHINKU_AT_BAD]] [[SHINKU_AT_1]] [[SHINKU_AT_2]] [[SHINKU_AT_3]]。",
            record_memory=False,
            track_chat_flow=False,
        )

        self.assertTrue(result.sent)
        self.assertNotIn("SHINKU_AT", result.final_text)
        mentions = [segment.target_id for segment in sent[0].segments if segment.kind == "at"]
        self.assertEqual(mentions, ["10001", "30003"])
        self.assertNotIn("SHINKU_AT", "".join(segment.text for segment in sent[0].segments))

    def test_mention_marker_survives_sentence_splitting(self) -> None:
        sent: list[OutgoingMessage] = []

        def sender(message: OutgoingMessage) -> DeliveryResult:
            sent.append(message)
            return DeliveryResult(True, "sent")

        bridge = QQAgentBridge(runtime=_Runtime(), system_prompt="test", sender=sender, bot_ids=("90000",))
        result = bridge._deliver_text(
            _group_turn(),
            "我先说明一下。[[SHINKU_AT_3]]你来看看这个。",
            record_memory=False,
            track_chat_flow=False,
        )

        self.assertTrue(result.sent)
        self.assertEqual(len(sent), 2)
        self.assertEqual([segment.target_id for segment in sent[0].segments if segment.kind == "at"], ["20002"])
        self.assertTrue(all(segment.kind != "at" for segment in sent[1].segments))
        self.assertNotIn("SHINKU_AT", result.final_text)

    def test_private_turn_strips_mention_markers_and_never_emits_at(self) -> None:
        sent: list[OutgoingMessage] = []

        def sender(message: OutgoingMessage) -> DeliveryResult:
            sent.append(message)
            return DeliveryResult(True, "sent")

        bridge = QQAgentBridge(runtime=_Runtime(), system_prompt="test", sender=sender)
        result = bridge._deliver_text(
            _private_turn(),
            "嗯，[[SHINKU_AT_1]]继续。",
            record_memory=False,
            track_chat_flow=False,
        )

        self.assertTrue(result.sent)
        self.assertNotIn("SHINKU_AT", result.final_text)
        self.assertTrue(all(segment.kind != "at" for segment in sent[0].segments))

    def test_plain_at_name_does_not_ping_named_bystander_or_duplicate_source_quote(self) -> None:
        sent: list[OutgoingMessage] = []

        def sender(message: OutgoingMessage) -> DeliveryResult:
            sent.append(message)
            return DeliveryResult(True, "sent")

        bridge = QQAgentBridge(runtime=_Runtime(), system_prompt="test", sender=sender, bot_ids=("90000",))
        result = bridge._deliver_text(
            _group_turn(),
            "请 @青空 看看这个。",
            record_memory=False,
            track_chat_flow=False,
        )

        self.assertTrue(result.sent)
        self.assertEqual(result.final_text, "请 @青空 看看这个。")
        self.assertFalse(any(segment.kind == "at" for segment in sent[0].segments))
        self.assertEqual(sent[0].reply_to_message_id, "m-1")

    def test_at_segment_requires_group_and_numeric_member_id(self) -> None:
        mention = OutgoingSegment(kind="at", target_id="20002")
        with self.assertRaises(ValueError):
            OutgoingMessage("10001", (mention,), conversation_type="private")
        with self.assertRaises(ValueError):
            OutgoingSegment(kind="at", target_id="all")
        with self.assertRaises(ValueError):
            OutgoingSegment(kind="at", target_id="not-a-qq-id")


if __name__ == "__main__":
    unittest.main()
