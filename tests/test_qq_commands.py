from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shinku.qq.adapter import QQIngressAdapter
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.attention import AttentionBatcher, AttentionPolicy
from shinku.qq.commands import (
    QQCommandInterpreter,
    QQCommandState,
    command_text_for_conversation,
    parse_slash_command,
)
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage, MessageSegment
from shinku.qq.routing import ReplyRoutePolicy
from shinku.qq.turns import QQTurn


def _message(
    text: str,
    *,
    conversation_type: str = "group",
    at_bot: bool = False,
) -> IncomingMessage:
    segments = []
    raw_event = {}
    if at_bot and conversation_type == "group":
        segments.append(MessageSegment(kind="at", payload={"data": {"qq": "bot-1"}}))
        raw_event["self_id"] = "bot-1"
    segments.append(MessageSegment(kind="text", text=text))
    return IncomingMessage(
        message_id="m-1",
        sender_id="u-1",
        conversation_id="g-1" if conversation_type == "group" else "u-1",
        conversation_type=conversation_type,
        segments=tuple(segments),
        raw_event=raw_event,
    )


class SlashParserTests(unittest.TestCase):
    def test_slash_must_be_the_literal_first_character(self) -> None:
        self.assertEqual(parse_slash_command("/stream on").args, ("on",))
        self.assertEqual(parse_slash_command("/at reply off").name, "at_reply")
        self.assertIsNone(parse_slash_command("普通话里提到 /stream on"))
        self.assertIsNone(parse_slash_command(" /stream on"))

    def test_unknown_slash_is_still_control_plane(self) -> None:
        parsed = parse_slash_command("/not-a-real-command")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.name, "not-a-real-command")

    def test_conversation_boundary_keeps_private_and_group_commands_separate(self) -> None:
        self.assertEqual(command_text_for_conversation("/help", "private"), "/help")
        self.assertEqual(
            command_text_for_conversation("/help", "group", mentioned_bot=True),
            "/help",
        )
        self.assertEqual(command_text_for_conversation("/help", "group"), "")
        self.assertEqual(
            command_text_for_conversation("/help", "group", mentioned_bot=False),
            "",
        )


class CommandStateTests(unittest.TestCase):
    def test_private_master_list_is_the_command_owner_source(self) -> None:
        state = QQCommandState.from_environment(
            {
                "SHINKU_MEMORY_MASTER_USER_IDS": "master",
                "SHINKU_QQ_COMMAND_USER_IDS": "legacy-other",
            }
        )
        self.assertEqual(state.owner_ids, frozenset({"master"}))

    def test_required_owner_gate_fails_closed_when_owner_list_is_empty(self) -> None:
        interpreter = QQCommandInterpreter(QQCommandState(), require_owner=True)
        result = interpreter.execute("/stream on", sender_id="anyone")
        self.assertTrue(result.quiet)
        self.assertEqual(result.reason, "slash_command_forbidden")

    def test_switches_are_shared_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "command_state.json"
            state = QQCommandState(state_path=path)
            interpreter = QQCommandInterpreter(state)
            self.assertEqual(interpreter.execute("/stream on").status, "stream_on")
            self.assertTrue(state.is_stream_enabled())
            self.assertEqual(interpreter.execute("/at reply on").status, "attention_reply_on")
            self.assertTrue(state.is_autonomous_reply_enabled())
            restored = QQCommandState(state_path=path)
            self.assertTrue(restored.is_stream_enabled())
            self.assertTrue(restored.is_autonomous_reply_enabled())

    def test_onlooker_is_scoped_to_one_group(self) -> None:
        state = QQCommandState()
        interpreter = QQCommandInterpreter(state)
        interpreter.execute("/onlooker off", conversation_id="g-1", conversation_type="group")
        self.assertFalse(state.is_onlooker_enabled("g-1"))
        self.assertTrue(state.is_onlooker_enabled("g-2"))


class CommandRouteTests(unittest.TestCase):
    def test_unaddressed_group_slash_message_does_not_enter_command_route(self) -> None:
        result = QQIngressAdapter(
            route_policy=ReplyRoutePolicy(bot_ids=frozenset({"bot-1"})),
            batcher=AttentionBatcher(window_seconds=1.0),
        ).ingest_message(_message("/help"), at=1.0)
        self.assertFalse(result.scheduled)
        self.assertEqual(result.route.reason, "unaddressed_group_message")

    def test_addressed_group_slash_message_enters_command_route(self) -> None:
        result = QQIngressAdapter(
            route_policy=ReplyRoutePolicy(bot_ids=frozenset({"bot-1"})),
            batcher=AttentionBatcher(window_seconds=1.0),
        ).ingest_message(_message("/help", at_bot=True), at=1.0)
        self.assertTrue(result.scheduled)
        self.assertEqual(result.route.reason, "slash_command")

    def test_group_slash_to_another_user_does_not_enter_command_route(self) -> None:
        message = IncomingMessage(
            message_id="m-other",
            sender_id="u-1",
            conversation_id="g-1",
            conversation_type="group",
            segments=(
                MessageSegment(kind="at", payload={"data": {"qq": "user-2"}}),
                MessageSegment(kind="text", text="/help"),
            ),
            raw_event={"self_id": "bot-1"},
        )
        result = QQIngressAdapter(
            route_policy=ReplyRoutePolicy(bot_ids=frozenset({"bot-1"})),
            batcher=AttentionBatcher(window_seconds=1.0),
        ).ingest_message(message, at=1.0)
        self.assertFalse(result.scheduled)
        self.assertEqual(result.route.reason, "unaddressed_group_message")

    def test_live_at_reply_switch_changes_attention_without_rebuild(self) -> None:
        state = QQCommandState()
        adapter = QQIngressAdapter(
            route_policy=ReplyRoutePolicy(bot_ids=frozenset({"bot-1"})),
            attention_policy=AttentionPolicy(autonomous_probability=1.0, random_source=lambda: 0.0),
            batcher=AttentionBatcher(window_seconds=1.0),
        )
        adapter.set_command_state(state)
        self.assertFalse(adapter.ingest_message(_message("普通群聊"), at=1.0).scheduled)
        state.set_autonomous_reply_enabled(True)
        self.assertTrue(adapter.ingest_message(_message("普通群聊"), at=2.0).scheduled)


class BridgeCommandTests(unittest.TestCase):
    def test_command_does_not_call_runtime_or_enter_memory(self) -> None:
        class Runtime:
            def __init__(self) -> None:
                self.calls = 0

            def call_chat_json(self, **kwargs):
                self.calls += 1
                return {"speech": "普通回复"}

        sent: list[str] = []

        def sender(message):
            sent.append(message.segments[0].text)
            return DeliveryResult(True, "sent", ("out-1",))

        runtime = Runtime()
        bridge = QQAgentBridge(runtime=runtime, system_prompt="persona", sender=sender)
        turn = QQTurn(
            conversation_id="g-1",
            conversation_type="group",
            primary_message_id="m-1",
            message_ids=("m-1",),
            combined_text="/stream on",
            messages=(_message("/stream on", at_bot=True),),
        )
        reply = bridge.handle_turn(turn)
        self.assertEqual(reply.agent_status, "command")
        self.assertEqual(runtime.calls, 0)
        self.assertEqual(len(sent), 1)
        self.assertIn("流式回复开了", sent[0])


if __name__ == "__main__":
    unittest.main()
