from __future__ import annotations
from reply_review_fixture import dialogue_review_result

import unittest

from shinku.hosts.tool_host import ToolHost
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.attention import AttentionBatch, AttentionDecision
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.trace import TurnTraceStore
from shinku.qq.turns import NapCatTurnDispatcher, QQTurn
from shinku.tools.registry import ToolRegistry


class _Runtime:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        review = dialogue_review_result(kwargs)
        if review is not None:
            return review
        if kwargs.get("system_prompt") == "群聊说话人与任务归属核验":
            return {"verdict": "allow", "relation": "addressed", "task_claim": "none", "task_id": ""}
        return self.response


def _message(message_id: str, text: str, *, conversation_type: str = "group") -> IncomingMessage:
    event = {
        "post_type": "message",
        "message_type": conversation_type,
        "user_id": "u-1",
        "message_id": message_id,
        "message": [{"type": "text", "data": {"text": text}}],
    }
    if conversation_type == "group":
        event["group_id"] = "g-1"
    return IncomingMessage.from_event(event)


def _turn(text: str, *, mode: str = "autonomous", conversation_type: str = "group") -> QQTurn:
    message = _message("m-p3", text, conversation_type=conversation_type)
    return QQTurn(
        conversation_id=message.conversation_id,
        conversation_type=message.conversation_type,
        primary_message_id=message.message_id,
        message_ids=(message.message_id,),
        combined_text=text,
        messages=(message,),
        attention_mode=mode,
    )


class P3AutonomousSilenceTests(unittest.TestCase):
    def test_dispatcher_marks_mixed_batch_immediate(self) -> None:
        first = _message("m-auto", "群友之间的闲聊")
        second = _message("m-addressed", "真红你怎么看")
        batch = AttentionBatch(
            conversation_id="g-1",
            messages=(first, second),
            decisions=(
                AttentionDecision(True, False, 20, "autonomous_probability_hit"),
                AttentionDecision(True, True, 100, "mentioned_bot"),
            ),
            primary_index=1,
        )
        dispatcher = NapCatTurnDispatcher(handler=lambda turn: None, flush_batch=lambda: None)

        turn = dispatcher._build_turn_locked(batch, now=1.0)

        self.assertEqual(turn.attention_mode, "immediate")

    def test_dispatcher_marks_all_non_immediate_scheduled_messages_autonomous(self) -> None:
        message = _message("m-auto", "群友之间的闲聊")
        batch = AttentionBatch(
            conversation_id="g-1",
            messages=(message,),
            decisions=(AttentionDecision(True, False, 20, "autonomous_probability_hit"),),
            primary_index=0,
        )
        dispatcher = NapCatTurnDispatcher(handler=lambda turn: None, flush_batch=lambda: None)

        turn = dispatcher._build_turn_locked(batch, now=1.0)

        self.assertEqual(turn.attention_mode, "autonomous")

    def test_autonomous_group_turn_can_suppress_exact_silence_marker(self) -> None:
        runtime = _Runtime({"speech": "  <SILENCE>  "})
        sent = []
        trace = TurnTraceStore()
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("out-1",)),
            trace_store=trace,
        )

        result = bridge.handle_turn(_turn("他们正在讨论别的事"))

        self.assertEqual((result.agent_status, result.delivery_status), ("silent", "suppressed"))
        self.assertEqual(result.reason, "attention_silence")
        self.assertEqual(result.final_text, "")
        self.assertEqual(sent, [])
        self.assertTrue(
            any("自主群聊的沉默选择" in block for block in runtime.calls[0]["system_extra_blocks"])
        )
        row = trace.snapshot()[0]
        self.assertTrue(any(stage["name"] == "attention_silence" for stage in row["stages"]))

    def test_immediate_and_private_turns_do_not_receive_silence_instruction(self) -> None:
        for turn in (
            _turn("真红，你在吗", mode="immediate"),
            _turn("你在吗", mode="autonomous", conversation_type="private"),
        ):
            with self.subTest(conversation_type=turn.conversation_type, mode=turn.attention_mode):
                runtime = _Runtime({"speech": "我听见了。"})
                sent = []
                bridge = QQAgentBridge(
                    runtime=runtime,
                    system_prompt="persona",
                    sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("out-1",)),
                )

                result = bridge.handle_turn(turn)

                blocks = runtime.calls[0]["system_extra_blocks"]
                self.assertFalse(any("自主群聊的沉默选择" in block for block in blocks))
                self.assertEqual(result.final_text, "我听见了。")
                self.assertEqual(len(sent), 1)

    def test_ineligible_model_marker_is_replaced_instead_of_leaking(self) -> None:
        runtime = _Runtime({"speech": "<SILENCE>"})
        sent = []
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="persona",
            sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("out-1",)),
        )

        result = bridge.handle_turn(_turn("真红，你在吗", mode="immediate"))

        self.assertEqual(result.final_text, "……刚才那句我没接好。你再说一遍？")
        self.assertGreaterEqual(len(sent), 1)
        delivered_text = "".join(
            segment.text
            for message in sent
            for segment in message.segments
        )
        self.assertNotIn("<SILENCE>", delivered_text)

    def test_tool_decision_is_not_suppressed_by_silence_gate(self) -> None:
        from shinku.agent.loop import AgentRunResult

        class EchoHandler:
            tool_type = "echo"

            def normalize_call(self, call):
                return call if isinstance(call, dict) else None

            def execute(self, *, call, context):
                del context
                return {"text": call.get("text", "")}

        runtime = _Runtime(
            {"kind": "tool", "tool_call": {"name": "echo", "arguments": {"text": "继续处理"}}}
        )
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="persona",
            tool_host=ToolHost(ToolRegistry({"echo": EchoHandler()})),
            allowed_tool_names={"echo"},
        )
        agent_calls = []

        def fake_agent_run(**kwargs):
            agent_calls.append(kwargs)
            return AgentRunResult(status="completed", rounds=1, final_text="已经处理。")

        bridge._run_agent = fake_agent_run

        item = _turn("帮我处理一下", mode="autonomous")
        item.messages[0].raw_event["_shinku_group_context"] = {"current": {
            "message_id": item.primary_message_id, "sender_id": "u-1", "bot_exchange_confirmed": True}}
        result = bridge.handle_turn(item)

        self.assertEqual(result.final_text, "已经处理。")
        self.assertEqual(len(agent_calls), 1)
        self.assertEqual(agent_calls[0]["first_decision"].kind, "tool")


if __name__ == "__main__":
    unittest.main()
