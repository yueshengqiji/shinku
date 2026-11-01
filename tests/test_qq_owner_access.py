from __future__ import annotations
from reply_review_fixture import dialogue_review_result

import tempfile
import time
import unittest
from pathlib import Path

from shinku.hosts.tool_host import ToolHost
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.tools.registry import ToolRegistry


class _SearchHandler:
    tool_type = "search"

    def __init__(self) -> None:
        self.calls = 0

    def normalize_call(self, call):
        return call if call.get("query") else None

    def execute(self, *, call, context):
        self.calls += 1
        return {"tool_type": "search", "followup_context": f"查到：{call['query']}"}


class _SpeechRuntime:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict[str, object]] = []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        review = dialogue_review_result(kwargs)
        if review is not None:
            return review
        return self.response


class _ToolThenSpeechRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        review = dialogue_review_result(kwargs)
        if review is not None:
            return review
        if len(self.calls) == 1:
            return {
                "tool_calls": [
                    {"id": "unauthorized", "function": {"name": "search", "arguments": '{"query":"x"}'}},
                ]
            }
        return {"speech": "……这类工具操作只听主人的。"}


class _ApprovalRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        review = dialogue_review_result(kwargs)
        if review is not None:
            return review
        if "授权" in str(kwargs.get("system_prompt") or ""):
            return {"decision": "approve", "confidence": 0.99, "reason": "明确同意"}
        if len(self.calls) == 1:
            return {
                "tool_calls": [
                    {"id": "owner-call", "function": {"name": "search", "arguments": '{"query":"x"}'}},
                ]
            }
        return {"speech": "结果已经拿到了。"}


def _turn(sender: str, text: str, *, conversation_id: str = "conversation-1", conversation_type: str = "private") -> QQTurn:
    event = {
        "post_type": "message",
        "message_type": conversation_type,
        "user_id": sender,
        "message_id": f"message-{sender}-{text}",
        "message": [{"type": "text", "data": {"text": text}}],
    }
    if conversation_type == "group":
        event["group_id"] = conversation_id
    message = IncomingMessage.from_event(event)
    return QQTurn(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        primary_message_id=message.message_id,
        message_ids=(message.message_id,),
        combined_text=text,
        messages=(message,),
    )


def _group_message(sender: str, text: str, *, group_id: str = "group-1") -> IncomingMessage:
    return IncomingMessage.from_event(
        {
            "post_type": "message",
            "message_type": "group",
            "group_id": group_id,
            "user_id": sender,
            "sender": {"nickname": sender},
            "message_id": f"{group_id}-{sender}-{text}",
            "message": [{"type": "text", "data": {"text": text}}],
        }
    )


def _group_message_at_bot(sender: str, text: str, *, group_id: str = "group-1") -> IncomingMessage:
    return IncomingMessage.from_event(
        {
            "post_type": "message",
            "message_type": "group",
            "group_id": group_id,
            "self_id": "bot-1",
            "user_id": sender,
            "sender": {"nickname": sender},
            "message_id": f"{group_id}-{sender}-{text}",
            "message": [
                {"type": "at", "data": {"qq": "bot-1", "name": "真红"}},
                {"type": "text", "data": {"text": text}},
            ],
        }
    )


class QQOwnerAccessTests(unittest.TestCase):
    def test_private_greeting_can_pick_up_volatile_group_exchange(self) -> None:
        runtime = _SpeechRuntime({"speech": "嗯……刚才那件事还没说完？"})
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            master_user_ids=("master",),
        )
        bridge.observe_message(_group_message("master", "我把本地向量检索接好了"))
        bridge.observe_message(_group_message("friend", "还得看索引覆盖率有没有漏"))

        result = bridge.handle_turn(_turn("master", "真红，在吗"))

        self.assertEqual(result.agent_status, "completed")
        prompt = str(runtime.calls[0].get("user_prompt") or "")
        self.assertIn("本地向量检索接好了", prompt)
        self.assertIn("索引覆盖率有没有漏", prompt)
        self.assertIn("不进入长期记忆", prompt)

    def test_volatile_group_handoff_is_one_shot(self) -> None:
        runtime = _SpeechRuntime({"speech": "嗯，我在。"})
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            master_user_ids=("master",),
        )
        bridge.observe_message(_group_message("master", "刚才在群里说的临时话题"))
        bridge.handle_turn(_turn("master", "真红，在吗"))
        bridge.handle_turn(_turn("master", "真红？"))

        second_prompt = str(runtime.calls[1].get("user_prompt") or "")
        self.assertNotIn("临时话题", second_prompt)

    def test_private_greeting_can_pick_up_recent_owner_topic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            memory = MemoryService(
                store=MemoryStore(Path(raw) / "memory.sqlite3"),
                router=MemoryRouter(),
                policy=MemoryPolicy(master_user_ids=("master",)),
            )
            memory.record_turn(
                conversation_id="group-1",
                conversation_type="group",
                message_id="prior-topic",
                text="刚才聊的本地语义检索还要看索引覆盖率。",
                metadata={"sender_id": "master", "shinku_involved": True, "at_targets": ["bot"]},
            )
            runtime = _SpeechRuntime({"speech": "……刚才那件事还没说完？"})
            bridge = QQAgentBridge(
                runtime=runtime,
                system_prompt="真红主人设",
                master_user_ids=("master",),
                memory_service=memory,
            )

            result = bridge.handle_turn(_turn("master", "真红，在吗"))

            self.assertEqual(result.agent_status, "completed")
            prompt = str(runtime.calls[0].get("user_prompt") or "")
            self.assertIn("本地语义检索还要看索引覆盖率", prompt)
            self.assertIn("主人最近在桌面与 QQ 的共同对话", prompt)

    def test_private_handoff_context_is_never_added_to_group_greeting(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            memory = MemoryService(
                store=MemoryStore(Path(raw) / "memory.sqlite3"),
                router=MemoryRouter(),
                policy=MemoryPolicy(master_user_ids=("master",)),
            )
            memory.record_turn(
                conversation_id="master",
                conversation_type="private",
                message_id="private-topic",
                text="这句只说在私聊里，不要带进群。",
                metadata={"sender_id": "master"},
            )
            runtime = _SpeechRuntime({"speech": "嗯，怎么了？"})
            bridge = QQAgentBridge(
                runtime=runtime,
                system_prompt="真红主人设",
                master_user_ids=("master",),
                memory_service=memory,
            )
            bridge.observe_message(_group_message("master", "这段临时群聊也不能带进群"))

            result = bridge.handle_turn(
                _turn("master", "真红，在吗", conversation_id="group-1", conversation_type="group")
            )

            self.assertEqual(result.agent_status, "completed")
            prompt = str(runtime.calls[0].get("user_prompt") or "")
            self.assertNotIn("这句只说在私聊里", prompt)
            self.assertNotIn("临时群聊也不能带进群", prompt)

    def test_history_needs_greeting_uses_fresh_sources_but_not_volatile_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            clock = [int(time.time())]
            memory = MemoryService(store=MemoryStore(Path(raw) / "memory.sqlite3"), router=MemoryRouter(),
                policy=MemoryPolicy(master_user_ids=("master",), history_needs_enabled=True),
                clock=lambda: clock[0])
            memory.record_turn(conversation_id="group-1", conversation_type="group", message_id="old",
                text="保留旧话题标记", metadata={"sender_id": "master", "shinku_involved": True})
            runtime = _SpeechRuntime({"speech": "嗯，我在。"})
            bridge = QQAgentBridge(runtime=runtime, system_prompt="真红主人设", master_user_ids=("master",),
                memory_service=memory)
            bridge.observe_message(_group_message("master", "临时旧话题标记"))
            for greeting in ("真红，在吗", "真红？"):
                bridge.handle_turn(_turn("master", greeting))
                prompt = str(runtime.calls[-1].get("user_prompt") or "")
                self.assertIn("保留旧话题标记", prompt)
                self.assertNotIn("临时旧话题标记", prompt)
            clock[0] += 301
            bridge.handle_turn(_turn("master", "真红，在吗"))
            self.assertNotIn("保留旧话题标记", str(runtime.calls[-1].get("user_prompt") or ""))

    def test_required_owner_gate_fails_closed_when_owner_ids_are_missing(self) -> None:
        runtime = _ToolThenSpeechRuntime()
        handler = _SearchHandler()
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            tool_host=ToolHost(ToolRegistry({"search": handler})),
            owner_gate_required=True,
        )

        result = bridge.handle_turn(_turn("anyone", "查一下"))

        self.assertEqual((result.agent_status, result.final_text), ("blocked", "……这类工具操作只听主人的。别的事情可以继续说。"))
        self.assertEqual(handler.calls, 0)

    def test_master_gets_a_compact_master_audience_block(self) -> None:
        runtime = _SpeechRuntime({"speech": "嗯，知道了。"})
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            master_user_ids=("master",),
        )

        result = bridge.handle_turn(_turn("master", "陪我说会儿话"))

        self.assertEqual(result.agent_status, "completed")
        prompt = "\n".join(runtime.calls[0].get("system_extra_blocks", []))
        self.assertIn("当前交互对象：主人", prompt)
        self.assertIn("自然回应他的具体话题", prompt)

    def test_group_owner_call_gets_positive_familiarity_guidance(self) -> None:
        runtime = _SpeechRuntime({"speech": "嗯，知道了。"})
        message = _group_message_at_bot("master", "陪我说会儿话")
        turn = QQTurn(
            conversation_id="group-1",
            conversation_type="group",
            primary_message_id=message.message_id,
            message_ids=(message.message_id,),
            combined_text="陪我说会儿话",
            messages=(message,),
        )
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            master_user_ids=("master",),
            bot_ids=("bot-1",),
        )

        result = bridge.handle_turn(turn)

        self.assertEqual(result.agent_status, "completed")
        system_context = "\n".join(runtime.calls[0].get("system_extra_blocks", []))
        self.assertIn("主人在群聊中的优先感", system_context)
        self.assertIn("适度偏心", system_context)
        self.assertIn("主人（master）", str(runtime.calls[0].get("user_prompt") or ""))

    def test_mixed_group_batch_marks_owner_without_granting_primary_friend_tools(self) -> None:
        runtime = _SpeechRuntime({"speech": "先听你说完。"})
        owner_message = _group_message_at_bot("master", "真红，你觉得呢？")
        friend_message = _group_message_at_bot("friend", "帮我查一下")
        turn = QQTurn(
            conversation_id="group-1",
            conversation_type="group",
            primary_message_id=friend_message.message_id,
            message_ids=(owner_message.message_id, friend_message.message_id),
            combined_text="真红，你觉得呢？\n帮我查一下",
            messages=(owner_message, friend_message),
        )
        handler = _SearchHandler()
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            tool_host=ToolHost(ToolRegistry({"search": handler})),
            master_user_ids=("master",),
            bot_ids=("bot-1",),
        )

        audience = bridge._audience_context(turn)
        result = bridge.handle_turn(turn)

        self.assertEqual(audience["role"], "other")
        self.assertFalse(audience["is_master"])
        self.assertTrue(audience["master_present"])
        self.assertTrue(audience["master_addressed_to_bot"])
        self.assertEqual(result.agent_status, "completed")
        user_prompt = str(runtime.calls[0].get("user_prompt") or "")
        history_turns = runtime.calls[0].get("history_turns") or []
        dialogue_context = "\n".join(
            [user_prompt]
            + [str(item.get("content") or "") for item in history_turns if isinstance(item, dict)]
        )
        system_context = "\n".join(runtime.calls[0].get("system_extra_blocks", []))
        self.assertIn("主人（master）", dialogue_context)
        self.assertIn("群聊中的主人消息归属", system_context)
        self.assertIn("当前主请求来自群友", system_context)
        self.assertIn("同样明确地叫你", system_context)
        self.assertIsNone(runtime.calls[0].get("native_tools"))
        self.assertEqual(handler.calls, 0)

    def test_non_master_sees_no_tools_and_gets_distance_guidance(self) -> None:
        runtime = _SpeechRuntime({"speech": "这件事我不替别人做。"})
        handler = _SearchHandler()
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            tool_host=ToolHost(ToolRegistry({"search": handler})),
            master_user_ids=("master",),
        )

        result = bridge.handle_turn(_turn("other", "帮我查一下"))

        self.assertEqual(result.agent_status, "completed")
        self.assertEqual(handler.calls, 0)
        self.assertIsNone(runtime.calls[0].get("native_tools"))
        prompt = "\n".join(runtime.calls[0].get("system_extra_blocks", []))
        self.assertIn("当前交互对象：其他人", prompt)
        self.assertIn("暧昧", prompt)
        self.assertIn("没有工具执行权限", prompt)

    def test_malicious_tool_call_from_non_master_cannot_reach_handler(self) -> None:
        runtime = _ToolThenSpeechRuntime()
        handler = _SearchHandler()
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            tool_host=ToolHost(ToolRegistry({"search": handler})),
            master_user_ids=("master",),
        )

        result = bridge.handle_turn(_turn("other", "查一下"))

        self.assertEqual((result.agent_status, result.final_text), ("blocked", "……这类工具操作只听主人的。别的事情可以继续说。"))
        self.assertEqual(handler.calls, 0)

    def test_non_master_cannot_approve_or_resume_master_pending_tool(self) -> None:
        runtime = _ApprovalRuntime()
        handler = _SearchHandler()
        sent = []
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            tool_host=ToolHost(ToolRegistry({"search": handler})),
            sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("out",)),
            master_user_ids=("master",),
        )

        waiting = bridge.handle_turn(_turn("master", "查一下", conversation_id="group-1", conversation_type="group"))
        rejected = bridge.handle_turn(_turn("other", "可以，执行吧", conversation_id="group-1", conversation_type="group"))
        resumed = bridge.handle_turn(_turn("master", "可以，执行吧", conversation_id="group-1", conversation_type="group"))

        self.assertEqual(waiting.agent_status, "waiting_user")
        self.assertEqual((rejected.agent_status, rejected.reason), ("rejected", "tool_owner_only"))
        self.assertEqual((resumed.agent_status, resumed.final_text), ("completed", "结果已经拿到了。"))
        self.assertEqual(handler.calls, 1)
        # The short refusal may be delivered as two sentence segments by the
        # normal reply host; the important boundary is that all three turns
        # produced a delivery and only the owner's final approval executed.
        self.assertGreaterEqual(len(sent), 3)


if __name__ == "__main__":
    unittest.main()
