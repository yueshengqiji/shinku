from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from shinku.hosts.tool_host import ToolHost
from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry


class MockQueryHandler:
    tool_type = "mock_query"

    def tool_metadata(self):
        return SimpleNamespace(input_schema={"type": "object", "properties": {"target": {"type": "string"}}})

    def build_prompt_instruction(self):
        return "查询目标。"

    def normalize_call(self, call):
        return call if call.get("target") else None

    def execute(self, *, call, context):
        return {"tool_type": "mock_query", "followup_context": f"已读取 {call['target']}，里面包含 5 个配置项。"}


class MultiTurnScriptedRuntime:
    def __init__(self, step_responses: list[dict[str, Any]]) -> None:
        self.step_responses = list(step_responses)
        self.calls: list[dict[str, Any]] = []

    def call_chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.step_responses:
            return self.step_responses.pop(0)
        return {"kind": "final", "text": "完成。"}


def _turn(user_text: str, conversation_id: str = "conv_test_1", message_id: str = "msg_1") -> QQTurn:
    event = {
        "post_type": "message",
        "message_type": "private",
        "user_id": "master_1",
        "message_id": message_id,
        "message": [{"type": "text", "data": {"text": user_text}}],
    }
    msg = IncomingMessage.from_event(event)
    return QQTurn(
        conversation_id=conversation_id,
        conversation_type="private",
        primary_message_id=message_id,
        message_ids=(message_id,),
        combined_text=user_text,
        messages=(msg,),
    )


class AgentCrossTurnMemoryTests(unittest.TestCase):
    def test_tool_execution_metadata_persisted_in_memory(self) -> None:
        """测试工具调用成果作为结构化元数据持久化存入 MemoryService。"""
        with tempfile.TemporaryDirectory() as raw_dir:
            store = MemoryStore(Path(raw_dir) / "memory.sqlite3")
            memory = MemoryService(
                store=store,
                router=MemoryRouter(),
                policy=MemoryPolicy(master_user_ids=("master_1",)),
                clock=lambda: 2_000_000_000,
            )
            registry = ToolRegistry({"mock_query": MockQueryHandler()})
            runtime = MultiTurnScriptedRuntime([
                # Step 1: 调用工具
                {
                    "kind": "tool",
                    "tool_calls": [{"id": "c1", "name": "mock_query", "arguments": {"target": "package.json"}}],
                },
                # Step 2: 交付最终回复
                {
                    "kind": "final",
                    "speech": "package.json 读取完毕，包含 5 个配置项。",
                },
            ])

            bridge = QQAgentBridge(
                runtime=runtime,
                system_prompt="测试助手",
                tool_host=ToolHost(registry),
                memory_service=memory,
                policy=ExecutionPolicy(risk_by_tool={"mock_query": "low"}),
                master_user_ids=("master_1",),
                sender=lambda msg: DeliveryResult(True, "sent", ("out",)),
            )

            reply = bridge.handle_turn(_turn("帮我查一下 package.json"))
            self.assertEqual(reply.agent_status, "completed")
            self.assertIn("package.json", reply.final_text)

            # 验证 memory timeline 中记录了 assistant 回复及其 tools_used 结构化元数据
            timeline = store.timeline(scope="private:conv_test_1", limit=10, now=2_000_000_000, include_facts=False)
            assistant_events = [e for e in timeline if e.metadata.get("role") == "assistant"]
            self.assertEqual(len(assistant_events), 1)
            self.assertIn("mock_query", assistant_events[0].metadata.get("tools_used", []))
            self.assertEqual(assistant_events[0].metadata.get("tool_count"), 1)

    def test_cross_turn_task_continuation_injects_snapshot_context(self) -> None:
        """测试用户在第二轮发送代词承接短语时，上一任务快照作为延续上下文成功注入 Prompt。"""
        with tempfile.TemporaryDirectory() as raw_dir:
            store = MemoryStore(Path(raw_dir) / "memory.sqlite3")
            memory = MemoryService(
                store=store,
                router=MemoryRouter(),
                policy=MemoryPolicy(master_user_ids=("master_1",)),
            )
            registry = ToolRegistry({"mock_query": MockQueryHandler()})
            runtime = MultiTurnScriptedRuntime([
                # Turn 1: 查 package.json
                {
                    "kind": "tool",
                    "tool_calls": [{"id": "c1", "name": "mock_query", "arguments": {"target": "package.json"}}],
                },
                {"kind": "final", "speech": "package.json 分析完成。"},
                # Turn 2: 承接
                {"kind": "final", "speech": "第二节的配置项是 scripts 和 dependencies。"},
            ])

            bridge = QQAgentBridge(
                runtime=runtime,
                system_prompt="测试助手",
                tool_host=ToolHost(registry),
                memory_service=memory,
                policy=ExecutionPolicy(risk_by_tool={"mock_query": "low"}),
                master_user_ids=("master_1",),
                sender=lambda msg: DeliveryResult(True, "sent", ("out",)),
            )

            # 轮次 1: 执行查询
            turn_1 = _turn("帮我查一下 package.json", message_id="m1")
            reply_1 = bridge.handle_turn(turn_1)
            self.assertEqual(reply_1.agent_status, "completed")

            # 轮次 2: 使用承接代词“把它第二节告诉我”
            turn_2 = _turn("把它第二节的内容告诉我", message_id="m2")
            reply_2 = bridge.handle_turn(turn_2)
            self.assertEqual(reply_2.agent_status, "completed")

            # 验证 Turn 2 模型接收的 prompt 中注入了【上一任务延续上下文】
            # Turn 1 有 2 次调用，Turn 2 只有 1 次调用
            turn_2_call = runtime.calls[2]
            turn_2_prompt = str(turn_2_call["user_prompt"])
            turn_2_system = "\n".join(turn_2_call.get("system_extra_blocks", []))
            self.assertIn("【候选任务，仅在语义相关时承接】", turn_2_prompt)
            self.assertIn("mock_query", turn_2_prompt)
            self.assertIn("package.json 分析完成", turn_2_prompt)
            self.assertIn("不要靠某个固定词判断", turn_2_system)
            self.assertIn("闲聊或新目标不要强行续接", turn_2_system)

    def test_non_continuation_turn_does_not_inject_snapshot(self) -> None:
        """测试非承接意图的新话题不会误注入上一任务延续上下文。"""
        with tempfile.TemporaryDirectory() as raw_dir:
            store = MemoryStore(Path(raw_dir) / "memory.sqlite3")
            memory = MemoryService(
                store=store,
                router=MemoryRouter(),
                policy=MemoryPolicy(master_user_ids=("master_1",)),
            )
            registry = ToolRegistry({"mock_query": MockQueryHandler()})
            runtime = MultiTurnScriptedRuntime([
                # Turn 1
                {"kind": "tool", "tool_calls": [{"id": "c1", "name": "mock_query", "arguments": {"target": "data"}}]},
                {"kind": "final", "speech": "查好了。"},
                # Turn 2: 独立新话题
                {"kind": "final", "speech": "明天晴天。"},
            ])

            bridge = QQAgentBridge(
                runtime=runtime,
                system_prompt="测试助手",
                tool_host=ToolHost(registry),
                memory_service=memory,
                policy=ExecutionPolicy(risk_by_tool={"mock_query": "low"}),
                master_user_ids=("master_1",),
                sender=lambda msg: DeliveryResult(True, "sent", ("out",)),
            )

            bridge.handle_turn(_turn("查一下 data", message_id="m1"))
            bridge.handle_turn(_turn("明天天气好吗", message_id="m2"))

            turn_2_prompt = str(runtime.calls[2]["user_prompt"])
            self.assertNotIn("【上一任务延续上下文】", turn_2_prompt)

    def test_task_snapshot_keeps_only_successes_and_is_conversation_scoped(self) -> None:
        bridge = QQAgentBridge(
            runtime=MultiTurnScriptedRuntime([]),
            system_prompt="测试助手",
            task_snapshot_ttl_seconds=60,
            task_snapshot_max_entries=2,
        )
        turn = _turn("查一下 package.json", conversation_id="same-id")
        result = SimpleNamespace(
            status="completed",
            tool_envelopes=(
                SimpleNamespace(status="ok", data={"tool_type": "mock_query"}),
                SimpleNamespace(status="error", data={"tool_type": "failed_write"}),
            ),
        )

        bridge._remember_task_snapshot(turn, result, task_id="task-1", final_text="查到了。", now=100)

        snapshot = bridge._task_snapshot_for_turn(turn, now=120)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["tools_used"], ["mock_query"])
        group_turn = QQTurn(
            conversation_id=turn.conversation_id,
            conversation_type="group",
            primary_message_id=turn.primary_message_id,
            message_ids=turn.message_ids,
            combined_text="继续",
            messages=turn.messages,
        )
        self.assertIsNone(bridge._task_snapshot_for_turn(group_turn, now=120))

    def test_task_snapshot_expires_and_capacity_evicts_oldest(self) -> None:
        bridge = QQAgentBridge(
            runtime=MultiTurnScriptedRuntime([]),
            system_prompt="测试助手",
            task_snapshot_ttl_seconds=60,
            task_snapshot_max_entries=2,
        )
        result = SimpleNamespace(
            status="completed",
            tool_envelopes=(SimpleNamespace(status="ok", data={"tool_type": "mock_query"}),),
        )
        turns = [_turn("查询", conversation_id=f"conv-{index}") for index in range(3)]
        for index, turn in enumerate(turns):
            bridge._remember_task_snapshot(turn, result, task_id=f"task-{index}", final_text="完成", now=100 + index)

        self.assertIsNone(bridge._task_snapshot_for_turn(turns[0], now=102))
        self.assertIsNotNone(bridge._task_snapshot_for_turn(turns[1], now=102))
        self.assertIsNotNone(bridge._task_snapshot_for_turn(turns[2], now=102))
        self.assertIsNone(bridge._task_snapshot_for_turn(turns[2], now=163))

    def test_failed_task_retains_separate_failure_and_prior_task(self) -> None:
        bridge = QQAgentBridge(runtime=MultiTurnScriptedRuntime([]), system_prompt="测试助手")
        turn = _turn("继续", conversation_id="conv-failed")
        success = SimpleNamespace(
            status="completed",
            tool_envelopes=(SimpleNamespace(status="ok", data={"tool_type": "mock_query"}),),
        )
        bridge._remember_task_snapshot(turn, success, task_id="task-old", final_text="旧结果", now=100)
        failure = SimpleNamespace(status="failed", tool_envelopes=())
        bridge._remember_task_snapshot(turn, failure, task_id="task-new", final_text="", now=101)
        snapshot = bridge._task_snapshot_for_turn(turn, now=101)
        self.assertEqual(snapshot["status"], "failed")
        self.assertEqual(snapshot["tools_used"], [])
        tasks = bridge.task_store.candidates(requester="master_1", session="private:conv-failed", now=101)
        self.assertEqual({task["task_id"] for task in tasks}, {"task-old", "task-new"})


if __name__ == "__main__":
    unittest.main()
