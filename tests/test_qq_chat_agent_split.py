from __future__ import annotations

import unittest

from shinku.agent.loop import AgentRunResult
from shinku.hosts.tool_host import ToolHost
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.delivery import DeliveryResult
from shinku.qq.message import IncomingMessage
from shinku.qq.turns import QQTurn
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry


class _Runtime:
    def __init__(self, response, *, native_tools: bool = False) -> None:
        self.response = response
        self.native_tools = native_tools
        self.calls = []

    def chat_supports_native_tools(self):
        return self.native_tools

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class _EchoHandler:
    tool_type = "echo"

    def normalize_call(self, call):
        return call if isinstance(call, dict) else None

    def execute(self, *, call, context):
        del context
        return {"tool_type": "echo", "followup_context": str(call.get("text") or "")}


def _turn(text: str = "午安。", *, sender_id: str = "u-19") -> QQTurn:
    message = IncomingMessage.from_event(
        {
            "post_type": "message",
            "message_type": "private",
            "user_id": sender_id,
            "message_id": "m-chat-split",
            "message": [{"type": "text", "data": {"text": text}}],
        }
    )
    return QQTurn(
        conversation_id=sender_id,
        conversation_type="private",
        primary_message_id="m-chat-split",
        message_ids=("m-chat-split",),
        combined_text=text,
        messages=(message,),
    )


class QQChatAgentSplitTests(unittest.TestCase):
    def test_ordinary_chat_is_delivered_without_entering_agent_loop(self) -> None:
        runtime = _Runtime({"speech": "嗯……午安。"})
        bridge = QQAgentBridge(runtime=runtime, system_prompt="真红主人设")

        def unexpected_agent_call(**kwargs):
            raise AssertionError("ordinary chat must not enter AgentLoop")

        bridge._run_agent = unexpected_agent_call
        result = bridge.handle_turn(_turn())

        self.assertEqual((result.agent_status, result.final_text), ("completed", "嗯……午安。"))
        self.assertEqual(len(runtime.calls), 1)
        self.assertIsNone(runtime.calls[0].get("native_tools"))
        self.assertNotIn("Agent 决策规则", "\n".join(runtime.calls[0]["system_extra_blocks"]))

    def test_disabling_agent_keeps_normal_replies_enabled_and_hides_tools(self) -> None:
        state = QQCommandState()
        state.set_agent_enabled(False)
        runtime = _Runtime({"speech": "我听着。"}, native_tools=True)
        bridge = QQAgentBridge(runtime=runtime, system_prompt="persona", command_state=state)

        result = bridge.handle_turn(_turn("在吗"))

        self.assertTrue(state.responses_enabled())
        self.assertFalse(state.is_agent_enabled())
        self.assertEqual(result.final_text, "我听着。")
        self.assertIsNone(runtime.calls[0].get("native_tools"))

    def test_model_selected_tool_is_handed_to_agent_as_the_first_decision(self) -> None:
        runtime = _Runtime(
            {"kind": "tool", "tool_call": {"name": "echo", "arguments": {"text": "检查一下"}}},
            native_tools=True,
        )
        host = ToolHost(ToolRegistry({"echo": _EchoHandler()}))
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="persona",
            tool_host=host,
            allowed_tool_names={"echo"},
            master_user_ids=("u-19",),
            owner_gate_required=True,
        )
        captured = {}

        def fake_agent_run(**kwargs):
            captured.update(kwargs)
            return AgentRunResult(status="waiting_user", rounds=1, final_text="要执行这一步吗？")

        bridge._run_agent = fake_agent_run
        result = bridge.handle_turn(_turn("帮我检查一下"))

        self.assertEqual(result.agent_status, "waiting_user")
        self.assertEqual(captured["first_decision"].kind, "tool")
        self.assertEqual(captured["first_decision"].invocation.name, "echo")
        self.assertEqual(captured["first_decision"].invocation.arguments, {"text": "检查一下"})
        self.assertEqual(runtime.calls[0]["native_tool_choice"], "auto")
        self.assertEqual(
            [item["function"]["name"] for item in runtime.calls[0]["native_tools"]],
            ["echo"],
        )

    def test_non_master_cannot_receive_or_execute_tool_decisions(self) -> None:
        runtime = _Runtime(
            {"kind": "tool", "tool_call": {"name": "echo", "arguments": {"text": "x"}}},
            native_tools=True,
        )
        host = ToolHost(ToolRegistry({"echo": _EchoHandler()}))
        sent = []
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="persona",
            tool_host=host,
            allowed_tool_names={"echo"},
            master_user_ids=("master-id",),
            owner_gate_required=True,
            sender=lambda message: sent.append(message) or DeliveryResult(True, "sent", ("out-1",)),
        )

        result = bridge.handle_turn(_turn("帮我检查一下", sender_id="other-id"))

        self.assertEqual(result.reason, "tool_unavailable_or_unauthorized")
        self.assertIn("只听主人的", result.final_text)
        self.assertIsNone(runtime.calls[0].get("native_tools"))
        self.assertTrue(sent)
        self.assertIn("只听主人的", "".join(message.segments[0].text for message in sent))

    def test_heavy_tool_runs_without_synthetic_early_acknowledgment(self) -> None:
        class _HeavyEchoHandler:
            tool_type = "private_web_search"
            description = "search"
            input_schema = {"type": "object"}

            def normalize_call(self, call):
                return call if isinstance(call, dict) else None

            def execute(self, *, call, context):
                del context
                return {"tool_type": "private_web_search", "followup_context": ""}

        runtime = _Runtime(
            {"kind": "tool", "tool_call": {"name": "private_web_search", "arguments": {"query": "shinku"}}},
            native_tools=True,
        )
        host = ToolHost(ToolRegistry({"private_web_search": _HeavyEchoHandler()}))
        sent_messages = []
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="persona",
            tool_host=host,
            allowed_tool_names={"private_web_search"},
            master_user_ids=("u-19",),
            owner_gate_required=True,
            sender=lambda message: sent_messages.append(message) or DeliveryResult(True, "sent", (f"out-{len(sent_messages)}",)),
        )

        def fake_agent_run(**kwargs):
            # Assert no synthetic canned early ack is sent before _run_agent finishes
            self.assertEqual(len(sent_messages), 0)
            return AgentRunResult(status="completed", rounds=1, final_text="搜索结果是……")

        bridge._run_agent = fake_agent_run
        result = bridge.handle_turn(_turn("帮我查一下"))

        self.assertEqual(result.agent_status, "completed")
        self.assertEqual(len(sent_messages), 1)
        self.assertIn("搜索结果是……", sent_messages[0].segments[0].text)

    def test_lightweight_tool_skips_early_acknowledgment(self) -> None:
        runtime = _Runtime(
            {"kind": "tool", "tool_call": {"name": "echo", "arguments": {"text": "hello"}}},
            native_tools=True,
        )
        host = ToolHost(ToolRegistry({"echo": _EchoHandler()}))
        sent_messages = []
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="persona",
            tool_host=host,
            allowed_tool_names={"echo"},
            master_user_ids=("u-19",),
            owner_gate_required=True,
            sender=lambda message: sent_messages.append(message) or DeliveryResult(True, "sent", ("out-1",)),
        )

        def fake_agent_run(**kwargs):
            # No early ack for lightweight echo tool
            self.assertEqual(len(sent_messages), 0)
            return AgentRunResult(status="completed", rounds=1, final_text="完成。")

        bridge._run_agent = fake_agent_run
        result = bridge.handle_turn(_turn("帮我回声"))

        self.assertEqual(result.agent_status, "completed")
        self.assertEqual(len(sent_messages), 1)
        self.assertEqual(sent_messages[0].segments[0].text, "完成。")

    def test_lightweight_perception_tool_bypasses_contract_and_runs_fast_track(self) -> None:
        executed_calls = []

        class _PerceptionHandler:
            tool_type = "private_current_time"
            agent_permission_class = "read"
            agent_progress_class = "observe"
            agent_completion_effects = ()

            def normalize_call(self, call):
                return call if isinstance(call, dict) else {}

            def execute(self, *, call, context):
                del context
                executed_calls.append(call)
                return {"tool_type": "private_current_time", "followup_context": "2026-10-02 14:00:00"}

        runtime = _Runtime(
            {"kind": "tool", "tool_call": {"name": "private_current_time", "arguments": {}}},
            native_tools=True,
        )
        host = ToolHost(ToolRegistry({"private_current_time": _PerceptionHandler()}))
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="二阶堂真红",
            tool_host=host,
            policy=ExecutionPolicy(risk_by_tool={"private_current_time": "low"}),
            allowed_tool_names={"private_current_time"},
            master_user_ids=("u-19",),
            owner_gate_required=True,
        )

        def unexpected_heavy_call(**kwargs):
            raise AssertionError("Lightweight perception must bypass contract preparation and AgentLoop!")

        bridge._prepare_agent_task = unexpected_heavy_call
        bridge._run_agent = unexpected_heavy_call

        call_count = 0

        def dynamic_call_chat_json(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"kind": "tool", "tool_call": {"name": "private_current_time", "arguments": {}}}
            self.assertIn("2026-10-02 14:00:00", kwargs["user_prompt"])
            self.assertIsNone(kwargs.get("native_tools"))
            return {"kind": "final", "text": "现在是下午两点整。"}

        runtime.call_chat_json = dynamic_call_chat_json

        result = bridge.handle_turn(_turn("现在几点了"))

        self.assertEqual(result.agent_status, "completed")
        self.assertEqual(result.reason, "lightweight_perception_fast_track")
        self.assertEqual(result.final_text, "现在是下午两点整。")
        self.assertEqual(len(executed_calls), 1)
        self.assertEqual(call_count, 2)

    def test_mutating_or_side_effect_tool_does_not_use_fast_track(self) -> None:
        class _MutatingHandler:
            tool_type = "private_file_write"
            agent_permission_class = "change"
            agent_progress_class = "act"
            agent_completion_effects = ("write",)

            def normalize_call(self, call):
                return call if isinstance(call, dict) else {}

            def execute(self, *, call, context):
                del context
                return {"tool_type": "private_file_write", "followup_context": "written"}

        runtime = _Runtime(
            {"kind": "tool", "tool_call": {"name": "private_file_write", "arguments": {"path": "a.txt"}}},
            native_tools=True,
        )
        host = ToolHost(ToolRegistry({"private_file_write": _MutatingHandler()}))
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="persona",
            tool_host=host,
            allowed_tool_names={"private_file_write"},
            master_user_ids=("u-19",),
            owner_gate_required=True,
        )

        agent_invoked = False

        def fake_agent_run(**kwargs):
            nonlocal agent_invoked
            agent_invoked = True
            return AgentRunResult(status="completed", rounds=1, final_text="已写入。")

        bridge._run_agent = fake_agent_run
        result = bridge.handle_turn(_turn("帮我写入文件"))

        self.assertTrue(agent_invoked)
        self.assertEqual(result.final_text, "已写入。")

    def test_continuing_task_link_does_not_use_fast_track(self) -> None:
        class _PerceptionHandler:
            tool_type = "private_current_time"
            agent_permission_class = "read"
            agent_progress_class = "observe"
            agent_completion_effects = ()

            def normalize_call(self, call):
                return call if isinstance(call, dict) else {}

            def execute(self, *, call, context):
                del context
                return {"tool_type": "private_current_time", "followup_context": "2026-10-02 14:00:00"}

        runtime = _Runtime(
            {
                "kind": "tool",
                "tool_call": {"name": "private_current_time", "arguments": {}},
                "task_link": {"mode": "continue", "task_id": "existing-task-1"},
            },
            native_tools=True,
        )
        host = ToolHost(ToolRegistry({"private_current_time": _PerceptionHandler()}))
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="persona",
            tool_host=host,
            allowed_tool_names={"private_current_time"},
            master_user_ids=("u-19",),
            owner_gate_required=True,
        )

        prepare_invoked = False
        agent_invoked = False

        def fake_prepare(turn, decision, context, task_id, trace_id):
            nonlocal prepare_invoked
            prepare_invoked = True
            return task_id, decision

        def fake_agent_run(**kwargs):
            nonlocal agent_invoked
            agent_invoked = True
            return AgentRunResult(status="completed", rounds=1, final_text="已续接并完成。")

        bridge._prepare_agent_task = fake_prepare
        bridge._run_agent = fake_agent_run
        result = bridge.handle_turn(_turn("继续看时间"))

        self.assertTrue(prepare_invoked)
        self.assertTrue(agent_invoked)
        self.assertEqual(result.final_text, "已续接并完成。")

    def test_lightweight_perception_parallel_execution(self) -> None:
        executed_calls = []

        class _HandlerA:
            tool_type = "private_current_time"
            agent_permission_class = "read"
            agent_progress_class = "observe"
            agent_completion_effects = ()

            def normalize_call(self, call):
                return call if isinstance(call, dict) else {}

            def execute(self, *, call, context):
                del context
                executed_calls.append("A")
                return {"tool_type": "private_current_time", "followup_context": "2026-10-02 14:00:00"}

        class _HandlerB:
            tool_type = "private_character_context"
            agent_permission_class = "read"
            agent_progress_class = "observe"
            agent_completion_effects = ()

            def normalize_call(self, call):
                return call if isinstance(call, dict) else {}

            def execute(self, *, call, context):
                del context
                executed_calls.append("B")
                return {"tool_type": "private_character_context", "followup_context": "真红/爱丽丝游戏"}

        runtime = _Runtime(
            {
                "kind": "tool",
                "tool_calls": [
                    {"name": "private_current_time", "arguments": {}},
                    {"name": "private_character_context", "arguments": {}},
                ],
            },
            native_tools=True,
        )
        host = ToolHost(ToolRegistry({"private_current_time": _HandlerA(), "private_character_context": _HandlerB()}))
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="二阶堂真红",
            tool_host=host,
            policy=ExecutionPolicy(risk_by_tool={"private_current_time": "low", "private_character_context": "low"}),
            allowed_tool_names={"private_current_time", "private_character_context"},
            master_user_ids=("u-19",),
            owner_gate_required=True,
        )

        call_count = 0

        def dynamic_call_chat_json(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "kind": "tool",
                    "tool_calls": [
                        {"name": "private_current_time", "arguments": {}},
                        {"name": "private_character_context", "arguments": {}},
                    ],
                }
            self.assertIn("2026-10-02 14:00:00", kwargs["user_prompt"])
            self.assertIn("真红/爱丽丝游戏", kwargs["user_prompt"])
            return {"kind": "final", "text": "我是真红，现在是下午两点。"}

        runtime.call_chat_json = dynamic_call_chat_json
        result = bridge.handle_turn(_turn("查看当前时间和状态"))

        self.assertEqual(result.agent_status, "completed")
        self.assertEqual(result.reason, "lightweight_perception_fast_track")
        self.assertEqual(result.final_text, "我是真红，现在是下午两点。")
        self.assertEqual(set(executed_calls), {"A", "B"})

    def test_lightweight_perception_handles_tool_error_gracefully(self) -> None:
        class _FailingHandler:
            tool_type = "private_file_read"
            agent_permission_class = "read"
            agent_progress_class = "observe"
            agent_completion_effects = ()

            def normalize_call(self, call):
                return call if isinstance(call, dict) else {}

            def execute(self, *, call, context):
                del context
                raise RuntimeError("Sensor IO failure")

        runtime = _Runtime(
            {"kind": "tool", "tool_call": {"name": "private_file_read", "arguments": {"path": "missing.txt"}}},
            native_tools=True,
        )
        host = ToolHost(ToolRegistry({"private_file_read": _FailingHandler()}))
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="二阶堂真红",
            tool_host=host,
            policy=ExecutionPolicy(risk_by_tool={"private_file_read": "low"}),
            allowed_tool_names={"private_file_read"},
            master_user_ids=("u-19",),
            owner_gate_required=True,
        )

        call_count = 0

        def dynamic_call_chat_json(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"kind": "tool", "tool_call": {"name": "private_file_read", "arguments": {"path": "missing.txt"}}}
            self.assertTrue(any("部分感知工具回执显示错误" in block for block in kwargs.get("system_extra_blocks", [])))
            return {"kind": "final", "text": "探测遇到了些阻碍，没有读取到该文件。"}

        runtime.call_chat_json = dynamic_call_chat_json
        result = bridge.handle_turn(_turn("帮我读一下文件"))

        self.assertEqual(result.agent_status, "completed")
        self.assertEqual(result.reason, "lightweight_perception_fast_track")
        self.assertEqual(result.final_text, "探测遇到了些阻碍，没有读取到该文件。")

    def test_lightweight_perception_handles_read_self_state(self) -> None:
        from shinku.agent.self_state import SelfStateTool

        state_tool = SelfStateTool(provider=lambda: {"model": "test-model", "persona": "二阶堂真红"})
        runtime = _Runtime(
            {"kind": "tool", "tool_call": {"name": "read_self_state", "arguments": {"scope": "all"}}},
            native_tools=True,
        )
        host = ToolHost(ToolRegistry({"read_self_state": state_tool}))
        bridge = QQAgentBridge(
            runtime=runtime,
            system_prompt="二阶堂真红",
            tool_host=host,
            policy=ExecutionPolicy(risk_by_tool={"read_self_state": "low"}),
            allowed_tool_names={"read_self_state"},
            master_user_ids=("u-19",),
            owner_gate_required=True,
        )

        call_count = 0

        def dynamic_call_chat_json(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"kind": "tool", "tool_call": {"name": "read_self_state", "arguments": {"scope": "all"}}}
            self.assertIn("当前实例的只读状态如下", kwargs.get("user_prompt", ""))
            return {"kind": "final", "text": "我是二阶堂真红，当前使用 test-model 运行。"}

        runtime.call_chat_json = dynamic_call_chat_json
        result = bridge.handle_turn(_turn("那你是什么"))

        self.assertEqual(result.agent_status, "completed")
        self.assertEqual(result.reason, "lightweight_perception_fast_track")
        self.assertEqual(result.final_text, "我是二阶堂真红，当前使用 test-model 运行。")


if __name__ == "__main__":
    unittest.main()
