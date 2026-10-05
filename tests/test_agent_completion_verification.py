from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from shinku.agent.loop import AgentLoop, AgentState
from shinku.agent.planner import LLMPlanner, build_agent_step_prompt
from shinku.agent.verification import SmartCompletionVerifier
from shinku.hosts.tool_host import ToolHost
from shinku.tools.registry import ToolRegistry


class FileReaderHandler:
    tool_type = "file_read"

    def tool_metadata(self):
        return SimpleNamespace(input_schema={"type": "object", "properties": {"path": {"type": "string"}}})

    def build_prompt_instruction(self):
        return "读取文件。"

    def normalize_call(self, call):
        return call if call.get("path") else None

    def execute(self, *, call, context):
        return {"tool_type": "file_read", "followup_context": '{"debug": true, "version": "1.0"}'}


class FileWriterHandler:
    tool_type = "file_write"

    def tool_metadata(self):
        return SimpleNamespace(input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}})

    def build_prompt_instruction(self):
        return "写入文件。"

    def normalize_call(self, call):
        return call if call.get("path") and call.get("content") else None

    def execute(self, *, call, context):
        return {"tool_type": "file_write", "followup_context": "写入成功"}


class TaggedWriterHandler(FileWriterHandler):
    tool_type = "custom_apply"
    agent_completion_effects = ("write",)
    agent_completion_target_arguments = ("path",)


class ScriptedMultiStepRuntime:
    def __init__(self, step_responses: list[dict[str, Any]]) -> None:
        self.step_responses = list(step_responses)
        self.calls: list[dict[str, Any]] = []

    def call_chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.step_responses:
            return self.step_responses.pop(0)
        return {"kind": "final", "text": "无法继续规划。"}


class AgentCompletionVerificationTests(unittest.TestCase):
    def test_verifier_blocks_hallucinated_write_action_and_drives_real_execution(self) -> None:
        """测试模型谎称已修改但未调用写工具时被门禁准确拦截，并引导其在下一轮实际执行写工具。"""
        registry = ToolRegistry({"file_read": FileReaderHandler(), "file_write": FileWriterHandler()})
        runtime = ScriptedMultiStepRuntime([
            # Step 1: 模型调读文件工具
            {
                "kind": "tool",
                "tool_calls": [
                    {"id": "call-1", "name": "file_read", "arguments": {"path": "config.json"}}
                ],
            },
            # Step 2: 模型收到文件内容后，未调写入工具就谎称已修改 (幻觉假完成)
            {
                "kind": "final",
                "speech": "我已经帮你把 config.json 里的 debug 参数修改为 false 了！",
            },
            # Step 3: 受到门禁拦截与原因提示后，模型自愈，真正调用 file_write
            {
                "kind": "tool",
                "tool_calls": [
                    {"id": "call-2", "name": "file_write", "arguments": {"path": "config.json", "content": '{"debug": false}'}}
                ],
            },
            # Step 4: 写入成功后，交付真实结果
            {
                "kind": "final",
                "speech": "修改已实际保存至 config.json，debug 现已变更为 false。",
            },
        ])

        planner = LLMPlanner(
            runtime=runtime,
            system_prompt="你是执行助手。",
            tool_specs=registry.native_specs(),
        )
        verifier = SmartCompletionVerifier(allowed_tool_names=list(registry.names()))
        loop = AgentLoop(
            planner=planner,
            handlers=registry.view(),
            max_rounds=6,
            completion_gate=verifier,
        )

        result = loop.run(
            task_id="task::hallucinated_write",
            initial_transcript=[{"role": "user", "content": "帮我把 config.json 里的 debug 改成 false"}],
        )

        self.assertEqual(result.status, "completed")
        self.assertIn("实际保存至 config.json", result.final_text)

        # 验证门禁在 Step 2 注入了精准拒绝原因
        system_reject_messages = [
            item.get("content", "")
            for item in result.transcript
            if item.get("role") == "system"
        ]
        self.assertTrue(any("未实际执行任何写入/保存类工具" in msg for msg in system_reject_messages))

        # 验证最终确实执行了 file_write
        called_names = [
            item.get("name")
            for item in result.transcript
            if item.get("role") == "tool"
        ]
        self.assertIn("file_write", called_names)

    def test_verifier_passes_legitimate_write_action(self) -> None:
        """测试正常先读后写时，门禁顺利通过。"""
        registry = ToolRegistry({"file_write": FileWriterHandler()})
        runtime = ScriptedMultiStepRuntime([
            {
                "kind": "tool",
                "tool_calls": [
                    {"id": "call-write", "name": "file_write", "arguments": {"path": "a.txt", "content": "hello"}}
                ],
            },
            {
                "kind": "final",
                "speech": "内容已成功写入 a.txt。",
            },
        ])

        planner = LLMPlanner(runtime=runtime, system_prompt="助手", tool_specs=registry.native_specs())
        verifier = SmartCompletionVerifier(allowed_tool_names=["file_write"])
        loop = AgentLoop(planner=planner, handlers=registry.view(), completion_gate=verifier)

        result = loop.run(initial_transcript=[{"role": "user", "content": "新建 a.txt 写入 hello"}])
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.rounds, 2)

    def test_verifier_allows_plain_inquiry_and_chat(self) -> None:
        """测试普通只读问答与闲聊不被误拦。"""
        verifier = SmartCompletionVerifier(allowed_tool_names=["file_write"])
        state = AgentState(
            transcript=[{"role": "user", "content": "今天天气真好，你觉得呢？"}]
        )
        self.assertTrue(verifier(state, "确实是个好天气。"))

    def test_verifier_allows_honest_failure_reporting(self) -> None:
        """测试模型如实汇报无法完成或权限受限时放行，不陷入死循环。"""
        verifier = SmartCompletionVerifier(allowed_tool_names=["file_write"])
        state = AgentState(
            transcript=[{"role": "user", "content": "修改系统文件 /etc/hosts"}]
        )
        # 模型如实报告无法修改
        self.assertTrue(verifier(state, "抱歉，由于权限不足，我无法修改该系统保护文件。"))

    def test_failed_write_tool_does_not_satisfy_completion_claim(self) -> None:
        verifier = SmartCompletionVerifier(allowed_tool_names=["file_write"])
        state = AgentState(
            transcript=[{"role": "user", "content": "帮我修改配置文件"}],
            tool_calls=[{"name": "file_write", "status": "error"}],
        )
        result = verifier(state, "配置已经成功保存。")
        self.assertIsInstance(result, str)
        self.assertIn("未实际执行任何写入", result)

    def test_successful_write_must_match_requested_target(self) -> None:
        verifier = SmartCompletionVerifier(allowed_tool_names=["file_write"])
        state = AgentState(
            transcript=[{"role": "user", "content": "帮我把 config.json 里的 debug 改成 false"}],
            tool_calls=[{
                "name": "file_write",
                "arguments": {"path": "other.json", "content": "{}"},
                "status": "ok",
            }],
        )
        result = verifier(state, "config.json 已成功修改并保存。")
        self.assertIsInstance(result, str)
        self.assertIn("config.json", result)

    def test_successful_write_to_matching_basename_passes(self) -> None:
        verifier = SmartCompletionVerifier(allowed_tool_names=["file_write"])
        state = AgentState(
            transcript=[{"role": "user", "content": "帮我把 config.json 里的 debug 改成 false"}],
            tool_calls=[{
                "name": "file_write",
                "arguments": {"path": r"C:\\app\\config.json", "content": "{}"},
                "status": "ok",
            }],
        )
        self.assertTrue(verifier(state, "修改已实际保存至 config.json。"))

    def test_verifier_aggregates_user_messages_but_ignores_post_tool_approval(self) -> None:
        verifier = SmartCompletionVerifier(allowed_tool_names=["file_write"])
        state = AgentState(
            transcript=[
                {"role": "user", "content": "先检查一下这个设置"},
                {"role": "user", "content": "再把 config.json 里的 debug 改成 false"},
                {"role": "assistant", "kind": "tool", "tool_calls": [{"name": "file_write"}]},
                {"role": "user", "content": "好，同意执行"},
            ],
            tool_calls=[{
                "name": "file_write",
                "arguments": {"path": "config.json", "content": "{}"},
                "status": "ok",
            }],
        )
        self.assertTrue(verifier(state, "debug 已修改并成功保存到 config.json。"))

    def test_similarly_named_unregistered_tool_is_not_completion_evidence(self) -> None:
        verifier = SmartCompletionVerifier(allowed_tool_names=["my_write_report"])
        state = AgentState(
            transcript=[{"role": "user", "content": "帮我修改 config.json"}],
            tool_calls=[{
                "name": "my_write_report",
                "arguments": {"path": "config.json"},
                "status": "ok",
            }],
        )
        result = verifier(state, "config.json 已经修改完成。")
        self.assertIsInstance(result, str)

    def test_tool_host_passes_declared_handler_capabilities_to_verifier(self) -> None:
        registry = ToolRegistry({"custom_apply": TaggedWriterHandler()})
        runtime = ScriptedMultiStepRuntime([
            {"kind": "tool", "tool_calls": [{"name": "custom_apply", "arguments": {"path": "config.json", "content": "{}"}}]},
            {"kind": "final", "speech": "config.json 已修改并保存。"},
        ])
        result = ToolHost(registry).run(
            runtime=runtime,
            system_prompt="助手",
            initial_transcript=[{"role": "user", "content": "帮我修改 config.json"}],
        )
        self.assertEqual(result.status, "completed")

    def test_information_request_about_write_tool_is_not_an_action_request(self) -> None:
        verifier = SmartCompletionVerifier(allowed_tool_names=["file_write"])
        state = AgentState(
            transcript=[{"role": "user", "content": "解释一下删除文件工具的工作原理"}]
        )
        self.assertTrue(verifier(state, "它会先校验路径，再按授权范围执行删除。"))

    def test_information_request_with_action_word_is_not_an_action_request(self) -> None:
        verifier = SmartCompletionVerifier(allowed_tool_names=["file_write"])
        state = AgentState(
            transcript=[{"role": "user", "content": "我想知道删除文件工具有什么风险"}]
        )
        self.assertTrue(verifier(state, "它可能造成数据丢失，所以执行前应核对路径并保留备份。"))

    def test_explicitly_negated_action_does_not_require_a_tool(self) -> None:
        verifier = SmartCompletionVerifier(allowed_tool_names=["file_write"])
        state = AgentState(
            transcript=[{"role": "user", "content": "不要修改配置，只告诉我修改它的原理"}]
        )
        self.assertTrue(verifier(state, "配置文件通常由程序读取，具体字段含义要看对应程序的文档。"))

    def test_failure_and_success_claim_cannot_use_failure_bypass(self) -> None:
        verifier = SmartCompletionVerifier(allowed_tool_names=["file_write"])
        state = AgentState(
            transcript=[{"role": "user", "content": "帮我修改配置文件"}],
            tool_calls=[{"name": "file_write", "status": "error"}],
        )
        result = verifier(state, "权限不足，无法修改；不过配置已经成功保存。")
        self.assertIsInstance(result, str)
        self.assertIn("未实际执行任何写入", result)

    def test_goal_anchoring_in_prompt(self) -> None:
        """测试多轮大文本下，原始任务目标在工具反馈阶段被正确提取与置顶。"""
        transcript = [
            {"role": "user", "content": "帮我把生产环境数据库配置修改为本地代理"},
            {"role": "assistant", "kind": "tool", "content": "调用工具 read"},
            {"role": "tool", "name": "file_read", "status": "ok", "content": "一大段几千字的长文本数据..."},
        ]
        prompt = build_agent_step_prompt(transcript)
        self.assertIn("【工具执行结果反馈与推进】", prompt)
        self.assertIn("[原始任务目标]\n帮我把生产环境数据库配置修改为本地代理", prompt)
        self.assertIn("【交付自检】", prompt)


if __name__ == "__main__":
    unittest.main()
