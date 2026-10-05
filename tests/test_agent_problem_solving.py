from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from shinku.agent.loop import AgentLoop, AgentDecision, AgentState
from shinku.agent.planner import LLMPlanner, build_agent_step_prompt
from shinku.tools.registry import ToolRegistry


class DocSearchHandler:
    tool_type = "search_docs"

    def tool_metadata(self):
        return SimpleNamespace(input_schema={"type": "object", "properties": {"query": {"type": "string"}}})

    def build_prompt_instruction(self):
        return "按关键词搜索文档。"

    def normalize_call(self, call):
        return call if call.get("query") else None

    def execute(self, *, call, context):
        query = call.get("query", "")
        if "101" in query:
            return {"tool_type": "search_docs", "followup_context": '{"doc_id": "101", "title": "系统架构规范"}'}
        return {"tool_type": "search_docs", "followup_context": '{"results": []}'}


class DocDetailHandler:
    tool_type = "get_doc_detail"

    def tool_metadata(self):
        return SimpleNamespace(input_schema={"type": "object", "properties": {"doc_id": {"type": "string"}}})

    def build_prompt_instruction(self):
        return "获取指定文档的详细内容。"

    def normalize_call(self, call):
        return call if call.get("doc_id") else None

    def execute(self, *, call, context):
        doc_id = call.get("doc_id", "")
        if doc_id == "101":
            return {"tool_type": "get_doc_detail", "followup_context": "系统架构规范核心要求必须实现三层解耦与可观测审计。"}
        return {"tool_type": "get_doc_detail", "followup_context": "未找到文档"}


class FlakyHandler:
    tool_type = "flaky_tool"

    def tool_metadata(self):
        return SimpleNamespace(input_schema={"type": "object", "properties": {"attempt": {"type": "integer"}}})

    def build_prompt_instruction(self):
        return "可能失败的工具。"

    def normalize_call(self, call):
        return call

    def execute(self, *, call, context):
        attempt = call.get("attempt", 0)
        if attempt < 2:
            return {"status": "error", "error": "attempt_too_low", "message": "参数 attempt 过小，至少为 2"}
        return {"status": "ok", "followup_context": "已修复参数并成功获取配置"}


class MultiStepScriptedRuntime:
    """模拟多步 ReAct 规划的测试 Runtime。"""

    def __init__(self, step_responses: list[dict[str, Any]]) -> None:
        self.step_responses = list(step_responses)
        self.calls: list[dict[str, Any]] = []

    def call_chat_json(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.step_responses:
            return self.step_responses.pop(0)
        return {"kind": "final", "text": "无法继续规划。"}


class AgentProblemSolvingLoopTests(unittest.TestCase):
    def test_multi_step_tool_chain_problem_solving(self) -> None:
        """测试多步工具依赖链：搜索 -> 读取详情 -> 给出最终完整答复。"""
        registry = ToolRegistry({"search_docs": DocSearchHandler(), "get_doc_detail": DocDetailHandler()})
        runtime = MultiStepScriptedRuntime([
            # Step 1: 模型决定搜索文档 101
            {
                "kind": "tool",
                "tool_calls": [
                    {"id": "call-search", "name": "search_docs", "arguments": {"query": "文档 101"}}
                ],
            },
            # Step 2: 模型收到搜索结果，决定调用 get_doc_detail 获取详情
            {
                "kind": "tool",
                "tool_calls": [
                    {"id": "call-detail", "name": "get_doc_detail", "arguments": {"doc_id": "101"}}
                ],
            },
            # Step 3: 模型收到详细内容，给出解决问题的最终答复
            {
                "kind": "final",
                "speech": "根据文档 101《系统架构规范》，核心要求是必须实现三层解耦与可观测审计。",
            },
        ])

        planner = LLMPlanner(
            runtime=runtime,
            system_prompt="你是问题解决助手。",
            tool_specs=registry.native_specs(),
        )
        loop = AgentLoop(planner=planner, handlers=registry.view(), max_rounds=5)
        result = loop.run(
            task_id="task::multi_step_chain",
            initial_transcript=[{"role": "user", "content": "帮我查一下文档 101 的核心要求并总结"}],
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.rounds, 3)
        self.assertIn("三层解耦与可观测审计", result.final_text)
        self.assertEqual(len(result.tool_envelopes), 2)

        # 验证时序记录的完整性与因果链
        roles = [item.get("role") for item in result.transcript]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant", "tool"])

        # 反馈规则位于 system 附加块，工具输出作为 user 侧证据传入。
        second_call_prompt = runtime.calls[1]["user_prompt"]
        self.assertIn("【Agent 工具结果处理】", "\n".join(runtime.calls[1]["system_extra_blocks"]))
        self.assertIn("search_docs", second_call_prompt)
        self.assertIn("系统架构规范", second_call_prompt)
        self.assertNotIn("根据当前任务继续处理", second_call_prompt)
        self.assertIn("帮我查一下文档 101 的核心要求并总结", runtime.calls[1]["history_turns"][0]["content"])

        # 验证第三轮规划时的 user_prompt 包含第二步工具输出
        third_call_prompt = runtime.calls[2]["user_prompt"]
        self.assertIn("【Agent 工具结果处理】", "\n".join(runtime.calls[2]["system_extra_blocks"]))
        self.assertIn("get_doc_detail", third_call_prompt)
        self.assertIn("三层解耦与可观测审计", third_call_prompt)

    def test_tool_error_recovery_in_loop(self) -> None:
        """测试工具执行失败后的错误反馈引导与自愈恢复。"""
        registry = ToolRegistry({"flaky_tool": FlakyHandler()})
        runtime = MultiStepScriptedRuntime([
            # Step 1: 尝试调用参数 attempt=1，将触发工具报错
            {
                "kind": "tool",
                "tool_calls": [
                    {"id": "call-flaky-1", "name": "flaky_tool", "arguments": {"attempt": 1}}
                ],
            },
            # Step 2: 模型看到错误反馈，自愈调整参数为 attempt=2
            {
                "kind": "tool",
                "tool_calls": [
                    {"id": "call-flaky-2", "name": "flaky_tool", "arguments": {"attempt": 2}}
                ],
            },
            # Step 3: 工具成功，交付结果
            {
                "kind": "final",
                "speech": "配置已通过重试成功获取。",
            },
        ])

        planner = LLMPlanner(
            runtime=runtime,
            system_prompt="你是自愈助手。",
            tool_specs=registry.native_specs(),
        )
        loop = AgentLoop(planner=planner, handlers=registry.view(), max_rounds=5)
        result = loop.run(
            task_id="task::error_recovery",
            initial_transcript=[{"role": "user", "content": "获取目标配置"}],
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.rounds, 3)
        self.assertIn("重试成功获取", result.final_text)

        # 验证第二轮 prompt 包含了错误状态或错误反馈提示
        second_call_prompt = runtime.calls[1]["user_prompt"]
        self.assertIn("【Agent 工具结果处理】", "\n".join(runtime.calls[1]["system_extra_blocks"]))
        self.assertIn("flaky_tool", second_call_prompt)
        self.assertIn("参数 attempt 过小", second_call_prompt)
        self.assertEqual([item.status for item in result.tool_envelopes], ["error", "ok"])

    def test_build_agent_step_prompt_formatting(self) -> None:
        """测试构建 prompt 时的分段逻辑与工具 tag 状态显示。"""
        # 初次进入任务，没有工具反馈
        initial_transcript = [{"role": "user", "content": "你好"}]
        prompt_1 = build_agent_step_prompt(initial_transcript)
        self.assertIn("根据当前任务继续处理", prompt_1)
        self.assertNotIn("【工具执行结果反馈与推进】", prompt_1)

        # 刚执行完工具，最近一条为 tool
        tool_transcript = [
            {"role": "user", "content": "查数据"},
            {"role": "assistant", "kind": "tool", "content": "调用工具 `data_fetch`"},
            {"role": "tool", "name": "data_fetch", "status": "ok", "content": "数据结果: 100"},
        ]
        prompt_2 = build_agent_step_prompt(tool_transcript)
        self.assertIn("【工具执行结果反馈与推进】", prompt_2)
        self.assertIn("[工具返回数据 tool=data_fetch]", prompt_2)
        self.assertIn("数据结果: 100", prompt_2)


if __name__ == "__main__":
    unittest.main()
