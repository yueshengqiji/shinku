from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from shinku.hosts.tool_host import ToolHost
from shinku.persona.loader import load_persona
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.tool_selection import compact_tool_dispatch_spec
from shinku.tools.registry import ToolRegistry


class _Runtime:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def chat_supports_native_tools(self):
        return True

    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class _Handler:
    def __init__(self, name: str):
        self.tool_type = name

    def tool_metadata(self):
        description = f"检查并处理本机的{name_words(self.tool_type)}相关任务，按请求返回准确结果。"
        return SimpleNamespace(
            description=description,
            input_schema={
                "type": "object",
                "description": description,
                "properties": {
                    f"field_{index}": {
                        "type": "string",
                        "description": "这是一个说明较长的参数，用来明确本字段实际要传递的内容和格式。",
                    }
                    for index in range(6)
                },
                "required": ["field_0"],
                "additionalProperties": False,
            },
        )

    def normalize_call(self, call):
        return call

    def execute(self, *, call, context):
        del context
        return {"tool_type": self.tool_type, "followup_context": str(call)}


def name_words(value: str) -> str:
    return value.replace("_", " ")


class CompactToolRouterTests(unittest.TestCase):
    def _bridge(self, runtime: _Runtime, names: tuple[str, ...]) -> QQAgentBridge:
        registry = ToolRegistry({name: _Handler(name) for name in names})
        return QQAgentBridge(
            runtime=runtime,
            system_prompt="真红主人设",
            tool_host=ToolHost(registry),
            allowed_tool_names=set(names),
            master_user_ids=("master",),
            owner_gate_required=True,
        )

    @staticmethod
    def _context() -> dict:
        return {"audience_context": {"is_master": True, "role": "master"}}

    def test_large_catalog_is_compact_and_model_choice_maps_to_exact_tool(self) -> None:
        names = tuple(f"private_tool_{index}" for index in range(8))
        runtime = _Runtime(
            {
                "kind": "tool",
                "tool_call": {
                    "name": "__shinku_tool_dispatch__",
                    "arguments": {
                        "tool_name": "private_tool_5",
                        "arguments": {"field_0": "从模型的选择原样传递"},
                    },
                },
            }
        )
        bridge = self._bridge(runtime, names)
        trace_id = bridge.trace_store.begin(
            SimpleNamespace(
                conversation_type="private",
                conversation_id="master",
                primary_message_id="test-compact-route",
                message_ids=("test-compact-route",),
                combined_text="帮我完成一个需要本机工具的任务",
            )
        )

        decision, available = bridge._run_chat_decision(
            task_id="test-compact-route",
            context=self._context(),
            initial_transcript=[{"role": "user", "content": "帮我完成一个需要本机工具的任务"}],
            trace_id=trace_id,
        )

        self.assertEqual(decision.kind, "tool")
        self.assertEqual(decision.invocation.name, "private_tool_5")
        self.assertEqual(decision.invocation.arguments, {"field_0": "从模型的选择原样传递"})
        self.assertEqual(available, set(names))
        self.assertEqual(len(runtime.calls), 1)
        offered = runtime.calls[0]["native_tools"]
        self.assertEqual(len(offered), 1)
        self.assertEqual(offered[0]["function"]["name"], "__shinku_tool_dispatch__")
        system_blocks = "\n".join(runtime.calls[0]["system_extra_blocks"])
        self.assertIn("private_tool_5", system_blocks)
        self.assertIn("按当前任务的真实需要自主判断", system_blocks)
        chat_start = next(
            stage
            for stage in bridge.trace_store.snapshot()[0]["stages"]
            if stage["name"] == "chat_start"
        )
        self.assertEqual(chat_start["tool_route"], "compact_selector")
        self.assertGreater(chat_start["tool_context_chars_saved"], 0)

    def test_invalid_compact_selection_retries_with_canonical_tools_and_never_executes_it(self) -> None:
        names = tuple(f"private_tool_{index}" for index in range(8))
        runtime = _Runtime(
            {
                "kind": "tool",
                "tool_call": {
                    "name": "__shinku_tool_dispatch__",
                    "arguments": {"tool_name": "not_registered", "arguments": {}},
                },
            },
            {
                "kind": "tool",
                "tool_call": {"name": "private_tool_2", "arguments": {"field_0": "recovered"}},
            },
        )
        bridge = self._bridge(runtime, names)

        decision, _ = bridge._run_chat_decision(
            task_id="test-invalid-route",
            context=self._context(),
            initial_transcript=[{"role": "user", "content": "调用适当工具处理这项任务"}],
        )

        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual(decision.kind, "tool")
        self.assertEqual(decision.invocation.name, "private_tool_2")
        self.assertEqual(runtime.calls[1]["native_tools"][0]["function"]["name"], "private_tool_0")


    def test_turn_boundary_prioritizes_a_correction_over_previous_assistant_interpretation(self) -> None:
        runtime = _Runtime({"speech": "嗯，我明白了。"})
        bridge = QQAgentBridge(runtime=runtime, system_prompt="真红主人设")

        bridge._run_chat_decision(
            task_id="test-correction",
            context=self._context(),
            initial_transcript=[
                {"role": "assistant", "content": "想笑就笑吧，我又没生气。"},
                {"role": "user", "content": "没事，不是在嘲笑你。"},
            ],
        )

        system_blocks = "\n".join(runtime.calls[0]["system_extra_blocks"])
        self.assertIn("以这条新信息更新旧推测", system_blocks)
        self.assertIn("不要重复旧回应来替自己辩解", system_blocks)

    def test_compact_selection_with_json_string_arguments_and_target_tool_alias(self) -> None:
        names = tuple(f"private_tool_{index}" for index in range(8))
        runtime = _Runtime(
            {
                "kind": "tool",
                "tool_call": {
                    "name": "__shinku_tool_dispatch__",
                    "arguments": {
                        "target_tool": "private_tool_3",
                        "arguments": '{"field_0": "stringified_json_param"}',
                    },
                },
            }
        )
        bridge = self._bridge(runtime, names)
        decision, _ = bridge._run_chat_decision(
            task_id="test-string-args",
            context=self._context(),
            initial_transcript=[{"role": "user", "content": "测试字符串参数"}],
        )
        self.assertEqual(decision.kind, "tool")
        self.assertEqual(decision.invocation.name, "private_tool_3")
        self.assertEqual(decision.invocation.arguments, {"field_0": "stringified_json_param"})

    def test_compact_selection_fallback_triggers_canonical_schema_retry(self) -> None:
        names = tuple(f"private_tool_{index}" for index in range(8))
        runtime = _Runtime(
            {"kind": "final", "text": "……刚才那句我没接好。你再说一遍？"},
            {
                "kind": "tool",
                "tool_call": {"name": "private_tool_1", "arguments": {"field_0": "recovered_after_malformed"}},
            },
        )
        bridge = self._bridge(runtime, names)
        decision, _ = bridge._run_chat_decision(
            task_id="test-fallback-retry",
            context=self._context(),
            initial_transcript=[{"role": "user", "content": "下载浏览器自动化"}],
        )
        self.assertEqual(len(runtime.calls), 2)
        self.assertEqual(decision.kind, "tool")
        self.assertEqual(decision.invocation.name, "private_tool_1")
        self.assertEqual(decision.invocation.arguments, {"field_0": "recovered_after_malformed"})

    def test_full_schema_is_selected_when_compact_directory_costs_more(self) -> None:
        names = ("single_small_tool",)
        runtime = _Runtime({"kind":"final", "text":"直接回答"})
        bridge = self._bridge(runtime, names)
        trace_id = bridge.trace_store.begin(SimpleNamespace(
            conversation_type="private", conversation_id="master",
            primary_message_id="test-full-schema-size", message_ids=("test-full-schema-size",),
            combined_text="直接回答即可"))

        decision, _ = bridge._run_chat_decision(
            task_id="test-full-schema-size",
            context=self._context(),
            initial_transcript=[{"role":"user", "content":"直接回答即可"}],
            trace_id=trace_id,
        )

        self.assertEqual(decision.kind, "final")
        self.assertEqual([tool["function"]["name"] for tool in runtime.calls[0]["native_tools"]],
                         ["single_small_tool"])
        stage = next(stage for stage in bridge.trace_store.snapshot()[0]["stages"]
                     if stage["name"] == "chat_start")
        self.assertEqual(stage["tool_route"], "native_schema")
        self.assertEqual(stage["tool_context_chars_after"], stage["tool_context_chars_before"])

    def test_required_nested_parameters_past_optional_preview_are_preserved_generically(self) -> None:
        date_shape = {
            "type":"object", "description":"另名的日期范围，不得截断结构说明。",
            "properties": {
                "from_day":{"type":"string", "maxLength":10, "description":"包含的起点。"},
                "until_day":{"type":"string", "maxLength":10, "description":"排除的终点。"},
            }, "required":["from_day", "until_day"], "additionalProperties":False,
        }
        step_shape = {
            "type":"array", "description":"执行步骤列表，嵌套结构也必须保留。",
            "items":{"type":"object", "properties":{
                "operation":{"type":"string", "enum":["read", "check"]},
            }, "required":["operation"], "additionalProperties":False},
        }
        properties = {
            **{f"optional_{index}":{"type":"string", "description":f"普通可选参数{index}"}
               for index in range(13)},
            "custom_window":date_shape,
            "plan_steps":step_shape,
        }
        spec = {"type":"function", "function":{"name":"renamed_calendar_reader",
            "description":"读取指定范围并执行轻量校验。", "parameters":{
                "type":"object", "properties":properties,
                "required":["custom_window", "plan_steps"], "additionalProperties":False}}}
        before = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))

        _, catalog = compact_tool_dispatch_spec([spec])

        self.assertIn("custom_window (必填，完整参数定义):" +
                      json.dumps(date_shape, ensure_ascii=False, separators=(",", ":")), catalog)
        self.assertIn("plan_steps (必填，完整参数定义):" +
                      json.dumps(step_shape, ensure_ascii=False, separators=(",", ":")), catalog)
        self.assertIn("另有1项", catalog)
        self.assertNotIn("optional_12:", catalog)
        self.assertEqual(json.dumps(spec, ensure_ascii=False, separators=(",", ":")), before)


if __name__ == "__main__":
    unittest.main()
