"""Context budget is metadata-only by default and never trims protected input."""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shinku.agent.planner import build_agent_step_prompt
from shinku.llm.context_budget import (
    ContextBudgetPolicy,
    assess_and_trim_payload,
    estimate_text_tokens,
    measure_payload,
)
from test_contract_c2_5 import FakeResponse, FakeStreamChunk, _make_runtime
from test_contract_c2_5b import _runtime


class ContextBudgetTests(unittest.TestCase):
    @staticmethod
    def _payload() -> dict:
        return {
            "model": "model-a",
            "messages": [
                {"role": "system", "content": "fixed " * 30},
                {"role": "user", "content": "old " * 400},
                {"role": "assistant", "content": "earlier " * 300},
                {"role": "user", "content": "recent " * 20},
                {"role": "user", "content": "current request"},
            ],
            "tools": [{"type": "function", "function": {"name": "lookup", "description": "read only"}}],
        }

    def test_observe_projects_trim_without_mutating_request(self) -> None:
        payload = self._payload()
        original = json.dumps(payload, ensure_ascii=False)
        policy = ContextBudgetPolicy(mode="observe", window_tokens=600, output_reserve_tokens=0, safety_margin_tokens=0)
        result, record = assess_and_trim_payload(payload, policy)
        self.assertIs(result, payload)
        self.assertEqual(json.dumps(payload, ensure_ascii=False), original)
        self.assertEqual(record["status"], "would_trim_history")
        self.assertGreater(record["history_turns_removed"], 0)
        self.assertLessEqual(record["after"]["total"], 600)

    def test_enforce_trims_history_but_preserves_system_current_and_tools(self) -> None:
        payload = self._payload()
        policy = ContextBudgetPolicy(mode="enforce", window_tokens=600, output_reserve_tokens=0, safety_margin_tokens=0)
        result, record = assess_and_trim_payload(payload, policy)
        self.assertEqual(record["status"], "trimmed_history")
        self.assertIsNot(result, payload)
        self.assertEqual(result["messages"][0], payload["messages"][0])
        self.assertEqual(result["messages"][-1], payload["messages"][-1])
        self.assertEqual(result["tools"], payload["tools"])
        self.assertEqual(len(payload["messages"]), 5)
        self.assertNotIn(payload["messages"][1], result["messages"])
        self.assertLessEqual(measure_payload(result, policy)["total"], 600)

    def test_unresolvable_budget_does_not_delete_fixed_or_current_text(self) -> None:
        payload = {"model": "model-a", "messages": [
            {"role": "system", "content": "S" * 6000},
            {"role": "user", "content": "current"},
        ]}
        policy = ContextBudgetPolicy(mode="enforce", window_tokens=100, output_reserve_tokens=0, safety_margin_tokens=0)
        result, record = assess_and_trim_payload(payload, policy)
        self.assertIs(result, payload)
        self.assertEqual(record["status"], "unresolvable_preserved")
        self.assertEqual(record["after"], record["before"])
        self.assertIn("projected_after", record)
        self.assertEqual(record["history_messages_removed"], 0)

    def test_old_user_and_assistant_are_trimmed_as_a_turn(self) -> None:
        old_user = {"role": "user", "content": "X" * 200}
        old_answer = {"role": "assistant", "content": "short answer"}
        payload = {"model": "model-a", "messages": [
            {"role": "system", "content": "fixed"}, old_user, old_answer,
            {"role": "user", "content": "now"},
        ]}
        policy = ContextBudgetPolicy(mode="enforce", window_tokens=70, output_reserve_tokens=0, safety_margin_tokens=0)
        result, record = assess_and_trim_payload(payload, policy)
        self.assertEqual(record["status"], "trimmed_history")
        self.assertNotIn(old_user, result["messages"])
        self.assertNotIn(old_answer, result["messages"])
        self.assertEqual(record["history_turns_removed"], 1)
        self.assertEqual(record["history_messages_removed"], 2)

    def test_tool_schema_and_image_cost_are_included_without_base64_counting(self) -> None:
        policy = ContextBudgetPolicy(image_estimate_tokens=1234)
        payload = {"messages": [
            {"role": "system", "content": "fixed"},
            {"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 100000}},
            ]},
        ], "tools": [{"function": {"description": "a long description"}}]}
        measured = measure_payload(payload, policy)
        self.assertEqual(measured["image_count"], 1)
        self.assertEqual(measured["images"], 1234)
        self.assertGreater(measured["tools"], 0)
        self.assertLess(measured["total"], 2000)

    def test_model_specific_window_overrides_generic_window(self) -> None:
        policy = ContextBudgetPolicy.from_environment({
            "SHINKU_LLM_CONTEXT_WINDOW_TOKENS": "16000",
            "SHINKU_LLM_MODEL_CONTEXT_WINDOWS_JSON": '{"Model-A":32768}',
            "SHINKU_LLM_OUTPUT_RESERVE_TOKENS": "2048",
            "SHINKU_LLM_CONTEXT_MARGIN_TOKENS": "1024",
        })
        self.assertEqual(policy.input_limit("model-a"), 29696)
        self.assertEqual(policy.input_limit("other"), 12928)

    def test_non_ascii_scripts_are_not_priced_as_ascii(self) -> None:
        self.assertGreaterEqual(estimate_text_tokens("こんにちは"), 5)
        self.assertGreaterEqual(estimate_text_tokens("안녕하세요"), 5)
        self.assertGreaterEqual(estimate_text_tokens("🙂"), 2)

    def test_mixed_history_stress_preserves_invariants(self) -> None:
        generator = random.Random(5721)
        for _ in range(200):
            messages = [{"role": "system", "content": "persona" * generator.randint(1, 20)}]
            for _ in range(generator.randint(0, 12)):
                messages.extend([
                    {"role": "user", "content": "过往问题" * generator.randint(1, 50)},
                    {"role": "assistant", "content": "previous answer " * generator.randint(1, 60)},
                ])
            messages.append({"role": "user", "content": "当前请求：请继续处理"})
            payload = {"model": "model-a", "messages": messages, "tools": [{"name": "read"}]}
            original = json.dumps(payload, ensure_ascii=False)
            policy = ContextBudgetPolicy(
                mode="enforce", window_tokens=generator.randint(80, 1500),
                output_reserve_tokens=0, safety_margin_tokens=0,
            )
            result, record = assess_and_trim_payload(payload, policy)
            self.assertEqual(json.dumps(payload, ensure_ascii=False), original)
            self.assertEqual(result["messages"][0], messages[0])
            self.assertEqual(result["messages"][-1], messages[-1])
            self.assertEqual(result["tools"], payload["tools"])
            if record["status"] == "trimmed_history":
                self.assertLessEqual(measure_payload(result, policy)["total"], policy.input_limit("model-a"))
            else:
                self.assertIs(result, payload)

    def test_agent_tool_feedback_compacts_only_in_enforce_mode(self) -> None:
        transcript = [
            {"role": "user", "content": "find the issue"},
            {"role": "tool", "name": "one", "content": "A" * 400},
            {"role": "tool", "name": "two", "content": "B" * 400 + "LATEST_TAIL"},
        ]
        with mock.patch.dict(os.environ, {
            "SHINKU_LLM_CONTEXT_BUDGET_MODE": "observe",
            "SHINKU_AGENT_TOOL_RESULT_MAX_CHARS": "120",
            "SHINKU_AGENT_FEEDBACK_MAX_CHARS": "170",
        }):
            self.assertIn("A" * 400, build_agent_step_prompt(transcript))
        with mock.patch.dict(os.environ, {
            "SHINKU_LLM_CONTEXT_BUDGET_MODE": "enforce",
            "SHINKU_AGENT_TOOL_RESULT_MAX_CHARS": "120",
            "SHINKU_AGENT_FEEDBACK_MAX_CHARS": "170",
        }):
            prompt = build_agent_step_prompt(transcript)
        self.assertNotIn("A" * 400, prompt)
        self.assertNotIn("B" * 400, prompt)
        self.assertIn("LATEST_TAIL", prompt)
        self.assertEqual(transcript[-1]["content"], "B" * 400 + "LATEST_TAIL")

    def test_agent_goal_in_history_is_protected_during_tool_feedback(self) -> None:
        payload = {"model": "model-a", "messages": [
            {"role": "system", "content": "fixed"},
            {"role": "user", "content": "finish the original task"},
            {"role": "assistant", "content": "A" * 800},
            {"role": "user", "content": "older followup " * 100},
            {"role": "user", "content": "【工具执行结果反馈与推进】\n[工具 lookup 返回]: data"},
        ]}
        policy = ContextBudgetPolicy(mode="enforce", window_tokens=200, output_reserve_tokens=0, safety_margin_tokens=0)
        result, record = assess_and_trim_payload(payload, policy)
        self.assertEqual(record["status"], "trimmed_history")
        self.assertIn(payload["messages"][1], result["messages"])
        self.assertNotIn(payload["messages"][2], result["messages"])
        self.assertGreater(record["history_messages_removed"], record["history_turns_removed"])

    def test_runtime_records_only_metadata_and_respects_observe(self) -> None:
        runtime = _runtime()
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {
                "SHINKU_LOG_DIR": root,
                "SHINKU_LLM_CONTEXT_BUDGET_MODE": "observe",
                "SHINKU_LLM_CONTEXT_WINDOW_TOKENS": "500",
                "SHINKU_LLM_OUTPUT_RESERVE_TOKENS": "0",
                "SHINKU_LLM_CONTEXT_MARGIN_TOKENS": "0",
            }):
                payload = runtime._build_payload(
                    bundle=runtime.chat,
                    system_prompt="SECRET PERSONA",
                    user_prompt="SECRET CURRENT REQUEST",
                    temperature=0.2,
                    history_turns=[{"role": "user", "content": "SECRET OLD HISTORY " * 150}],
                )
            files = list(Path(root).glob("llm_context_budget/*.jsonl"))
            self.assertEqual(len(files), 1)
            logged = files[0].read_text(encoding="utf-8")
        self.assertIn("SECRET OLD HISTORY", str(payload))
        self.assertNotIn("SECRET PERSONA", logged)
        self.assertNotIn("SECRET CURRENT REQUEST", logged)
        self.assertNotIn("SECRET OLD HISTORY", logged)
        self.assertEqual(json.loads(logged)["status"], "would_trim_history")

    def test_budget_log_uses_service_log_root_when_legacy_dir_is_unset(self) -> None:
        runtime = _runtime()
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {
                "SHINKU_LOG_DIR": "",
                "SHINKU_LOG_ROOT": root,
                "SHINKU_LLM_CONTEXT_BUDGET_MODE": "observe",
            }):
                runtime._build_payload(
                    bundle=runtime.chat,
                    system_prompt="fixed",
                    user_prompt="current",
                    temperature=0.2,
                )
            self.assertEqual(len(list(Path(root).glob("llm_context_budget/*.jsonl"))), 1)

    def test_off_mode_writes_no_budget_log(self) -> None:
        runtime = _runtime()
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {
                "SHINKU_LOG_DIR": root,
                "SHINKU_LLM_CONTEXT_BUDGET_MODE": "off",
            }):
                runtime._build_payload(
                    bundle=runtime.chat, system_prompt="fixed", user_prompt="now", temperature=0.2,
                )
            self.assertFalse(Path(root, "llm_context_budget").exists())

    def test_stream_json_and_anthropic_extra_blocks_are_measured(self) -> None:
        runtime = _runtime(protocol="anthropic")
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"SHINKU_LOG_DIR": root, "SHINKU_LLM_CONTEXT_BUDGET_MODE": "observe"}):
                payload = runtime._build_payload(
                    bundle=runtime.chat, system_prompt="fixed", user_prompt="describe",
                    temperature=0.2, stream=True, json_mode=True,
                    system_extra_blocks=["runtime hint"],
                    user_images=[{"data_url": "data:image/png;base64," + "A" * 300}],
                )
            records = [json.loads(line) for file in Path(root).glob("llm_context_budget/*.jsonl") for line in file.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 1)
        self.assertTrue(payload["stream"])
        self.assertEqual(records[0]["before"]["image_count"], 1)
        self.assertGreater(records[0]["before"]["system"], 0)

    def test_provider_usage_is_joined_to_estimate_by_request_id(self) -> None:
        runtime = _make_runtime(scripts=[
            ("response", FakeResponse('{"ok": true}', usage={"prompt_tokens": 31, "completion_tokens": 4}))
        ])
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"SHINKU_LOG_DIR": root, "SHINKU_LLM_CONTEXT_BUDGET_MODE": "observe"}):
                result = runtime.call_chat_json(
                    system_prompt="fixed", user_prompt="test", fallback={"ok": False},
                    prompt_cache_key="chat:gray",
                )
            records = [json.loads(line) for file in Path(root).glob("llm_context_budget/*.jsonl") for line in file.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(result["ok"])
        self.assertEqual([item["kind"] for item in records], ["estimate", "usage"])
        self.assertEqual(records[0]["request_id"], records[1]["request_id"])
        self.assertEqual(records[1]["actual_input_tokens"], 31)
        self.assertEqual(records[1]["actual_output_tokens"], 4)
        self.assertTrue(records[1]["usage_available"])
        self.assertEqual(records[0]["channel"], "chat")
        self.assertEqual(records[1]["channel"], "chat")
        self.assertGreater(records[1]["estimate_to_actual_ratio"], 0)

    def test_qq_cache_key_still_counts_as_chat_usage(self) -> None:
        runtime = _make_runtime(scripts=[
            ("response", FakeResponse('{"ok": true}', usage={
                "prompt_tokens": 31, "completion_tokens": 4,
                "prompt_cache_hit_tokens": 12, "prompt_cache_miss_tokens": 19,
            }))
        ])
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"SHINKU_LOG_DIR": root, "SHINKU_LLM_CONTEXT_BUDGET_MODE": "observe"}):
                runtime.call_chat_json(
                    system_prompt="fixed", user_prompt="test", fallback={"ok": False},
                    prompt_cache_key="shinku:qq:v1",
                )
            records = [json.loads(line) for file in Path(root).glob("llm_context_budget/*.jsonl") for line in file.read_text(encoding="utf-8").splitlines()]
        metrics = runtime.snapshot_metrics()
        self.assertEqual(metrics["input_tokens_chat"], 31)
        self.assertEqual(metrics["cache_read_tokens_chat"], 12)
        self.assertEqual(metrics["cache_read_tokens_aux"], 0)
        self.assertEqual(records[0]["channel"], "chat")
        self.assertEqual(records[1]["channel"], "chat")

    def test_stream_usage_is_joined_without_duplicate_event(self) -> None:
        chunk = FakeStreamChunk('{"type": "done"}\n')
        chunk.usage = {"prompt_tokens": 19, "completion_tokens": 2}
        runtime = _make_runtime(scripts=[("stream", [chunk])])
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"SHINKU_LOG_DIR": root, "SHINKU_LLM_CONTEXT_BUDGET_MODE": "observe"}):
                runtime.call_aux_ndjson(system_prompt="fixed", user_prompt="test", prompt_cache_key="aux:gray")
            records = [json.loads(line) for file in Path(root).glob("llm_context_budget/*.jsonl") for line in file.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([item["kind"] for item in records], ["estimate", "usage"])
        self.assertEqual(records[1]["actual_input_tokens"], 19)
        self.assertEqual(records[0]["request_id"], records[1]["request_id"])
        self.assertEqual(records[1]["channel"], "aux")

    def test_stream_chat_usage_with_qq_cache_key_is_chat(self) -> None:
        content = FakeStreamChunk('{"speech":"ok"}')
        usage = FakeStreamChunk("")
        usage.usage = {"prompt_tokens": 41, "completion_tokens": 3}
        runtime = _make_runtime(scripts=[("stream", [content, usage])])
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"SHINKU_LOG_DIR": root, "SHINKU_LLM_CONTEXT_BUDGET_MODE": "observe"}):
                stream = runtime.stream_chat_json(
                    system_prompt="fixed", user_prompt="test", fallback={"speech": "fallback"},
                    prompt_cache_key="shinku:qq:v1",
                )
                while True:
                    try:
                        next(stream)
                    except StopIteration as stop:
                        result = stop.value
                        break
            records = [json.loads(line) for file in Path(root).glob("llm_context_budget/*.jsonl") for line in file.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(result.parsed["speech"], "ok")
        self.assertEqual(runtime.snapshot_metrics()["input_tokens_chat"], 41)
        self.assertEqual([item["kind"] for item in records], ["estimate", "usage"])
        self.assertEqual(records[1]["actual_input_tokens"], 41)
        self.assertEqual(records[1]["channel"], "chat")

    def test_enforce_mode_trims_actual_outbound_payload(self) -> None:
        runtime = _make_runtime(scripts=[
            ("response", FakeResponse('{"ok": true}', usage={"prompt_tokens": 31, "completion_tokens": 4}))
        ])
        old = "old turn " * 400
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {
                "SHINKU_LOG_DIR": root,
                "SHINKU_LLM_CONTEXT_BUDGET_MODE": "enforce",
                "SHINKU_LLM_CONTEXT_WINDOW_TOKENS": "200",
                "SHINKU_LLM_OUTPUT_RESERVE_TOKENS": "0",
                "SHINKU_LLM_CONTEXT_MARGIN_TOKENS": "0",
            }):
                result = runtime.call_chat_json(
                    system_prompt="fixed rule", user_prompt="current request", fallback={"ok": False},
                    history_turns=[{"role": "user", "content": old}, {"role": "assistant", "content": "old answer"}],
                )
            records = [json.loads(line) for file in Path(root).glob("llm_context_budget/*.jsonl") for line in file.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(result["ok"])
        outbound = runtime.chat.client.payloads[0]
        self.assertTrue(all(item["role"] == "system" for item in outbound["messages"][:-1]))
        self.assertEqual(outbound["messages"][-1]["role"], "user")
        self.assertEqual(outbound["messages"][0]["content"], "fixed rule")
        self.assertEqual(outbound["messages"][-1]["content"], "current request")
        self.assertEqual(records[0]["status"], "trimmed_history")
        self.assertEqual(records[1]["estimated_input_tokens"], records[0]["after"]["total"])
        self.assertGreater(records[0]["before"]["total"], records[1]["estimated_input_tokens"])


if __name__ == "__main__":
    unittest.main()
