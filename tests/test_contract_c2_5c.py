"""C2-5c 契约测试：提示词审计、token/缓存用量与缓存提示。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_contract_c2_5b import _runtime


class UsageMetricTests(unittest.TestCase):
    def test_token_aliases_sum_total_and_record_chat_suffix(self) -> None:
        runtime = _runtime()
        self.assertTrue(
            runtime._record_tokens(
                {"prompt_tokens": 7, "completion_tokens": 3},
                metric_suffix="chat",
            )
        )
        metrics = runtime.snapshot_metrics()
        self.assertEqual(metrics["input_tokens"], 7)
        self.assertEqual(metrics["output_tokens"], 3)
        self.assertEqual(metrics["total_tokens"], 10)
        self.assertEqual(metrics["total_tokens_chat"], 10)
        self.assertEqual(metrics["token_usage_reports_chat"], 1)

    def test_duplicate_stream_usage_is_counted_once(self) -> None:
        runtime = _runtime()
        seen: set[str] = set()
        usage = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
        self.assertTrue(runtime._record_tokens(usage, metric_suffix="chat", seen=seen))
        self.assertTrue(runtime._record_tokens(usage, metric_suffix="chat", seen=seen))
        self.assertEqual(runtime.snapshot_metrics()["total_tokens_chat"], 10)
        self.assertEqual(runtime.snapshot_metrics()["token_usage_reports_chat"], 1)

    def test_object_and_gemini_style_usage_fields_are_supported(self) -> None:
        runtime = _runtime()
        usage = SimpleNamespace(promptTokenCount=11, candidatesTokenCount=4)
        self.assertTrue(runtime._record_tokens(usage, metric_suffix="aux"))
        metrics = runtime.snapshot_metrics()
        self.assertEqual(metrics["input_tokens_aux"], 11)
        self.assertEqual(metrics["output_tokens_aux"], 4)
        self.assertEqual(metrics["total_tokens_aux"], 15)

    def test_cache_metrics_support_anthropic_and_deepseek_aliases(self) -> None:
        runtime = _runtime()
        anthropic = SimpleNamespace(
            usage={
                "input_tokens": 12,
                "output_tokens": 5,
                "cache_read_input_tokens": 9,
                "cache_creation_input_tokens": 3,
            }
        )
        deepseek = {"usage": {"prompt_cache_hit_tokens": 4, "prompt_cache_miss_tokens": 2}}
        self.assertTrue(runtime._record_cache(anthropic, prompt_cache_key="chat:final"))
        self.assertTrue(runtime._record_cache(deepseek, prompt_cache_key="aux:probe"))
        metrics = runtime.snapshot_metrics()
        self.assertEqual(metrics["cache_read_tokens_chat"], 9)
        self.assertEqual(metrics["cache_creation_tokens_chat"], 3)
        self.assertEqual(metrics["cache_read_tokens_aux"], 4)
        self.assertEqual(metrics["cache_creation_tokens_aux"], 2)

    def test_usage_without_cache_fields_is_still_a_report(self) -> None:
        runtime = _runtime()
        response = SimpleNamespace(usage={"prompt_tokens": 2, "completion_tokens": 1})
        self.assertTrue(runtime._record_cache(response, prompt_cache_key="chat:final"))
        metrics = runtime.snapshot_metrics()
        self.assertEqual(metrics["cache_usage_reports"], 1)
        self.assertEqual(metrics["cache_fields_unavailable_reports"], 1)

    def test_missing_usage_is_not_reported(self) -> None:
        runtime = _runtime()
        self.assertFalse(runtime._record_cache(SimpleNamespace(usage=None)))


class AuditTests(unittest.TestCase):
    def test_audit_writes_metadata_only_for_final_chat(self) -> None:
        runtime = _runtime()
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(
                os.environ,
                {
                    "SHINKU_LLM_PROMPT_AUDIT_ENABLED": "1",
                    "SHINKU_LOG_DIR": root,
                    "SHINKU_LLM_PROMPT_AUDIT_INCLUDE_AUX": "0",
                },
            ):
                runtime._maybe_record_audit(
                    bundle=runtime.chat,
                    prompt_cache_key="chat:final",
                    messages=[{"role": "system", "content": "secret persona"}],
                    system_extra_blocks=["runtime rule"],
                    history_turns=[{"role": "user", "content": "old question"}],
                    user_prompt="new question",
                    user_image_count=1,
                    prompt_audit_sections=[{"name": "source", "text": "source text"}],
                    stream=True,
                    json_mode=True,
                    native_tool_count=2,
                )
            files = list(Path(root).glob("llm_prompt_audit/*.jsonl"))
            self.assertEqual(len(files), 1)
            audit_text = files[0].read_text(encoding="utf-8")
            record = json.loads(audit_text)
        self.assertEqual(record["prompt_cache_key"], "chat:final")
        self.assertEqual(record["user_image_count"], 1)
        self.assertEqual(record["native_tool_count"], 2)
        self.assertNotIn("secret persona", audit_text)
        self.assertEqual(record["payload_totals"]["chars"] > 0, True)

    def test_audit_disabled_and_aux_default_do_not_write(self) -> None:
        runtime = _runtime()
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.dict(os.environ, {"SHINKU_LOG_DIR": root}, clear=False):
                self.assertFalse(runtime._audit_enabled("chat:final"))
                self.assertFalse(runtime._audit_enabled("aux:probe"))
                runtime._maybe_record_audit(
                    bundle=runtime.chat,
                    prompt_cache_key="chat:final",
                    messages=[],
                    system_extra_blocks=[],
                    history_turns=None,
                    user_prompt="ignored",
                    user_image_count=0,
                    prompt_audit_sections=None,
                    stream=False,
                    json_mode=False,
                    native_tool_count=0,
                )
            self.assertFalse(Path(root, "llm_prompt_audit").exists())

    def test_aux_audit_requires_explicit_include_flag(self) -> None:
        runtime = _runtime()
        with mock.patch.dict(
            os.environ,
            {
                "SHINKU_LLM_PROMPT_AUDIT_ENABLED": "true",
                "SHINKU_LLM_PROMPT_AUDIT_INCLUDE_AUX": "on",
            },
        ):
            self.assertTrue(runtime._audit_enabled("aux:probe"))

    def test_audit_sections_hash_text_and_normalize_inputs(self) -> None:
        runtime = _runtime()
        sections = runtime._coerce_audit_sections(
            [{"name": "one", "text": "中文"}, "plain"]
        )
        self.assertEqual([item["name"] for item in sections], ["one", "section_1"])
        self.assertEqual(sections[0]["chars"], 2)
        self.assertEqual(len(sections[0]["sha256_16"]), 16)
        self.assertEqual(runtime._sum_sections(sections)["chars"], 7)


class CacheHintTests(unittest.TestCase):
    def test_official_openai_cache_hints_use_namespace_and_retention(self) -> None:
        runtime = _runtime(base_url="https://api.openai.com/v1")
        with mock.patch.dict(
            os.environ,
            {
                "SHINKU_PROMPT_CACHE_HINTS_ENABLED": "1",
                "SHINKU_PROMPT_CACHE_NAMESPACE": "prod",
                "SHINKU_PROMPT_CACHE_RETENTION": "24h",
            },
        ):
            self.assertEqual(
                runtime._build_cache_kwargs(bundle=runtime.chat, prompt_cache_key=":chat:final:"),
                {"prompt_cache_key": "prod:chat:final", "prompt_cache_retention": "24h"},
            )

    def test_nonofficial_gateway_requires_force_flag(self) -> None:
        runtime = _runtime(base_url="https://gateway.example/v1")
        env = {
            "SHINKU_PROMPT_CACHE_HINTS_ENABLED": "1",
            "SHINKU_PROMPT_CACHE_NAMESPACE": "shinku",
            "SHINKU_PROMPT_CACHE_RETENTION": "in_memory",
        }
        with mock.patch.dict(os.environ, env):
            self.assertEqual(runtime._build_cache_kwargs(bundle=runtime.chat, prompt_cache_key="chat:final"), {})
        with mock.patch.dict(os.environ, {**env, "SHINKU_PROMPT_CACHE_HINTS_FORCE": "yes"}):
            self.assertEqual(
                runtime._build_cache_kwargs(bundle=runtime.chat, prompt_cache_key="chat:final"),
                {"prompt_cache_key": "shinku:chat:final", "prompt_cache_retention": "in_memory"},
            )

    def test_cache_hint_switch_and_retention_validation(self) -> None:
        runtime = _runtime(base_url="https://api.openai.com/v1")
        with mock.patch.dict(
            os.environ,
            {
                "SHINKU_PROMPT_CACHE_HINTS_ENABLED": "off",
                "SHINKU_PROMPT_CACHE_RETENTION": "forever",
            },
        ):
            self.assertEqual(runtime._build_cache_kwargs(bundle=runtime.chat, prompt_cache_key="x"), {})
            self.assertEqual(runtime._coerce_cache_retention("forever"), "")
