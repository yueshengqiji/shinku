"""C2-5a 契约测试：LLM 调用引擎传输核心（`shinku.llm.runtime`）。

对象是 ``RuntimeCore`` / 组合根 ``LLMRuntime`` 的**外部可观察行为**
（契约文档 ``docs/contracts/c2_5_runtime.md``）。传输层一律换成记录用的
假件（patch ``shinku.llm.runtime_core.build_llm_client``），全程不发起
任何真实网络请求；配置走 ``SHINKU_*`` 环境变量。

本批的测试面覆盖**桩等价路径**（不启用原生工具、不开审计、不带缓存
提示、非 DeepSeek 思考控制）；原生工具画像 / allowlist / 审计落盘 /
token 用量 / 缓存提示的完整行为属 C2-5b / C2-5c 的测试面。

边界类（``C2_5BoundaryTests``）沿用每批口径：外来标识 / 旧私有名 /
旧整句不得复现；新模块只 import 本仓模块 + 标准库。
"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from shinku.llm import runtime as runtime_mod
from shinku.llm import runtime_core as core_mod
from shinku.llm.runtime import (
    ChatJSONStreamResult,
    LLMRuntime,
    ModelBundle,
    NDJSONCallResult,
    ProviderToolProfile,
    StreamTap,
    normalize_reply_medium,
)
from shinku.tools.invocation import NATIVE_TOOL_CALL_FIELD

SRC_FILES = [
    Path(core_mod.__file__),
    Path(runtime_mod.__file__),
    Path(core_mod.__file__).with_name("runtime_capabilities.py"),
    Path(core_mod.__file__).with_name("runtime_audit.py"),
]

#: 旧实现的私有名样例（tokenize 精确比对口径；全集由 _kit namegap 核）。
LEGACY_NAMES = [
    "_RetryableLLMError",
    "_llm_error_is_retryable",
    "_TopLevelJSONStreamTap",
    "_call_json",
    "_call_ndjson",
    "_stream_chat_json",
    "_create_completion",
    "_create_completion_once",
    "_record_metric",
    "_record_error_detail",
    "_sanitize_error_message",
    "_extract_json",
    "_repair_json",
    "_extract_first_json_object",
    "_recover_partial_chat_json",
    "_build_completion_kwargs",
    "_note_truncation",
    "_note_parse_fallback",
    "_metrics",
    "_last_error",
]

FOREIGN_TOKENS = ("companion_v01", "code_shared", "akane", "COMPANION_", "SHINKU_SERVER_")


# --------------------------------------------------------------------------- #
# 假传输层
# --------------------------------------------------------------------------- #


class FakeCompletions:
    """按脚本队列应答 ``create()``：流 / 非流 / 异常三形态。"""

    def __init__(self, client: "FakeClient"):
        self._client = client

    def create(self, **payload: Any):
        self._client.payloads.append(payload)
        if not self._client.scripts:
            raise AssertionError("fake transport: no scripted response left")
        kind, value = self._client.scripts.pop(0)
        if kind == "raise":
            raise value
        if kind == "stream":
            # 惰性交回：list / 迭代器原样返回，迭代（及迭代中抛的异常）
            # 发生在引擎的 for 循环里，才覆盖得到流中重试路径。
            return value
        return value


class FakeClient:
    def __init__(self, *, protocol: str = "openai", base_url: str = "", scripts=None):
        self._shinku_protocol = protocol
        self.base_url = base_url
        self.scripts = list(scripts or [])
        self.payloads: list[dict[str, Any]] = []
        completions = FakeCompletions(self)
        # 真实调用路径是 client.chat.completions.create；completions 保留同一实例。
        self.chat = SimpleNamespace(completions=completions)
        self.completions = completions


class FakeMessage:
    def __init__(self, content: Any, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, content: Any, finish_reason: str = "stop", usage=None):
        self.choices = [FakeChoice(FakeMessage(content), finish_reason)]
        self.usage = usage


class FakeDelta:
    def __init__(self, content: Any):
        self.content = content


class FakeStreamChunk:
    def __init__(self, content: Any):
        self.choices = [SimpleNamespace(delta=FakeDelta(content))]


def _make_runtime(
    *,
    protocol: str = "openai",
    base_url: str = "https://api.example.test/v1",
    chat_model: str = "chat-model",
    scripts=None,
) -> LLMRuntime:
    captured: list[dict[str, Any]] = []

    def fake_build(**kwargs: Any) -> FakeClient:
        captured.append(dict(kwargs))
        return FakeClient(protocol=protocol, base_url=base_url, scripts=scripts)

    env = {
        "SHINKU_CHAT_MODEL_NAME": chat_model,
        "SHINKU_AUX_MODEL_NAME": "aux-model",
    }
    with mock.patch("shinku.llm.runtime_core.build_llm_client", side_effect=fake_build):
        with mock.patch.dict(os.environ, env):
            runtime = LLMRuntime()
    runtime._build_calls = captured  # 测试侦察口
    return runtime


def _empty_scripts_holder() -> list:
    return []


def _metrics(runtime: LLMRuntime, key: str) -> int:
    return runtime.snapshot_metrics().get(key, 0)


# --------------------------------------------------------------------------- #
# bundle 构建与配置
# --------------------------------------------------------------------------- #


class BundleConfigTests(unittest.TestCase):
    def test_chat_bundle_reads_shinku_env_and_uses_chat_timeout(self) -> None:
        runtime = _make_runtime(chat_model="glm-4")
        self.assertEqual(runtime.chat.model, "glm-4")
        self.assertEqual(len(runtime._build_calls), 2)
        chat_call = runtime._build_calls[1]
        self.assertEqual(chat_call["timeout"], 120.0)
        self.assertEqual(chat_call["max_retries"], 0)

    def test_aux_bundle_comes_first_with_aux_timeout(self) -> None:
        runtime = _make_runtime()
        aux_call = runtime._build_calls[0]
        self.assertEqual(aux_call["timeout"], 90.0)
        self.assertEqual(aux_call["max_retries"], 0)
        self.assertEqual(runtime.aux.model, "aux-model")

    def test_reload_from_config_rebuilds_both_bundles(self) -> None:
        runtime = _make_runtime(chat_model="old-model")
        with mock.patch.dict(
            os.environ,
            {"SHINKU_CHAT_MODEL_NAME": "new-model", "SHINKU_AUX_MODEL_NAME": "new-aux"},
        ):
            result = runtime.reload_from_config()
        self.assertEqual(
            result,
            {"status": "reloaded", "auxModel": "new-aux", "chatModel": "new-model"},
        )
        self.assertEqual(runtime.chat.model, "new-model")


# --------------------------------------------------------------------------- #
# 度量与错误通道
# --------------------------------------------------------------------------- #


class MetricsTests(unittest.TestCase):
    def test_initial_counter_keys(self) -> None:
        runtime = _make_runtime()
        snapshot = runtime.snapshot_metrics()
        for key in (
            "aux_json_calls",
            "chat_json_calls",
            "chat_stream_rescue_attempts",
            "native_tool_call_extracted",
            "cache_read_tokens",
            "token_usage_reports",
        ):
            self.assertIn(key, snapshot)
        self.assertTrue(all(value == 0 for value in snapshot.values()))

    def test_record_metric_increments_by_amount(self) -> None:
        runtime = _make_runtime()
        runtime.record_metric("input_tokens", 7)
        runtime.record_metric("input_tokens", 3)
        self.assertEqual(_metrics(runtime, "input_tokens"), 10)

    def test_recent_metric_days_and_metadata_without_store(self) -> None:
        runtime = _make_runtime()
        self.assertEqual(runtime.recent_metric_days(), [])
        self.assertEqual(runtime.metrics_persistence_metadata(), {"persistent": False})
        runtime.close()  # 无持久化件时 close 是 no-op


class ErrorChannelTests(unittest.TestCase):
    def test_redact_secrets_masks_bearer_api_key_and_sk_tokens(self) -> None:
        runtime = _make_runtime()
        text = runtime._redact_secrets(
            "Authorization: Bearer sk-testsecret123456 api_key=sk-othersecret123456"
        )
        self.assertNotIn("sk-testsecret", text)
        self.assertNotIn("sk-othersecret", text)
        self.assertIn("[redacted]", text)

    def test_redact_secrets_caps_length(self) -> None:
        runtime = _make_runtime()
        text = runtime._redact_secrets("x" * 2000, max_chars=100)
        self.assertLessEqual(len(text), 100)
        self.assertTrue(text.endswith("..."))

    def test_note_error_records_redacted_detail(self) -> None:
        runtime = _make_runtime()
        runtime._note_error(RuntimeError("Bearer sk-abcdef12345678"), phase="call_json")
        detail = runtime.snapshot_last_error()
        self.assertEqual(detail["phase"], "call_json")
        self.assertEqual(detail["type"], "RuntimeError")
        self.assertNotIn("sk-abcdef", detail["message"])

    def test_note_truncated_records_length_stop(self) -> None:
        runtime = _make_runtime()
        runtime._note_truncated(
            FakeResponse('{"speech": "半', finish_reason="length"), phase="call_json"
        )
        detail = runtime.snapshot_last_error()
        self.assertEqual(detail["type"], "ResponseTruncated")
        self.assertIn("finish_reason=length", detail["message"])
        self.assertEqual(_metrics(runtime, "response_truncated"), 1)

    def test_note_truncated_ignores_normal_stop(self) -> None:
        runtime = _make_runtime()
        runtime._note_truncated(FakeResponse('{"a": 1}'), phase="call_json")
        self.assertEqual(runtime.snapshot_last_error(), {})
        self.assertEqual(_metrics(runtime, "response_truncated"), 0)

    def test_note_parse_failure_records_head_sample(self) -> None:
        runtime = _make_runtime()
        runtime._note_parse_failure("这不是 JSON", phase="call_json")
        detail = runtime.snapshot_last_error()
        self.assertEqual(detail["type"], "ChatJSONFallback")
        self.assertIn("这不是 JSON", detail["message"])
        self.assertIn("head=", detail["message"])


# --------------------------------------------------------------------------- #
# 回复 JSON 解析 / 修复 / 恢复
# --------------------------------------------------------------------------- #


class JsonParsingTests(unittest.TestCase):
    def test_parse_json_direct_and_wrapped(self) -> None:
        runtime = _make_runtime()
        self.assertEqual(runtime._parse_json('{"a": 1}'), {"a": 1})
        self.assertEqual(runtime._parse_json('好的 {"a": 1} 完毕'), {"a": 1})
        self.assertIsNone(runtime._parse_json(""))
        self.assertIsNone(runtime._parse_json("[1, 2]"))

    def test_first_json_object_respects_prefix_context(self) -> None:
        runtime = _make_runtime()
        self.assertEqual(runtime._first_json_object('x {"k": 2} y'), {"k": 2})
        # 前一个字符是 ":" / "[" / "," 的花括号属于外层结构，不能起算。
        self.assertIsNone(runtime._first_json_object('"k": {"a": 1}'))

    def test_heal_json_never_raises(self) -> None:
        runtime = _make_runtime()
        result = runtime._heal_json('{"a": ')
        self.assertTrue(result is None or isinstance(result, dict))

    def test_recover_partial_restores_emotion_and_speech(self) -> None:
        runtime = _make_runtime()
        recovered = runtime._recover_partial(
            '{"emotion": "calm", "speech": "半截的话',
            fallback={"fallback": 1},
        )
        self.assertEqual(recovered["emotion"], "calm")
        self.assertEqual(recovered["speech"], "半截的话")
        self.assertEqual(recovered["speech_segments"], [])

    def test_recover_partial_with_nothing_usable_returns_none(self) -> None:
        runtime = _make_runtime()
        self.assertIsNone(runtime._recover_partial('{"other": 1', fallback={"f": 1}))

    def test_read_json_value_extracts_segments(self) -> None:
        runtime = _make_runtime()
        value = runtime._read_json_value(
            '{"speech_segments": [{"speech": "一"}, "二"]}', "speech_segments"
        )
        self.assertEqual(value, [{"speech": "一"}, "二"])


class ReplyMediumTests(unittest.TestCase):
    def test_alias_matrix(self) -> None:
        cases = {
            "text": "text",
            "文字": "text",
            "voice": "voice",
            "语音": "voice",
            "audio": "voice",
            "both": "both",
            "双发": "both",
            "text-voice": "both",
            "unknown": "",
            "": "",
            None: "",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_reply_medium(raw), expected)


class RescueDecisionTests(unittest.TestCase):
    def test_decision_matrix(self) -> None:
        runtime = _make_runtime()
        cases = [
            (None, True),
            ("not a dict", True),
            ({"tool_call": {"type": "x"}}, False),
            ({NATIVE_TOOL_CALL_FIELD: {"type": "x"}}, False),
            ({"status": "skip"}, False),
            ({"status": "silent"}, False),
            ({"status": "decline"}, False),
            ({"status": "final"}, True),
            ({"speech": "hi"}, False),
            ({"speech_segments": ["一", ""]}, False),
            ({"speech_segments": []}, True),
            ({"speech_segments": "not-list"}, True),
        ]
        for parsed, expected in cases:
            with self.subTest(parsed=parsed):
                self.assertEqual(runtime._chat_reply_needs_rescue(parsed), expected)


# --------------------------------------------------------------------------- #
# StreamTap
# --------------------------------------------------------------------------- #


class StreamTapTests(unittest.TestCase):
    def test_incremental_emotion_speech_and_medium(self) -> None:
        tap = StreamTap()
        first = tap.feed('{"emotion": "happy", "speech": "')
        second = tap.feed('你好呀", "reply_medium": "语音"}')
        self.assertEqual(first, [{"type": "ui", "emotion": "happy"}])
        types = [event["type"] for event in second]
        self.assertIn("delivery_hint", types)
        self.assertIn("speech_chunk", types)
        self.assertEqual(tap.latest_emotion, "happy")
        self.assertEqual(tap.latest_speech, "你好呀")
        self.assertEqual(tap.latest_reply_medium, "voice")

    def test_unicode_escapes_decode(self) -> None:
        tap = StreamTap()
        tap.feed('{"speech": "你\\u597d"}')
        self.assertEqual(tap.latest_speech, "你好")

    def test_speech_segments_emit_capped_and_dedup(self) -> None:
        tap = StreamTap()
        events: list[dict[str, Any]] = []
        text = '{"speech_segments": ["第一句。", "第一句。", "第二句。", "第三句。", "第四句。"]}'
        events.extend(tap.feed(text))
        segments = [event for event in events if event["type"] == "speech_segment"]
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0]["index"], 0)
        self.assertEqual(segments[0]["text"], "第一句。")
        # 重复的句子不重复发事件（去重键 = 去空白全文）。

    def test_top_level_speech_supersedes_segments(self) -> None:
        tap = StreamTap()
        tap.feed('{"speech_segments": ["段一"], "speech": "整')
        tap.feed('句"}')
        self.assertEqual(tap.latest_speech, "整句")


# --------------------------------------------------------------------------- #
# payload 组装（桩等价路径）
# --------------------------------------------------------------------------- #


class PayloadTests(unittest.TestCase):
    def _runtime_with(self, protocol: str, base_url: str = "") -> LLMRuntime:
        return _make_runtime(protocol=protocol, base_url=base_url)

    def test_json_mode_for_ollama_adds_response_format_and_hint(self) -> None:
        runtime = self._runtime_with("ollama")
        payload = runtime._build_payload(
            bundle=runtime.chat,
            system_prompt="system",
            user_prompt="user",
            temperature=0.1,
            stream=True,
            json_mode=True,
        )
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertIs(payload["stream"], True)
        # JSON 提示插在第一条 system 之后，不改写稳定首条。
        messages = payload["messages"]
        self.assertEqual(messages[0]["content"], "system")
        self.assertIn("JSON", messages[1]["content"])
        self.assertEqual(messages[1]["role"], "system")

    def test_json_mode_openai_still_forces_response_format(self) -> None:
        runtime = self._runtime_with("openai")
        payload = runtime._build_payload(
            bundle=runtime.chat,
            system_prompt="system",
            user_prompt="user",
            temperature=0.1,
            json_mode=True,
        )
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_anthropic_gets_no_response_format_but_payload_blocks(self) -> None:
        runtime = self._runtime_with("anthropic")
        payload = runtime._build_payload(
            bundle=runtime.chat,
            system_prompt="system",
            user_prompt="user",
            temperature=0.1,
            json_mode=True,
            system_extra_blocks=["extra"],
        )
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["system_extra_blocks"], ["extra"])

    def test_unknown_protocol_merges_extra_blocks_into_first_system(self) -> None:
        runtime = self._runtime_with("some-gateway")
        payload = runtime._build_payload(
            bundle=runtime.chat,
            system_prompt="system",
            user_prompt="user",
            temperature=0.1,
            system_extra_blocks=["extra"],
        )
        messages = payload["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], "system\n\nextra")
        self.assertNotIn("system_extra_blocks", payload)

    def test_openai_keeps_extra_blocks_as_separate_system_messages(self) -> None:
        runtime = self._runtime_with("openai")
        payload = runtime._build_payload(
            bundle=runtime.chat,
            system_prompt="system",
            user_prompt="user",
            temperature=0.1,
            system_extra_blocks=["extra-1", "extra-2"],
        )
        roles = [message["role"] for message in payload["messages"]]
        self.assertEqual(roles, ["system", "system", "system", "user"])
        self.assertEqual(payload["messages"][1]["content"], "extra-1")
        self.assertEqual(payload["messages"][2]["content"], "extra-2")

    def test_history_turns_filtered_and_appended(self) -> None:
        runtime = self._runtime_with("openai")
        payload = runtime._build_payload(
            bundle=runtime.chat,
            system_prompt="system",
            user_prompt="user",
            temperature=0.1,
            history_turns=[
                {"role": "user", "content": "早前提问"},
                {"role": "assistant", "content": "早前回答"},
                {"role": "tool", "content": "被过滤"},
                {"role": "user", "content": ""},
            ],
        )
        turns = payload["messages"][1:]
        self.assertEqual(
            turns,
            [
                {"role": "user", "content": "早前提问"},
                {"role": "assistant", "content": "早前回答"},
                {"role": "user", "content": "user"},
            ],
        )

    def test_base_payload_shape_and_no_tools_without_native_support(self) -> None:
        runtime = self._runtime_with("openai")
        payload = runtime._build_payload(
            bundle=runtime.chat,
            system_prompt="system",
            user_prompt="user",
            temperature=0.5,
            native_tools=[{"type": "function"}],  # 桩路径：规范化后为空
        )
        self.assertEqual(
            payload,
            {
                "model": "chat-model",
                "temperature": 0.5,
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "user"},
                ],
            },
        )
        self.assertNotIn("tools", payload)


# --------------------------------------------------------------------------- #
# 三条调用通道（假传输层全流程）
# --------------------------------------------------------------------------- #


class CallChannelTests(unittest.TestCase):
    def test_call_chat_json_parses_valid_reply(self) -> None:
        runtime = _make_runtime(scripts=[("response", FakeResponse('{"speech": "好"}'))])
        result = runtime.call_chat_json(
            system_prompt="s", user_prompt="u", fallback={"fallback": 1}
        )
        self.assertEqual(result, {"speech": "好"})
        self.assertEqual(_metrics(runtime, "chat_json_calls"), 1)
        self.assertEqual(_metrics(runtime, "chat_json_fallbacks"), 0)

    def test_call_chat_json_falls_back_on_garbage(self) -> None:
        runtime = _make_runtime(scripts=[("response", FakeResponse("不是 JSON"))])
        result = runtime.call_chat_json(
            system_prompt="s", user_prompt="u", fallback={"fallback": 1}
        )
        self.assertEqual(result, {"fallback": 1})
        self.assertEqual(_metrics(runtime, "chat_json_fallbacks"), 1)
        self.assertEqual(runtime.snapshot_last_error()["type"], "ChatJSONFallback")

    def test_call_chat_json_falls_back_on_transport_error(self) -> None:
        runtime = _make_runtime(scripts=[("raise", RuntimeError("boom"))])
        result = runtime.call_chat_json(
            system_prompt="s", user_prompt="u", fallback={"fallback": 1}
        )
        self.assertEqual(result, {"fallback": 1})
        self.assertEqual(_metrics(runtime, "errors"), 1)
        self.assertEqual(runtime.snapshot_last_error()["type"], "RuntimeError")

    def test_retryable_error_is_retried_once(self) -> None:
        runtime = _make_runtime(
            scripts=[
                ("raise", RuntimeError("429 too many requests")),
                ("response", FakeResponse('{"speech": "重试成功"}')),
            ]
        )
        result = runtime.call_chat_json(
            system_prompt="s", user_prompt="u", fallback={"fallback": 1}
        )
        self.assertEqual(result, {"speech": "重试成功"})
        self.assertEqual(_metrics(runtime, "llm_http_retry"), 1)
        self.assertEqual(_metrics(runtime, "errors"), 0)

    def test_retryable_error_exhaustion_falls_back(self) -> None:
        runtime = _make_runtime(
            scripts=[
                ("raise", RuntimeError("503 service unavailable")),
                ("raise", RuntimeError("503 service unavailable")),
            ]
        )
        result = runtime.call_chat_json(
            system_prompt="s", user_prompt="u", fallback={"fallback": 1}
        )
        self.assertEqual(result, {"fallback": 1})
        self.assertEqual(_metrics(runtime, "llm_http_retry"), 1)

    def test_type_error_without_cache_hints_propagates_to_fallback(self) -> None:
        runtime = _make_runtime(scripts=[("raise", TypeError("unexpected keyword"))])
        result = runtime.call_chat_json(
            system_prompt="s", user_prompt="u", fallback={"fallback": 1}
        )
        self.assertEqual(result, {"fallback": 1})

    def test_call_aux_json_uses_aux_channel(self) -> None:
        runtime = _make_runtime(scripts=[("response", FakeResponse('{"tag": "aux"}'))])
        result = runtime.call_aux_json(
            system_prompt="s", user_prompt="u", fallback={"fallback": 1}
        )
        self.assertEqual(result, {"tag": "aux"})
        self.assertEqual(_metrics(runtime, "aux_json_calls"), 1)
        self.assertEqual(runtime.aux.client.payloads[0]["model"], "aux-model")

    def test_call_aux_ndjson_drains_lines_and_tail(self) -> None:
        chunks = [
            FakeStreamChunk('{"type": "a"}\n{"type":'),
            FakeStreamChunk(' "b"}'),
        ]
        runtime = _make_runtime(
            protocol="openai", scripts=[("stream", chunks)]
        )
        result = runtime.call_aux_ndjson(system_prompt="s", user_prompt="u")
        self.assertIsInstance(result, NDJSONCallResult)
        self.assertEqual(
            [event["type"] for event in result.events], ["a", "b"]
        )
        self.assertTrue(result.completed_stream)
        self.assertEqual(result.error, "")
        self.assertEqual(len(result.event_timings), 2)
        self.assertEqual(_metrics(runtime, "aux_ndjson_calls"), 1)

    def test_call_aux_ndjson_stops_early_on_callback(self) -> None:
        chunks = [
            FakeStreamChunk('{"type": "stop-now"}\n'),
            FakeStreamChunk('{"type": "never-seen"}\n'),
        ]
        runtime = _make_runtime(scripts=[("stream", chunks)])
        result = runtime.call_aux_ndjson(
            system_prompt="s",
            user_prompt="u",
            on_event=lambda event: event.get("type") == "stop-now",
        )
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.stop_event_type, "stop-now")
        self.assertFalse(result.completed_stream)


class StreamChannelTests(unittest.TestCase):
    def _run_stream(
        self, runtime: LLMRuntime, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], ChatJSONStreamResult]:
        """跑完整个流式生成器：收齐事件，再从 StopIteration 取返回值。"""

        gen = runtime.stream_chat_json(
            system_prompt=kwargs.pop("system_prompt", "s"),
            user_prompt=kwargs.pop("user_prompt", "u"),
            **kwargs,
        )
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(next(gen))
            except StopIteration as stop:
                return events, stop.value

    def test_stream_yields_events_and_parses_full_object(self) -> None:
        chunks = [
            FakeStreamChunk('{"emotion": "happy", '),
            FakeStreamChunk('"speech": "你好'),
            FakeStreamChunk('呀", "reply_medium": "voice"}'),
        ]
        runtime = _make_runtime(scripts=[("stream", chunks)])
        events, result = self._run_stream(runtime, fallback={"fallback": 1})
        types = [event["type"] for event in events]
        self.assertIn("ui", types)
        self.assertIn("speech_chunk", types)
        self.assertEqual(result.parsed.get("speech"), "你好呀")
        self.assertEqual(result.latest_emotion, "happy")
        self.assertEqual(result.latest_reply_medium, "voice")
        self.assertFalse(result.rescue_attempted)
        self.assertEqual(_metrics(runtime, "chat_stream_calls"), 1)

    def test_stream_rescue_succeeds_on_empty_stream(self) -> None:
        runtime = _make_runtime(
            scripts=[
                ("stream", []),
                ("response", FakeResponse('{"speech": "救回来了"}')),
            ]
        )
        events, result = self._run_stream(runtime, fallback={"fallback": 1})
        self.assertEqual(events, [])
        self.assertTrue(result.rescue_attempted)
        self.assertTrue(result.rescue_succeeded)
        self.assertEqual(result.parsed.get("speech"), "救回来了")
        self.assertEqual(result.error, "")
        self.assertEqual(_metrics(runtime, "chat_stream_rescue_attempts"), 1)
        self.assertEqual(_metrics(runtime, "chat_stream_rescue_successes"), 1)

    def test_stream_rescue_failure_reports_error(self) -> None:
        runtime = _make_runtime(
            scripts=[
                ("stream", []),
                ("response", FakeResponse("也不是 JSON")),
            ]
        )
        events, result = self._run_stream(runtime, fallback={"fallback": 1})
        self.assertTrue(result.rescue_attempted)
        self.assertFalse(result.rescue_succeeded)
        self.assertEqual(result.parsed, {"fallback": 1})
        self.assertTrue(result.error)
        self.assertEqual(_metrics(runtime, "chat_stream_rescue_failures"), 1)

    def test_stream_early_tool_call_stops_stream(self) -> None:
        chunks = [
            FakeStreamChunk('{"tool_call": {"type": "note'),
            FakeStreamChunk('_taker", "arguments": {}}"}'),
        ]
        runtime = _make_runtime(scripts=[("stream", chunks)])
        events, result = self._run_stream(
            runtime,
            fallback={"fallback": 1},
            early_tool_call_validator=lambda call: True,
        )
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.early_tool_call.get("type"), "note_taker")
        self.assertEqual(
            result.parsed,
            {"tool_call": {"type": "note_taker", "arguments": {}}},
        )
        self.assertEqual(events, [])

    def test_stream_transport_error_then_retry_success(self) -> None:
        chunks = [
            FakeStreamChunk('{"speech": "第二'),
            FakeStreamChunk('次成功"}'),
        ]
        runtime = _make_runtime(scripts=[("stream", None)])
        # 首轮流式连接在迭代中直接炸（"connection" 命中可重试 token），次轮正常。
        class ExplodingStream:
            def __iter__(self):
                return self

            def __next__(self):
                raise RuntimeError("connection reset mid-stream")

        runtime.chat.client.scripts[0] = ("stream", ExplodingStream())
        runtime.chat.client.scripts.append(("stream", chunks))
        with mock.patch("time.sleep"):
            events, result = self._run_stream(runtime, fallback={"fallback": 1})
        self.assertEqual(result.parsed.get("speech"), "第二次成功")
        self.assertEqual(_metrics(runtime, "chat_stream_retries"), 1)

    def test_stream_rejects_validator_keeps_streaming(self) -> None:
        chunks = [
            FakeStreamChunk('{"tool_call": {"type": "rejected"'),
            FakeStreamChunk('}, "speech": "继续正文"}'),
        ]
        runtime = _make_runtime(scripts=[("stream", chunks)])
        events, result = self._run_stream(
            runtime,
            fallback={"fallback": 1},
            early_tool_call_validator=lambda call: False,
        )
        self.assertFalse(result.stopped_early)
        self.assertIsNone(result.early_tool_call)
        self.assertEqual(result.parsed.get("speech"), "继续正文")


# --------------------------------------------------------------------------- #
# 流内早停探测
# --------------------------------------------------------------------------- #


class ScanToolCallTests(unittest.TestCase):
    def test_scan_states(self) -> None:
        runtime = _make_runtime()
        cases = [
            ("", ("pending", None)),
            ('{"tool_call"', ("pending", None)),
            ('{"other": 1}', ("none", None)),
            ('{"tool_call": null}', ("null", None)),
            ('{"tool_call": {"type": "x"}}', ("object", {"type": "x"})),
            ('{"tool_call": "str"}', ("none", None)),
            ("plain text", ("none", None)),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(runtime._scan_stream_tool_call(text), expected)
        self.assertEqual(
            runtime._scan_leading_tool_call('{"tool_call": {"type": "y"}}'),
            ("object", {"type": "y"}),
        )


# --------------------------------------------------------------------------- #
# 数据类契约
# --------------------------------------------------------------------------- #


class DataclassContractTests(unittest.TestCase):
    def test_provider_tool_profile_defaults(self) -> None:
        profile = ProviderToolProfile()
        self.assertFalse(profile.supports_native_tools)
        self.assertFalse(profile.native_tools_coexist_with_forced_json)
        self.assertEqual(profile.native_call_shape, "openai_tool_calls")
        self.assertFalse(profile.verified)
        self.assertTrue(profile.force_response_json)
        self.assertEqual(profile.notes, "")

    def test_result_types_field_shape(self) -> None:
        stream_result = ChatJSONStreamResult(
            parsed={}, raw_text="", elapsed_ms=0.0, error="",
            latest_emotion="", latest_speech="", latest_reply_medium="",
        )
        self.assertFalse(stream_result.stopped_early)
        self.assertIsNone(stream_result.early_tool_call)
        self.assertFalse(stream_result.rescue_attempted)
        self.assertFalse(stream_result.rescue_succeeded)
        bundle = ModelBundle(client=object(), model="m")
        self.assertEqual(bundle.model, "m")


# --------------------------------------------------------------------------- #
# 边界：旧私有名 / 外来标识 / 旧整句 / import 面
# --------------------------------------------------------------------------- #


class C2_5BoundaryTests(unittest.TestCase):
    def test_no_legacy_private_names(self) -> None:
        import io
        import tokenize as py_tokenize

        for path in SRC_FILES:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                tokens = {
                    token.string
                    for token in py_tokenize.generate_tokens(io.StringIO(source).readline)
                    if token.type == py_tokenize.NAME
                }
                hits = sorted(set(LEGACY_NAMES) & tokens)
                self.assertEqual(hits, [])

    def test_no_foreign_identifiers(self) -> None:
        for path in SRC_FILES:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8").lower()
                for token in FOREIGN_TOKENS:
                    self.assertNotIn(token.lower(), source)

    def test_old_prose_not_reproduced(self) -> None:
        old_sentences = [
            "Expose provider-aware model configuration while keeping v1 routes alive.",
            "Return whether a streamed result is unusable for a reply turn.",
            "Recover a failed stream with one complete JSON request.",
            "Whether the selected protocol supports multiple system messages.",
            "Surface silent length-truncation through the metric + last-error",
            "Record provider-reported input/output tokens once per LLM call.",
        ]
        for path in SRC_FILES:
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                for sentence in old_sentences:
                    self.assertNotIn(sentence, source)

    def test_imports_restricted_to_shinku_and_stdlib(self) -> None:
        import ast as py_ast

        stdlib_imports = (
            "json", "logging", "os", "re", "threading", "time", "hashlib",
        )
        stdlib_from = (
            "json", "logging", "os", "re", "threading", "time",
            "dataclasses", "pathlib", "typing", "urllib", "hashlib",
            "__future__",
        )
        for path in SRC_FILES:
            with self.subTest(path=path.name):
                tree = py_ast.parse(path.read_text(encoding="utf-8"))
                for node in py_ast.walk(tree):
                    if isinstance(node, py_ast.Import):
                        for alias in node.names:
                            root = alias.name.split(".")[0]
                            # json_repair 是惰性可选依赖（失败即返回 None）。
                            self.assertIn(root, stdlib_imports + ("json_repair",))
                    elif isinstance(node, py_ast.ImportFrom) and node.level == 0 and node.module:
                        root = node.module.split(".")[0]
                        if root not in stdlib_from:
                            self.assertTrue(node.module.startswith("shinku."))

    def test_optional_persistent_metrics_import_is_guarded(self) -> None:
        source = (Path(core_mod.__file__).read_text(encoding="utf-8"))
        self.assertIn("try:", source[:4000])
        self.assertIn("except ImportError", source[:4000])


if __name__ == "__main__":
    unittest.main()
