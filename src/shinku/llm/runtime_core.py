"""LLM 调用引擎 —— 传输核心（C2-5a 洁净室重写）。

职责切分
--------
本模块承载 ``LLMRuntime`` 的骨架与底层通道：

- 模型 bundle 的构建与热切换（aux / chat 两路）；
- 三种调用通道：非流式 JSON、NDJSON 事件流、流式 chat JSON；
- 请求 payload 组装（消息序列、多 system 块、图片项、历史轮次）；
- 瞬时错误重试（限流 / 网关 5xx / 超时重试一次）与熔断记账；
- 流式顶层 JSON 的增量解析（``StreamTap``）；
- 回复 JSON 的修复、截断恢复与兜底；
- 度量计数与最近一次错误的脱敏记录。

另外两片在后续批次落地：

- ``runtime_capabilities``（C2-5b）：原生工具、思考控制、协议能力协商；
- ``runtime_audit``（C2-5c）：prompt 审计、token / 缓存用量度量。
``runtime.py`` 把三者合成为公开的 ``LLMRuntime``。

本仓不 import 旧项目的配置模块：一切可调项从 ``SHINKU_*`` 环境变量
读取（入口是 ``RuntimeCore._env``），键名沿用旧配置属性名（契约事实，
待 C4/C6 复查）。完整契约见 ``docs/contracts/c2_5_runtime.md``。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generator

from shinku.llm.client import build_llm_client
from shinku.tools.invocation import (
    NATIVE_OPENAI,
    NATIVE_TOOL_CALL_FIELD,
    TOOL_INVOCATION_ID_FIELD,
    TOOL_SOURCE_FIELD,
)

try:  # 持久化计数件是可选件：缺失时退回内存计数（对拍基线同样不带它）。
    from shinku.llm.persistent_metrics import PersistentCounterStore
except ImportError:  # pragma: no cover - 可选依赖缺失
    PersistentCounterStore = None  # type: ignore[assignment]


logger = logging.getLogger("shinku.llm_debug")

#: 一次调用最多打两次（瞬时错误重试一次）；再多就交给上层兜底。
_HTTP_ATTEMPTS = 2
_RETRY_BACKOFF = 0.8
_RETRYABLE_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_RETRYABLE_TOKENS = (
    "429",
    "rate limit",
    "too many requests",
    "timed out",
    "timeout",
    "connection",
    "temporarily",
    "overloaded",
    "unavailable",
    "500",
    "502",
    "503",
    "504",
)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

#: 错误明细进入 last-error 通道前先过这三条正则打码。
SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)([A-Za-z0-9._~+/=-]{8,})"),
    re.compile(r"(?i)((?:api[_-]?key|authorization|x-api-key)\s*[:=]\s*)([^\s,;]+)"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)

#: 流式 tap 解析 JSON 转义序列用的映射表。
JSON_ESCAPE_MAP = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

#: 回复媒介口径的用户输入别名（含中文俗称）归一到三种取值。
REPLY_MEDIUM_ALIASES = {
    "text": "text",
    "文字": "text",
    "文本": "text",
    "voice": "voice",
    "audio": "voice",
    "record": "voice",
    "语音": "voice",
    "both": "both",
    "all": "both",
    "text_voice": "both",
    "voice_text": "both",
    "文字语音": "both",
    "双发": "both",
}


class TransientLLMError(Exception):
    """内部信号：本次失败属于瞬时故障，重试有望救回。"""


def is_retryable_llm_error(exc: Exception) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    try:
        if status is not None and int(status) in _RETRYABLE_CODES:
            return True
    except (TypeError, ValueError):
        pass
    text = f"{type(exc).__name__} {exc}".casefold()
    return any(token in text for token in _RETRYABLE_TOKENS)


def normalize_reply_medium(value: Any) -> str:
    """把用户输入的媒介词归一成 ``text`` / ``voice`` / ``both``，认不出给空串。"""

    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return ""
    return REPLY_MEDIUM_ALIASES.get(text, "")


@dataclass
class ModelBundle:
    client: Any
    model: str


@dataclass(frozen=True)
class ProviderToolProfile:
    """一个 (host, model) 组合的原生工具能力画像。"""

    supports_native_tools: bool = False
    native_tools_coexist_with_forced_json: bool = False
    native_call_shape: str = "openai_tool_calls"
    verified: bool = False
    force_response_json: bool = True
    notes: str = ""


DEFAULT_PROVIDER_TOOL_PROFILE = ProviderToolProfile()
CONFIG_ALLOWLISTED_PROVIDER_TOOL_PROFILE = ProviderToolProfile(
    supports_native_tools=True,
    native_tools_coexist_with_forced_json=False,
    verified=False,
    notes=(
        "Enabled by NATIVE_TOOL_PROVIDER_ALLOWLIST. Treat as OpenAI-compatible "
        "prompt-only JSON until a live probe verifies response_format coexistence."
    ),
)
PROVIDER_TOOL_PROFILES: dict[tuple[str, str], ProviderToolProfile] = {
    (
        "api.deepseek.com",
        "deepseek-v4-flash",
    ): ProviderToolProfile(
        supports_native_tools=True,
        native_tools_coexist_with_forced_json=False,
        verified=True,
        force_response_json=False,
        notes=(
            "DeepSeek supports native tool_calls on simple prompts. With native tools "
            "enabled, forced tool calls use tool_choice=required and the engine parses "
            "tool_calls from the response; keep forced response_format=json_object off "
            "(force_response_json=False) because forced JSON makes DeepSeek drop "
            "tool_calls entirely. Verified by live probes 2026-08-27."
        ),
    ),
    (
        "api.deepseek.com",
        "deepseek-v4-pro",
    ): ProviderToolProfile(
        supports_native_tools=True,
        native_tools_coexist_with_forced_json=True,
        verified=True,
        notes="Project probe: tools and response_format=json_object can coexist.",
    ),
}


@dataclass
class NDJSONCallResult:
    events: list[dict[str, Any]]
    event_timings: list[dict[str, Any]]
    elapsed_ms: float
    stopped_early: bool
    completed_stream: bool
    stop_event_type: str
    error: str


@dataclass
class ChatJSONStreamResult:
    parsed: dict[str, Any]
    raw_text: str
    elapsed_ms: float
    error: str
    latest_emotion: str
    latest_speech: str
    latest_reply_medium: str
    stopped_early: bool = False
    early_tool_call: dict[str, Any] | None = None
    rescue_attempted: bool = False
    rescue_succeeded: bool = False


class StreamTap:
    """流式顶层 JSON 的增量抽取器。

    逐字符扫过流式片段，在对象还没拼完整时就把 ``emotion`` / ``speech`` /
    ``reply_medium`` / ``speech_segments`` 抽出来变成 UI 事件，让桌宠和
    gal 前端能边收边播。``feed()`` 返回本次新产生的事件列表。
    """

    def __init__(self):
        self.depth = 0
        self.in_string = False
        self.string_role = ""
        self.string_buffer: list[str] = []
        self.current_key = ""
        self.captured_value_key: str | None = None
        self.expecting_key = False
        self.expecting_colon = False
        self.expecting_value = False
        self.in_primitive = False
        self.escape_pending = False
        self.unicode_buffer: list[str] | None = None
        self.latest_emotion = ""
        self.latest_speech = ""
        self.latest_reply_medium = ""
        self._ui_hinted = False
        self._medium_hinted = False
        self._segment_count = 0
        self._segment_cursor = 0
        self._segment_limit = 3
        self._segments_array_depth: int | None = None
        self._emitted_segment_keys: set[str] = set()
        self._inline_speech_seen = False
        self._speech_segments: list[str] = []

    def feed(self, text: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        speech_delta: list[str] = []
        for char in str(text or ""):
            if self.in_string:
                self._consume_char(char, events, speech_delta)
                continue

            if self.in_primitive:
                if self.depth == 1 and char == ",":
                    self._reset_pair()
                    continue
                if self.depth == 1 and char == "}":
                    self.in_primitive = False
                    self._clear_pair()
                    self.depth = max(0, self.depth - 1)
                    continue
                if char in "{[":
                    self.depth += 1
                elif char in "}]":
                    self.depth = max(0, self.depth - 1)
                continue

            if char in " \t\r\n":
                continue

            if self._segments_array_depth is not None:
                if char == '"' and self.depth == self._segments_array_depth:
                    self._begin_string("speech_segment")
                    continue
                if char == "]" and self.depth == self._segments_array_depth:
                    self._segments_array_depth = None
                    self.depth = max(0, self.depth - 1)
                    continue

            if char == "{":
                self.depth += 1
                if self.depth == 1:
                    self.expecting_key = True
                    self.expecting_colon = False
                    self.expecting_value = False
                    self.current_key = ""
                elif self.expecting_value:
                    self.expecting_value = False
                continue

            if char == "[":
                self.depth += 1
                if self.expecting_value:
                    if self.current_key == "speech_segments" and self.depth == 2:
                        self._segments_array_depth = self.depth
                    self.expecting_value = False
                continue

            if char == "}":
                if self.depth == 1:
                    self._clear_pair()
                self.depth = max(0, self.depth - 1)
                continue

            if char == "]":
                self.depth = max(0, self.depth - 1)
                continue

            if self.depth != 1:
                continue

            if char == ",":
                self._reset_pair()
                continue

            if self.expecting_key:
                if char == '"':
                    self._begin_string("key")
                continue

            if self.expecting_colon:
                if char == ":":
                    self.expecting_colon = False
                    self.expecting_value = True
                continue

            if self.expecting_value:
                if char == '"':
                    self._begin_string("value")
                    self.captured_value_key = self.current_key
                else:
                    self.expecting_value = False
                    self.in_primitive = True
                continue

        if speech_delta:
            delta_text = "".join(speech_delta)
            if delta_text:
                self.latest_speech += delta_text
                events.append({"type": "speech_chunk", "text": delta_text})
                if self._segment_count < self._segment_limit:
                    remaining = self.latest_speech[self._segment_cursor:]
                    match = re.search(r"[。！？!?\n]", remaining)
                    if match:
                        end = self._segment_cursor + match.end()
                        self._emit_segment(
                            events,
                            self.latest_speech[self._segment_cursor:end],
                        )
                        self._segment_cursor = end
        return events

    def _begin_string(self, role: str) -> None:
        self.in_string = True
        self.string_role = role
        self.string_buffer = []
        self.escape_pending = False
        self.unicode_buffer = None

    def _consume_char(
        self,
        char: str,
        events: list[dict[str, Any]],
        speech_delta: list[str],
    ) -> None:
        if self.unicode_buffer is not None:
            if char.lower() in "0123456789abcdef":
                self.unicode_buffer.append(char)
                if len(self.unicode_buffer) == 4:
                    decoded = ""
                    try:
                        decoded = chr(int("".join(self.unicode_buffer), 16))
                    except Exception:
                        decoded = ""
                    self.unicode_buffer = None
                    if decoded:
                        self._append_char(decoded, speech_delta)
                return

            self.unicode_buffer = None
            self._append_char("u", speech_delta)

        if self.escape_pending:
            self.escape_pending = False
            if char == "u":
                self.unicode_buffer = []
                return
            self._append_char(JSON_ESCAPE_MAP.get(char, char), speech_delta)
            return

        if char == "\\":
            self.escape_pending = True
            return

        if char == '"':
            self.in_string = False
            text = "".join(self.string_buffer)
            if self.string_role == "key":
                self.current_key = text
                self.expecting_key = False
                self.expecting_colon = True
            elif self.string_role == "speech_segment":
                segment_text = self._tidy_segment_text(text)
                if segment_text:
                    self._speech_segments.append(segment_text)
                    if not self._inline_speech_seen:
                        self.latest_speech = "\n".join(self._speech_segments)
                    self._emit_segment(events, segment_text)
            else:
                if self.captured_value_key == "emotion":
                    self.latest_emotion = text
                    if text and not self._ui_hinted:
                        self._ui_hinted = True
                        events.append({"type": "ui", "emotion": text})
                elif self.captured_value_key == "reply_medium":
                    medium = normalize_reply_medium(text)
                    if medium:
                        self.latest_reply_medium = medium
                        if not self._medium_hinted:
                            self._medium_hinted = True
                            events.append({"type": "delivery_hint", "medium": medium})
                self.expecting_value = False
                self.captured_value_key = None
            self.string_role = ""
            self.string_buffer = []
            return

        self._append_char(char, speech_delta)

    def _append_char(self, char: str, speech_delta: list[str]) -> None:
        self.string_buffer.append(char)
        if self.string_role == "value" and self.captured_value_key == "speech":
            # 顶层 speech 与 speech_segments 二选一：见到前者就清掉后者的缓冲。
            if not self._inline_speech_seen and self._speech_segments:
                self.latest_speech = ""
            self._inline_speech_seen = True
            speech_delta.append(char)

    def _emit_segment(self, events: list[dict[str, Any]], text: str) -> bool:
        if self._segment_count >= self._segment_limit:
            return False
        segment_text = self._tidy_segment_text(text)
        if not segment_text:
            return False
        key = re.sub(r"\s+", "", segment_text)
        if not key or key in self._emitted_segment_keys:
            return False
        self._emitted_segment_keys.add(key)
        self._segment_count += 1
        events.append({
            "type": "speech_segment",
            "index": self._segment_count - 1,
            "text": segment_text,
        })
        return True

    def _tidy_segment_text(self, text: Any) -> str:
        return " ".join(str(text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()

    def _reset_pair(self) -> None:
        self.current_key = ""
        self.captured_value_key = None
        self.expecting_key = True
        self.expecting_colon = False
        self.expecting_value = False
        self.in_primitive = False

    def _clear_pair(self) -> None:
        self.current_key = ""
        self.captured_value_key = None
        self.expecting_key = False
        self.expecting_colon = False
        self.expecting_value = False
        self.in_primitive = False


class RuntimeCore:
    """``LLMRuntime`` 的传输骨架（详见模块 docstring 的职责切分）。"""

    def __init__(self, *, metrics_path: Path | None = None):
        self._swap_lock = threading.RLock()
        self.aux = self._make_aux_bundle()
        self.chat = self._make_chat_bundle()
        self._counters_lock = threading.RLock()
        self._counters = {
            "aux_json_calls": 0,
            "chat_json_calls": 0,
            "aux_ndjson_calls": 0,
            "chat_stream_calls": 0,
            "errors": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "cache_read_tokens_chat": 0,
            "cache_creation_tokens_chat": 0,
            "cache_usage_reports_chat": 0,
            "cache_read_tokens_aux": 0,
            "cache_creation_tokens_aux": 0,
            "cache_usage_reports_aux": 0,
            "cache_usage_reports": 0,
            "cache_usage_unavailable_calls": 0,
            "cache_usage_incomplete_streams": 0,
            "cache_fields_unavailable_reports": 0,
            "cache_read_zero_reports": 0,
            "cache_creation_zero_reports": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "input_tokens_chat": 0,
            "output_tokens_chat": 0,
            "total_tokens_chat": 0,
            "input_tokens_aux": 0,
            "output_tokens_aux": 0,
            "total_tokens_aux": 0,
            "token_usage_reports": 0,
            "token_usage_reports_chat": 0,
            "token_usage_reports_aux": 0,
            "chat_json_fallbacks": 0,
            "chat_stream_rescue_attempts": 0,
            "chat_stream_rescue_successes": 0,
            "chat_stream_rescue_failures": 0,
            "native_tool_decision_sent": 0,
            "native_tool_provider_unsupported": 0,
            "native_tool_call_extracted": 0,
            "native_tool_calls_extra": 0,
            "native_tool_no_call": 0,
            "native_tool_forced_json_suppressed": 0,
        }
        self._counter_store = None
        if metrics_path is not None and PersistentCounterStore is not None:
            self._counter_store = PersistentCounterStore(metrics_path, defaults=self._counters)
        self._error_lock = threading.RLock()
        self._error_state: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    # 配置入口：SHINKU_* 环境变量（键名沿用旧配置属性名）
    # ------------------------------------------------------------------ #

    def _env(self, key: str, default: str = "") -> str:
        return os.environ.get(f"SHINKU_{key}", default)

    # ------------------------------------------------------------------ #
    # bundle 构建与热切换
    # ------------------------------------------------------------------ #

    def reload_from_config(self) -> dict[str, str]:
        aux = self._make_aux_bundle()
        chat = self._make_chat_bundle()
        with self._swap_lock:
            self.aux = aux
            self.chat = chat
        return {
            "status": "reloaded",
            "auxModel": aux.model,
            "chatModel": chat.model,
        }

    def _make_aux_bundle(self) -> ModelBundle:
        client = build_llm_client(
            api_key=self._env("AUX_API_KEY"),
            base_url=self._env("AUX_BASE_URL"),
            protocol=self._env("AUX_API_PROTOCOL", "auto"),
            timeout=90.0,
            max_retries=0,
        )
        return ModelBundle(client=client, model=self._env("AUX_MODEL_NAME"))

    def _make_chat_bundle(self) -> ModelBundle:
        client = build_llm_client(
            api_key=self._env("CHAT_API_KEY"),
            base_url=self._env("CHAT_BASE_URL"),
            protocol=self._env("CHAT_API_PROTOCOL", "auto"),
            timeout=120.0,
            max_retries=0,
        )
        return ModelBundle(client=client, model=self._env("CHAT_MODEL_NAME"))

    # ------------------------------------------------------------------ #
    # 公开调用面
    # ------------------------------------------------------------------ #

    def call_aux_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
        temperature: float = 0.2,
        prompt_cache_key: str = "",
    ) -> dict[str, Any]:
        self._add_metric("aux_json_calls")
        return self._run_json_call(
            bundle=self.aux,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
        )

    def call_chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
        temperature: float = 0.7,
        prompt_cache_key: str = "",
        user_images: list[dict[str, Any]] | None = None,
        native_tools: list[dict[str, Any]] | None = None,
        native_tool_choice: Any = "",
        system_extra_blocks: list[str] | None = None,
        history_turns: list[dict[str, str]] | None = None,
        prompt_audit_sections: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._add_metric("chat_json_calls")
        return self._run_json_call(
            bundle=self.chat,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
            user_images=user_images,
            native_tools=native_tools,
            native_tool_choice=native_tool_choice,
            system_extra_blocks=system_extra_blocks,
            history_turns=history_turns,
            prompt_audit_sections=prompt_audit_sections,
        )

    def chat_supports_native_tools(self) -> bool:
        with self._swap_lock:
            bundle = self.chat
        return self._wants_native_tools(bundle)

    def record_metric(self, key: str, amount: int = 1) -> None:
        self._add_metric(key, amount)

    def call_aux_ndjson(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        on_event: Callable[[dict[str, Any]], bool] | None = None,
        temperature: float = 0.2,
        prompt_cache_key: str = "",
    ) -> NDJSONCallResult:
        self._add_metric("aux_ndjson_calls")
        return self._run_ndjson_call(
            bundle=self.aux,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            on_event=on_event,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
        )

    def stream_chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
        temperature: float = 0.7,
        early_tool_call_validator: Callable[[dict[str, Any]], bool] | None = None,
        prompt_cache_key: str = "",
        user_images: list[dict[str, Any]] | None = None,
        native_tools: list[dict[str, Any]] | None = None,
        native_tool_choice: Any = "",
        system_extra_blocks: list[str] | None = None,
        history_turns: list[dict[str, str]] | None = None,
        prompt_audit_sections: list[dict[str, Any]] | None = None,
    ) -> Generator[dict[str, Any], None, ChatJSONStreamResult]:
        self._add_metric("chat_stream_calls")
        return self._stream_chat(
            bundle=self.chat,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            temperature=temperature,
            early_tool_call_validator=early_tool_call_validator,
            prompt_cache_key=prompt_cache_key,
            user_images=user_images,
            native_tools=native_tools,
            native_tool_choice=native_tool_choice,
            system_extra_blocks=system_extra_blocks,
            history_turns=history_turns,
            prompt_audit_sections=prompt_audit_sections,
        )

    # ------------------------------------------------------------------ #
    # 非流式 JSON 通道
    # ------------------------------------------------------------------ #

    def _run_json_call(
        self,
        *,
        bundle: ModelBundle,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
        temperature: float,
        prompt_cache_key: str,
        user_images: list[dict[str, Any]] | None = None,
        native_tools: list[dict[str, Any]] | None = None,
        native_tool_choice: Any = "",
        system_extra_blocks: list[str] | None = None,
        history_turns: list[dict[str, str]] | None = None,
        prompt_audit_sections: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        native_requested = bool(self._coerce_tools(native_tools))
        try:
            response = self._dispatch_completion(
                bundle=bundle,
                payload=self._build_payload(
                    bundle=bundle,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    json_mode=True,
                    prompt_cache_key=prompt_cache_key,
                    user_images=user_images,
                    native_tools=native_tools,
                    native_tool_choice=native_tool_choice,
                    system_extra_blocks=system_extra_blocks,
                    history_turns=history_turns,
                    prompt_audit_sections=prompt_audit_sections,
                ),
            )
            if not self._record_cache(response, prompt_cache_key=prompt_cache_key):
                self._add_metric("cache_usage_unavailable_calls")
            native_tool_call = self._read_native_tool_call(response)
            if native_tool_call is not None:
                self._add_metric("native_tool_call_extracted")
                return {NATIVE_TOOL_CALL_FIELD: native_tool_call, "tool_call": None}
            if native_requested:
                self._add_metric("native_tool_no_call")
            self._note_truncated(response, phase="call_json")
            content = self._read_content_text(response)
            parsed = self._parse_json(content)
            if isinstance(parsed, dict):
                return parsed
            recovered = self._recover_partial(content, fallback=fallback)
            if isinstance(recovered, dict):
                return recovered
            self._note_parse_failure(content, phase="call_json")
        except Exception as exc:
            self._add_metric("errors")
            self._note_error(exc, phase="call_json")
        self._add_metric("chat_json_fallbacks")
        return dict(fallback)

    # ------------------------------------------------------------------ #
    # NDJSON 事件流通道
    # ------------------------------------------------------------------ #

    def _run_ndjson_call(
        self,
        *,
        bundle: ModelBundle,
        system_prompt: str,
        user_prompt: str,
        on_event: Callable[[dict[str, Any]], bool] | None,
        temperature: float,
        prompt_cache_key: str,
    ) -> NDJSONCallResult:
        events: list[dict[str, Any]] = []
        event_timings: list[dict[str, Any]] = []
        response: Any = None
        buffer = ""
        start_at = time.perf_counter()
        stopped_early = False
        cache_usage_reported = False
        token_usage_seen: set[str] = set()
        stop_event_type = ""
        error = ""
        try:
            response = self._dispatch_completion(
                bundle=bundle,
                payload=self._build_payload(
                    bundle=bundle,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    stream=True,
                    prompt_cache_key=prompt_cache_key,
                ),
            )
            for chunk in response:
                cache_usage_reported = self._record_cache(
                    chunk,
                    prompt_cache_key=prompt_cache_key,
                    token_usage_seen=token_usage_seen,
                ) or cache_usage_reported
                text = self._read_stream_text(chunk)
                if not text:
                    continue
                buffer += text
                buffer, parsed_events = self._drain_buffer(buffer)
                for event in parsed_events:
                    events.append(event)
                    event_type = str(event.get("type") or "").strip().lower()
                    event_timings.append(
                        {
                            "index": len(events) - 1,
                            "type": event_type or "unknown",
                            "elapsed_ms": round((time.perf_counter() - start_at) * 1000, 1),
                        }
                    )
                    if on_event and on_event(event):
                        stopped_early = True
                        stop_event_type = event_type or "unknown"
                        break
                if stopped_early:
                    break

            tail_event = self._parse_line(buffer) if not stopped_early else None
            if tail_event is not None:
                events.append(tail_event)
                event_type = str(tail_event.get("type") or "").strip().lower()
                event_timings.append(
                    {
                        "index": len(events) - 1,
                        "type": event_type or "unknown",
                        "elapsed_ms": round((time.perf_counter() - start_at) * 1000, 1),
                    }
                )
                if on_event:
                    if on_event(tail_event):
                        stopped_early = True
                        stop_event_type = event_type or "unknown"
        except Exception as exc:
            error = str(exc or "").strip()
            self._add_metric("errors")
        finally:
            self._close_response(response)
        if not cache_usage_reported:
            self._add_metric(
                "cache_usage_incomplete_streams" if stopped_early else "cache_usage_unavailable_calls"
            )
        elapsed_ms = round((time.perf_counter() - start_at) * 1000, 1)
        return NDJSONCallResult(
            events=events,
            event_timings=event_timings,
            elapsed_ms=elapsed_ms,
            stopped_early=stopped_early,
            completed_stream=(not stopped_early and not error),
            stop_event_type=stop_event_type,
            error=error,
        )

    # ------------------------------------------------------------------ #
    # 流式 chat JSON 通道
    # ------------------------------------------------------------------ #

    def _stream_chat(
        self,
        *,
        bundle: ModelBundle,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
        temperature: float,
        early_tool_call_validator: Callable[[dict[str, Any]], bool] | None,
        prompt_cache_key: str,
        user_images: list[dict[str, Any]] | None = None,
        native_tools: list[dict[str, Any]] | None = None,
        native_tool_choice: Any = "",
        system_extra_blocks: list[str] | None = None,
        history_turns: list[dict[str, str]] | None = None,
        prompt_audit_sections: list[dict[str, Any]] | None = None,
    ) -> Generator[dict[str, Any], None, ChatJSONStreamResult]:
        native_requested = bool(self._coerce_tools(native_tools))
        # 强制工具调用时先发一次非流式请求再进流式：部分供应商（GLM 实测）
        # 在流式通道上会丢掉 required 约束改回纯文本，非流式没有这个问题。
        normalized_tool_choice = self._coerce_tool_choice(native_tool_choice)
        if native_requested and normalized_tool_choice == "required":
            try:
                non_stream = self._run_json_call(
                    bundle=bundle,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    fallback=fallback,
                    temperature=temperature,
                    prompt_cache_key=prompt_cache_key,
                    user_images=user_images,
                    native_tools=native_tools,
                    native_tool_choice=native_tool_choice,
                    system_extra_blocks=system_extra_blocks,
                    history_turns=history_turns,
                    prompt_audit_sections=prompt_audit_sections,
                )
                native_call = non_stream.get(NATIVE_TOOL_CALL_FIELD)
                if native_call is not None:
                    return ChatJSONStreamResult(
                        parsed={NATIVE_TOOL_CALL_FIELD: native_call, "tool_call": None},
                        raw_text="",
                        elapsed_ms=0.0,
                        error="",
                        latest_emotion="",
                        latest_speech="",
                        latest_reply_medium="",
                        stopped_early=True,
                        early_tool_call=None,
                    )
            except Exception as exc:
                error = str(exc or "").strip()
                self._add_metric("errors")
                logger.error(
                    "FORCE_TOOL non-stream fallback failed: %s", error[:200],
                )
        response: Any = None
        error = ""
        raw_parts: list[str] = []
        native_tool_parts: dict[int, dict[str, Any]] = {}
        tap = StreamTap()
        start_at = time.perf_counter()
        stopped_early = False
        early_tool_call: dict[str, Any] | None = None
        tool_probe_disabled = False
        # 连接层兜底：整个流还没产出任何内容时才重试一次，
        # 已经流出去的内容不能重复发。
        max_stream_attempts = 2
        stream_attempt = 0
        yielded_any = False
        cache_usage_reported = False
        while True:
            stream_attempt += 1
            response = None
            error = ""
            raw_parts = []
            native_tool_parts = {}
            tap = StreamTap()
            start_at = time.perf_counter()
            stopped_early = False
            early_tool_call = None
            tool_probe_disabled = False
            token_usage_seen: set[str] = set()
            try:
                response = self._dispatch_completion(
                    bundle=bundle,
                    payload=self._build_payload(
                        bundle=bundle,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        stream=True,
                        json_mode=True,
                        prompt_cache_key=prompt_cache_key,
                        user_images=user_images,
                        native_tools=native_tools,
                        native_tool_choice=native_tool_choice,
                        system_extra_blocks=system_extra_blocks,
                        history_turns=history_turns,
                        prompt_audit_sections=prompt_audit_sections,
                    ),
                )
                for chunk in response:
                    cache_usage_reported = self._record_cache(
                        chunk,
                        prompt_cache_key=prompt_cache_key,
                        token_usage_seen=token_usage_seen,
                    ) or cache_usage_reported
                    self._collect_stream_tool_parts(chunk, native_tool_parts)
                    text = self._read_stream_text(chunk)
                    if not text:
                        continue
                    raw_parts.append(text)
                    for event in tap.feed(text):
                        yielded_any = True
                        yield event
                    if not tool_probe_disabled:
                        probe_state, probe_call = self._scan_stream_tool_call("".join(raw_parts))
                        if probe_state == "object" and isinstance(probe_call, dict):
                            if early_tool_call_validator is None or early_tool_call_validator(probe_call):
                                early_tool_call = dict(probe_call)
                                stopped_early = True
                                break
                            tool_probe_disabled = True
                        elif probe_state in {"null", "none"}:
                            tool_probe_disabled = True
                    if stopped_early:
                        break
            except Exception as exc:
                error = str(exc or "").strip()
                self._add_metric("errors")
                # 只在还没产出任何内容（连接层失败）时重试一次
                if stream_attempt < max_stream_attempts and not yielded_any and not raw_parts:
                    self._add_metric("chat_stream_retries")
                    time.sleep(1.0)
                    continue
            finally:
                cache_usage_reported = self._record_cache(
                    response,
                    prompt_cache_key=prompt_cache_key,
                    token_usage_seen=token_usage_seen,
                ) or cache_usage_reported
                self._close_response(response)
            break

        if not cache_usage_reported:
            self._add_metric(
                "cache_usage_incomplete_streams" if stopped_early else "cache_usage_unavailable_calls"
            )

        raw_text = "".join(raw_parts)
        native_tool_call = self._assemble_stream_tool_call(native_tool_parts)
        if native_tool_call is not None:
            self._add_metric("native_tool_call_extracted")
            parsed = {NATIVE_TOOL_CALL_FIELD: native_tool_call, "tool_call": None}
        elif native_requested:
            self._add_metric("native_tool_no_call")
            parsed = self._parse_json(raw_text)
        elif early_tool_call is not None:
            parsed = {"tool_call": early_tool_call}
        else:
            parsed = self._parse_json(raw_text)

        rescue_attempted = False
        rescue_succeeded = False
        if self._chat_reply_needs_rescue(parsed):
            rescue_attempted = True
            rescued, rescue_error = self._rescue_stream_via_json(
                bundle=bundle,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                prompt_cache_key=prompt_cache_key,
                user_images=user_images,
                native_tools=native_tools,
                native_tool_choice=native_tool_choice,
                system_extra_blocks=system_extra_blocks,
                history_turns=history_turns,
                prompt_audit_sections=prompt_audit_sections,
            )
            if rescued is not None:
                parsed = rescued
                rescue_succeeded = True
                # 非流式的完整响应是权威结果：失败流的半截 tap 状态不能渗回。
                tap.latest_emotion = ""
                tap.latest_speech = ""
                tap.latest_reply_medium = ""
                error = ""
            elif not error and rescue_error:
                error = rescue_error
        if not isinstance(parsed, dict):
            self._add_metric("chat_json_fallbacks")
            self._note_parse_failure(raw_text, phase="stream_chat_json")
            parsed = dict(fallback)
        else:
            parsed = dict(parsed)

        if tap.latest_emotion and not parsed.get("emotion"):
            parsed["emotion"] = tap.latest_emotion
        if tap.latest_speech and not parsed.get("speech"):
            parsed["speech"] = tap.latest_speech
        if tap.latest_reply_medium and not parsed.get("reply_medium"):
            parsed["reply_medium"] = tap.latest_reply_medium

        elapsed_ms = round((time.perf_counter() - start_at) * 1000, 1)
        return ChatJSONStreamResult(
            parsed=parsed,
            raw_text=raw_text,
            elapsed_ms=elapsed_ms,
            error=error,
            latest_emotion=tap.latest_emotion,
            latest_speech=tap.latest_speech,
            latest_reply_medium=tap.latest_reply_medium,
            stopped_early=stopped_early,
            early_tool_call=early_tool_call,
            rescue_attempted=rescue_attempted,
            rescue_succeeded=rescue_succeeded,
        )

    @staticmethod
    def _chat_reply_needs_rescue(parsed: Any) -> bool:
        """判断流式结果还能不能当一轮回复用。

        供应商可能干净地关掉流却一个字都没给，也可能吐半截 / 非 JSON 正文。
        这两种过去会直接滑进 persona 兜底。注意「有意沉默」是合法结果，
        不能再额外打一次模型。
        """

        if not isinstance(parsed, dict):
            return True
        if parsed.get("tool_call") or parsed.get(NATIVE_TOOL_CALL_FIELD):
            return False
        status = str(parsed.get("status") or "final").strip().lower()
        if status in {"skip", "silent", "decline"}:
            return False
        if str(parsed.get("speech") or "").strip():
            return False
        segments = parsed.get("speech_segments")
        return not isinstance(segments, list) or not any(str(item or "").strip() for item in segments)

    def _rescue_stream_via_json(
        self,
        *,
        bundle: ModelBundle,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        prompt_cache_key: str,
        user_images: list[dict[str, Any]] | None,
        native_tools: list[dict[str, Any]] | None,
        native_tool_choice: Any,
        system_extra_blocks: list[str] | None,
        history_turns: list[dict[str, str]] | None,
        prompt_audit_sections: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, Any] | None, str]:
        """流式废了就用一次完整的非流式请求把它救回来。

        只在流式产出不可用时跑一次。用哨兵 fallback 区分「真的解析成功」
        和 ``_run_json_call`` 自己的兜底，不暴露实现字段。
        """

        self._add_metric("chat_stream_rescue_attempts")
        sentinel_key = "__shinku_stream_rescue_fallback__"
        try:
            rescued = self._run_json_call(
                bundle=bundle,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback={sentinel_key: True},
                temperature=temperature,
                prompt_cache_key=prompt_cache_key,
                user_images=user_images,
                native_tools=native_tools,
                native_tool_choice=native_tool_choice,
                system_extra_blocks=system_extra_blocks,
                history_turns=history_turns,
                prompt_audit_sections=prompt_audit_sections,
            )
        except Exception as exc:  # pragma: no cover - 防御：_run_json_call 通常自己吞错
            self._add_metric("chat_stream_rescue_failures")
            return None, self._redact_secrets(str(exc or ""))

        if isinstance(rescued, dict) and sentinel_key not in rescued:
            self._add_metric("chat_stream_rescue_successes")
            return dict(rescued), ""

        self._add_metric("chat_stream_rescue_failures")
        detail = self.snapshot_last_error()
        return None, str(detail.get("message") or "stream_json_rescue_failed")

    # ------------------------------------------------------------------ #
    # 流内早停探测：在对象没收尾时判断 tool_call
    # ------------------------------------------------------------------ #

    def _scan_stream_tool_call(self, text: str) -> tuple[str, dict[str, Any] | None]:
        raw = str(text or "")
        length = len(raw)
        idx = self._skip_ws(raw, 0)
        if idx >= length:
            return "pending", None
        if raw[idx] != "{":
            return "none", None
        decoder = json.JSONDecoder()
        idx += 1

        while True:
            idx = self._skip_ws(raw, idx)
            if idx >= length:
                return "pending", None
            if raw[idx] == "}":
                return "none", None
            if raw[idx] != '"':
                return "pending", None
            try:
                key, key_end = decoder.raw_decode(raw, idx)
            except json.JSONDecodeError:
                return "pending", None
            if not isinstance(key, str):
                return "none", None
            idx = self._skip_ws(raw, key_end)
            if idx >= length:
                return "pending", None
            if raw[idx] != ":":
                return "pending", None
            idx = self._skip_ws(raw, idx + 1)
            if idx >= length:
                return "pending", None
            try:
                value, value_end = decoder.raw_decode(raw, idx)
            except json.JSONDecodeError:
                return "pending", None
            if key == "tool_call":
                if value is None:
                    return "null", None
                return ("object", value) if isinstance(value, dict) else ("none", None)
            idx = self._skip_ws(raw, value_end)
            if idx >= length:
                return "pending", None
            if raw[idx] == ",":
                idx += 1
                continue
            if raw[idx] == "}":
                return "none", None
            return "pending", None

    def _scan_leading_tool_call(self, text: str) -> tuple[str, dict[str, Any] | None]:
        return self._scan_stream_tool_call(text)

    def _skip_ws(self, text: str, start: int) -> int:
        idx = int(start)
        while idx < len(text) and text[idx] in " \t\r\n":
            idx += 1
        return idx

    def _scan_json_value_end(self, text: str, start: int) -> int | None:
        depth = 0
        in_string = False
        escape_pending = False
        for idx in range(int(start), len(text)):
            char = text[idx]
            if in_string:
                if escape_pending:
                    escape_pending = False
                elif char == "\\":
                    escape_pending = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char in "{[":
                depth += 1
                continue
            if char in "}]":
                depth -= 1
                if depth == 0:
                    return idx + 1
                if depth < 0:
                    return None
        return None

    # ------------------------------------------------------------------ #
    # 响应读取
    # ------------------------------------------------------------------ #

    def _read_content_text(self, response: Any) -> str:
        try:
            return self._flatten_content(response.choices[0].message.content).strip()
        except Exception:
            return ""

    def _flatten_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text is None:
                        text = item.get("content")
                    if text is None and isinstance(item.get("input_text"), str):
                        text = item.get("input_text")
                else:
                    text = getattr(item, "text", None)
                    if text is None:
                        text = getattr(item, "content", None)
                if text is not None:
                    parts.append(str(text))
            return "".join(parts)
        return str(content or "")

    def _read_stream_text(self, chunk: Any) -> str:
        try:
            choice = chunk.choices[0]
        except Exception:
            return ""

        delta = getattr(choice, "delta", None)
        if delta is None and isinstance(choice, dict):
            delta = choice.get("delta")

        if isinstance(delta, dict):
            content = delta.get("content")
        else:
            content = getattr(delta, "content", None)

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = str(item.get("text") or item.get("content") or "").strip()
                else:
                    text = str(getattr(item, "text", "") or getattr(item, "content", "") or "").strip()
                if text:
                    parts.append(text)
            return "".join(parts)
        return str(content or "")

    # ------------------------------------------------------------------ #
    # payload 组装
    # ------------------------------------------------------------------ #

    def _build_payload(
        self,
        *,
        bundle: ModelBundle,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        stream: bool = False,
        json_mode: bool = False,
        prompt_cache_key: str = "",
        user_images: list[dict[str, Any]] | None = None,
        native_tools: list[dict[str, Any]] | None = None,
        native_tool_choice: Any = "",
        system_extra_blocks: list[str] | None = None,
        history_turns: list[dict[str, str]] | None = None,
        prompt_audit_sections: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        user_content: str | list[dict[str, Any]]
        image_items = self._coerce_image_items(user_images)
        if image_items:
            user_content = [{"type": "text", "text": user_prompt}, *image_items]
        else:
            user_content = user_prompt
        filtered_system_extra_blocks = self._coerce_system_blocks(system_extra_blocks)
        effective_system_prompt = str(system_prompt or "")
        messages: list[dict[str, Any]] = [{"role": "system", "content": effective_system_prompt}]
        if filtered_system_extra_blocks and self._supports_multi_system(bundle):
            messages.extend(
                {"role": "system", "content": block}
                for block in filtered_system_extra_blocks
            )
        elif filtered_system_extra_blocks and not self._is_anthropic(bundle):
            # 部分兼容端点只认一条 system 消息：保留拼接回退。
            messages[0]["content"] = "\n\n".join(
                part for part in [effective_system_prompt.strip(), *filtered_system_extra_blocks] if part
            )
        for turn in history_turns or []:
            role = str(turn.get("role", "") or "").strip().lower()
            content = str(turn.get("content", "") or "").strip()
            if content and role in {"user", "assistant"}:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_content})
        payload: dict[str, Any] = {
            "model": bundle.model,
            "temperature": temperature,
            "messages": messages,
        }
        if stream:
            payload["stream"] = True
            if self._wants_stream_usage(bundle):
                payload["stream_options"] = {"include_usage": True}
        normalized_tools = self._coerce_tools(native_tools)
        native_tool_profile = self._tool_profile(bundle)
        should_send_native_tools = bool(normalized_tools and native_tool_profile.supports_native_tools)
        # 强制工具调用与 JSON 响应格式互斥：后者会把模型的输出引向
        # 纯 JSON 正文，挤掉工具调用通道。
        normalized_tool_choice = self._coerce_tool_choice(native_tool_choice)
        force_tool_call = bool(normalized_tools and should_send_native_tools and normalized_tool_choice == "required")
        should_use_response_json = bool(
            json_mode
            and self._wants_response_json(bundle)
            and not force_tool_call
        )
        if json_mode and should_send_native_tools and not force_tool_call:
            if native_tool_profile.native_tools_coexist_with_forced_json:
                should_use_response_json = True
            else:
                should_use_response_json = False
                self._add_metric("native_tool_forced_json_suppressed")
        if should_use_response_json:
            payload["response_format"] = {"type": "json_object"}
            self._ensure_json_hint(messages)
        if normalized_tools and should_send_native_tools:
            payload["tools"] = normalized_tools
            self._add_metric("native_tool_decision_sent")
            if normalized_tool_choice:
                payload["tool_choice"] = normalized_tool_choice
        elif normalized_tools:
            self._add_metric("native_tool_provider_unsupported")
        payload.update(self._build_thinking_kwargs(bundle=bundle))
        payload.update(
            self._build_cache_kwargs(
                bundle=bundle,
                prompt_cache_key=prompt_cache_key,
            )
        )
        if filtered_system_extra_blocks and self._is_anthropic(bundle):
            payload["system_extra_blocks"] = filtered_system_extra_blocks
        self._maybe_record_audit(
            bundle=bundle,
            prompt_cache_key=prompt_cache_key,
            messages=messages,
            system_extra_blocks=filtered_system_extra_blocks if self._is_anthropic(bundle) else [],
            history_turns=history_turns,
            user_prompt=user_prompt,
            user_image_count=len(image_items),
            prompt_audit_sections=prompt_audit_sections,
            stream=stream,
            json_mode=json_mode,
            native_tool_count=len(normalized_tools),
        )
        return payload

    def _coerce_system_blocks(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item or "").strip() for item in value if str(item or "").strip()]

    def _is_anthropic(self, bundle: ModelBundle) -> bool:
        protocol = str(getattr(bundle.client, "_shinku_protocol", getattr(bundle.client, "protocol", "")) or "").strip().lower()
        return protocol == "anthropic"

    def _supports_multi_system(self, bundle: ModelBundle) -> bool:
        """所选协议能不能吃多条 system 消息。

        OpenAI 兼容端（含 DeepSeek、Ollama）可以保住第一条 system 稳定、
        后面再补运行期规则。未知协议沿用单条回退。
        """
        protocol = str(
            getattr(bundle.client, "_shinku_protocol", getattr(bundle.client, "protocol", ""))
            or ""
        ).strip().lower()
        return protocol in {"openai", "ollama"}

    def _wants_stream_usage(self, bundle: ModelBundle) -> bool:
        return self._supports_thinking_control(bundle)

    # ------------------------------------------------------------------ #
    # 发送与重试
    # ------------------------------------------------------------------ #

    def _dispatch_completion(self, *, bundle: ModelBundle, payload: dict[str, Any]) -> Any:
        try:
            logger.debug(
                "SEND model=%s tools=%s response_format=%s",
                payload.get("model"),
                bool(payload.get("tools")),
                bool(payload.get("response_format")),
            )
        except Exception:
            pass
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._attempt_completion(bundle=bundle, payload=payload, attempt=attempt)
            except TransientLLMError:
                # 429 / 5xx / 超时是瞬时的：多试一次能救回一整轮回复
                # （2026-09-13 一次限流风暴里 61/111 轮直接吞回答）。
                self._add_metric("llm_http_retry")
                if attempt < _HTTP_ATTEMPTS:
                    time.sleep(_RETRY_BACKOFF * attempt)
                    continue
                raise

    def _attempt_completion(self, *, bundle: ModelBundle, payload: dict[str, Any], attempt: int) -> Any:
        try:
            result = bundle.client.chat.completions.create(**payload)
            try:
                from shinku.llm.circuit_breaker import get_llm_circuit_breaker

                get_llm_circuit_breaker().record_success()
            except Exception:
                pass
            return result
        except TypeError:
            stripped = self._strip_cache_hints(payload)
            if stripped != payload:
                result = bundle.client.chat.completions.create(**stripped)
                try:
                    from shinku.llm.circuit_breaker import get_llm_circuit_breaker

                    get_llm_circuit_breaker().record_success()
                except Exception:
                    pass
                return result
            raise
        except Exception as exc:
            if attempt < _HTTP_ATTEMPTS and is_retryable_llm_error(exc):
                # 瞬时错误：记明细但不记熔断失败，交给上层重试循环。
                self._note_error(exc, phase=f"http_retry_{attempt}")
                try:
                    logger.warning(
                        "LLM call retryable failure (attempt %s/%s): %s | %s",
                        attempt,
                        _HTTP_ATTEMPTS,
                        type(exc).__name__,
                        str(exc)[:200],
                    )
                except Exception:
                    pass
                raise TransientLLMError(str(exc)) from exc
            # 排查 400 用：请求键 + 响应体（凭据已由 _redact_secrets 打码）
            try:
                logger.error(
                    "LLM call failed: %s | payload_keys=%s | model=%s | extra_body=%s | body=%s",
                    type(exc).__name__,
                    sorted(payload.keys()),
                    payload.get("model"),
                    json.dumps(payload.get("extra_body") or {}, ensure_ascii=False)[:300],
                    getattr(getattr(exc, "response", None), "text", "")[:800],
                )
            except Exception:
                pass
            # 熔断记账：连续失败到阈值后，上层冷却期内静默，避免逐条事件刷兜底话术
            try:
                from shinku.llm.circuit_breaker import get_llm_circuit_breaker

                get_llm_circuit_breaker().record_failure()
            except Exception:
                pass
            if self._retry_without_cache_hints(exc):
                stripped = self._strip_cache_hints(payload)
                if stripped != payload:
                    return bundle.client.chat.completions.create(**stripped)
            raise

    def _strip_cache_hints(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "prompt_cache_key" not in payload and "prompt_cache_retention" not in payload:
            return dict(payload)
        sanitized = dict(payload)
        sanitized.pop("prompt_cache_key", None)
        sanitized.pop("prompt_cache_retention", None)
        return sanitized

    def _retry_without_cache_hints(self, exc: Exception) -> bool:
        message = str(exc or "").strip().lower()
        if not message:
            return False
        return (
            "prompt_cache_key" in message
            or "prompt_cache_retention" in message
            or "unexpected keyword" in message
            or "extra_forbidden" in message
            or "unknown parameter" in message
            or "unrecognized request argument" in message
        )

    # ------------------------------------------------------------------ #
    # NDJSON 行解析 / 流关闭
    # ------------------------------------------------------------------ #

    def _drain_buffer(self, buffer: str) -> tuple[str, list[dict[str, Any]]]:
        remaining = str(buffer or "")
        events: list[dict[str, Any]] = []
        while True:
            newline_index = remaining.find("\n")
            if newline_index < 0:
                break
            line = remaining[:newline_index]
            remaining = remaining[newline_index + 1:]
            parsed = self._parse_line(line)
            if parsed is not None:
                events.append(parsed)
        return remaining, events

    def _parse_line(self, line: Any) -> dict[str, Any] | None:
        raw = str(line or "").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _close_response(self, response: Any) -> None:
        if response is None:
            return
        close = getattr(response, "close", None)
        if callable(close):
            try:
                close()
                return
            except Exception:
                pass
        raw_response = getattr(response, "_response", None)
        close = getattr(raw_response, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # 度量 / 最近错误
    # ------------------------------------------------------------------ #

    def _add_metric(self, key: str, amount: int = 1) -> None:
        persistent = getattr(self, "_counter_store", None)
        if persistent is not None:
            persistent.incr(key, amount)
            return
        lock = getattr(self, "_counters_lock", None)
        metrics = getattr(self, "_counters", None)
        if lock is None or not isinstance(metrics, dict):
            return
        with lock:
            metrics[key] = int(metrics.get(key, 0)) + int(amount)

    def snapshot_metrics(self) -> dict[str, int]:
        persistent = getattr(self, "_counter_store", None)
        if persistent is not None:
            return {key: int(value) for key, value in persistent.snapshot().items()}
        with self._counters_lock:
            return {key: int(value) for key, value in self._counters.items()}

    def recent_metric_days(self, days: int = 7) -> list[dict[str, Any]]:
        persistent = getattr(self, "_counter_store", None)
        return persistent.recent_days(days) if persistent is not None else []

    def metrics_persistence_metadata(self) -> dict[str, Any]:
        persistent = getattr(self, "_counter_store", None)
        return persistent.metadata() if persistent is not None else {"persistent": False}

    def close(self) -> None:
        persistent = getattr(self, "_counter_store", None)
        if persistent is not None:
            persistent.flush()

    def snapshot_last_error(self) -> dict[str, str]:
        lock = getattr(self, "_error_lock", None)
        last_error = getattr(self, "_error_state", None)
        if lock is None or not isinstance(last_error, dict):
            return {}
        with lock:
            return {str(key): str(value) for key, value in last_error.items()}

    def _note_error(self, exc: Exception, *, phase: str = "") -> None:
        lock = getattr(self, "_error_lock", None)
        if lock is None:
            return
        detail = {
            "phase": str(phase or ""),
            "type": exc.__class__.__name__,
            "message": self._redact_secrets(str(exc or "")),
        }
        with lock:
            self._error_state = detail

    def _note_truncated(self, response: Any, *, phase: str) -> None:
        """静默截断走度量 + last-error 两条通道暴露出来。

        非流式 payload 不发 max_tokens，供应商侧的默认上限（Anthropic 垫片
        默认 1024）会把回复拦腰截断，截出来的 JSON 还能被当完整的解析掉。
        Anthropic 的 stop reason 映射到 finish_reason="length"；不记这笔，
        截断回复和正常短回复 / 兜底就分不开。
        """
        try:
            choice = response.choices[0]
            finish_reason = (
                str(choice.get("finish_reason") or "")
                if isinstance(choice, dict)
                else str(getattr(choice, "finish_reason", "") or "")
            )
        except Exception:
            return
        if finish_reason.strip().lower() != "length":
            return
        sample = self._read_content_text(response)
        self._add_metric("response_truncated")
        lock = getattr(self, "_error_lock", None)
        if lock is None:
            return
        detail = {
            "phase": str(phase or ""),
            "type": "ResponseTruncated",
            "message": self._redact_secrets(
                f"finish_reason=length, output cut off (chars={len(sample)}); tail={sample[-200:]!r}"
            ),
        }
        with lock:
            self._error_state = detail

    def _note_parse_failure(self, raw_text: Any, *, phase: str) -> None:
        """解析失败滑进兜底时记一份脱敏样本。

        兜底次数由 chat_json_fallbacks 计；没有样本的话，坏回复无声变兜底、
        想复现都没材料。样本过 _redact_secrets 打码 + 截长；取头部不取尾
        ——诊断价值在前半段（正文当 JSON 吐、形状不对之类）。
        """
        lock = getattr(self, "_error_lock", None)
        if lock is None:
            return
        sample = str(raw_text or "")
        detail = {
            "phase": str(phase or ""),
            "type": "ChatJSONFallback",
            "message": self._redact_secrets(
                f"response not valid JSON, used fallback (chars={len(sample)}); head={sample[:200]!r}"
            ),
        }
        with lock:
            self._error_state = detail

    def _redact_secrets(self, message: str, *, max_chars: int = 1000) -> str:
        text = str(message or "")
        for pattern in SECRET_PATTERNS:
            if pattern.groups >= 2:
                text = pattern.sub(lambda match: f"{match.group(1)}[redacted]", text)
            else:
                text = pattern.sub("[redacted]", text)
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        return text

    # ------------------------------------------------------------------ #
    # 回复 JSON 的解析 / 修复 / 部分恢复
    # ------------------------------------------------------------------ #

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else None
        except Exception:
            pass
        parsed = self._first_json_object(raw)
        if isinstance(parsed, dict):
            return parsed
        match = JSON_RE.search(raw)
        if not match:
            repaired = self._heal_json(raw)
            return repaired
        try:
            payload = json.loads(match.group(0))
            return payload if isinstance(payload, dict) else None
        except Exception:
            repaired = self._heal_json(match.group(0))
            return repaired

    def _heal_json(self, text: str) -> dict[str, Any] | None:
        try:
            import json_repair

            repaired = json_repair.repair_json(text, return_objects=True)
            return repaired if isinstance(repaired, dict) else None
        except Exception:
            return None

    def _first_json_object(self, text: str) -> dict[str, Any] | None:
        raw = str(text or "")
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", raw):
            start = match.start()
            prefix = raw[:start].rstrip()
            if prefix and prefix[-1] in {":", "[", ","}:
                continue
            try:
                payload, _stop = decoder.raw_decode(raw[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    def _recover_partial(self, text: str, *, fallback: dict[str, Any]) -> dict[str, Any] | None:
        tap = StreamTap()
        tap.feed(text)
        speech = str(tap.latest_speech or "").strip()
        emotion = str(tap.latest_emotion or "").strip()
        reply_medium = normalize_reply_medium(tap.latest_reply_medium)
        raw_segments = self._read_json_value(text, "speech_segments")
        segments: list[str] = []
        if isinstance(raw_segments, list):
            for item in raw_segments:
                value = (item.get("speech") or item.get("text") or "") if isinstance(item, dict) else item
                segment = " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()
                if segment:
                    segments.append(segment[:500])
                if len(segments) >= 3:
                    break
        if not speech and not emotion and not segments and not reply_medium:
            return None
        recovered = dict(fallback)
        if emotion:
            recovered["emotion"] = emotion
        if reply_medium:
            recovered["reply_medium"] = reply_medium
        if segments:
            recovered["speech"] = "\n".join(segments)
            recovered["speech_segments"] = segments
        elif speech:
            recovered["speech"] = speech
            recovered["speech_segments"] = []
        return recovered

    def _read_json_value(self, text: str, key: str) -> Any:
        raw = str(text or "")
        key_text = str(key or "").strip()
        if not key_text:
            return None
        pattern = re.compile(r'"' + re.escape(key_text) + r'"\s*:\s*')
        for match in pattern.finditer(raw):
            start = self._skip_ws(raw, match.end())
            if start >= len(raw):
                continue
            end = self._scan_json_value_end(raw, start)
            if end is None:
                continue
            try:
                return json.loads(raw[start:end])
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    # 通用小件：dict / object 双形态取值（各 mixin 共用）
    # ------------------------------------------------------------------ #

    def _pick(self, value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)
