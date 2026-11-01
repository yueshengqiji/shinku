"""LLM 结果类型、供应商能力配置及无状态错误分类。

标准库与其它职责的依赖由使用方直接导入，避免本模块成为隐式依赖中转站。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any


logger = logging.getLogger("shinku.llm_debug")

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
    (
        "api.pinaic.com",
        "gemini-2.5-pro",
    ): ProviderToolProfile(
        supports_native_tools=True,
        native_tools_coexist_with_forced_json=False,
        verified=True,
        force_response_json=False,
        notes=(
            "Pinaic forwarded an OpenAI-compatible native tool_call for this exact "
            "host/model in a harmless live probe on 2026-09-22. Keep forced "
            "response_format=json_object off when tools are available until JSON "
            "coexistence is separately verified."
        ),
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
    first_chunk_ms: float | None = None
    transport_chunks: int = 0
