"""LLM 运行时门面：持有运行状态，向调用、传输和响应解析模块提供回调。

公开类型由 runtime.py 继续导出；time 保留为既有离线测试的传输时钟替换点。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Generator

from shinku.llm.client import build_llm_client
from shinku.llm.policy import RuntimePolicy

from . import (
    runtime_chat_stream,
    runtime_json_call,
    runtime_ndjson_call,
    runtime_payload,
    runtime_response,
    runtime_transport,
)
from .runtime_chat_stream import RuntimeChatStreamDependencies
from .runtime_json_call import RuntimeJsonCallDependencies
from .runtime_ndjson_call import RuntimeNdjsonCallDependencies
from .runtime_payload import RuntimePayloadDependencies
from .runtime_reply_format import normalize_reply_medium
from .runtime_response import RuntimeResponseDependencies
from .runtime_stream_parser import StreamTap
from .runtime_transport import RuntimeTransportDependencies
from .runtime_types import (
    CONFIG_ALLOWLISTED_PROVIDER_TOOL_PROFILE,
    DEFAULT_PROVIDER_TOOL_PROFILE,
    PROVIDER_TOOL_PROFILES,
    SECRET_PATTERNS,
    ChatJSONStreamResult,
    ModelBundle,
    NDJSONCallResult,
    ProviderToolProfile,
    is_retryable_llm_error,
    logger,
)

try:  # Optional persistence remains replaceable at the original facade.
    from shinku.llm.persistent_metrics import PersistentCounterStore
except ImportError:  # pragma: no cover
    PersistentCounterStore = None


class RuntimeCore:
    """``LLMRuntime`` 的传输骨架（详见模块 docstring 的职责切分）。"""

    def __init__(
        self,
        *,
        metrics_path: Path | None = None,
        runtime_policy: RuntimePolicy | None = None,
    ):
        self.runtime_policy = runtime_policy or RuntimePolicy.from_environment()
        # 标准运行时支持完整JSON流聚合；跳过构造器的不透明适配器不自动宣称支持。
        self.incremental_json_supported = True
        self._runtime_policy_from_environment = runtime_policy is None
        self._apply_circuit_policy()
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
            "chat_json_plain_text_recoveries": 0,
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

    def reload_from_config(self, *, runtime_policy: RuntimePolicy | None = None) -> dict[str, str]:
        if runtime_policy is not None:
            self.runtime_policy = runtime_policy
            self._runtime_policy_from_environment = False
        elif self._runtime_policy_from_environment:
            self.runtime_policy = RuntimePolicy.from_environment()
        self._apply_circuit_policy()
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

    def _apply_circuit_policy(self) -> None:
        """让进程级熔断器跟随本次 runtime 策略，而不是固定在模块默认值。"""

        try:
            from shinku.llm.circuit_breaker import get_llm_circuit_breaker

            get_llm_circuit_breaker().configure(
                failure_threshold=self.runtime_policy.circuit_failure_threshold,
                base_cooldown_seconds=self.runtime_policy.circuit_base_cooldown_seconds,
                max_cooldown_seconds=self.runtime_policy.circuit_max_cooldown_seconds,
            )
        except Exception:
            logger.debug("unable to apply circuit policy", exc_info=True)

    def _make_aux_bundle(self) -> ModelBundle:
        client = build_llm_client(
            api_key=self._env("AUX_API_KEY"),
            base_url=self._env("AUX_BASE_URL"),
            protocol=self._env("AUX_API_PROTOCOL", "auto"),
            timeout=self.runtime_policy.aux_timeout_seconds,
            max_retries=0,
            default_max_tokens=self.runtime_policy.default_max_tokens,
            system_cache_slots=self.runtime_policy.system_cache_slots,
        )
        return ModelBundle(client=client, model=self._env("AUX_MODEL_NAME"))

    def _make_chat_bundle(self) -> ModelBundle:
        client = build_llm_client(
            api_key=self._env("CHAT_API_KEY"),
            base_url=self._env("CHAT_BASE_URL"),
            protocol=self._env("CHAT_API_PROTOCOL", "auto"),
            timeout=self.runtime_policy.chat_timeout_seconds,
            max_retries=0,
            default_max_tokens=self.runtime_policy.default_max_tokens,
            system_cache_slots=self.runtime_policy.system_cache_slots,
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
        attempt_budget: dict[str, int] | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        self._add_metric("aux_json_calls")
        with runtime_transport.completion_deadline(timeout_seconds):
            return self._run_json_call(
                bundle=self.aux,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                fallback=fallback,
                temperature=temperature,
                prompt_cache_key=prompt_cache_key,
                attempt_budget=attempt_budget,
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
        native_history: list[dict[str, Any]] | None = None,
        provider_message_sink: Any = None,
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
            **({"native_history": native_history, "provider_message_sink": provider_message_sink}
               if native_history is not None or provider_message_sink is not None else {}),
        )

    def native_message_scope(self):
        with self._swap_lock:
            bundle = self.chat
        if self._protocol_of(bundle) != "openai":
            from .native_messages import NativeProtocolError
            raise NativeProtocolError("native_history_protocol_unsupported")
        return (id(bundle.client), str(bundle.client.base_url), bundle.model)

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
        attempt_budget: dict[str, int] | None = None,
        native_history: list[dict[str, Any]] | None = None,
        provider_message_sink: Any = None,
    ) -> dict[str, Any]:
        return runtime_json_call.run_json_call(
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
            attempt_budget=attempt_budget,
            deps=self._runtime_json_call_dependencies(),
            **({"native_history": native_history, "provider_message_sink": provider_message_sink}
               if native_history is not None or provider_message_sink is not None else {}),
        )

    def _recover_agent_plain_text(
        self,
        content: Any,
        *,
        fallback: dict[str, Any],
    ) -> dict[str, Any] | None:
        return runtime_json_call.recover_agent_plain_text(
            content,
            fallback=fallback,
        )

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
        return runtime_ndjson_call.run_ndjson_call(
            bundle=bundle,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            on_event=on_event,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
            deps=self._runtime_ndjson_call_dependencies(),
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
        return runtime_chat_stream.stream_chat(
            bundle=bundle,
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
            deps=self._runtime_chat_stream_dependencies(),
        )

    @staticmethod
    def _chat_reply_needs_rescue(parsed: Any) -> bool:
        return runtime_response.chat_reply_needs_rescue(
            parsed,
        )

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
        attempt_budget: dict[str, int] | None = None,
    ) -> tuple[dict[str, Any] | None, str]:
        return runtime_chat_stream.rescue_stream_via_json(
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
            attempt_budget=attempt_budget,
            deps=self._runtime_chat_stream_dependencies(),
        )

    # ------------------------------------------------------------------ #
    # 流内早停探测：在对象没收尾时判断 tool_call
    # ------------------------------------------------------------------ #

    def _scan_stream_tool_call(self, text: str) -> tuple[str, dict[str, Any] | None]:
        return runtime_response.scan_stream_tool_call(
            text,
            deps=self._runtime_response_dependencies(),
        )

    def _scan_leading_tool_call(self, text: str) -> tuple[str, dict[str, Any] | None]:
        return runtime_response.scan_leading_tool_call(
            text,
            deps=self._runtime_response_dependencies(),
        )

    def _skip_ws(self, text: str, start: int) -> int:
        return runtime_response.skip_ws(
            text,
            start,
        )

    # ------------------------------------------------------------------ #
    # 响应读取
    # ------------------------------------------------------------------ #

    def _read_content_text(self, response: Any) -> str:
        return runtime_response.read_content_text(
            response,
            deps=self._runtime_response_dependencies(),
        )

    def _flatten_content(self, content: Any) -> str:
        return runtime_response.flatten_content(
            content,
        )

    def _read_stream_text(self, chunk: Any) -> str:
        return runtime_response.read_stream_text(
            chunk,
        )

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
        budget_context: dict[str, Any] | None = None,
        native_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return runtime_payload.build_payload(
            bundle=bundle,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            stream=stream,
            json_mode=json_mode,
            prompt_cache_key=prompt_cache_key,
            user_images=user_images,
            native_tools=native_tools,
            native_tool_choice=native_tool_choice,
            system_extra_blocks=system_extra_blocks,
            history_turns=history_turns,
            prompt_audit_sections=prompt_audit_sections,
            budget_context=budget_context,
            deps=self._runtime_payload_dependencies(),
            **({"native_history": native_history} if native_history is not None else {}),
        )

    def _coerce_system_blocks(self, value: Any) -> list[str]:
        return runtime_payload.coerce_system_blocks(
            value,
        )

    def _is_anthropic(self, bundle: ModelBundle) -> bool:
        return runtime_payload.is_anthropic(
            bundle,
        )

    def _supports_multi_system(self, bundle: ModelBundle) -> bool:
        return runtime_payload.supports_multi_system(
            bundle,
        )

    def _wants_stream_usage(self, bundle: ModelBundle) -> bool:
        return runtime_payload.wants_stream_usage(
            bundle,
            deps=self._runtime_payload_dependencies(),
        )

    # ------------------------------------------------------------------ #
    # 发送与重试
    # ------------------------------------------------------------------ #

    def _dispatch_completion(self, *, bundle: ModelBundle, payload: dict[str, Any],
                             attempt_budget: dict[str, int] | None = None) -> Any:
        return runtime_transport.dispatch_completion(
            bundle=bundle,
            payload=payload,
            attempt_budget=attempt_budget,
            deps=self._runtime_transport_dependencies(),
        )

    def _attempt_completion(self, *, bundle: ModelBundle, payload: dict[str, Any], attempt: int,
                            attempt_budget: dict[str, int] | None = None) -> Any:
        return runtime_transport.attempt_completion(
            bundle=bundle,
            payload=payload,
            attempt=attempt,
            attempt_budget=attempt_budget,
            deps=self._runtime_transport_dependencies(),
        )

    def _strip_cache_hints(self, payload: dict[str, Any]) -> dict[str, Any]:
        return runtime_transport.strip_cache_hints(
            payload,
        )

    def _retry_without_cache_hints(self, exc: Exception) -> bool:
        return runtime_transport.retry_without_cache_hints(
            exc,
        )

    # ------------------------------------------------------------------ #
    # NDJSON 行解析 / 流关闭
    # ------------------------------------------------------------------ #

    def _drain_buffer(self, buffer: str) -> tuple[str, list[dict[str, Any]]]:
        return runtime_response.drain_buffer(
            buffer,
            deps=self._runtime_response_dependencies(),
        )

    def _parse_line(self, line: Any) -> dict[str, Any] | None:
        return runtime_response.parse_line(
            line,
        )

    def _close_response(self, response: Any) -> None:
        return runtime_response.close_response(
            response,
        )

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
        return runtime_response.parse_json(
            text,
            deps=self._runtime_response_dependencies(),
        )

    def _heal_json(self, text: str) -> dict[str, Any] | None:
        return runtime_response.heal_json(
            text,
        )

    def _first_json_object(self, text: str) -> dict[str, Any] | None:
        return runtime_response.first_json_object(
            text,
        )

    def _recover_partial(self, text: str, *, fallback: dict[str, Any]) -> dict[str, Any] | None:
        return runtime_response.recover_partial(
            text,
            fallback=fallback,
            deps=self._runtime_response_dependencies(),
        )

    def _read_json_value(self, text: str, key: str) -> Any:
        return runtime_response.read_json_value(
            text,
            key,
            deps=self._runtime_response_dependencies(),
        )

    def _runtime_json_call_dependencies(self) -> RuntimeJsonCallDependencies:
        return RuntimeJsonCallDependencies(
            add_metric=self._add_metric,
            build_payload=self._build_payload,
            coerce_tools=self._coerce_tools,
            dispatch_completion=self._dispatch_completion,
            note_error=self._note_error,
            note_parse_failure=self._note_parse_failure,
            note_truncated=self._note_truncated,
            parse_json=self._parse_json,
            read_content_text=self._read_content_text,
            read_native_tool_call=self._read_native_tool_call,
            record_cache=self._record_cache,
            record_context_budget_usage=self._record_context_budget_usage,
            recover_agent_plain_text=self._recover_agent_plain_text,
            recover_partial=self._recover_partial,
            chat=lambda: self.chat,
            optional_is_malformed_native_tool_call=lambda: getattr(self, '_is_malformed_native_tool_call', lambda r: False),
        )

    def _runtime_ndjson_call_dependencies(self) -> RuntimeNdjsonCallDependencies:
        return RuntimeNdjsonCallDependencies(
            add_metric=self._add_metric,
            build_payload=self._build_payload,
            close_response=self._close_response,
            dispatch_completion=self._dispatch_completion,
            drain_buffer=self._drain_buffer,
            parse_line=self._parse_line,
            read_stream_text=self._read_stream_text,
            record_cache=self._record_cache,
            record_context_budget_usage=self._record_context_budget_usage,
        )

    def _runtime_chat_stream_dependencies(self) -> RuntimeChatStreamDependencies:
        return RuntimeChatStreamDependencies(
            add_metric=self._add_metric,
            assemble_stream_tool_call=self._assemble_stream_tool_call,
            build_payload=self._build_payload,
            chat_reply_needs_rescue=self._chat_reply_needs_rescue,
            close_response=self._close_response,
            coerce_tool_choice=self._coerce_tool_choice,
            coerce_tools=self._coerce_tools,
            collect_stream_tool_parts=self._collect_stream_tool_parts,
            dispatch_completion=self._dispatch_completion,
            note_parse_failure=self._note_parse_failure,
            parse_json=self._parse_json,
            read_stream_text=self._read_stream_text,
            record_cache=self._record_cache,
            record_context_budget_usage=self._record_context_budget_usage,
            redact_secrets=self._redact_secrets,
            rescue_stream_via_json=self._rescue_stream_via_json,
            run_json_call=self._run_json_call,
            scan_stream_tool_call=self._scan_stream_tool_call,
            runtime_policy=lambda: self.runtime_policy,
            snapshot_last_error=self.snapshot_last_error,
        )

    def _runtime_payload_dependencies(self) -> RuntimePayloadDependencies:
        return RuntimePayloadDependencies(
            add_metric=self._add_metric,
            build_cache_kwargs=self._build_cache_kwargs,
            build_thinking_kwargs=self._build_thinking_kwargs,
            coerce_image_items=self._coerce_image_items,
            coerce_system_blocks=self._coerce_system_blocks,
            coerce_tool_choice=self._coerce_tool_choice,
            coerce_tools=self._coerce_tools,
            ensure_json_hint=self._ensure_json_hint,
            is_anthropic=self._is_anthropic,
            maybe_record_audit=self._maybe_record_audit,
            record_context_budget=self._record_context_budget,
            supports_multi_system=self._supports_multi_system,
            supports_thinking_control=self._supports_thinking_control,
            tool_profile=self._tool_profile,
            wants_response_json=self._wants_response_json,
            wants_stream_usage=self._wants_stream_usage,
            chat=lambda: self.chat,
        )

    def _runtime_transport_dependencies(self) -> RuntimeTransportDependencies:
        return RuntimeTransportDependencies(
            add_metric=self._add_metric,
            attempt_completion=self._attempt_completion,
            note_error=self._note_error,
            retry_without_cache_hints=self._retry_without_cache_hints,
            strip_cache_hints=self._strip_cache_hints,
            runtime_policy=lambda: self.runtime_policy,
        )

    def _runtime_response_dependencies(self) -> RuntimeResponseDependencies:
        return RuntimeResponseDependencies(
            first_json_object=self._first_json_object,
            flatten_content=self._flatten_content,
            heal_json=self._heal_json,
            parse_line=self._parse_line,
            read_json_value=self._read_json_value,
            scan_stream_tool_call=self._scan_stream_tool_call,
            skip_ws=self._skip_ws,
        )
