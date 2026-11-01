"""LLM runtime facade; channels, payload, transport and response parsing are separate responsibilities: runtime_transport."""

from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable

from shinku.agent.budget import TaskBudgetExceeded

from .runtime_types import (
    ModelBundle,
    TransientLLMError,
    is_retryable_llm_error,
    logger,
)

_completion_deadline = ContextVar("llm_completion_deadline", default=None)


@contextmanager
def completion_deadline(seconds: float | None):
    """Bound all attempts in this call without mutating a shared SDK client."""
    if seconds is None:
        yield
        return
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("invalid_completion_timeout")
    previous = _completion_deadline.get()
    until = time.monotonic() + seconds
    token = _completion_deadline.set(min(previous, until) if previous is not None else until)
    try:
        yield
    finally:
        _completion_deadline.reset(token)


def _remaining():
    until = _completion_deadline.get()
    if until is None:
        return None
    remaining = until - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("llm_completion_deadline")
    return remaining


@dataclass(frozen=True)
class RuntimeTransportDependencies:
    """Bound callbacks and live getters for this responsibility."""
    add_metric: Callable[..., Any]
    attempt_completion: Callable[..., Any]
    note_error: Callable[..., Any]
    retry_without_cache_hints: Callable[..., Any]
    strip_cache_hints: Callable[..., Any]
    runtime_policy: Callable[..., Any]


def dispatch_completion(
    *,
    bundle: ModelBundle,
    payload: dict[str, Any],
    attempt_budget: dict[str, int] | None=None,
    deps: RuntimeTransportDependencies,
) -> Any:
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
    attempt_budget = attempt_budget if attempt_budget is not None else {"remaining": deps.runtime_policy().http_attempts}
    while True:
        attempt += 1
        try:
            return deps.attempt_completion(
                bundle=bundle,
                payload=payload,
                attempt=attempt,
                attempt_budget=attempt_budget,
            )
        except TransientLLMError:
            # 429 / 5xx / 超时是瞬时的：多试一次能救回一整轮回复
            # （2026-09-13 一次限流风暴里 61/111 轮直接吞回答）。
            deps.add_metric("llm_http_retry")
            if attempt < deps.runtime_policy().http_attempts and attempt_budget["remaining"] > 0:
                if deps.runtime_policy().retry_backoff_seconds:
                    pause = deps.runtime_policy().retry_backoff_seconds * attempt
                    remaining = _remaining()
                    if remaining is not None and pause >= remaining:
                        raise TimeoutError("llm_completion_deadline")
                    time.sleep(pause)
                continue
            raise


def attempt_completion(
    *,
    bundle: ModelBundle,
    payload: dict[str, Any],
    attempt: int,
    attempt_budget: dict[str, int] | None=None,
    deps: RuntimeTransportDependencies,
) -> Any:
    from shinku.agent.budget import metered_completion
    attempt_budget = attempt_budget if attempt_budget is not None else {"remaining": deps.runtime_policy().http_attempts}
    def send(request_payload):
        remaining = _remaining()
        if attempt_budget["remaining"] <= 0:
            raise RuntimeError("llm_transport_attempt_limit")
        if remaining is not None:
            configured = request_payload.get("timeout")
            timeout = min(remaining, configured) if isinstance(configured, (float, int)) else remaining
            request_payload = {**request_payload, "timeout": timeout}
        attempt_budget["remaining"] -= 1
        deps.add_metric("llm_http_attempt")
        # 预算拒绝的请求尚未派发；计费观察只包住实际 SDK create，并覆盖每次重试。
        from contextlib import nullcontext

        from shinku.agent.budget import output_reservation

        from .context_compaction import compaction_enabled
        from .model_limits import model_limits
        from .output_budget import current_output_budget
        from .request_ledger import observed_completion
        stage = current_output_budget()
        unlimited = compaction_enabled() or (stage is not None and stage.max_tokens is None)
        guard = output_reservation(model_limits(bundle)['max_output_tokens']) if unlimited else nullcontext()
        with guard:
            return metered_completion(lambda **kw: observed_completion(bundle.client.chat.completions.create, kw), request_payload)
    try:
        result = send(payload)
        try:
            from shinku.llm.circuit_breaker import get_llm_circuit_breaker

            get_llm_circuit_breaker().record_success()
        except Exception:
            pass
        return result
    except TypeError:
        stripped = deps.strip_cache_hints(payload)
        if stripped != payload:
            result = send(stripped)
            try:
                from shinku.llm.circuit_breaker import get_llm_circuit_breaker

                get_llm_circuit_breaker().record_success()
            except Exception:
                pass
            return result
        raise
    except TaskBudgetExceeded:
        raise
    except Exception as exc:
        if attempt < deps.runtime_policy().http_attempts and is_retryable_llm_error(exc):
            # 瞬时错误：记明细但不记熔断失败，交给上层重试循环。
            deps.note_error(exc, phase=f"http_retry_{attempt}")
            try:
                logger.warning(
                    "LLM call retryable failure (attempt %s/%s): %s | %s",
                    attempt,
                    deps.runtime_policy().http_attempts,
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
        if deps.retry_without_cache_hints(exc):
            stripped = deps.strip_cache_hints(payload)
            if stripped != payload:
                return send(stripped)
        raise


def strip_cache_hints(payload: dict[str, Any]) -> dict[str, Any]:
    if "prompt_cache_key" not in payload and "prompt_cache_retention" not in payload:
        return dict(payload)
    sanitized = dict(payload)
    sanitized.pop("prompt_cache_key", None)
    sanitized.pop("prompt_cache_retention", None)
    return sanitized


def retry_without_cache_hints(exc: Exception) -> bool:
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
