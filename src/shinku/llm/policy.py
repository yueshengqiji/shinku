"""LLM 传输层的统一运行策略。

默认值保持现有行为；环境变量只负责覆盖运行参数，不改变 provider 协议判断
和请求格式。这样 RuntimeCore、传输客户端和灰度环境不会各自维护一套超时/重试值。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

__all__ = ["RuntimePolicy"]


def _float_value(
    env: Mapping[str, str],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(str(env.get(key, default) or default).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _int_value(
    env: Mapping[str, str],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(str(env.get(key, default) or default).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class RuntimePolicy:
    """LLM 调用链的可调参数；构造后不可变，适合热重载时整体替换。"""

    aux_timeout_seconds: float = 90.0
    chat_timeout_seconds: float = 120.0
    http_attempts: int = 2
    retry_backoff_seconds: float = 0.8
    stream_attempts: int = 2
    stream_retry_delay_seconds: float = 1.0
    default_max_tokens: int = 1024
    system_cache_slots: int = 4
    circuit_failure_threshold: int = 3
    circuit_base_cooldown_seconds: float = 60.0
    circuit_max_cooldown_seconds: float = 600.0

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "RuntimePolicy":
        env = os.environ if environ is None else environ
        return cls(
            aux_timeout_seconds=_float_value(
                env, "SHINKU_LLM_AUX_TIMEOUT_SECONDS", 90.0, minimum=1.0, maximum=3600.0
            ),
            chat_timeout_seconds=_float_value(
                env, "SHINKU_LLM_CHAT_TIMEOUT_SECONDS", 120.0, minimum=1.0, maximum=3600.0
            ),
            http_attempts=_int_value(
                env, "SHINKU_LLM_HTTP_ATTEMPTS", 2, minimum=1, maximum=8
            ),
            retry_backoff_seconds=_float_value(
                env, "SHINKU_LLM_RETRY_BACKOFF_SECONDS", 0.8, minimum=0.0, maximum=60.0
            ),
            stream_attempts=_int_value(
                env, "SHINKU_LLM_STREAM_ATTEMPTS", 2, minimum=1, maximum=8
            ),
            stream_retry_delay_seconds=_float_value(
                env, "SHINKU_LLM_STREAM_RETRY_DELAY_SECONDS", 1.0, minimum=0.0, maximum=60.0
            ),
            default_max_tokens=_int_value(
                env, "SHINKU_LLM_DEFAULT_MAX_TOKENS", 1024, minimum=1, maximum=32768
            ),
            system_cache_slots=_int_value(
                env, "SHINKU_LLM_SYSTEM_CACHE_SLOTS", 4, minimum=0, maximum=32
            ),
            circuit_failure_threshold=_int_value(
                env, "SHINKU_LLM_CIRCUIT_FAILURE_THRESHOLD", 3, minimum=1, maximum=100
            ),
            circuit_base_cooldown_seconds=_float_value(
                env,
                "SHINKU_LLM_CIRCUIT_BASE_COOLDOWN_SECONDS",
                60.0,
                minimum=1.0,
                maximum=86400.0,
            ),
            circuit_max_cooldown_seconds=_float_value(
                env,
                "SHINKU_LLM_CIRCUIT_MAX_COOLDOWN_SECONDS",
                600.0,
                minimum=1.0,
                maximum=604800.0,
            ),
        )

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "aux_timeout_seconds": self.aux_timeout_seconds,
            "chat_timeout_seconds": self.chat_timeout_seconds,
            "http_attempts": self.http_attempts,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "stream_attempts": self.stream_attempts,
            "stream_retry_delay_seconds": self.stream_retry_delay_seconds,
            "default_max_tokens": self.default_max_tokens,
            "system_cache_slots": self.system_cache_slots,
            "circuit_failure_threshold": self.circuit_failure_threshold,
            "circuit_base_cooldown_seconds": self.circuit_base_cooldown_seconds,
            "circuit_max_cooldown_seconds": self.circuit_max_cooldown_seconds,
        }
