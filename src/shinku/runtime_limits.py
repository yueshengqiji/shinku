"""跨宿主的运行限额与时间窗口。

这些值属于部署策略，不属于 QQ 协议或会话接口本身。默认值保持现有行为；
需要调节时通过 ``SHINKU_*`` 环境变量覆盖，避免把同一个数字散落在多个装配点。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

__all__ = ["RuntimeLimits"]


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
class RuntimeLimits:
    """QQ 合并窗口和会话读取上限。"""

    qq_batch_window_seconds: float = 1.5
    session_list_limit: int = 50
    session_message_limit: int = 120

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "RuntimeLimits":
        env = os.environ if environ is None else environ
        return cls(
            qq_batch_window_seconds=_float_value(
                env,
                "SHINKU_QQ_BATCH_WINDOW_SECONDS",
                1.5,
                minimum=0.0,
                maximum=60.0,
            ),
            session_list_limit=_int_value(
                env,
                "SHINKU_SESSION_LIST_LIMIT",
                50,
                minimum=1,
                maximum=1000,
            ),
            session_message_limit=_int_value(
                env,
                "SHINKU_SESSION_MESSAGE_LIMIT",
                120,
                minimum=1,
                maximum=5000,
            ),
        )

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "qq_batch_window_seconds": self.qq_batch_window_seconds,
            "session_list_limit": self.session_list_limit,
            "session_message_limit": self.session_message_limit,
        }
