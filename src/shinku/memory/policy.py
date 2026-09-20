"""可调的记忆生命周期策略。

记忆策略不是角色设定，也不是数据库结构。把保留、检索和注入边界集中在这里，
可以在不改 ``MemoryStore`` 查询代码的情况下做灰度调参。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Mapping

__all__ = ["MemoryPolicy"]


def _int_value(env: Mapping[str, str], key: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(str(env.get(key, default) or default).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _flag(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = str(env.get(key, "") or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MemoryPolicy:
    """一轮记忆处理使用的边界参数。"""

    event_ttl_seconds: int = 30 * 86400
    retrieval_limit: int = 5
    max_query_terms: int = 12
    event_scan_limit: int = 800
    fact_scan_limit: int = 120
    max_results: int = 12
    deterministic_guards: bool = True
    recent_context_chars: int = 1800
    max_query_chars: int = 240
    render_record_chars: int = 360
    render_total_chars: int = 2400

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "MemoryPolicy":
        env = os.environ if environ is None else environ
        return cls(
            event_ttl_seconds=_int_value(env, "SHINKU_MEMORY_EVENT_TTL_SECONDS", 30 * 86400, minimum=0, maximum=10 * 365 * 86400),
            retrieval_limit=_int_value(env, "SHINKU_MEMORY_RETRIEVAL_LIMIT", 5, minimum=1, maximum=50),
            max_query_terms=_int_value(env, "SHINKU_MEMORY_MAX_QUERY_TERMS", 12, minimum=1, maximum=64),
            event_scan_limit=_int_value(env, "SHINKU_MEMORY_EVENT_SCAN_LIMIT", 800, minimum=10, maximum=10000),
            fact_scan_limit=_int_value(env, "SHINKU_MEMORY_FACT_SCAN_LIMIT", 120, minimum=10, maximum=5000),
            max_results=_int_value(env, "SHINKU_MEMORY_MAX_RESULTS", 12, minimum=1, maximum=100),
            deterministic_guards=_flag(env, "SHINKU_MEMORY_DETERMINISTIC_GUARDS", True),
            recent_context_chars=_int_value(env, "SHINKU_MEMORY_RECENT_CONTEXT_CHARS", 1800, minimum=100, maximum=10000),
            max_query_chars=_int_value(env, "SHINKU_MEMORY_MAX_QUERY_CHARS", 240, minimum=20, maximum=2000),
            render_record_chars=_int_value(env, "SHINKU_MEMORY_RENDER_RECORD_CHARS", 360, minimum=80, maximum=2000),
            render_total_chars=_int_value(env, "SHINKU_MEMORY_RENDER_TOTAL_CHARS", 2400, minimum=200, maximum=20000),
        )
