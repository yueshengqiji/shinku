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


def _float_value(env: Mapping[str, str], key: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(str(env.get(key, default) or default).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _id_tuple(env: Mapping[str, str], key: str) -> tuple[str, ...]:
    """Read a bounded comma/semicolon separated owner-id list."""

    raw = str(env.get(key, "") or "")
    values = raw.replace("，", ",").replace("；", ";").replace("\n", ",")
    result: list[str] = []
    for item in values.replace(";", ",").split(","):
        value = item.strip()
        if value and value not in result:
            result.append(value)
    return tuple(result[:32])


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
    recent_limit: int = 6
    recent_window_seconds: int = 15 * 60
    topic_window_seconds: int = 10 * 60
    recent_render_total_chars: int = 1800
    passive_observe_group: bool = True
    passive_observe_private: bool = False
    # The private owner memory is opt-in and only accepts these QQ identities.
    # It is deliberately separate from the command-authorisation list.
    master_user_ids: tuple[str, ...] = ()
    # Semantic retrieval is opt-in in the public runtime and enabled by the
    # private profile.  The heavyweight model remains an optional dependency.
    semantic_enabled: bool = False
    semantic_provider: str = "huggingface"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = ""
    embedding_cache_dir: str = ""
    embedding_local_files_only: bool = False
    semantic_backfill_batch_size: int = 128
    semantic_index_interval_seconds: float = 3.0
    semantic_min_score: float = 0.22

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
            recent_limit=_int_value(env, "SHINKU_MEMORY_RECENT_LIMIT", 6, minimum=1, maximum=20),
            recent_window_seconds=_int_value(env, "SHINKU_MEMORY_RECENT_WINDOW_SECONDS", 15 * 60, minimum=30, maximum=7 * 86400),
            topic_window_seconds=_int_value(env, "SHINKU_MEMORY_TOPIC_WINDOW_SECONDS", 10 * 60, minimum=30, maximum=7 * 86400),
            recent_render_total_chars=_int_value(env, "SHINKU_MEMORY_RECENT_RENDER_TOTAL_CHARS", 1800, minimum=200, maximum=20000),
            passive_observe_group=_flag(env, "SHINKU_MEMORY_PASSIVE_OBSERVE_GROUP", True),
            passive_observe_private=_flag(env, "SHINKU_MEMORY_PASSIVE_OBSERVE_PRIVATE", False),
            master_user_ids=_id_tuple(env, "SHINKU_MEMORY_MASTER_USER_IDS"),
            semantic_enabled=_flag(env, "SHINKU_MEMORY_SEMANTIC_ENABLED", False),
            semantic_provider=str(env.get("SHINKU_MEMORY_SEMANTIC_PROVIDER", "huggingface") or "huggingface").strip().lower() or "huggingface",
            embedding_model=str(env.get("SHINKU_MEMORY_EMBEDDING_MODEL", "BAAI/bge-m3") or "BAAI/bge-m3").strip() or "BAAI/bge-m3",
            embedding_device=str(env.get("SHINKU_MEMORY_EMBEDDING_DEVICE", "") or "").strip(),
            embedding_cache_dir=str(env.get("SHINKU_MEMORY_EMBEDDING_CACHE_DIR", "") or "").strip(),
            embedding_local_files_only=_flag(env, "SHINKU_MEMORY_EMBEDDING_LOCAL_FILES_ONLY", False),
            semantic_backfill_batch_size=_int_value(env, "SHINKU_MEMORY_SEMANTIC_BACKFILL_BATCH_SIZE", 128, minimum=8, maximum=1000),
            semantic_index_interval_seconds=_float_value(env, "SHINKU_MEMORY_SEMANTIC_INDEX_INTERVAL_SECONDS", 3.0, minimum=0.5, maximum=60.0),
            semantic_min_score=_float_value(env, "SHINKU_MEMORY_SEMANTIC_MIN_SCORE", 0.22, minimum=0.0, maximum=1.0),
        )
