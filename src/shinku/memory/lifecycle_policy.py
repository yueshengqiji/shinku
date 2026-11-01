"""Versioned, bounded engineering policy, not a human memory probability model."""

from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class LifecyclePolicy:
    # Shadow collects independent evidence and proposals, never replaces text,
    # changes expiry, or changes real tiers. Off preserves the legacy pipeline.
    mode: str = "shadow"
    version: str = "ebbinghaus-v1"
    l1_seconds: int = 7 * 86400
    l2_seconds: int = 30 * 86400
    minimum_idle_turns: int = 12
    idle_rate: float = 0.08
    active_window_seconds: int = 3600
    repeat_interval_seconds: int = 12 * 3600
    daily_reinforcement_cap: int = 2
    raw_count: int = 3
    raw_days: int = 2
    reinforcement_window_seconds: int = 21 * 86400
    raw_protection_seconds: int = 21 * 86400
    l3_count: int = 6
    l3_days: int = 4
    l3_span_seconds: int = 14 * 86400
    summary_grace_seconds: int = 7 * 86400
    migration_grace_seconds: int = 14 * 86400
    batch_size: int = 64
    scan_interval_seconds: float = 60.0
    summary_max_chars: int = 1200
    audit_limit: int = 2000
    evidence_retention_seconds: int = 60 * 86400
    auto_raw_budget_bytes: int = 64 * 1024 * 1024
    # True 将 L1 持久向量生成延后到原有后台索引器；前台仅临时编码有界缺口。
    # 已建立的派生向量跟随原文有效期，不影响等级、强化或原文回收授权。
    defer_l1_embeddings: bool = False
    # Additional source excerpts may reach the already configured AUX service.
    # Opt-in independently of retention/reclamation; no new model is selected.
    semantic_review_enabled: bool = False
    review_candidates: int = 6
    review_queue_size: int = 16
    review_timeout_seconds: float = 20.0
    review_daily_calls: int = 64
    review_min_confidence: float = 0.9

    def __post_init__(self) -> None:
        for field in fields(self):
            value, default = getattr(self, field.name), field.default
            if isinstance(default, bool):
                valid = isinstance(value, bool)
            elif isinstance(default, str):
                valid = isinstance(value, str)
            elif isinstance(default, int):
                valid = isinstance(value, int) and not isinstance(value, bool)
            else:
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            if not valid:
                raise ValueError(f"invalid_lifecycle_parameter_type:{field.name}")
        if self.mode not in {"off", "shadow", "active"}:
            raise ValueError("invalid_lifecycle_mode")
        if not self.version or len(self.version) > 64:
            raise ValueError("invalid_lifecycle_version")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or isinstance(value, str):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid_lifecycle_parameter:{field.name}")
        if self.idle_rate > 1 or self.batch_size > 1000 or self.summary_max_chars > 10000:
            raise ValueError("unbounded_lifecycle_parameter")
        if self.l2_seconds <= self.l1_seconds or self.l3_days > self.l3_count:
            raise ValueError("inconsistent_lifecycle_thresholds")
        if self.raw_days > self.raw_count or self.evidence_retention_seconds < self.reinforcement_window_seconds:
            raise ValueError("inconsistent_lifecycle_window")
        if (self.review_candidates > 12 or self.review_queue_size > 64 or self.review_daily_calls > 1000
            or self.review_timeout_seconds > 60 or self.review_min_confidence > 1):
            raise ValueError("unbounded_lifecycle_review")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def revision(self) -> str:
        values = asdict(self)
        values.pop("mode")
        # Mode is an activation gate, not a new retention equation. Threshold
        # changes are detected even if an operator forgets to rename version.
        digest = hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
        return f"{self.version}:{digest}"

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> "LifecyclePolicy":
        defaults = cls()
        values = {}
        for field in fields(cls):
            default = getattr(defaults, field.name)
            raw = environ.get(f"SHINKU_MEMORY_LIFECYCLE_{field.name.upper()}")
            if raw is None:
                continue
            try:
                if isinstance(default, bool):
                    parsed = str(raw).strip().lower()
                    if parsed not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
                        raise ValueError("invalid_boolean")
                    value = parsed in {"1", "true", "yes", "on"}
                elif isinstance(default, int):
                    value = int(raw)
                elif isinstance(default, float):
                    value = float(raw)
                else:
                    value = str(raw).strip()
                values[field.name] = value
            except (TypeError, ValueError):
                raise ValueError(f"invalid_lifecycle_environment:{field.name}") from None
        return cls(**values)
