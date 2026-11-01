"""Projection contracts keep evidence, uncertainty and authority separate."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import time
from typing import Any, Mapping


@dataclass(frozen=True)
class ProjectionPolicy:
    enabled: bool = False
    semantic_enabled: bool = False
    candidate_limit: int = 64
    source_limit: int = 24
    recent_limit: int = 12
    max_chars: int = 10000
    record_chars: int = 2400
    retrieval_ms: int = 3000
    recent_seconds: int = 2 * 86400
    semantic_timeout_ms: int = 30000
    semantic_daily_calls: int = 64
    cache_entries: int = 128
    cache_seconds: int = 3600
    maintenance_batch: int = 256
    # 兼容旧配置名；现在限制每批读取数量（实现再封顶 256），不按行号截断总候选。
    # 整体向量扫描由 retrieval_ms 对应的 Deadline 限制。
    vector_scan_limit: int = 50000
    # Comparison only; the default remains the judge path.
    ambiguity_strategy: str = "judge"
    # Opt-in: the existing conversation model requests history through the
    # scoped memory tool. No per-message classifier or lexical admission gate.
    retrieval_strategy: str = "legacy"

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(item.default, bool):
                if not isinstance(value, bool):
                    raise ValueError("invalid_projection_flag")
            elif isinstance(item.default, int):
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError("invalid_projection_budget")
        if (self.candidate_limit > 256 or self.source_limit > 64 or self.recent_limit > 64
                or self.max_chars > 30000 or self.record_chars > self.max_chars
                or self.retrieval_ms > 10000 or self.semantic_timeout_ms > 30000
                or self.cache_entries > 1024 or self.vector_scan_limit > 100000
                or self.maintenance_batch > 1000 or self.semantic_daily_calls > 1000):
            raise ValueError("unbounded_projection_policy")
        if self.ambiguity_strategy not in {"judge", "main_retry"}:
            raise ValueError("unknown_projection_strategy")
        if self.retrieval_strategy not in {"legacy", "model"}:
            raise ValueError("unknown_projection_retrieval_strategy")

    @classmethod
    def from_environment(cls, env: Mapping[str, str]):
        defaults = cls()
        values = {}
        for item in fields(cls):
            raw = env.get("SHINKU_CONTEXT_PROJECTION_" + item.name.upper())
            if raw is None or not str(raw).strip():
                continue
            default = getattr(defaults, item.name)
            if isinstance(default, bool):
                if str(raw).lower() not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                    raise ValueError("invalid_projection_flag")
                values[item.name] = str(raw).lower() in {"true", "1", "yes", "on"}
            elif isinstance(default, int):
                values[item.name] = int(raw)
            else:
                values[item.name] = str(raw)
        return cls(**values)

    @property
    def version(self):
        return "projection-v2:" + digest(asdict(self))[:16]

    @property
    def read_timeout_seconds(self):
        # One bounded extraction may share the existing audited read call.
        semantic_ms = self.semantic_timeout_ms if self.semantic_enabled else 0
        return min(30., (self.retrieval_ms + semantic_ms + 500) / 1000)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass(frozen=True)
class ProjectionRequest:
    scope: str
    allowed_scopes: tuple[str, ...]
    requester: str
    self_id: str
    query: str
    message_id: str
    now: int
    history_mode: str = "none"
    seed_handles: tuple[tuple[str, str, str], ...] = ()
    reply_message_ids: tuple[str, ...] = ()
    operational_state: Mapping[str, Any] = field(default_factory=dict)
    flow_reason: str = ""
    start_at: int | None = None
    end_at: int | None = None
    # Re-read this turn's source selection instead of replacing it with newly
    # arrived background messages. Locators carry no text or authorization.
    # None discovers a fresh selection; an empty tuple preserves an empty one.
    continuity_handles: tuple[tuple[str, str, str], ...] | None = None
    # The existing router's context-aware query is a locator, never authority.
    # Keep the user's original question for semantic interpretation and dialogue.
    retrieval_query: str = ""
    # 仅宿主从当前配置填写；参与者仍是第三方作者，不获得主人或群聊读取权限。
    participant_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.scope or self.scope not in self.allowed_scopes or not self.requester:
            raise ValueError("projection_requires_authorized_scope_and_requester")
        if len(self.allowed_scopes) > 8 or len(self.seed_handles) > 256 or len(self.continuity_handles or ()) > 64:
            raise ValueError("projection_input_unbounded")
        if not isinstance(self.retrieval_query, str) or len(self.retrieval_query) > 2000:
            raise ValueError("projection_retrieval_query_unbounded")

    @property
    def search_query(self):
        return self.retrieval_query.strip() or self.query


@dataclass
class ProjectionView:
    records: list[dict[str, Any]] = field(default_factory=list)
    entities: list[dict[str, Any]] = field(default_factory=list)
    topic: dict[str, Any] = field(default_factory=dict)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    source_relations: list[dict[str, Any]] = field(default_factory=list)
    rendered: str = ""
    version: str = "projection-v2"
    execution_authority: bool = False
    # 最终有效来源在渲染前登记；展示预算不能删除未展示来源的版本依赖。
    # 这里只保存定位符，不携带正文、授权或可跨轮复用的渲染快照。
    source_dependencies: tuple[tuple[str, str, str], ...] = ()
    # 只含来源位置与署名时间，不能当成原文或模型已经核实的答案。
    source_catalog: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


class Deadline:
    def __init__(self, milliseconds: int):
        self.started = time.monotonic()
        self.until = self.started + milliseconds / 1000

    @property
    def remaining(self):
        return max(0.0, self.until - time.monotonic())

    def check(self):
        if not self.remaining:
            raise TimeoutError("projection_budget_exhausted")

    @property
    def elapsed_ms(self):
        return round((time.monotonic() - self.started) * 1000, 3)
