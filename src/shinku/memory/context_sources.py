"""模型上下文的来源依赖契约：定位、版本和诊断，不保存正文或授权。

提示组装持有 PromptSourceDependencies；模型调用门在生成前后重新取证。
来源读取器只反馈每个定位符的结果与原因，诊断不授予读取权限，也不证明观点正确。
本模块不访问数据库、模型或时钟；实际身份、作用域和时效由每次读取重新决定。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re

from .models import MemoryRecord

CONTEXT_SOURCE_KEY = "_prompt_source_dependencies"
CONTEXT_DIAGNOSTIC_KEY = "context_source_diagnostics"
SourceHandle = tuple[str, str, str]
CONTEXT_SOURCE_READ_PATHS = frozenset({"", "unified_projection", "projection_not_selected", "projection_disabled",
               "projection_unavailable", "legacy_volatile_handoff", "recent_owner_sources",
               "legacy_memory_reference", "legacy_adapter", "projection_not_applicable"})
# 仅用于诊断落盘的固定词表；未知原因不当作安全文本保存，也不改变来源准入结果。
# 新增读取器拒绝原因时同步维护此表，正文、来源ID或外部异常不得作为原因名。
CONTEXT_SOURCE_REJECTION_REASONS = frozenset({
    "source_budget", "source_reader_unavailable", "requester_unavailable", "source_read_failed",
    "source_turn_changed", "invalid_source_contract", "source_window_expired",
    "owner_sources_unavailable", "owner_dependency_changed", "source_dependency_budget",
    "invalid_source_handle", "source_missing", "scope_not_allowed", "source_revoked",
    "source_expired", "source_inactive", "source_not_readable", "source_dependency_invalid",
    "source_dependency_changed", "owner_origin_unavailable", "owner_origin_changed",
    "source_after_current_turn", "source_version_changed", "source_unavailable",
})
_STATES = {"selected", "empty", "disabled", "unavailable", "not_applicable", "untracked", "invalid"}


@dataclass(frozen=True)
class ContextSourceGroup:
    """一种上下文来源的固定版本；signature 可覆盖未展示的更正或长文。"""

    kind: str
    handles: tuple[SourceHandle, ...] = ()
    signature: str | None = None
    expires_at: int = 0
    omitted_sources: int = 0
    read_path: str = ""
    state: str = "selected"
    reason_codes: tuple[str, ...] = ()
    selection_at: int = 0

    def __post_init__(self):
        if self.kind not in {"projection", "owner_continuity", "legacy"}:
            raise ValueError("unknown_context_source_kind")
        if (not isinstance(self.handles, tuple) or len(self.handles) > 64
                or any(not isinstance(h, tuple) or len(h) != 3
                       or any(not isinstance(v, str) or len(v) > 1024 for v in h)
                       or not h[0] or not h[1] for h in self.handles)):
            raise ValueError("invalid_context_source_handles")
        if (self.signature is not None and (not isinstance(self.signature, str)
                                           or not self.signature or len(self.signature) > 256)):
            raise ValueError("invalid_context_source_signature")
        if any(type(v) is not int or v < 0 for v in (self.expires_at, self.omitted_sources, self.selection_at)):
            raise ValueError("invalid_context_source_budget")
        if (self.read_path not in CONTEXT_SOURCE_READ_PATHS or self.state not in _STATES
                or not isinstance(self.reason_codes, tuple) or len(self.reason_codes) > 16
                or any(not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
                       for code in self.reason_codes)):
            raise ValueError("invalid_context_source_diagnostics")


@dataclass(frozen=True)
class PromptSourceDependencies:
    """只属于本轮身份与消息的定位信息；不能跨轮复用渲染文本或扩大范围。"""

    binding: str
    groups: tuple[ContextSourceGroup, ...] = ()

    def __post_init__(self):
        if (not isinstance(self.binding, str) or not re.fullmatch(r"[0-9a-f]{64}", self.binding)
                or not isinstance(self.groups, tuple) or len(self.groups) > 4
                or any(not isinstance(g, ContextSourceGroup) for g in self.groups)
                or len({g.kind for g in self.groups}) != len(self.groups)):
            raise ValueError("invalid_context_source_groups")

    def with_group(self, group: ContextSourceGroup):
        return replace(self, groups=tuple(g for g in self.groups if g.kind != group.kind) + (group,))

    @property
    def handles(self):
        # 不静默裁去超出总限额的依赖；读取入口会明确拒绝，避免展示了却未核验。
        return tuple(dict.fromkeys(h for group in self.groups for h in group.handles))

    def diagnostics(self):
        """宿主诊断仅包含固定路径、状态和计数；不输出来源 ID、正文或身份。"""
        return {"version": "context-sources-v1", "source_count": len(self.handles),
                "groups": [{"kind": g.kind, "read_path": g.read_path, "state": g.state,
                            "source_count": len(g.handles), "omitted_sources": g.omitted_sources,
                            "reason_codes": list(g.reason_codes)} for g in self.groups],
                "coverage": "tracked_sources_only", "absence_means": "unknown"}


@dataclass(frozen=True)
class SourceReadBatch:
    """按输入槽位反馈来源；失效项为 None，原因只描述本次校验观察到的事实。"""

    records: tuple[MemoryRecord | None, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ContextSourceCheck:
    """一次重新取证的结果；快照不携带正文，只供生成前后比较。"""

    valid: bool
    snapshot: tuple[SourceHandle, ...] = ()
    reason: str = ""
    read_paths: tuple[str, ...] = ()
    rejected: int = 0
    reason_counts: tuple[tuple[str, int], ...] = ()

    def diagnostics(self):
        return {"state": "valid" if self.valid else "invalid", "reason": self.reason,
                "read_paths": list(self.read_paths), "source_count": len(self.snapshot),
                "rejected": self.rejected, "reason_counts": dict(self.reason_counts),
                "absence_means": "unknown"}
