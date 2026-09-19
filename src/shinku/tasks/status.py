"""任务状态协议。

这一层只描述任务状态的词汇和合法迁移，不保存任务，也不触碰数据库或执行器。
Agent、工作区和 HTTP 适配器都可以依赖它；真正的持久化由上层负责。
"""

from __future__ import annotations

from collections.abc import Mapping

WORKFLOW_STATUSES = frozenset(
    {"queued", "running", "waiting_user", "completed", "failed", "blocked", "canceled", "cleaned"}
)
STEP_STATUSES = frozenset({"queued", "running", "done", "failed", "waiting_user"})
EVENT_STATUSES = frozenset({"pending", "handled"})
TERMINAL_WORKFLOW_STATUSES = frozenset({"completed", "failed", "canceled", "cleaned"})

_WORKFLOW_ALIASES = {
    "todo": "queued",
    "pending": "queued",
    "working": "running",
    "done": "completed",
    "ok": "completed",
    "error": "failed",
    "cancelled": "canceled",
    "deleted": "cleaned",
    "cleared": "cleaned",
}

_WORKFLOW_EDGES: Mapping[str, frozenset[str]] = {
    "queued": frozenset({"queued", "running", "waiting_user", "blocked", "completed", "failed", "canceled", "cleaned"}),
    "running": frozenset({"running", "waiting_user", "blocked", "completed", "failed", "canceled", "cleaned"}),
    "waiting_user": frozenset({"waiting_user", "queued", "running", "blocked", "completed", "failed", "canceled", "cleaned"}),
    "blocked": frozenset({"blocked", "queued", "running", "waiting_user", "completed", "failed", "canceled", "cleaned"}),
    "completed": frozenset({"completed", "cleaned"}),
    "failed": frozenset({"failed", "cleaned"}),
    "canceled": frozenset({"canceled", "cleaned"}),
    "cleaned": frozenset({"cleaned"}),
}


def _clean(value: object) -> str:
    return str(value or "").strip().lower()


def _normalize(value: object, allowed: frozenset[str], *, default: str, aliases: Mapping[str, str] | None = None) -> str:
    candidate = _clean(value)
    if aliases is not None:
        candidate = aliases.get(candidate, candidate)
    if candidate in allowed:
        return candidate
    fallback = _clean(default)
    if aliases is not None:
        fallback = aliases.get(fallback, fallback)
    if fallback in allowed:
        return fallback
    # Invalid defaults are programmer errors in practice, but returning a stable
    # member keeps this boundary total for malformed external payloads.
    return sorted(allowed)[0]


def normalize_workflow_status(value: object, *, default: str = "queued") -> str:
    """把工作流状态和历史别名归一化为当前协议词汇。"""

    return _normalize(value, WORKFLOW_STATUSES, default=default, aliases=_WORKFLOW_ALIASES)


def normalize_step_status(value: object, *, default: str = "queued") -> str:
    return _normalize(value, STEP_STATUSES, default=default)


def normalize_event_status(value: object, *, default: str = "pending") -> str:
    return _normalize(value, EVENT_STATUSES, default=default)


def validate_workflow_transition(
    current: object,
    target: object,
    *,
    allow_reopen: bool = False,
    allow_cleanup: bool = False,
) -> str:
    """校验一次任务迁移并返回归一化后的目标状态。

    重新打开已结束任务和清理任务是两个需要调用方明确声明的动作，默认不会
    因为外部 payload 写入而发生。
    """

    source = normalize_workflow_status(current)
    destination = normalize_workflow_status(target)
    permitted = destination in _WORKFLOW_EDGES[source]
    permitted = permitted or (allow_reopen and source in {"completed", "failed", "canceled"} and destination == "queued")
    permitted = permitted or (allow_cleanup and destination == "cleaned")
    if not permitted:
        raise ValueError(f"invalid task workflow transition: {source} -> {destination}")
    return destination


# 迁移期别名：保留语义名，不保留旧项目的平铺模块名。
TASK_WORKFLOW_STATUSES = WORKFLOW_STATUSES
TASK_STEP_STATUSES = STEP_STATUSES
TASK_EVENT_STATUSES = EVENT_STATUSES
TERMINAL_TASK_WORKFLOW_STATUSES = TERMINAL_WORKFLOW_STATUSES
TASK_WORKFLOW_TRANSITIONS = _WORKFLOW_EDGES
normalize_workspace_status = normalize_workflow_status
transition_workspace_status = validate_workflow_transition


__all__ = [
    "EVENT_STATUSES",
    "STEP_STATUSES",
    "TASK_EVENT_STATUSES",
    "TASK_STEP_STATUSES",
    "TASK_WORKFLOW_STATUSES",
    "TASK_WORKFLOW_TRANSITIONS",
    "TERMINAL_TASK_WORKFLOW_STATUSES",
    "TERMINAL_WORKFLOW_STATUSES",
    "WORKFLOW_STATUSES",
    "normalize_event_status",
    "normalize_step_status",
    "normalize_workspace_status",
    "normalize_workflow_status",
    "transition_workspace_status",
    "validate_workflow_transition",
]
