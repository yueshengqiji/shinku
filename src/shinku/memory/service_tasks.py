"""Service responsibilities: tasks."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any


from .service_models import MemoryServiceTasksDependencies


def record_agent_task_summary(
    *,
    conversation_id: str,
    conversation_type: str,
    requester_id: str,
    task_id: str,
    goal: str,
    status: str,
    outcome: str,
    reason: str='',
    rounds: int=0,
    tool_outcomes: Sequence[Mapping[str, Any]]=(),
    metadata: Mapping[str, Any] | None=None,
    deps: MemoryServiceTasksDependencies,
) -> bool:
    """Persist a compact, owner-only index card for a terminal Agent task.

        This compact index card stores the goal, final user-facing outcome,
        status, reason, tool names/statuses, and task ID. A separate, expiring
        detail record may keep the ordered tool names/statuses. Raw tool
        arguments, outputs, and model reasoning are intentionally excluded.
        The index card is non-expiring; the owner's normal forget/reset path
        remains its retention control.
        """

    task_key = str(task_id or "").strip()[:240]
    normalized_status = str(status or "").strip().lower()
    if not task_key or normalized_status not in {"completed", "failed", "blocked"}:
        return False
    source_scope = deps.scope_for(conversation_id, conversation_type)
    task_metadata = dict(metadata or {})
    task_metadata["sender_id"] = str(requester_id or task_metadata.get("sender_id") or "").strip()
    if not deps.owner_memory_allowed(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        sender_id=task_metadata["sender_id"],
        metadata=task_metadata,
        role="user",
    ):
        return False

    def clean(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").split())
        text = deps.task_secret_pattern().sub(r"\1\2[已省略]", text)
        text = deps.task_bearer_pattern().sub("Bearer [已省略]", text)
        return text[:limit]

    clean_goal = clean(goal, 600)
    clean_outcome = clean(outcome, 900)
    clean_reason = clean(reason, 240)
    if not clean_goal:
        return False

    normalized_tools: list[dict[str, str]] = []
    for item in tuple(tool_outcomes or ())[:16]:
        if not isinstance(item, Mapping):
            continue
        name = clean(item.get("name"), 80)
        if not name:
            continue
        normalized_tools.append({"name": name, "status": clean(item.get("status"), 24)})

    status_label = {
        "completed": "已完成",
        "failed": "失败",
        "blocked": "受阻",
    }[normalized_status]
    lines = [
        f"Agent任务目标：{clean_goal}",
        f"执行状态：{status_label}",
    ]
    if clean_outcome:
        lines.append(f"最终结果：{clean_outcome}")
    if clean_reason:
        lines.append(f"停止原因：{clean_reason}")
    if normalized_tools:
        tool_text = "、".join(
            f"{item['name']}（{item['status'] or '状态未记录'}）" for item in normalized_tools
        )
        lines.append(f"涉及工具：{tool_text}")
    lines.append(f"任务档案 ID：{task_key}")

    now = int(deps.clock())
    owner_metadata = {
        "source": "agent_task_summary",
        "record_type": "agent_task_summary",
        "role": "agent_task",
        "task_id": task_key,
        "task_status": normalized_status,
        "task_reason": clean_reason,
        "rounds": max(0, min(1000, int(rounds or 0))),
        "tool_outcomes": normalized_tools,
        "owner_memory": True,
        "owner_source_scope": source_scope,
        "shinku_involved": bool(task_metadata.get("shinku_involved")),
    }
    return deps.store().record_event(
        conversation_id="master",
        scope=deps.owner_scope(),
        text="\n".join(lines),
        source_id=f"owner:agent-task:{task_key}",
        layer="episodic",
        created_at=now,
        importance=0.72 if normalized_status != "completed" else 0.62,
        confidence=0.82,
        expires_at=0,
        metadata=owner_metadata,
    )


def inspect_agent_task(
    *,
    requester_id: str,
    task_id: str,
    deps: MemoryServiceTasksDependencies,
) -> dict[str, Any]:
    """Open an exact owner task card without searching or exposing other scopes."""

    task_key = str(task_id or "").strip()
    if not deps.is_master_user(requester_id) or not task_key or len(task_key) > 240:
        return {
            "found": False,
            "task_id": task_key[:240],
            "rendered": "没有找到这条任务档案。",
        }
    deps.store().expire(now=int(deps.clock()))
    record = deps.store().event_by_source_id(
        source_id=f"owner:agent-task:{task_key}",
        scope=deps.owner_scope(),
    )
    metadata = record.metadata if record is not None and isinstance(record.metadata, Mapping) else {}
    if (
        record is None
        or metadata.get("record_type") != "agent_task_summary"
        or metadata.get("owner_memory") is not True
        or str(metadata.get("task_id") or "") != task_key
    ):
        return {
            "found": False,
            "task_id": task_key,
            "rendered": "没有找到这条任务档案。",
        }
    if metadata.get("experience_schema") == 1:
        projector = getattr(deps.store(), "task_experience", None)
        if projector is None or not projector.validate_record(record):
            return {"found": False, "task_id": task_key,
                    "rendered": "这条任务经历的来源当前无法核实。"}
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(record.created_at))
    rendered = f"【主人 Agent 任务档案】\n任务 ID：{task_key}\n记录时间：{stamp}\n{record.text}"
    detail = None
    detail_available = False
    if deps.policy().agent_task_detail_enabled and metadata.get("experience_schema") != 1:
        candidate = deps.store().event_by_source_id(
            source_id=f"owner:agent-task-detail:{task_key}",
            scope=deps.owner_scope(),
        )
        detail_metadata = (
            candidate.metadata
            if candidate is not None and isinstance(candidate.metadata, Mapping)
            else {}
        )
        if (
            candidate is not None
            and detail_metadata.get("record_type") == "agent_task_detail"
            and detail_metadata.get("owner_memory") is True
            and str(detail_metadata.get("task_id") or "") == task_key
        ):
            detail = candidate
            detail_available = True
            rendered += "\n\n" + candidate.text[: min(6000, deps.policy().render_total_chars)]
    return {
        "found": True,
        "task_id": task_key,
        "status": str(metadata.get("task_status") or "unknown"),
        "created_at": int(record.created_at),
        "detail_available": detail_available,
        "record": record.as_dict(),
        "detail_record": detail.as_dict() if detail is not None else None,
        "rendered": rendered,
    }


def record_agent_task_detail(
    *,
    conversation_id: str,
    conversation_type: str,
    requester_id: str,
    task_id: str,
    status: str,
    rounds: int,
    steps: Sequence[Mapping[str, Any]]=(),
    metadata: Mapping[str, Any] | None=None,
    deps: MemoryServiceTasksDependencies,
) -> bool:
    """Store a bounded owner-only tool sequence without raw arguments/results."""

    task_key = str(task_id or "").strip()[:240]
    normalized_status = str(status or "").strip().lower()
    if (
        not deps.policy().agent_task_detail_enabled
        or not task_key
        or normalized_status not in {"completed", "failed", "blocked"}
    ):
        return False
    source_scope = deps.scope_for(conversation_id, conversation_type)
    task_metadata = dict(metadata or {})
    task_metadata["sender_id"] = str(requester_id or task_metadata.get("sender_id") or "").strip()
    if not deps.owner_memory_allowed(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        sender_id=task_metadata["sender_id"],
        metadata=task_metadata,
        role="user",
    ):
        return False

    safe_steps: list[dict[str, str]] = []
    for item in tuple(steps or ())[: max(1, min(64, int(deps.policy().agent_task_detail_max_steps or 32)))]:
        if not isinstance(item, Mapping):
            continue
        name = " ".join(str(item.get("name") or "").split())[:80]
        if not name:
            continue
        step_status = " ".join(str(item.get("status") or "unknown").split())[:24]
        safe_steps.append({"name": name, "status": step_status or "unknown"})
    if not safe_steps:
        return False

    status_label = {"completed": "已完成", "failed": "失败", "blocked": "受阻"}[normalized_status]
    lines = [
        "Agent任务过程（目标与最终结果见长期摘要卡片）：",
        f"结束状态：{status_label}",
        f"规划轮数：{max(0, min(1000, int(rounds or 0)))}",
        "工具步骤（仅顺序、工具名和状态；不含参数、原始输出或思维过程）：",
    ]
    for index, item in enumerate(safe_steps, start=1):
        lines.append(f"{index}. {item['name']}：{item['status']}")

    now = int(deps.clock())
    stored = deps.store().record_event(
        conversation_id="master",
        scope=deps.owner_scope(),
        text="\n".join(lines),
        source_id=f"owner:agent-task-detail:{task_key}",
        layer="episodic",
        created_at=now,
        importance=0.42,
        confidence=0.9,
        expires_at=now + max(3600, int(deps.policy().agent_task_detail_ttl_seconds or 30 * 86400)),
        metadata={
            "source": "agent_task_detail",
            "record_type": "agent_task_detail",
            "role": "agent_task_detail",
            "task_id": task_key,
            "task_status": normalized_status,
            "rounds": max(0, min(1000, int(rounds or 0))),
            "tool_steps": safe_steps,
            "owner_memory": True,
            "owner_source_scope": source_scope,
            "shinku_involved": bool(task_metadata.get("shinku_involved")),
        },
    )
    deps.store().prune_agent_task_details(
        scope=deps.owner_scope(),
        max_records=deps.policy().agent_task_detail_max_records,
    )
    return stored
