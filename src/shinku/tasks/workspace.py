"""Agent 任务快照与事件的领域边界。

``TaskWorkspaceService`` 不拥有数据库连接，也不假定某一种存储格式。它把一次
任务创建、状态更新和生命周期事件组织成稳定的调用面；存储实现通过协议注入。
工具执行器可以依赖这个边界，但不能反向把 QQ、模型 SDK 或线程调度塞进来。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol

from .payloads import normalize_artifacts, normalize_steps, normalize_text_list
from .status import normalize_event_status, normalize_workspace_status

__all__ = ["TaskWorkspaceService", "WorkspaceStore"]


class WorkspaceStore(Protocol):
    """任务工作区所需的最小存储面。

    具体实现可以是 SQLite、内存假件或远程适配器；方法返回 JSON 形状的字典，
    从而保持这个领域层与存储实现解耦。
    """

    def add_task_workspace(self, **fields: Any) -> dict[str, Any]: ...

    def get_task_workspace(self, task_id: str) -> dict[str, Any] | None: ...

    def list_task_workspaces(self, **filters: Any) -> list[dict[str, Any]]: ...

    def list_pending_agent_workspaces(self, **filters: Any) -> list[dict[str, Any]]: ...

    def update_task_workspace(self, **fields: Any) -> dict[str, Any] | None: ...

    def append_task_workspace_event(self, **fields: Any) -> dict[str, Any]: ...

    def list_task_workspace_events(self, **filters: Any) -> list[dict[str, Any]]: ...

    def mark_task_workspace_event_handled(self, **fields: Any) -> dict[str, Any] | None: ...


def _now(value: int | None) -> int:
    return int(value) if value is not None else int(time.time())


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _items(value: Sequence[Any] | None) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


class TaskWorkspaceService:
    """组织任务快照和事件，不直接执行工具。"""

    def __init__(self, store: WorkspaceStore) -> None:
        self.store = store

    def create_task(
        self,
        *,
        profile_user_id: str,
        session_id: str,
        raw_request_text: str,
        source_message_id: str = "",
        normalized_goal: str = "",
        success_criteria: Sequence[Any] | None = None,
        constraints: Sequence[Any] | None = None,
        steps: Sequence[Any] | None = None,
        artifacts: Sequence[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        owner: str = "frontstage",
        status: str = "queued",
        timestamp: int | None = None,
        task_id: str = "",
    ) -> dict[str, Any]:
        created_at = _now(timestamp)
        raw_request = {
            "text": str(raw_request_text or "").strip(),
            "source_message_id": str(source_message_id or "").strip(),
        }
        criteria = normalize_text_list(_items(success_criteria))
        limits = normalize_text_list(_items(constraints))
        task = self.store.add_task_workspace(
            profile_user_id=str(profile_user_id or "").strip(),
            session_id=str(session_id or "").strip(),
            owner=str(owner or "frontstage").strip() or "frontstage",
            status=normalize_workspace_status(status),
            raw_request=raw_request,
            normalized_goal=str(normalized_goal or "").strip(),
            success_criteria=criteria,
            constraints=limits,
            steps=normalize_steps(_items(steps)),
            artifacts=normalize_artifacts(_items(artifacts)),
            metadata=_mapping(metadata),
            timestamp=created_at,
            task_id=str(task_id or "").strip(),
        )
        task_key = str(task.get("task_id") or task_id or "").strip()
        self.store.append_task_workspace_event(
            task_id=task_key,
            profile_user_id=str(profile_user_id or "").strip(),
            session_id=str(session_id or "").strip(),
            event_type="task_created",
            from_actor=str(owner or "frontstage").strip() or "frontstage",
            priority="normal",
            requires_user=False,
            message=str(normalized_goal or raw_request["text"]),
            payload={"source_message_id": raw_request["source_message_id"], "success_criteria": criteria},
            status="handled",
            timestamp=created_at,
        )
        return task

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self.store.get_task_workspace(str(task_id or "").strip())

    def list_tasks(
        self,
        *,
        profile_user_id: str,
        session_id: str | None = None,
        statuses: Sequence[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        normalized = [normalize_workspace_status(item) for item in _items(statuses)] if statuses is not None else None
        return self.store.list_task_workspaces(
            profile_user_id=str(profile_user_id or "").strip(),
            session_id=str(session_id or "").strip() if session_id is not None else None,
            statuses=normalized,
            limit=max(1, int(limit or 20)),
        )

    def list_pending_agent_tasks(self, *, after_task_id: str = "", limit: int = 100) -> list[dict[str, Any]]:
        getter = getattr(self.store, "list_pending_agent_workspaces", None)
        if not callable(getter):
            return []
        return getter(after_task_id=str(after_task_id or "").strip(), limit=max(1, int(limit or 100)))

    def update_task(
        self,
        *,
        task_id: str,
        status: str | None = None,
        normalized_goal: str | None = None,
        steps: Sequence[Any] | None = None,
        artifacts: Sequence[Any] | None = None,
        pending_question: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: int | None = None,
        allow_reopen: bool = False,
        allow_cleanup: bool = False,
    ) -> dict[str, Any] | None:
        fields: dict[str, Any] = {"task_id": str(task_id or "").strip(), "updated_at": timestamp}
        if status is not None:
            fields["status"] = normalize_workspace_status(status)
        if normalized_goal is not None:
            fields["normalized_goal"] = str(normalized_goal or "").strip()
        if steps is not None:
            fields["steps"] = normalize_steps(_items(steps))
        if artifacts is not None:
            fields["artifacts"] = normalize_artifacts(_items(artifacts))
        if pending_question is not None:
            fields["pending_question"] = _mapping(pending_question)
        if metadata is not None:
            fields["metadata"] = _mapping(metadata)
        fields["allow_reopen"] = bool(allow_reopen)
        fields["allow_cleanup"] = bool(allow_cleanup)
        return self.store.update_task_workspace(**fields)

    def append_event(
        self,
        *,
        task_id: str,
        event_type: str,
        from_actor: str = "",
        priority: str = "normal",
        requires_user: bool = False,
        message: str = "",
        payload: dict[str, Any] | None = None,
        status: str = "pending",
        timestamp: int | None = None,
    ) -> dict[str, Any]:
        task_key = str(task_id or "").strip()
        task = self.get_task(task_key)
        if not task:
            raise ValueError(f"task not found: {task_key}")
        return self.store.append_task_workspace_event(
            task_id=task_key,
            profile_user_id=str(task.get("profile_user_id") or "").strip(),
            session_id=str(task.get("session_id") or "").strip(),
            event_type=str(event_type or "").strip(),
            from_actor=str(from_actor or "").strip(),
            priority=str(priority or "normal").strip() or "normal",
            requires_user=bool(requires_user),
            message=str(message or "").strip(),
            payload=_mapping(payload),
            status=normalize_event_status(status),
            timestamp=timestamp,
        )

    def list_events(self, *, task_id: str, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return self.store.list_task_workspace_events(
            task_id=str(task_id or "").strip(),
            status=normalize_event_status(status) if status is not None else None,
            limit=max(1, int(limit or 50)),
        )

    def mark_event_handled(self, *, event_id: str, timestamp: int | None = None) -> dict[str, Any] | None:
        return self.store.mark_task_workspace_event_handled(
            event_id=str(event_id or "").strip(),
            status="handled",
            handled_at=timestamp,
        )

    def complete_task(
        self,
        *,
        task_id: str,
        artifacts: Sequence[Any] | None = None,
        message: str = "",
        timestamp: int | None = None,
    ) -> dict[str, Any] | None:
        completed_at = _now(timestamp)
        normalized_artifacts = normalize_artifacts(_items(artifacts)) if artifacts is not None else None
        updated = self.update_task(task_id=task_id, status="completed", artifacts=normalized_artifacts, timestamp=completed_at)
        if updated:
            self.append_event(
                task_id=task_id,
                event_type="task_completed",
                from_actor="system",
                message=message,
                payload={"artifacts": normalized_artifacts or []},
                status="handled",
                timestamp=completed_at,
            )
            updated = self.store.update_task_workspace(
                task_id=str(task_id or "").strip(), completed_at=completed_at, updated_at=completed_at
            ) or updated
        return updated

    def cleanup_task(
        self,
        *,
        task_id: str,
        mode: str = "clean_scratch",
        reason: str = "",
        timestamp: int | None = None,
    ) -> dict[str, Any] | None:
        task_key = str(task_id or "").strip()
        task = self.get_task(task_key)
        if not task:
            return None
        cleaned_at = _now(timestamp)
        metadata = _mapping(task.get("metadata"))
        metadata["cleanup"] = {
            "mode": str(mode or "clean_scratch").strip() or "clean_scratch",
            "reason": str(reason or "").strip(),
            "cleaned_at": cleaned_at,
        }
        updated = self.update_task(
            task_id=task_key,
            status="cleaned",
            metadata=metadata,
            timestamp=cleaned_at,
            allow_cleanup=True,
        )
        if updated:
            self.append_event(
                task_id=task_key,
                event_type="task_cleaned",
                from_actor="frontstage",
                message=reason,
                payload={"mode": metadata["cleanup"]["mode"]},
                status="handled",
                timestamp=cleaned_at,
            )
            updated = self.store.update_task_workspace(
                task_id=task_key, cleaned_at=cleaned_at, updated_at=cleaned_at
            ) or updated
        return updated
