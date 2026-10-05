"""轻量后台任务协调器。

它只管理任务生命周期，不替 Agent 偷开工具或扩大权限。任务执行函数由宿主
显式提交，并收到一个 ``threading.Event`` 作为协作式取消信号；任务状态和错误
会保留在内存中，并可选择镜像到 SQLite，供控制中枢和 ``/agent cancel|retry`` 查看。
持久化只保存任务记录，不保存/恢复可执行 handler；重启后未完成任务会标记失败，
避免把可能有外部副作用的工作自动执行第二次。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from typing import Any

from .status import TERMINAL_WORKFLOW_STATUSES, validate_workflow_transition

__all__ = ["AgentTaskCoordinator", "AgentTaskOutcome", "AgentTaskRecord", "AgentTaskContinuation"]


TaskHandler = Callable[[threading.Event], Any]


@dataclass(frozen=True)
class AgentTaskContinuation:
    """Yield the SAME attempt to the executor's queue tail; never persisted code."""
    handler: TaskHandler
    metadata: Mapping[str, Any] = field(default_factory=dict)
    on_cancel: Callable[[], None] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class AgentTaskOutcome:
    """Explicit host outcome; a returned failure is not successful execution."""

    status: str = "completed"
    summary: str = ""
    error: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed", "canceled", "blocked", "waiting_user"}:
            raise ValueError("invalid_task_outcome")


@dataclass
class AgentTaskRecord:
    task_id: str
    goal: str = ""
    profile_user_id: str = ""
    session_id: str = ""
    status: str = "queued"
    attempt: int = 1
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    error: str = ""
    result_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    worker_active: bool = False
    handler: TaskHandler | None = field(default=None, repr=False, compare=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    future: Future[Any] | None = field(default=None, repr=False, compare=False)
    on_cancel: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "profile_user_id": self.profile_user_id,
            "session_id": self.session_id,
            "status": self.status,
            "attempt": self.attempt,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result_summary": self.result_summary,
            "metadata": copy.deepcopy(self.metadata),
            "cancel_requested": self.cancel_requested,
            "worker_active": self.worker_active,
        }


class AgentTaskCoordinator:
    """线程安全、可查询、可取消和可重试的本地后台任务表。"""

    def __init__(
        self,
        *,
        max_workers: int = 2,
        clock: Callable[[], float] | None = None,
        executor: ThreadPoolExecutor | None = None,
        state_path: str | Path | None = None,
        max_persisted_records: int = 500,
        max_pending_tasks: int = 32,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.clock = clock or time.time
        self._executor = executor or ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="shinku-task")
        self._owns_executor = executor is None
        self._records: dict[str, AgentTaskRecord] = {}
        self._lock = threading.RLock()
        self._state_path = Path(state_path).expanduser() if state_path else None
        self._max_persisted_records = max(1, min(5000, int(max_persisted_records or 500)))
        self.max_pending_tasks = max(1, min(200, int(max_pending_tasks)))
        self._closed = False
        if self._state_path is not None:
            self._initialize_store()
            self._load_records()

    def submit(
        self,
        handler: TaskHandler,
        *,
        task_id: str = "",
        goal: str = "",
        profile_user_id: str = "",
        session_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        defer_start: bool = False,
        allow_reopen: bool = False,
        on_cancel: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        if not callable(handler):
            raise TypeError("task handler must be callable")
        identifier = str(task_id or "").strip() or f"task:{uuid.uuid4().hex}"
        now = float(self.clock())
        with self._lock:
            if self._closed:
                raise ValueError("task_processor_closed")
            if sum(item.status == "queued" or self._active(item) for item in self._records.values()) >= self.max_pending_tasks:
                raise ValueError("task_queue_full")
            previous = self._records.get(identifier)
            history: list[dict[str, Any]] = []
            if previous is not None:
                if not allow_reopen:
                    raise ValueError(f"task already exists: {identifier}")
                if (previous.profile_user_id != str(profile_user_id or "").strip()
                        or previous.session_id != str(session_id or "").strip()):
                    raise ValueError("task_scope_mismatch")
                if self._active(previous) or previous.status == "queued":
                    raise ValueError("task_already_active")
                if previous.status == "cleaned":
                    raise ValueError("task_not_reopenable")
                validate_workflow_transition(previous.status, "queued", allow_reopen=True)
                history = [*previous.metadata.get("attempt_history", []), {
                    "attempt": previous.attempt, "status": previous.status,
                    "started_at": previous.started_at, "finished_at": previous.finished_at,
                    "error": previous.error, "result_summary": previous.result_summary,
                }][-16:]
            record = AgentTaskRecord(
                task_id=identifier,
                goal=str(goal or "").strip()[:500],
                profile_user_id=str(profile_user_id or "").strip(),
                session_id=str(session_id or "").strip(),
                attempt=previous.attempt + 1 if previous is not None else 1,
                created_at=previous.created_at if previous is not None else now,
                updated_at=now,
                metadata=copy.deepcopy(dict(metadata or {})) if isinstance(metadata, Mapping) else {},
                handler=handler,
                on_cancel=on_cancel,
            )
            if history:
                record.metadata["attempt_history"] = history
            self._records[identifier] = record
            self._persist_record(record)
            if not defer_start:
                self._dispatch_record(record)
            return record.as_dict()

    def dispatch(self, task_id: str, *, profile_user_id: str = "", session_id: str = "") -> bool:
        """Start an admitted job once, after the foreground handoff is recorded."""
        with self._lock:
            record = self._records.get(str(task_id or "").strip())
            if (self._closed or record is None or not self._visible(record, profile_user_id, session_id)
                    or record.status != "queued" or record.future is not None or record.cancel_requested):
                return False
            self._dispatch_record(record)
            return True

    def _dispatch_record(self, record: AgentTaskRecord) -> None:
        try:
            record.future = self._executor.submit(self._run, record.task_id, record.attempt)
        except Exception:
            self._set_status(record, "failed")
            record.error = "task_dispatch_failed"
            record.finished_at = float(self.clock())
            self._persist_record(record)
            raise

    def get(self, task_id: str, *, profile_user_id: str = "", session_id: str = "") -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(str(task_id or "").strip())
            if record is None or not self._visible(record, profile_user_id, session_id):
                return None
            return record.as_dict()

    def list(self, *, profile_user_id: str = "", session_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        count = max(1, min(200, int(limit or 50)))
        with self._lock:
            records = [record for record in self._records.values() if self._visible(record, profile_user_id, session_id)]
            records.sort(key=lambda item: (item.updated_at, item.task_id), reverse=True)
            return [record.as_dict() for record in records[:count]]

    def cancel(self, task_id: str, *, profile_user_id: str = "", session_id: str = "") -> dict[str, Any]:
        finalize = None
        with self._lock:
            record = self._records.get(str(task_id or "").strip())
            if record is None or not self._visible(record, profile_user_id, session_id):
                return {"ok": False, "reason": "task_not_found"}
            if record.status in TERMINAL_WORKFLOW_STATUSES:
                return {"ok": False, "reason": f"task_already_{record.status}", "task": record.as_dict()}
            record.cancel_requested = True
            record.cancel_event.set()
            future = record.future
            if future is None or future.cancel():
                self._set_status(record, "canceled")
                record.finished_at = float(self.clock())
            elif not self._active(record):
                self._set_status(record, "canceled")
                record.finished_at = float(self.clock())
            if record.status == "canceled" and record.metadata.get("retry_policy") == "fresh_continuation_only":
                record.handler = None
            if record.status == "canceled":
                finalize, record.on_cancel = record.on_cancel, None
            # A running handler remains running + cancel_requested until it
            # actually exits. No false "stopped" receipt or overlapping retry.
            self._persist_record(record)
            response = {"ok": True, "task": record.as_dict()}
        if finalize:
            try:
                finalize()
            except Exception:
                pass  # Cancellation itself must not be undone by diagnostic cleanup.
        return response

    def retry(self, task_id: str, *, profile_user_id: str = "", session_id: str = "") -> dict[str, Any]:
        with self._lock:
            record = self._records.get(str(task_id or "").strip())
            if record is None or not self._visible(record, profile_user_id, session_id):
                return {"ok": False, "reason": "task_not_found"}
            if self._closed or self._active(record):
                return {"ok": False, "reason": "task_still_active", "task": record.as_dict()}
            if record.status not in {"failed", "canceled"}:
                return {"ok": False, "reason": "task_not_retryable", "task": record.as_dict()}
            if record.metadata.get("retry_policy") == "fresh_continuation_only":
                return {"ok": False, "reason": "fresh_continuation_required", "task": record.as_dict()}
            if record.handler is None:
                return {"ok": False, "reason": "retry_handler_missing", "task": record.as_dict()}
            if sum(item.status == "queued" or self._active(item) for item in self._records.values()) >= self.max_pending_tasks:
                return {"ok": False, "reason": "task_queue_full"}
            self._set_status(record, "queued", allow_reopen=True)
            record.attempt += 1
            record.updated_at = float(self.clock())
            record.started_at = None
            record.finished_at = None
            record.error = ""
            record.result_summary = ""
            record.cancel_requested = False
            record.cancel_event = threading.Event()
            self._persist_record(record)
            self._dispatch_record(record)
            return {"ok": True, "task": record.as_dict()}

    def control(self, action: str, task_id: str, profile_user_id: str = "", *, session_id: str = "") -> dict[str, Any]:
        operation = str(action or "").strip().casefold()
        if operation == "cancel":
            return self.cancel(task_id, profile_user_id=profile_user_id, session_id=session_id)
        if operation == "retry":
            return self.retry(task_id, profile_user_id=profile_user_id, session_id=session_id)
        if operation == "status":
            task = self.get(task_id, profile_user_id=profile_user_id, session_id=session_id)
            return {"ok": bool(task), "task": task} if task else {"ok": False, "reason": "task_not_found"}
        if operation == "list":
            return {"ok": True, "tasks": self.list(profile_user_id=profile_user_id, session_id=session_id, limit=10)}
        return {"ok": False, "reason": "unsupported_task_action"}

    def update_progress(self, task_id: str, *, attempt: int, metadata: Mapping[str, Any]) -> bool:
        """Host-owned, content-free progress; stale attempts cannot replace it."""
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.attempt != attempt or not record.worker_active:
                return False
            for key in ("rounds", "tool_calls", "current_tool", "last_stage", "slices"):
                if key in metadata:
                    value = metadata[key]
                    record.metadata[key] = value if isinstance(value, (int, bool)) else str(value or "")[:160]
            if isinstance(metadata.get("budget"), Mapping):
                record.metadata["budget"] = copy.deepcopy(dict(metadata["budget"]))
            record.updated_at = float(self.clock())
            self._persist_record(record)
            return True

    def claim_target(self, task_id: str, *, attempt: int, agent_task_id: str) -> bool:
        """One live attempt owns an Agent ledger; queued guesses grant nothing."""
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.attempt != attempt or not record.worker_active or not agent_task_id:
                return False
            for other in self._records.values():
                if (other.task_id != task_id and other.profile_user_id == record.profile_user_id
                        and other.session_id == record.session_id and (other.worker_active or other.status == "queued")
                        and other.metadata.get("target_claimed") and other.metadata.get("agent_task_id") == agent_task_id):
                    return False
            record.metadata.update({"agent_task_id": agent_task_id, "target_claimed": True})
            self._persist_record(record)
            return True

    def shutdown(self, *, wait: bool = False) -> None:
        with self._lock:
            self._closed = True
            if not wait:
                for record in list(self._records.values()):
                    if record.status == "queued" or self._active(record):
                        self.cancel(record.task_id)
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def storage_status(self) -> dict[str, Any]:
        return {
            "persistent": self._state_path is not None,
            "path": str(self._state_path) if self._state_path is not None else "",
            "records": len(self._records),
            "max_pending_tasks": self.max_pending_tasks,
            "closed": self._closed,
        }

    def _run(self, task_id: str, attempt: int) -> None:
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.attempt != attempt or record.cancel_requested:
                return
            self._set_status(record, "running")
            record.worker_active = True
            record.started_at = record.started_at or float(self.clock())
            handler = record.handler
            cancel_event = record.cancel_event
            self._persist_record(record)
        try:
            result = handler(cancel_event) if handler is not None else None
            with self._lock:
                record = self._records.get(task_id)
                if record is None or record.attempt != attempt:
                    return
                if isinstance(result, AgentTaskContinuation) and not cancel_event.is_set() and not self._closed:
                    record.worker_active = False
                    record.handler = result.handler
                    record.on_cancel = result.on_cancel
                    record.metadata.update(copy.deepcopy(dict(result.metadata)))
                    self._set_status(record, "queued")
                    self._persist_record(record)
                    self._dispatch_record(record)
                    return
                if record.status == "canceled" or cancel_event.is_set():
                    record.cancel_requested = True
                    self._set_status(record, "canceled")
                    record.result_summary = result.summary[:1000] if isinstance(result, AgentTaskOutcome) else ""
                    if isinstance(result, AgentTaskContinuation) and result.on_cancel:
                        result.on_cancel()
                elif isinstance(result, AgentTaskOutcome):
                    self._set_status(record, result.status)
                    record.result_summary = str(result.summary or "")[:1000]
                    record.error = str(result.error or "")[:1000]
                else:
                    self._set_status(record, "completed")
                    record.result_summary = _summary(result)
                if isinstance(result, AgentTaskOutcome):
                    record.metadata.update(copy.deepcopy(dict(result.metadata)))
                record.worker_active = False
                record.on_cancel = None
                if record.metadata.get("retry_policy") == "fresh_continuation_only":
                    record.handler = None
                record.finished_at = float(self.clock())
                record.updated_at = record.finished_at
                self._persist_record(record)
        except Exception as exc:
            with self._lock:
                record = self._records.get(task_id)
                if record is None or record.attempt != attempt:
                    return
                if cancel_event.is_set():
                    self._set_status(record, "canceled")
                elif record.status != "canceled":
                    self._set_status(record, "failed")
                    record.error = f"task_handler_error:{type(exc).__name__}"
                record.worker_active = False
                record.on_cancel = None
                if record.metadata.get("retry_policy") == "fresh_continuation_only":
                    record.handler = None
                record.finished_at = float(self.clock())
                record.updated_at = record.finished_at
                self._persist_record(record)

    def _initialize_store(self) -> None:
        assert self._state_path is not None
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with _database(self._state_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS agent_task_records ("
                "task_id TEXT PRIMARY KEY, updated_at REAL NOT NULL, payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_agent_task_records_updated "
                "ON agent_task_records(updated_at DESC)"
            )

    def _load_records(self) -> None:
        assert self._state_path is not None
        with _database(self._state_path) as connection:
            rows = connection.execute(
                "SELECT payload FROM agent_task_records ORDER BY updated_at DESC",
            ).fetchall()
        now = float(self.clock())
        recovered: dict[str, AgentTaskRecord] = {}
        for (payload,) in rows:
            try:
                value = json.loads(payload)
                if not isinstance(value, dict):
                    continue
                task_id = str(value.get("task_id") or "").strip()
                if not task_id:
                    continue
                status = str(value.get("status") or "failed")
                if status not in TERMINAL_WORKFLOW_STATUSES:
                    status = validate_workflow_transition(status, "failed")
                    value["status"] = status
                    value["updated_at"] = now
                    value["finished_at"] = now
                    value["error"] = "服务重启时任务尚未结束；为避免重复执行，已标记失败且不会自动重跑。"
                    value["metadata"] = {
                        **(value.get("metadata") if isinstance(value.get("metadata"), dict) else {}),
                        "recovered_after_restart": True,
                    }
                recovered[task_id] = AgentTaskRecord(
                    task_id=task_id,
                    goal=str(value.get("goal") or ""),
                    profile_user_id=str(value.get("profile_user_id") or ""),
                    session_id=str(value.get("session_id") or ""),
                    status=status,
                    attempt=max(1, int(value.get("attempt") or 1)),
                    created_at=float(value.get("created_at") or now),
                    updated_at=float(value.get("updated_at") or now),
                    started_at=_optional_float(value.get("started_at")),
                    finished_at=_optional_float(value.get("finished_at")),
                    error=str(value.get("error") or ""),
                    result_summary=str(value.get("result_summary") or ""),
                    metadata=value.get("metadata") if isinstance(value.get("metadata"), dict) else {},
                    cancel_requested=bool(value.get("cancel_requested", False)),
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        with self._lock:
            self._records.update(recovered)
            for record in recovered.values():
                self._persist_record(record)

    def _persist_record(self, record: AgentTaskRecord) -> None:
        # Bound completed history in RAM and on disk, never drop queued,
        # executing, blocked or approval-waiting work merely for being old.
        terminal = sorted((item for item in self._records.values()
                           if item.status in TERMINAL_WORKFLOW_STATUSES and not self._active(item)),
                          key=lambda item: (item.updated_at, item.task_id), reverse=True)
        pruned = [item.task_id for item in terminal[self._max_persisted_records:]]
        for task_id in pruned:
            self._records.pop(task_id, None)
        if self._state_path is None:
            return
        payload = json.dumps(record.as_dict(), ensure_ascii=False, separators=(",", ":"), default=str)
        with _database(self._state_path) as connection:
            if record.task_id in self._records:
                connection.execute(
                    "INSERT INTO agent_task_records(task_id, updated_at, payload) VALUES(?, ?, ?) "
                    "ON CONFLICT(task_id) DO UPDATE SET updated_at=excluded.updated_at, payload=excluded.payload",
                    (record.task_id, record.updated_at, payload),
                )
            if pruned:
                connection.executemany("DELETE FROM agent_task_records WHERE task_id = ?", ((key,) for key in pruned))

    def _visible(self, record: AgentTaskRecord, profile_user_id: str, session_id: str = "") -> bool:
        owner = str(profile_user_id or "").strip()
        session = str(session_id or "").strip()
        return (not owner or record.profile_user_id == owner) and (not session or record.session_id == session)

    @staticmethod
    def _active(record: AgentTaskRecord) -> bool:
        return record.worker_active or (
            record.status == "queued" and record.future is not None and not record.future.done()
        )

    def _set_status(self, record: AgentTaskRecord, target: str, *, allow_reopen: bool = False) -> None:
        record.status = validate_workflow_transition(record.status, target, allow_reopen=allow_reopen)
        record.updated_at = float(self.clock())


def _summary(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return str(value.get("summary") or value.get("status") or "任务完成")[:1000]
    return str(value)[:1000]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@contextmanager
def _database(path: Path):
    connection = sqlite3.connect(path, timeout=10.0)
    try:
        with connection:
            yield connection
    finally:
        connection.close()
