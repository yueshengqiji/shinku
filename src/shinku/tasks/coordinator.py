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
from dataclasses import dataclass, field
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from typing import Any

from .status import TERMINAL_WORKFLOW_STATUSES, validate_workflow_transition

__all__ = ["AgentTaskCoordinator", "AgentTaskRecord"]


TaskHandler = Callable[[threading.Event], Any]


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
    handler: TaskHandler | None = field(default=None, repr=False, compare=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    future: Future[Any] | None = field(default=None, repr=False, compare=False)

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
            "metadata": dict(self.metadata),
            "cancel_requested": self.cancel_requested,
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
    ) -> dict[str, Any]:
        if not callable(handler):
            raise TypeError("task handler must be callable")
        identifier = str(task_id or "").strip() or f"task:{uuid.uuid4().hex}"
        now = float(self.clock())
        with self._lock:
            if identifier in self._records:
                raise ValueError(f"task already exists: {identifier}")
            record = AgentTaskRecord(
                task_id=identifier,
                goal=str(goal or "").strip()[:500],
                profile_user_id=str(profile_user_id or "").strip(),
                session_id=str(session_id or "").strip(),
                created_at=now,
                updated_at=now,
                metadata=dict(metadata or {}) if isinstance(metadata, Mapping) else {},
                handler=handler,
            )
            self._records[identifier] = record
            self._persist_record(record)
            record.future = self._executor.submit(self._run, identifier, record.attempt)
            return record.as_dict()

    def get(self, task_id: str, *, profile_user_id: str = "") -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(str(task_id or "").strip())
            if record is None or not self._visible(record, profile_user_id):
                return None
            return record.as_dict()

    def list(self, *, profile_user_id: str = "", limit: int = 50) -> list[dict[str, Any]]:
        count = max(1, min(200, int(limit or 50)))
        with self._lock:
            records = [record for record in self._records.values() if self._visible(record, profile_user_id)]
            records.sort(key=lambda item: (item.updated_at, item.task_id), reverse=True)
            return [record.as_dict() for record in records[:count]]

    def cancel(self, task_id: str, *, profile_user_id: str = "") -> dict[str, Any]:
        with self._lock:
            record = self._records.get(str(task_id or "").strip())
            if record is None or not self._visible(record, profile_user_id):
                return {"ok": False, "reason": "task_not_found"}
            if record.status in TERMINAL_WORKFLOW_STATUSES:
                return {"ok": False, "reason": f"task_already_{record.status}", "task": record.as_dict()}
            record.cancel_requested = True
            record.cancel_event.set()
            future = record.future
            if future is not None and future.cancel():
                self._set_status(record, "canceled")
                record.finished_at = float(self.clock())
            else:
                # Running handlers are cooperative.  Marking the request now
                # makes the control plane truthful even if the handler has not
                # reached its next cancellation checkpoint yet.
                self._set_status(record, "canceled")
            self._persist_record(record)
            return {"ok": True, "task": record.as_dict()}

    def retry(self, task_id: str, *, profile_user_id: str = "") -> dict[str, Any]:
        with self._lock:
            record = self._records.get(str(task_id or "").strip())
            if record is None or not self._visible(record, profile_user_id):
                return {"ok": False, "reason": "task_not_found"}
            if record.status not in {"failed", "canceled"}:
                return {"ok": False, "reason": "task_not_retryable", "task": record.as_dict()}
            if record.handler is None:
                return {"ok": False, "reason": "retry_handler_missing", "task": record.as_dict()}
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
            record.future = self._executor.submit(self._run, record.task_id, record.attempt)
            return {"ok": True, "task": record.as_dict()}

    def control(self, action: str, task_id: str, profile_user_id: str = "") -> dict[str, Any]:
        operation = str(action or "").strip().casefold()
        if operation == "cancel":
            return self.cancel(task_id, profile_user_id=profile_user_id)
        if operation == "retry":
            return self.retry(task_id, profile_user_id=profile_user_id)
        return {"ok": False, "reason": "unsupported_task_action"}

    def shutdown(self, *, wait: bool = False) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def storage_status(self) -> dict[str, Any]:
        return {
            "persistent": self._state_path is not None,
            "path": str(self._state_path) if self._state_path is not None else "",
            "records": len(self._records),
        }

    def _run(self, task_id: str, attempt: int) -> None:
        with self._lock:
            record = self._records.get(task_id)
            if record is None or record.attempt != attempt or record.cancel_requested:
                return
            self._set_status(record, "running")
            record.started_at = float(self.clock())
            handler = record.handler
            cancel_event = record.cancel_event
            self._persist_record(record)
        try:
            result = handler(cancel_event) if handler is not None else None
            with self._lock:
                record = self._records.get(task_id)
                if record is None or record.attempt != attempt:
                    return
                if record.status == "canceled" or cancel_event.is_set():
                    record.cancel_requested = True
                else:
                    self._set_status(record, "completed")
                    record.result_summary = _summary(result)
                record.finished_at = float(self.clock())
                record.updated_at = record.finished_at
                self._persist_record(record)
        except Exception as exc:
            with self._lock:
                record = self._records.get(task_id)
                if record is None or record.attempt != attempt:
                    return
                if record.status != "canceled":
                    self._set_status(record, "failed")
                    record.error = f"{type(exc).__name__}: {exc}"[:1000]
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
                "SELECT payload FROM agent_task_records ORDER BY updated_at DESC LIMIT ?",
                (self._max_persisted_records,),
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
        if self._state_path is None:
            return
        payload = json.dumps(record.as_dict(), ensure_ascii=False, separators=(",", ":"), default=str)
        with _database(self._state_path) as connection:
            connection.execute(
                "INSERT INTO agent_task_records(task_id, updated_at, payload) VALUES(?, ?, ?) "
                "ON CONFLICT(task_id) DO UPDATE SET updated_at=excluded.updated_at, payload=excluded.payload",
                (record.task_id, record.updated_at, payload),
            )
            connection.execute(
                "DELETE FROM agent_task_records WHERE task_id IN ("
                "SELECT task_id FROM agent_task_records ORDER BY updated_at DESC LIMIT -1 OFFSET ?) ",
                (self._max_persisted_records,),
            )

    def _visible(self, record: AgentTaskRecord, profile_user_id: str) -> bool:
        owner = str(profile_user_id or "").strip()
        return not owner or not record.profile_user_id or record.profile_user_id == owner

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
