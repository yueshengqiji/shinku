"""待授权任务的可替换存储。

默认实现只保存在当前进程内；需要跨进程重启时，调用方可以显式注入 JSON
存储。相同会话中的多条记录按 FIFO 排队，避免后来的待授权调用覆盖先来的调用。
存储层只负责保存和恢复待授权记录，不参与授权判断，也不执行工具。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .approval import ToolApprovalRequest
from shinku.tools.invocation import ToolInvocation

__all__ = [
    "InMemoryPendingApprovalStore",
    "JsonPendingApprovalStore",
    "PendingApprovalRecord",
    "PendingApprovalStore",
]


@dataclass(frozen=True)
class PendingApprovalRecord:
    """一条可恢复的待授权记录。"""

    request: ToolApprovalRequest
    task_id: str
    context: Mapping[str, Any]
    transcript: tuple[dict[str, Any], ...]
    created_at: float


class PendingApprovalStore(Protocol):
    def get(self, conversation_id: str) -> PendingApprovalRecord | None: ...

    def put(self, conversation_id: str, record: PendingApprovalRecord) -> None: ...

    def delete(self, conversation_id: str) -> None: ...


class InMemoryPendingApprovalStore:
    """默认的会话级 FIFO 存储；进程退出后不会恢复待授权动作。"""

    def __init__(self) -> None:
        self._records: dict[str, list[PendingApprovalRecord]] = {}

    def get(self, conversation_id: str) -> PendingApprovalRecord | None:
        records = self._records.get(str(conversation_id or "").strip(), [])
        return records[0] if records else None

    def put(self, conversation_id: str, record: PendingApprovalRecord) -> None:
        key = str(conversation_id or "").strip()
        if key:
            self._records.setdefault(key, []).append(record)

    def delete(self, conversation_id: str) -> None:
        key = str(conversation_id or "").strip()
        records = self._records.get(key, [])
        if records:
            records.pop(0)
        if not records:
            self._records.pop(key, None)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _invocation_to_dict(invocation: ToolInvocation) -> dict[str, Any]:
    return {
        "name": invocation.name,
        "arguments": _json_safe(dict(invocation.arguments or {})),
        "source": invocation.source,
        "id": invocation.id,
    }


def _invocation_from_dict(value: Any) -> ToolInvocation | None:
    if not isinstance(value, Mapping):
        return None
    name = str(value.get("name") or "").strip()
    arguments = value.get("arguments")
    if not name or not isinstance(arguments, Mapping):
        return None
    return ToolInvocation(
        name=name,
        arguments=dict(arguments),
        source=str(value.get("source") or "legacy_json"),
        id=str(value.get("id") or ""),
    )


class JsonPendingApprovalStore:
    """显式启用的本地 JSON 存储。

    这是按会话 FIFO 排列的短期恢复队列，不是长期记忆库：只保存尚未获批的具体调用和恢复所需的短上下文。路径由
    使用者配置时应按本机敏感数据处理，因为工具参数可能包含用户主动提交的
    文件路径或凭据引用。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._records: dict[str, list[PendingApprovalRecord]] = {}
        self._load()

    def get(self, conversation_id: str) -> PendingApprovalRecord | None:
        records = self._records.get(str(conversation_id or "").strip(), [])
        return records[0] if records else None

    def put(self, conversation_id: str, record: PendingApprovalRecord) -> None:
        key = str(conversation_id or "").strip()
        if not key:
            return
        self._records.setdefault(key, []).append(record)
        self._flush()

    def delete(self, conversation_id: str) -> None:
        key = str(conversation_id or "").strip()
        records = self._records.get(key, [])
        if records:
            records.pop(0)
        if not records:
            self._records.pop(key, None)
        self._flush()

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            return
        if not isinstance(payload, Mapping):
            return
        for conversation_id, raw in payload.items():
            values = raw if isinstance(raw, list) else [raw]
            records = [
                record
                for item in values
                for record in (self._record_from_dict(item),)
                if record is not None
            ]
            if records:
                self._records[str(conversation_id)] = records

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: [self._record_to_dict(record) for record in records]
            for key, records in self._records.items()
        }
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    @staticmethod
    def _record_to_dict(record: PendingApprovalRecord) -> dict[str, Any]:
        return {
            "request_id": record.request.request_id,
            "task_id": record.request.task_id,
            "question": record.request.question,
            "created_at": record.created_at,
            "risk_by_tool": dict(record.request.risk_by_tool),
            "invocations": [_invocation_to_dict(item) for item in record.request.invocations],
            "context": _json_safe({key: value for key, value in record.context.items() if key != "qq_turn"}),
            "transcript": _json_safe(list(record.transcript)),
        }

    @staticmethod
    def _record_from_dict(value: Any) -> PendingApprovalRecord | None:
        if not isinstance(value, Mapping):
            return None
        calls = tuple(
            item
            for raw in (value.get("invocations") or ())
            for item in (_invocation_from_dict(raw),)
            if item is not None
        )
        if not calls:
            return None
        try:
            created_at = float(value.get("created_at", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        request = ToolApprovalRequest(
            request_id=str(value.get("request_id") or ""),
            task_id=str(value.get("task_id") or ""),
            invocation=calls[0],
            question=str(value.get("question") or ""),
            created_at=created_at,
            invocations=calls,
            risk_by_tool=value.get("risk_by_tool") if isinstance(value.get("risk_by_tool"), Mapping) else {},
        )
        context = value.get("context") if isinstance(value.get("context"), Mapping) else {}
        transcript = value.get("transcript") if isinstance(value.get("transcript"), list) else []
        return PendingApprovalRecord(
            request=request,
            task_id=str(value.get("task_id") or ""),
            context=dict(context),
            transcript=tuple(dict(item) for item in transcript if isinstance(item, Mapping)),
            created_at=created_at,
        )
