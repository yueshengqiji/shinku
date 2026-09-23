"""本机 QQ Agent 回合的结构化追踪缓冲区。

追踪用于控制中枢排查“这一轮为什么这样回复”，不是聊天记忆，也不是提示词
审计。它只保留有限的元数据，不保存用户原文、system prompt、工具参数或密钥。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import threading
import time
import uuid
from typing import Any

__all__ = ["TurnTraceStore"]


_SENSITIVE_KEYS = frozenset({
    "api_key",
    "arguments",
    "content",
    "cookie",
    "password",
    "prompt",
    "secret",
    "text",
    "token",
})


def _safe_detail(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        return value[:160]
    if isinstance(value, (list, tuple, set)):
        return [_safe_detail(item) for item in list(value)[:30]]
    if isinstance(value, Mapping):
        return {
            str(key): _safe_detail(item)
            for key, item in list(value.items())[:30]
            if str(key).strip().lower() not in _SENSITIVE_KEYS
        }
    return str(value)[:160]


class TurnTraceStore:
    """线程安全的最近回合追踪缓冲区。"""

    def __init__(self, *, max_items: int = 100, clock: Any = time.time) -> None:
        self.max_items = max(1, min(1000, int(max_items or 100)))
        self.clock = clock
        self._items: deque[dict[str, Any]] = deque(maxlen=self.max_items)
        self._lock = threading.RLock()

    def begin(self, turn: Any) -> str:
        trace_id = uuid.uuid4().hex
        now = float(self.clock())
        record = {
            "trace_id": trace_id,
            "started_at": now,
            "finished_at": None,
            "elapsed_ms": None,
            "conversation_type": str(getattr(turn, "conversation_type", "") or "").strip().lower(),
            "conversation_id": str(getattr(turn, "conversation_id", "") or "").strip(),
            "message_id": str(getattr(turn, "primary_message_id", "") or "").strip(),
            "status": "running",
            "agent_status": "",
            "reason": "",
            "sent": False,
            "delivery_status": "not_sent",
            "input_chars": len(str(getattr(turn, "combined_text", "") or "")),
            "stages": [],
        }
        with self._lock:
            self._items.append(record)
        return trace_id

    def stage(self, trace_id: str, name: str, **details: Any) -> None:
        identifier = str(trace_id or "").strip()
        if not identifier:
            return
        with self._lock:
            record = self._find(identifier)
            if record is None:
                return
            elapsed = round(max(0.0, (float(self.clock()) - float(record["started_at"])) * 1000), 1)
            clean = {
                str(key): _safe_detail(value)
                for key, value in details.items()
                if str(key).strip().lower() not in _SENSITIVE_KEYS
            }
            record["stages"].append({"name": str(name or "stage")[:80], "elapsed_ms": elapsed, **clean})

    def finish(self, trace_id: str, reply: Any = None, **details: Any) -> None:
        identifier = str(trace_id or "").strip()
        if not identifier:
            return
        with self._lock:
            record = self._find(identifier)
            if record is None:
                return
            now = float(self.clock())
            record["finished_at"] = now
            record["elapsed_ms"] = round(max(0.0, (now - float(record["started_at"])) * 1000), 1)
            if reply is not None:
                record["agent_status"] = str(getattr(reply, "agent_status", "") or "")
                record["sent"] = bool(getattr(reply, "sent", False))
                record["delivery_status"] = str(getattr(reply, "delivery_status", "not_sent") or "not_sent")
                record["reason"] = str(getattr(reply, "reason", "") or "")[:240]
                record["reply_chars"] = len(str(getattr(reply, "final_text", "") or ""))
                record["status"] = "completed" if record["agent_status"] in {"completed", "command"} else record["agent_status"] or "finished"
            for key, value in details.items():
                if str(key).strip().lower() not in _SENSITIVE_KEYS:
                    record[str(key)] = _safe_detail(value)
            if record["status"] == "running":
                record["status"] = "finished"

    def snapshot(self, *, limit: int = 30) -> list[dict[str, Any]]:
        wanted = max(1, min(self.max_items, int(limit or 30)))
        with self._lock:
            rows = list(self._items)[-wanted:]
            return [self._copy(row) for row in reversed(rows)]

    def _find(self, trace_id: str) -> dict[str, Any] | None:
        for item in reversed(self._items):
            if item.get("trace_id") == trace_id:
                return item
        return None

    @staticmethod
    def _copy(record: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(record)
        result["stages"] = [dict(item) for item in record.get("stages", []) if isinstance(item, Mapping)]
        return result
