"""Bounded, content-free diagnostics for QQ attention decisions."""

from __future__ import annotations

from collections import Counter, deque
import threading
import uuid
from typing import Any, Mapping

__all__ = ["AttentionTraceStore"]


class AttentionTraceStore:
    """Keep a short in-memory window of ingress and attention outcomes.

    Message text is deliberately never accepted by this API. The trace stores
    only identifiers, lengths, routing reasons, and compact topic-state facts.
    """

    def __init__(self, *, max_items: int = 500) -> None:
        self.max_items = max(1, min(5000, int(max_items or 500)))
        self._items: deque[dict[str, Any]] = deque(maxlen=self.max_items)
        self._lock = threading.RLock()

    def record(
        self,
        *,
        observed_at: float,
        conversation_id: str,
        conversation_type: str,
        message_id: str,
        sender_id: str,
        text_chars: int,
        has_media: bool,
        is_poke: bool,
        admission: Mapping[str, Any],
        route: Mapping[str, Any],
        attention: Mapping[str, Any],
        queued: bool,
        closed_batch_size: int = 0,
        closed_batch_primary_message_id: str = "",
        topic_state: Mapping[str, Any] | None = None,
    ) -> str:
        trace_id = uuid.uuid4().hex
        record = {
            "trace_id": trace_id,
            "observed_at": float(observed_at),
            "conversation_id": str(conversation_id or "")[:160],
            "conversation_type": str(conversation_type or "").strip().lower()[:24],
            "message_id": str(message_id or "")[:160],
            "sender_id": str(sender_id or "")[:160],
            "text_chars": max(0, int(text_chars or 0)),
            "has_media": bool(has_media),
            "is_poke": bool(is_poke),
            "admission": dict(admission),
            "route": dict(route),
            "attention": dict(attention),
            "queued": bool(queued),
            "closed_batch_size": max(0, int(closed_batch_size or 0)),
            "closed_batch_primary_message_id": str(closed_batch_primary_message_id or "")[:160],
            "topic_state": dict(topic_state or {}),
        }
        with self._lock:
            self._items.append(record)
        return trace_id

    def snapshot(self, *, limit: int = 80) -> dict[str, Any]:
        wanted = max(1, min(self.max_items, int(limit or 80)))
        with self._lock:
            rows = [dict(item) for item in list(self._items)[-wanted:]][::-1]
        admission_counts: Counter[str] = Counter()
        route_counts: Counter[str] = Counter()
        attention_counts: Counter[str] = Counter()
        scheduled = 0
        admitted = 0
        for item in rows:
            admission = item.get("admission") or {}
            route = item.get("route") or {}
            attention = item.get("attention") or {}
            admission_counts[str(admission.get("reason") or "unknown")] += 1
            route_counts[str(route.get("reason") or "unknown")] += 1
            attention_counts[str(attention.get("reason") or "unknown")] += 1
            admitted += int(bool(admission.get("accepted")))
            scheduled += int(bool(attention.get("should_schedule")))
        return {
            "summary": {
                "window_items": len(rows),
                "admitted": admitted,
                "filtered": len(rows) - admitted,
                "scheduled": scheduled,
                "not_scheduled": len(rows) - scheduled,
                "admission_reasons": dict(admission_counts),
                "route_reasons": dict(route_counts),
                "attention_reasons": dict(attention_counts),
            },
            "items": rows,
        }
