"""群聊说话人标签与短期话题交接。

群聊的旁观消息仍然可以被动入库，但只有当前消息已经叫到真红或明确回复
某条消息时，才把一小段本地话题快照交给 Agent。这里不调用模型，也不把
旁观者原话当成人设或长期事实。
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .message import IncomingMessage
from .routing import RouteDecision

__all__ = ["GroupContextTracker"]


_REFERENT_RE = re.compile(r"(?:他|她|它|那个|这个|刚才|前面|上面|继续|这件事|那件事|谁|哪句|哪一条)")


@dataclass(frozen=True)
class GroupContextItem:
    sender_id: str
    sender_label: str
    message_id: str
    text: str
    timestamp: float

    def render(self) -> str:
        label = self.sender_label or self.sender_id or "群成员"
        return f"{label}（{self.sender_id or 'unknown'}）：{self.text}"


class GroupContextTracker:
    """每个群保留一个有 TTL、有上限的本地话题快照。"""

    def __init__(self, *, ttl_seconds: float = 90.0, max_items: int = 8, max_chars: int = 1600) -> None:
        if ttl_seconds <= 0 or max_items <= 0 or max_chars <= 0:
            raise ValueError("group context limits must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.max_items = int(max_items)
        self.max_chars = int(max_chars)
        self._buckets: dict[str, list[GroupContextItem]] = {}
        self._lock = threading.RLock()

    def enrich(self, message: IncomingMessage, route: RouteDecision, *, now: float) -> None:
        """把当前消息的前置话题快照写回 raw event，再保存当前消息。"""

        if message.conversation_type != "group" or message.is_poke:
            return
        current = float(now)
        key = str(message.conversation_id or "").strip()
        if not key:
            return
        with self._lock:
            bucket = self._prune(key, current)
            context = self._select_context(message, route, bucket)
            if isinstance(message.raw_event, dict):
                message.raw_event["_shinku_group_context"] = {
                    "items": [
                        {
                            "sender_id": item.sender_id,
                            "sender_label": item.sender_label,
                            "message_id": item.message_id,
                            "text": item.text,
                            "timestamp": item.timestamp,
                        }
                        for item in context
                    ],
                    "reason": self._reason(message, route),
                }
            text = self._message_text(message)
            if text:
                bucket.append(
                    GroupContextItem(
                        sender_id=str(message.sender_id or "").strip(),
                        sender_label=str(message.sender_label or "").strip(),
                        message_id=str(message.message_id or "").strip(),
                        text=text,
                        timestamp=current,
                    )
                )
                self._trim(bucket)
            self._buckets[key] = bucket

    def snapshot(self, conversation_id: str, *, limit: int = 8, now: float | None = None) -> list[dict[str, Any]]:
        current = float(now if now is not None else 0.0)
        key = str(conversation_id or "").strip()
        with self._lock:
            bucket = self._prune(key, current) if current else list(self._buckets.get(key, []))
            return [self._as_dict(item) for item in bucket[-max(1, int(limit or 8)) :]]

    def _select_context(
        self,
        message: IncomingMessage,
        route: RouteDecision,
        bucket: list[GroupContextItem],
    ) -> list[GroupContextItem]:
        if not route.should_reply and not message.has_reply:
            return []
        reply_ids = set(message.reply_message_ids)
        if reply_ids:
            exact = [item for item in bucket if item.message_id and item.message_id in reply_ids]
            if exact:
                return exact[-3:]
        if route.addressed or _REFERENT_RE.search(message.text or ""):
            return bucket[-3:]
        return []

    def _reason(self, message: IncomingMessage, route: RouteDecision) -> str:
        if message.reply_message_ids:
            return "reply_chain"
        if route.addressed:
            return "addressed_group_turn"
        if _REFERENT_RE.search(message.text or ""):
            return "local_referent"
        return ""

    def _prune(self, key: str, now: float) -> list[GroupContextItem]:
        cutoff = now - self.ttl_seconds
        bucket = [item for item in self._buckets.get(key, []) if item.timestamp >= cutoff]
        self._buckets[key] = bucket
        return bucket

    def _trim(self, bucket: list[GroupContextItem]) -> None:
        del bucket[:-self.max_items]
        while len("\n".join(item.render() for item in bucket)) > self.max_chars and len(bucket) > 1:
            bucket.pop(0)

    @staticmethod
    def _message_text(message: IncomingMessage) -> str:
        text = str(message.text or "").strip()
        if message.has_media:
            text = f"{text}\n[附件]".strip()
        return text[:600]

    @staticmethod
    def _as_dict(item: GroupContextItem) -> dict[str, Any]:
        return {
            "sender_id": item.sender_id,
            "sender_label": item.sender_label,
            "message_id": item.message_id,
            "text": item.text,
            "timestamp": item.timestamp,
            "rendered": item.render(),
        }

