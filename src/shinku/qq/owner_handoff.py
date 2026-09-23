"""Short-lived, in-memory group-topic handoff for private owner greetings.

This buffer is deliberately separate from persistent memory. It is populated
only after a configured owner speaks in a group, bounded by time and size, and
consumed only by that owner in a private conversation.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from .message import IncomingMessage

__all__ = ["OwnerGroupHandoff", "OwnerHandoffLine"]


@dataclass(frozen=True)
class OwnerHandoffLine:
    speaker: str
    text: str
    timestamp: float


@dataclass
class _OwnerTopic:
    group_id: str
    owner_last_at: float
    last_at: float
    lines: list[OwnerHandoffLine] = field(default_factory=list)
    consumed: bool = False


class OwnerGroupHandoff:
    """Keep only a fresh group exchange that was anchored by the owner."""

    def __init__(
        self,
        *,
        owner_ids: Sequence[str] = (),
        bot_ids: Sequence[str] = (),
        ttl_seconds: float = 20 * 60,
        continuation_seconds: float = 10 * 60,
        max_items: int = 6,
        max_chars: int = 1200,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.owner_ids = frozenset(str(item or "").strip() for item in owner_ids if str(item or "").strip())
        self.bot_ids = frozenset(str(item or "").strip() for item in bot_ids if str(item or "").strip())
        self.ttl_seconds = max(30.0, min(2 * 60 * 60, float(ttl_seconds)))
        self.continuation_seconds = max(30.0, min(self.ttl_seconds, float(continuation_seconds)))
        self.max_items = max(2, min(12, int(max_items)))
        self.max_chars = max(200, min(4000, int(max_chars)))
        self._clock = clock or time.time
        self._topics: dict[str, _OwnerTopic] = {}
        self._lock = threading.RLock()

    def observe(self, message: IncomingMessage, *, now: float | None = None) -> bool:
        if not isinstance(message, IncomingMessage) or message.conversation_type != "group" or message.is_poke:
            return False
        group_id = str(message.conversation_id or "").strip()
        sender_id = str(message.sender_id or "").strip()
        if not group_id or not sender_id or sender_id in self.bot_ids:
            return False
        text = str(message.text or "").strip()
        if message.has_media:
            text = f"{text} [附有媒体]".strip()
        text = text[:360]
        if not text:
            return False

        stamp = float(self._clock() if now is None else now)
        with self._lock:
            self._prune(stamp)
            if sender_id in self.owner_ids:
                previous = self._topics.get(sender_id)
                if (
                    previous is not None
                    and not previous.consumed
                    and previous.group_id == group_id
                    and stamp - previous.last_at <= self.continuation_seconds
                    and stamp - previous.owner_last_at <= self.ttl_seconds
                ):
                    topic = previous
                    topic.owner_last_at = stamp
                else:
                    topic = _OwnerTopic(group_id=group_id, owner_last_at=stamp, last_at=stamp)
                    self._topics[sender_id] = topic
                speaker = "你"
            else:
                owner_id = self._owner_for_group(group_id, stamp)
                if not owner_id:
                    return False
                topic = self._topics[owner_id]
                speaker = str(message.sender_label or "群成员").strip()[:40] or "群成员"
            topic.lines.append(OwnerHandoffLine(speaker=speaker, text=text, timestamp=stamp))
            topic.last_at = stamp
            self._trim(topic)
            return True

    def consume(self, owner_id: str, *, now: float | None = None) -> tuple[OwnerHandoffLine, ...]:
        owner = str(owner_id or "").strip()
        if owner not in self.owner_ids:
            return ()
        stamp = float(self._clock() if now is None else now)
        with self._lock:
            self._prune(stamp)
            topic = self._topics.get(owner)
            if topic is None or topic.consumed or not topic.lines:
                return ()
            topic.consumed = True
            lines = tuple(topic.lines)
            self._topics.pop(owner, None)
            return lines

    def diagnostics(self) -> dict[str, Any]:
        stamp = float(self._clock())
        with self._lock:
            self._prune(stamp)
            return {
                "active_owner_topics": sum(1 for item in self._topics.values() if not item.consumed),
                "ttl_seconds": self.ttl_seconds,
                "max_items": self.max_items,
            }

    def _owner_for_group(self, group_id: str, now: float) -> str:
        for owner_id, topic in self._topics.items():
            if (
                topic.group_id == group_id
                and not topic.consumed
                and now - topic.last_at <= self.continuation_seconds
                and now - topic.owner_last_at <= self.ttl_seconds
            ):
                return owner_id
        return ""

    def _prune(self, now: float) -> None:
        expired = [
            owner_id
            for owner_id, topic in self._topics.items()
            if now - topic.owner_last_at > self.ttl_seconds
        ]
        for owner_id in expired:
            self._topics.pop(owner_id, None)

    def _trim(self, topic: _OwnerTopic) -> None:
        del topic.lines[:-self.max_items]
        while (
            sum(len(item.speaker) + len(item.text) for item in topic.lines) > self.max_chars
            and len(topic.lines) > 1
        ):
            topic.lines.pop(0)
