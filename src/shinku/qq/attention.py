"""模型调用前的注意力门与短窗口消息合并。"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .message import IncomingMessage
from .routing import RouteDecision

__all__ = ["AttentionBatch", "AttentionBatcher", "AttentionDecision", "AttentionPolicy"]


@dataclass(frozen=True)
class AttentionDecision:
    should_schedule: bool
    immediate: bool
    score: int
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_schedule": self.should_schedule,
            "immediate": self.immediate,
            "score": self.score,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AttentionPolicy:
    """不调用模型的注意力策略。"""

    autonomous_enabled: bool = False
    autonomous_probability: float = 0.0
    random_source: Callable[[], float] = random.random

    def evaluate(self, message: IncomingMessage, route: RouteDecision) -> AttentionDecision:
        if route.should_reply:
            return AttentionDecision(True, True, route.priority, route.reason)
        if not self.autonomous_enabled:
            return AttentionDecision(False, False, 0, "autonomous_reply_disabled")
        if not message.text.strip() and not message.has_media:
            return AttentionDecision(False, False, 0, "empty_message")
        probability = max(0.0, min(1.0, float(self.autonomous_probability)))
        try:
            draw = float(self.random_source())
        except Exception:
            draw = 1.0
        if draw < probability:
            return AttentionDecision(True, False, 20, "autonomous_probability_hit")
        return AttentionDecision(False, False, 0, "autonomous_probability_miss")


@dataclass(frozen=True)
class AttentionBatch:
    conversation_id: str
    messages: tuple[IncomingMessage, ...]
    decisions: tuple[AttentionDecision, ...]
    primary_index: int

    @property
    def primary_message(self) -> IncomingMessage:
        return self.messages[self.primary_index]

    @property
    def combined_text(self) -> str:
        return "\n".join(message.text.strip() for message in self.messages if message.text.strip())

    @property
    def reply_to_message_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.reason
            for decision in self.decisions
            if decision.reason == "reply_to_bot"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "message_ids": [message.message_id for message in self.messages],
            "primary_message_id": self.primary_message.message_id,
            "combined_text": self.combined_text,
            "decisions": [decision.as_dict() for decision in self.decisions],
        }


class AttentionBatcher:
    """把同一会话短时间内的已调度消息合并成一轮。"""

    def __init__(self, *, window_seconds: float = 1.5) -> None:
        self.window_seconds = max(0.0, float(window_seconds))
        self._conversation_id = ""
        self._messages: list[IncomingMessage] = []
        self._decisions: list[AttentionDecision] = []
        self._last_at: float | None = None

    def push(
        self,
        message: IncomingMessage,
        decision: AttentionDecision,
        *,
        at: float,
    ) -> AttentionBatch | None:
        timestamp = float(at)
        if not decision.should_schedule:
            return None
        closed: AttentionBatch | None = None
        if self._messages and (
            message.conversation_id != self._conversation_id
            or self._last_at is None
            or timestamp - self._last_at > self.window_seconds
        ):
            closed = self.flush()
        if not self._messages:
            self._conversation_id = message.conversation_id
        self._messages.append(message)
        self._decisions.append(decision)
        self._last_at = timestamp
        return closed

    def flush(self) -> AttentionBatch | None:
        if not self._messages:
            return None
        primary_index = max(
            range(len(self._decisions)),
            key=lambda index: (self._decisions[index].score, -index),
        )
        result = AttentionBatch(
            conversation_id=self._conversation_id,
            messages=tuple(self._messages),
            decisions=tuple(self._decisions),
            primary_index=primary_index,
        )
        self._conversation_id = ""
        self._messages.clear()
        self._decisions.clear()
        self._last_at = None
        return result
