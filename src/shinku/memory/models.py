"""Data contracts for Shinku's independent memory pipeline.

The memory layer deliberately keeps records separate from prompts.  A record
is historical evidence; only ``MemoryService`` may decide whether it is useful
for the current turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

HISTORY_MODES = ("none", "recent", "retrieve", "summary")

__all__ = ["HISTORY_MODES", "MemoryDecision", "MemoryRecord", "MemoryContext", "ForgetReport"]


@dataclass(frozen=True)
class MemoryDecision:
    """One turn's recall decision."""

    need_retrieval: bool
    memory_type: str = "none"
    scope: str = "conversation"
    query: str = ""
    time_range: str = ""
    confidence: float = 0.0
    reason: str = ""
    used_model: bool = False
    # ``need_retrieval`` remains for compatibility with the first contract.
    # New routing code uses the four-level history mode instead.
    history_mode: str = ""
    local_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mode = str(self.history_mode or "").strip().lower()
        if mode not in HISTORY_MODES:
            mode = "retrieve" if self.need_retrieval else "none"
        object.__setattr__(self, "history_mode", mode)
        if mode in {"retrieve", "summary"} and not self.need_retrieval:
            object.__setattr__(self, "need_retrieval", True)

    @property
    def needs_history(self) -> bool:
        return self.history_mode != "none"

    def as_dict(self) -> dict[str, Any]:
        return {
            "need_retrieval": self.need_retrieval,
            "memory_type": self.memory_type,
            "scope": self.scope,
            "query": self.query,
            "time_range": self.time_range,
            "confidence": self.confidence,
            "reason": self.reason,
            "used_model": self.used_model,
            "history_mode": self.history_mode,
            "needs_history": self.needs_history,
            "local_signals": list(self.local_signals),
        }


@dataclass(frozen=True)
class MemoryRecord:
    """A retrievable historical record with provenance."""

    record_id: str
    layer: str
    scope: str
    text: str
    created_at: int
    score: float = 0.0
    importance: float = 0.0
    confidence: float = 0.0
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "layer": self.layer,
            "scope": self.scope,
            "text": self.text,
            "created_at": self.created_at,
            "score": self.score,
            "importance": self.importance,
            "confidence": self.confidence,
            "source_id": self.source_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryContext:
    """The only memory payload allowed to enter one model turn."""

    decision: MemoryDecision
    records: tuple[MemoryRecord, ...] = ()
    rendered: str = ""
    retrieved_count: int = 0
    injected_chars: int = 0
    audit: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.as_dict(),
            "records": [record.as_dict() for record in self.records],
            "rendered": self.rendered,
            "retrieved_count": self.retrieved_count,
            "injected_chars": self.injected_chars,
            "audit": dict(self.audit),
        }


@dataclass(frozen=True)
class ForgetReport:
    """Result of an explicit forget operation."""

    query: str
    scope: str
    forgotten_count: int = 0
    purged_count: int = 0
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "scope": self.scope,
            "forgotten_count": self.forgotten_count,
            "purged_count": self.purged_count,
            "reason": self.reason,
        }
