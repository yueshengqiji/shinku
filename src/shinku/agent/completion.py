"""Typed completion receipts: ending a turn does not imply goal success."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompletionVerdict:
    accepted: bool
    status: str = "completed"
    reason: str = ""
    satisfied: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    advisory: tuple[dict[str, Any], ...] = ()
    layers: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.accepted and self.status == "completed":
            object.__setattr__(self, "status", "unverified")

    def as_dict(self) -> dict[str, Any]:
        result = {
            "accepted": self.accepted,
            "status": self.status,
            "reason": self.reason,
            "satisfied": list(self.satisfied),
            "missing": list(self.missing),
            "evidence": list(self.evidence),
        }
        # 仅分层任务附加字段，兼容既有严格任务与旧账本的导出协议。
        if self.layers:
            result.update(layers=self.layers, advisory=list(self.advisory))
        return result
