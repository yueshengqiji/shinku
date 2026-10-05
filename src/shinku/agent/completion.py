"""Typed completion receipts: ending a turn does not imply goal success."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CompletionVerdict:
    accepted: bool
    status: str = "completed"
    reason: str = ""
    satisfied: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.accepted and self.status == "completed":
            object.__setattr__(self, "status", "unverified")

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "reason": self.reason,
            "satisfied": list(self.satisfied),
            "missing": list(self.missing),
            "evidence": list(self.evidence),
        }
