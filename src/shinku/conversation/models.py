"""Host-owned lifecycle and evidence metadata; no executable authority."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field


def text_digest(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConversationChannel:
    conversation_type: str
    conversation_id: str

    def __post_init__(self) -> None:
        if self.conversation_type not in {"private", "group", "desktop"} or not self.conversation_id.strip():
            raise ValueError("conversation_channel_missing")

    @property
    def key(self) -> str:
        return json.dumps([self.conversation_type, self.conversation_id], separators=(",", ":"))


@dataclass(frozen=True)
class ConversationScope:
    conversation_type: str
    conversation_id: str
    requester: str

    def __post_init__(self) -> None:
        _ = self.channel  # Validate the shared channel boundary too.
        if not self.requester.strip() or self.requester == "unknown":
            raise ValueError("conversation_requester_missing")

    @property
    def channel(self) -> ConversationChannel:
        return ConversationChannel(self.conversation_type, self.conversation_id)

    @property
    def key(self) -> str:
        return json.dumps([self.conversation_type, self.conversation_id, self.requester], separators=(",", ":"))


@dataclass(frozen=True)
class AnswerEvidence:
    """Only a receipt-bound verifier creates positive evidence, never reply text."""

    kind: str = "none"
    disposition: str = "not_requested"
    validated: bool = False
    query_count: int = 0
    source_count: int = 0
    answer_sha256: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"none", "history", "public", "local", "state", "task"}:
            raise ValueError("invalid_answer_evidence_kind")
        if self.disposition not in {
            "not_requested", "unverified", "found", "verified", "ambiguous",
            "not_found", "scope_limited", "read_failed", "unconfirmed",
        }:
            raise ValueError("invalid_answer_evidence_disposition")
        if (type(self.validated) is not bool or type(self.query_count) is not int
                or type(self.source_count) is not int or not 0 <= self.query_count <= 128
                or not 0 <= self.source_count <= 128):
            raise ValueError("invalid_answer_evidence_counts")
        if self.answer_sha256 and not re.fullmatch(r"[a-f0-9]{64}", self.answer_sha256):
            raise ValueError("invalid_answer_evidence_digest")
        if self.validated and (not self.query_count or not self.answer_sha256):
            raise ValueError("answer_evidence_requires_read_and_answer")
        if self.validated and self.disposition in {"found", "verified", "ambiguous"} and not self.source_count:
            raise ValueError("answer_evidence_requires_source")

    @property
    def verified(self) -> bool:
        return self.validated and self.disposition in {"found", "verified"}

    @property
    def resolved(self) -> bool:
        return self.validated and self.disposition in {"found", "verified", "not_found", "scope_limited"}

    def bound_to(self, answer: str) -> AnswerEvidence:
        if self.validated and self.answer_sha256 != text_digest(answer):
            return AnswerEvidence(self.kind, "unverified", query_count=self.query_count,
                                  source_count=self.source_count)
        return self

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_history(cls, ledger) -> AnswerEvidence:
        check = ledger.answer_check
        disposition = str(check.get("disposition", "unverified"))
        if disposition not in {"found", "ambiguous", "not_found", "scope_limited", "read_failed", "unconfirmed"}:
            disposition = "unverified"
        return cls("history", disposition, check.get("valid") is True,
                   min(128, len(ledger.lookups)), min(128, len(ledger.sources)),
                   str(check.get("answer_sha256") or ""))

    @classmethod
    def from_source_ledger(cls, ledger, *, kind: str, query_count: int) -> AnswerEvidence:
        """Export the ledger's actual freshness/binding gate, not its model label."""
        check = ledger.answer_check
        answer_hash = str(check.get("answer_sha256") or "")
        complete = bool(query_count and answer_hash and ledger.satisfies({"min_sources": 1})
                        and check.get("verdict") == "supported" and not check.get("missing_fields"))
        return cls(kind, "verified" if complete else "unverified", complete,
                   min(128, query_count), min(128, len(ledger.sources)), answer_hash if complete else "")


@dataclass(frozen=True)
class ProactivePermit:
    channel: ConversationChannel
    revision: int
    checked_at: float
    allowed: bool
    reason: str = ""


_PHASES = {"context", "planning", "preparation", "query", "execution", "verification", "delivery", "finished"}
_STATUSES = {
    "running", "queued", "completed", "chat", "failed", "blocked", "partial",
    "waiting_user", "waiting_background", "canceled", "command", "ignored", "silent",
    "rejected", "stopped", "agent_disabled", "interrupted", "continued",
}
_ROUTES = {"chat", "history", "perception", "public_query", "agent", "command", "scheduled"}
_DELIVERIES = {
    "not_sent", "queued", "dry_run", "sent", "partial", "suppressed", "failed", "uncertain",
}


def safe_code(value: str) -> str:
    # Only structured control codes, not raw provider/transport error prose.
    value = str(value or "")
    return value if re.fullmatch(r"[a-zA-Z0-9_.:-]{0,96}", value) else "internal_error"


@dataclass
class ConversationTurnState:
    turn_id: str
    scope: ConversationScope
    started_at: float
    updated_at: float
    lease_until: float
    status: str = "running"
    phase: str = "context"
    route: str = "chat"
    origin: str = "user"
    evidence: AnswerEvidence = field(default_factory=AnswerEvidence)
    candidate_count: int = 0
    reason: str = ""
    delivery_status: str = "not_sent"
    sent: bool = False
    task_id: str = ""
    task_attempt: int = 0
    task_version: int = 0
    message_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not self.turn_id or self.status not in _STATUSES or self.phase not in _PHASES
                or self.route not in _ROUTES or self.origin not in {"user", "autonomous", "scheduled"}
                or self.delivery_status not in _DELIVERIES or type(self.sent) is not bool
                or type(self.candidate_count) is not int or not 0 <= self.candidate_count <= 128
                or type(self.task_attempt) is not int or self.task_attempt < 0
                or type(self.task_version) is not int or self.task_version < 0
                or not isinstance(self.message_ids, (list, tuple)) or len(self.message_ids) > 32
                or any(not isinstance(v, str) or len(v) > 256 for v in self.message_ids)
                or any(not math.isfinite(t) for t in (self.started_at, self.updated_at, self.lease_until))):
            raise ValueError("invalid_conversation_state")
        self.reason = safe_code(self.reason)

    @property
    def active(self) -> bool:
        return self.phase != "finished" and self.status in {"running", "queued", "waiting_background"}

    def as_dict(self) -> dict:
        value = asdict(self)
        return value

    @classmethod
    def from_dict(cls, value: dict) -> ConversationTurnState:
        value = dict(value)
        value["scope"] = ConversationScope(**value["scope"])
        value["evidence"] = AnswerEvidence(**value["evidence"])
        return cls(**value)
