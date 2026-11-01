"""Per-query memory coordination state; no persistence or host references."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from collections.abc import Mapping

if TYPE_CHECKING:
    from .models import MemoryDecision


@dataclass
class MemoryTurnQuery:
    conversation_id: str
    conversation_type: str
    user_message: str
    recent_context: str
    message_meta: Mapping[str, Any] | None
    now: int = 0
    scope: str = ""
    effective_meta: dict[str, Any] = field(default_factory=dict)
    message_id: str = ""
    decision: MemoryDecision | None = None
    owner_request: bool = False
    sender_id: str = ""
    natural_records: Any = None
    history_need: Any = None
    history_scopes: tuple[str, ...] = ()
    recent_dialogue: str = ""
    dialogue_audit: dict[str, Any] = field(default_factory=dict)
    topic_batch_id: str = ""
    fallback: str = ""
    records: Any = ()
    rendered: str = ""
    evidence_package: Any = None
    own_evidence: tuple[Any, ...] = ()


@dataclass
class MemoryInspectionQuery:
    conversation_id: str
    conversation_type: str
    requester_id: str
    query: str
    mode: str
    limit: int
    message_id: str
    user_text: str
    self_id: str
    now: int = 0
    scope: str = ""
    selected_mode: str = ""
    clean_query: str = ""
    wanted: int = 0
    owner_request: bool = False
    history_scopes: tuple[str, ...] = ()
    records: Any = ()
    fallback: str = ""
    rendered: str = ""
    evidence_package: Any = None
