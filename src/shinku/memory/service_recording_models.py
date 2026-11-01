"""Request-local observation and turn-write state, separate from host services."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MemoryObservedMessage:
    message: Any
    conversation_id: str = ""
    conversation_type: str = ""
    text: str = ""
    has_media: bool = False
    attachments: tuple[Any, ...] = ()
    quoted_text: str = ""
    quoted_attachments: tuple[Any, ...] = ()
    has_quoted_media: bool = False
    effective_has_media: bool = False
    now: int = 0
    scope: str = ""
    message_id: str = ""
    sender_id: str = ""
    sender_name: str = ""
    reply_to: str = ""
    at_targets: list[str] = field(default_factory=list)
    continuation: bool = False
    deictic_reference: bool = False
    topic_batch_id: str = ""


@dataclass
class MemoryRecordedTurn:
    conversation_id: str
    conversation_type: str
    message_id: str
    text: str
    role: str
    source_id: str
    metadata: Mapping[str, Any] | None
    body: str = ""
    now: int = 0
    scope: str = ""
    normalized_role: str = ""
    effective_source_id: str = ""
    turn_metadata: dict[str, Any] = field(default_factory=dict)
    owner_allowed: bool = False
    owner_source_id: str = ""
    owner_metadata: dict[str, Any] = field(default_factory=dict)
