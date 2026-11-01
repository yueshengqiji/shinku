"""Local embedding phase state; provider, locks and caches remain on the host."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import MemoryRecord


@dataclass
class MemoryEmbeddingBackfill:
    limit: int | None
    scope: str | None
    include_deferred_l1: bool = False
    model_key: str = ""
    wanted: int = 0
    rows: Any = ()
    vectors: Any = ()
    dimension: int = 0
    stamp: int = 0
    written: int = 0


@dataclass
class MemorySemanticQuery:
    query: str
    terms: list[str]
    scope: str
    selected_layers: tuple[str, ...]
    start_at: int | None
    end_at: int | None
    topic_batch_id: str
    sender_id: str
    now: int
    model_key: str = ""
    query_vector: Any = ()
    event_rows: Any = ()
    fact_rows: Any = ()
    l1_rows: Any = ()
    semantic: list[MemoryRecord] = field(default_factory=list)
