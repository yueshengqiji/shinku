"""Transient state for one scoped lexical/hybrid search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import MemoryRecord


@dataclass
class MemoryStoreSearch:
    query: str
    scope: str
    limit: int
    now: int | None
    layers: set[str] | tuple[str, ...] | None
    topic_batch_id: str
    sender_id: str
    start_at: int | None
    end_at: int | None
    require_term_overlap: bool
    query_text: str = ""
    terms: list[str] = field(default_factory=list)
    stamp: int = 0
    selected_layers: tuple[str, ...] = ()
    # 原始近期读取未填满两个窗口时，当前范围已读全，无需另开连接补查。
    lexical_window_complete: bool = False
    candidates: list[MemoryRecord] = field(default_factory=list)
    selected: list[MemoryRecord] = field(default_factory=list)
