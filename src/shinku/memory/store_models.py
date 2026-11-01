"""Explicit live dependencies for memory store collaborators."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .lifecycle import MemoryLifecycle
    from .models import MemoryRecord


@dataclass(frozen=True)
class MemoryStoreRecordsDependencies:
    connect: Callable[..., Any]
    event_record: Callable[..., MemoryRecord]
    json: Callable[..., str]
    loads: Callable[..., dict[str, Any]]
    lock: Callable[[], Any]
    lifecycle: Callable[[], MemoryLifecycle]


@dataclass(frozen=True)
class MemoryStoreQueriesDependencies:
    connect: Callable[..., Any]
    embedding_provider: Callable[[], Any | None]
    event_record: Callable[..., MemoryRecord]
    fact_record: Callable[..., MemoryRecord]
    loads: Callable[..., dict[str, Any]]
    lock: Callable[[], Any]
    score_event: Callable[..., MemoryRecord]
    score_fact: Callable[..., MemoryRecord]
    semantic_candidates: Callable[..., list[MemoryRecord]]
    terms: Callable[..., list[str]]
    backfill_embeddings: Callable[..., int]
    event_scan_limit: Callable[[], int]
    expire: Callable[..., int]
    fact_scan_limit: Callable[[], int]
    max_query_terms: Callable[[], int]
    max_results: Callable[[], int]
    search_index: Callable[[], Any | None]
    semantic_backfill_batch_size: Callable[[], int]
    semantic_enabled: Callable[[], bool]
    set_search_keys_last_status: Callable[[dict[str, Any]], None]
    term_pattern: Callable[[], re.Pattern[str]]
    search_keys_available: Callable[[], bool]


@dataclass(frozen=True)
class MemoryStoreSemanticDependencies:
    connect: Callable[..., Any]
    discard_cached_text: Callable[..., None]
    embedding_allowed: Callable[..., bool]
    embedding_lock: Callable[[], Any]
    embedding_provider: Callable[[], Any | None]
    embedding_worker: Callable[[], threading.Thread | None]
    embedding_worker_stop: Callable[[], threading.Event]
    hybrid_record: Callable[..., MemoryRecord]
    l1_semantic_cache: Callable[[], dict[tuple[str, str, str, str], list[float]]]
    lock: Callable[[], Any]
    score_event: Callable[..., MemoryRecord]
    score_fact: Callable[..., MemoryRecord]
    semantic_cleanup_done: Callable[[], bool]
    backfill_embeddings: Callable[..., int]
    embedding_error: Callable[[], str]
    embedding_model_key: Callable[[], str]
    embedding_status: Callable[[], str]
    event_scan_limit: Callable[[], int]
    fact_scan_limit: Callable[[], int]
    lifecycle: Callable[[], MemoryLifecycle]
    semantic_backfill_batch_size: Callable[[], int]
    semantic_enabled: Callable[[], bool]
    semantic_min_score: Callable[[], float]
    set_embedding_worker: Callable[[threading.Thread | None], None]
    set_semantic_cleanup_done: Callable[[bool], None]


@dataclass(frozen=True)
class MemoryStoreDeliveryDependencies:
    connect: Callable[..., Any]
    event_record: Callable[..., MemoryRecord]
    loads: Callable[..., dict[str, Any]]
    lock: Callable[[], Any]
    lifecycle: Callable[[], MemoryLifecycle]


@dataclass(frozen=True)
class MemoryStoreMaintenanceDependencies:
    clear_derived_caches: Callable[..., None]
    connect: Callable[..., Any]
    l1_semantic_cache: Callable[[], dict[tuple[str, str, str, str], list[float]]]
    loads: Callable[..., dict[str, Any]]
    lock: Callable[[], Any]
    terms: Callable[..., list[str]]
    lifecycle: Callable[[], MemoryLifecycle]
    record_event: Callable[..., bool]
    set_l1_semantic_cache: Callable[[dict[tuple[str, str, str, str], list[float]]], None]
