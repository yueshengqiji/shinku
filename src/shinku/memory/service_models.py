"""Explicit live dependencies for memory service collaborators."""

from __future__ import annotations

from shinku.character import CharacterProfile, SHINKU_CHARACTER

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .lifecycle_review import LifecycleReviewQueue
    from .models import MemoryRecord
    from .policy import MemoryPolicy
    from .router import MemoryRouter
    from .store import MemoryStore


@dataclass(frozen=True)
class MemoryServiceAccessDependencies:
    owner_scope: Callable[[], str]
    request_date_window: Callable[..., Any]
    clock: Callable[..., float]
    integrity_reader: Callable[[], Any]
    is_master_user: Callable[..., bool]
    policy: Callable[[], MemoryPolicy]
    scope_for: Callable[..., str]
    store: Callable[[], MemoryStore]

    character: CharacterProfile = SHINKU_CHARACTER


@dataclass(frozen=True)
class MemoryServiceQueriesDependencies:
    current_records: Callable[..., Any]
    evidence_records: Callable[..., Any]
    history_scopes: Callable[..., tuple[str, ...]]
    is_master_request: Callable[..., bool]
    merge_by_source: Callable[..., tuple[Any, ...]]
    own_utterance_evidence: Callable[..., tuple[Any, ...]]
    precise_day_request: Callable[..., bool]
    private_recent_dialogue: Callable[..., str]
    recall_query: Callable[..., str]
    record_audit: Callable[..., None]
    render_evidence: Callable[..., Any]
    render_own_utterances: Callable[..., str]
    request_date_window: Callable[..., Any]
    source_reader: Callable[[], Any | None]
    bind_usage_context: Callable[..., None]
    clock: Callable[..., float]
    history_workbench: Callable[[], Any | None]
    inspect_self_group_activity: Callable[..., dict[str, Any]]
    policy: Callable[[], MemoryPolicy]
    router: Callable[[], MemoryRouter]
    scope_for: Callable[..., str]
    store: Callable[[], MemoryStore]
    route_history_need: Callable[..., Any]

    character: CharacterProfile = SHINKU_CHARACTER


@dataclass(frozen=True)
class MemoryServiceEvidenceDependencies:
    evidence_ngrams: Callable[..., set[str]]
    merge_by_source: Callable[..., tuple[Any, ...]]
    render: Callable[..., str]
    request_date_window: Callable[..., Any]
    source_reader: Callable[[], Any | None]
    clock: Callable[..., float]
    current_understanding: Callable[[], Any | None]
    integrity_reader: Callable[[], Any]
    policy: Callable[[], MemoryPolicy]
    store: Callable[[], MemoryStore]
    set_current_understanding_status: Callable[[dict[str, Any]], None]

    character: CharacterProfile = SHINKU_CHARACTER


@dataclass(frozen=True)
class MemoryServiceRecordingDependencies:
    owner_scope: Callable[[], str]
    fact_pattern: Callable[[], re.Pattern[str]]
    forget_pattern: Callable[[], re.Pattern[str]]
    attachment_refs: Callable[..., list[dict[str, str]]]
    key: Callable[..., str]
    reinforce_rementions: Callable[..., int]
    topic_overlap: Callable[..., int]
    clock: Callable[..., float]
    owner_memory_allowed: Callable[..., bool]
    policy: Callable[[], MemoryPolicy]
    scope_for: Callable[..., str]
    store: Callable[[], MemoryStore]


@dataclass(frozen=True)
class MemoryServiceTasksDependencies:
    owner_scope: Callable[[], str]
    task_bearer_pattern: Callable[[], re.Pattern[str]]
    task_secret_pattern: Callable[[], re.Pattern[str]]
    clock: Callable[..., float]
    is_master_user: Callable[..., bool]
    owner_memory_allowed: Callable[..., bool]
    policy: Callable[[], MemoryPolicy]
    scope_for: Callable[..., str]
    store: Callable[[], MemoryStore]


@dataclass(frozen=True)
class MemoryServiceUsageDependencies:
    owner_scope: Callable[[], str]
    usage_contexts: Callable[[], dict[tuple[str, str], tuple[int, str, tuple[MemoryRecord, ...], str]]]
    usage_lock: Callable[[], Any]
    clock: Callable[..., float]
    is_master_user: Callable[..., bool]
    lifecycle_review: Callable[[], LifecycleReviewQueue]
    policy: Callable[[], MemoryPolicy]
    scope_for: Callable[..., str]
    store: Callable[[], MemoryStore]
    set_usage_contexts: Callable[[dict[tuple[str, str], tuple[int, str, tuple[MemoryRecord, ...], str]]], None]
