"""Small SQLite store for the independent Shinku memory layers."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .models import MemoryRecord
from .lifecycle import MemoryLifecycle, SCHEMA as LIFECYCLE_SCHEMA
from .lifecycle_policy import LifecyclePolicy
from .store_search_ranking import calibrated_candidate
from .store_relevance import QueryRelevance

from . import store_records, store_queries, store_semantic, store_delivery, store_maintenance
from .store_models import (
    MemoryStoreRecordsDependencies,
    MemoryStoreQueriesDependencies,
    MemoryStoreSemanticDependencies,
    MemoryStoreDeliveryDependencies,
    MemoryStoreMaintenanceDependencies,
)

__all__ = ["MemoryStore"]


_TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}")


class MemoryStore:
    """Persistence with explicit layers and lifecycle fields.

    The source database is never involved.  Migration writes through the same
    API so imported records get the same scope, status and provenance rules as
    new records.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        max_query_terms: int = 12,
        event_scan_limit: int = 800,
        fact_scan_limit: int = 120,
        max_results: int = 12,
        embedding_provider: Any | None = None,
        semantic_enabled: bool = False,
        semantic_backfill_batch_size: int = 128,
        semantic_min_score: float = 0.22,
        embedding_status: str = "configured",
        embedding_error: str = "",
        lifecycle_policy: LifecyclePolicy | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._embedding_lock = threading.RLock()
        self._embedding_provider = embedding_provider
        self.semantic_enabled = bool(semantic_enabled and embedding_provider is not None)
        self.semantic_backfill_batch_size = max(8, min(1000, int(semantic_backfill_batch_size or 128)))
        self.semantic_min_score = max(0.0, min(1.0, float(semantic_min_score)))
        self.embedding_status = str(embedding_status or "configured").strip() or "configured"
        self.embedding_error = str(embedding_error or "").strip()[:240]
        self._embedding_worker: threading.Thread | None = None
        self._embedding_worker_stop = threading.Event()
        self._semantic_cleanup_done = False
        self._l1_semantic_cache: dict[tuple[str, str, str, str], list[float]] = {}
        self._forget_listeners: list[Any] = []
        self.search_index = None
        self.search_keys_enabled = False
        self.search_keys_last_status: dict[str, Any] = {}
        self.configure(
            max_query_terms=max_query_terms,
            event_scan_limit=event_scan_limit,
            fact_scan_limit=fact_scan_limit,
            max_results=max_results,
        )
        self._initialize()
        self.lifecycle = MemoryLifecycle(self, lifecycle_policy)

    def configure(
        self,
        *,
        max_query_terms: int = 12,
        event_scan_limit: int = 800,
        fact_scan_limit: int = 120,
        max_results: int = 12,
    ) -> None:
        self.max_query_terms = max(1, min(64, int(max_query_terms or 12)))
        self.event_scan_limit = max(10, min(10000, int(event_scan_limit or 800)))
        self.fact_scan_limit = max(10, min(5000, int(fact_scan_limit or 120)))
        self.max_results = max(1, min(100, int(max_results or 12)))

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        observers = tuple(getattr(self, "_projection_observers", ()))
        tokens = []
        for observer in observers:
            try:
                tokens.append((observer, observer.before_commit()))
            except Exception:
                pass  # A derived locator must never block canonical storage.
        source_write = [False]
        if observers:
            pending_source = [False, connection.total_changes]
            def track(statement):
                if pending_source[0] and connection.total_changes > pending_source[1]:
                    source_write[0] = True
                writes_source = bool(re.search(r"\b(?:INSERT\s+(?:OR\s+\w+\s+)?INTO|UPDATE|DELETE\s+FROM)\s+[\"\[]?memory_(?:events|facts)\b", statement, re.I))
                usage_only = bool(re.match(r"\s*UPDATE\s+memory_(?:events|facts)\s+SET\s+last_used_at\s*=\s*\d+\s+WHERE\b", statement, re.I))
                pending_source[:] = [writes_source and not usage_only, connection.total_changes]
            connection.set_trace_callback(track)
        try:
            yield connection
            connection.commit()
            if connection.total_changes:
                for observer, token in tokens:
                    try:
                        observer.after_commit(token, source_write[0])
                    except Exception:
                        pass
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS memory_turns (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    source_id TEXT NOT NULL UNIQUE,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    text TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_used_at INTEGER NOT NULL DEFAULT 0,
                    importance REAL NOT NULL DEFAULT 0.25,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    expires_at INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    source_id TEXT NOT NULL UNIQUE,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS memory_facts (
                    fact_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    last_used_at INTEGER NOT NULL DEFAULT 0,
                    importance REAL NOT NULL DEFAULT 0.8,
                    confidence REAL NOT NULL DEFAULT 0.8,
                    status TEXT NOT NULL DEFAULT 'active',
                    expires_at INTEGER NOT NULL DEFAULT 0,
                    source_id TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_fact_key
                    ON memory_facts(scope, fact_key);
                CREATE TABLE IF NOT EXISTS memory_forget_log (
                    forget_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    query TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    action TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_routing_audit (
                    audit_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    history_mode TEXT NOT NULL,
                    need_retrieval INTEGER NOT NULL DEFAULT 0,
                    used_model INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    reason TEXT NOT NULL DEFAULT '',
                    retrieved_count INTEGER NOT NULL DEFAULT 0,
                    injected_chars INTEGER NOT NULL DEFAULT 0,
                    decision_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    record_type TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    model_key TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    UNIQUE(record_type, record_id, model_key)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_events_scope_time
                    ON memory_events(scope, status, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_facts_scope_status
                    ON memory_facts(scope, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_routing_audit_scope_time
                    ON memory_routing_audit(scope, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_embeddings_scope_model
                    ON memory_embeddings(scope, model_key, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_embeddings_record
                    ON memory_embeddings(record_type, record_id, model_key);
                """
            )

            conn.executescript(LIFECYCLE_SCHEMA)
            if "metadata_json" not in {row[1] for row in conn.execute("PRAGMA table_info(memory_facts)")}:
                conn.execute("ALTER TABLE memory_facts ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")

    def configure_lifecycle(self, policy: LifecyclePolicy) -> None:
        """Apply one versioned policy without deleting legacy history."""
        self.lifecycle.stop_worker()
        self.lifecycle.policy = policy
        self._l1_semantic_cache.clear()

    def on_forget(self, listener: Any) -> None:
        if not callable(listener):
            raise TypeError("forget_listener_must_be_callable")
        self._forget_listeners.append(listener)

    def configure_search_keys(self, *, enabled: bool) -> None:
        if enabled and self.search_index is None:
            from .search_index import MemorySearchIndex
            try:
                self.search_index = MemorySearchIndex(self)
            except Exception as exc:
                # A derived schema cannot take the source-only service down.
                self.search_keys_enabled = False
                self.search_keys_last_status = {"fallback": "source_search", "phase": "initialize",
                                               "error_type": type(exc).__name__, "remote_calls": 0}
                return
        self.search_keys_enabled = bool(enabled)

    def _clear_derived_caches(self, *, scope: str, record_ids: set[str]) -> None:
        self._l1_semantic_cache.clear()
        clear = getattr(self._embedding_provider, "clear_cache", None)
        if callable(clear):
            clear()
        for listener in self._forget_listeners:
            try:
                listener(scope=scope, record_ids=record_ids)
            except Exception:
                continue

    def _discard_cached_text(self, text: str) -> None:
        forget = getattr(self._embedding_provider, "forget_texts", None)
        if callable(forget):
            forget([text[:6000]])

    @property
    def embedding_provider(self) -> Any | None:
        return self._embedding_provider

    @property
    def embedding_model_key(self) -> str:
        provider = self._embedding_provider
        if provider is None:
            return ""
        try:
            return str(provider.collection_key() or "").strip()
        except Exception:
            name = str(getattr(provider, "name", "embedding") or "embedding")
            version = str(getattr(provider, "version", "v1") or "v1")
            dimension = int(getattr(provider, "dimension", 0) or 0)
            return f"{name}_{version}_{dimension}".strip("_").lower()

    def semantic_status(self) -> dict[str, Any]:
        """Return safe diagnostics for the active local semantic index."""
        return store_semantic.semantic_status(
            deps=self._store_semantic_dependencies(),
        )

    def start_embedding_worker(self, *, interval_seconds: float = 3.0) -> None:
        """Warm the local index in the background without blocking QQ turns."""
        return store_semantic.start_embedding_worker(
            interval_seconds=interval_seconds,
            deps=self._store_semantic_dependencies(),
        )

    def stop_embedding_worker(self) -> None:
        return store_semantic.stop_embedding_worker(
            deps=self._store_semantic_dependencies(),
        )

    def backfill_embeddings(self, *, limit: int | None = None, scope: str | None = None,
                            include_deferred_l1: bool = False) -> int:
        """Embed active records missing from the current model collection.

        The operation is idempotent and model-versioned.  It is deliberately
        separate from ``record_event`` so passive QQ observation never waits
        for a sentence-transformer model.
        后台传 include_deferred_l1 才处理延后索引的 L1；查询内回填保持原预算。
        """
        return store_semantic.backfill_embeddings(
            limit=limit,
            scope=scope,
            include_deferred_l1=include_deferred_l1,
            deps=self._store_semantic_dependencies(),
        )

    def _embedding_allowed(self, conn, record_type: str, record_id: str,
                           *, include_deferred_l1: bool = False) -> bool:
        return store_semantic.embedding_allowed(
            conn,
            record_type,
            record_id,
            include_deferred_l1=include_deferred_l1,
            deps=self._store_semantic_dependencies(),
        )

    @staticmethod
    def _json(value: Any) -> str:
        try:
            return json.dumps(value if isinstance(value, dict) else {}, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return "{}"

    def _terms(self, query: str) -> list[str]:
        return store_queries.terms(
            query,
            deps=self._store_queries_dependencies(),
        )

    def record_turn(
        self,
        *,
        conversation_id: str,
        scope: str,
        role: str,
        content: str,
        source_id: str,
        created_at: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        return store_records.record_turn(
            conversation_id=conversation_id,
            scope=scope,
            role=role,
            content=content,
            source_id=source_id,
            created_at=created_at,
            metadata=metadata,
            deps=self._store_records_dependencies(),
        )

    def record_routing_audit(
        self,
        *,
        conversation_id: str,
        scope: str,
        created_at: int,
        decision: dict[str, Any],
        retrieved_count: int = 0,
        injected_chars: int = 0,
    ) -> bool:
        """Persist one cheap, non-prompt audit row for memory routing."""
        return store_records.record_routing_audit(
            conversation_id=conversation_id,
            scope=scope,
            created_at=created_at,
            decision=decision,
            retrieved_count=retrieved_count,
            injected_chars=injected_chars,
            deps=self._store_records_dependencies(),
        )

    def recent_routing_audits(
        self,
        *,
        scope: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        return store_records.recent_routing_audits(
            scope=scope,
            limit=limit,
            deps=self._store_records_dependencies(),
        )

    def record_event(
        self,
        *,
        conversation_id: str,
        scope: str,
        text: str,
        source_id: str,
        layer: str = "episodic",
        created_at: int | None = None,
        importance: float = 0.25,
        confidence: float = 0.5,
        expires_at: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        return store_records.record_event(
            conversation_id=conversation_id,
            scope=scope,
            text=text,
            source_id=source_id,
            layer=layer,
            created_at=created_at,
            importance=importance,
            confidence=confidence,
            expires_at=expires_at,
            metadata=metadata,
            deps=self._store_records_dependencies(),
        )

    def event_by_source_id(self, *, source_id: str, scope: str) -> MemoryRecord | None:
        """Read one active event by its stable source key inside one scope."""
        return store_records.event_by_source_id(
            source_id=source_id,
            scope=scope,
            deps=self._store_records_dependencies(),
        )

    def prune_agent_task_details(self, *, scope: str, max_records: int) -> int:
        """Forget the oldest active detail archives past the configured cap."""
        return store_records.prune_agent_task_details(
            scope=scope,
            max_records=max_records,
            deps=self._store_records_dependencies(),
        )

    def promote_scope_to_owner(
        self,
        *,
        source_scope: str,
        owner_scope: str = "private:master",
    ) -> dict[str, int]:
        """Copy an already isolated private scope into the owner library.

        This is intentionally limited to a private scope whose identity is
        configured as the owner.  It never promotes group spectators, and the
        deterministic source prefix makes startup migration idempotent.
        """
        return store_records.promote_scope_to_owner(
            source_scope=source_scope,
            owner_scope=owner_scope,
            deps=self._store_records_dependencies(),
        )

    def upsert_fact(
        self,
        *,
        conversation_id: str,
        scope: str,
        fact_key: str,
        text: str,
        source_id: str = "",
        importance: float = 0.85,
        confidence: float = 0.9,
        expires_at: int = 0,
        now: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        return store_records.upsert_fact(
            conversation_id=conversation_id,
            scope=scope,
            fact_key=fact_key,
            text=text,
            source_id=source_id,
            importance=importance,
            confidence=confidence,
            expires_at=expires_at,
            now=now,
            metadata=metadata,
            deps=self._store_records_dependencies(),
        )

    def search(
        self,
        *,
        query: str,
        scope: str,
        limit: int = 5,
        now: int | None = None,
        layers: set[str] | tuple[str, ...] | None = None,
        topic_batch_id: str = "",
        sender_id: str = "",
        start_at: int | None = None,
        end_at: int | None = None,
        require_term_overlap: bool = False,
    ) -> list[MemoryRecord]:
        """Search one scope, optionally bounded to a calendar window.

        Normal retrieval keeps the small recency prior used by the original
        memory contract.  Date-specific recall can opt into
        ``require_term_overlap`` so a query such as "前天我问你的那件事"
        does not simply return the newest records from that day when its
        actual topic is present farther down the timeline.
        """
        return store_queries.search(
            query=query,
            scope=scope,
            limit=limit,
            now=now,
            layers=layers,
            topic_batch_id=topic_batch_id,
            sender_id=sender_id,
            start_at=start_at,
            end_at=end_at,
            require_term_overlap=require_term_overlap,
            deps=self._store_queries_dependencies(),
        )

    def _semantic_candidates(
        self,
        *,
        query: str,
        terms: list[str],
        scope: str,
        selected_layers: tuple[str, ...],
        start_at: int | None,
        end_at: int | None,
        topic_batch_id: str,
        sender_id: str,
        now: int,
    ) -> list[MemoryRecord]:
        """Return vector candidates already present in the local SQLite index."""
        return store_semantic.semantic_candidates(
            query=query,
            terms=terms,
            scope=scope,
            selected_layers=selected_layers,
            start_at=start_at,
            end_at=end_at,
            topic_batch_id=topic_batch_id,
            sender_id=sender_id,
            now=now,
            deps=self._store_semantic_dependencies(),
        )

    @staticmethod
    def _hybrid_record(
        record: MemoryRecord,
        similarity: float,
        *,
        terms: list[str],
        features: QueryRelevance | None = None,
    ) -> MemoryRecord:
        """语义扫描与最终合并共用分量评分；编码器的输入不作改写。"""
        return calibrated_candidate(record, terms=terms, similarity=similarity, features=features)

    def nearby_events(
        self,
        *,
        scope: str,
        center_at: int,
        limit: int = 6,
        window_seconds: int = 600,
        layers: set[str] | tuple[str, ...] = ("observation", "episodic"),
        start_at: int | None = None,
        end_at: int | None = None,
    ) -> list[MemoryRecord]:
        """Return a small chronological neighborhood around one hit.

        A date question often matches the user's question but not the
        assistant's answer.  Pulling the adjacent turn keeps that answer
        available without injecting an entire day's transcript.
        """
        return store_queries.nearby_events(
            scope=scope,
            center_at=center_at,
            limit=limit,
            window_seconds=window_seconds,
            layers=layers,
            start_at=start_at,
            end_at=end_at,
            deps=self._store_queries_dependencies(),
        )

    def recent_events(
        self,
        *,
        scope: str,
        limit: int = 6,
        now: int | None = None,
        max_age_seconds: int = 900,
        layers: set[str] | tuple[str, ...] = ("observation", "episodic"),
        topic_batch_id: str = "",
        include_assistant: bool = False,
    ) -> list[MemoryRecord]:
        """Return a bounded local snapshot without term-based retrieval.

        Normal ``recent`` context is for topic continuity, not for teaching the
        model how it should speak.  Generated assistant turns are therefore
        excluded by default; otherwise a cold or defensive answer can be
        immediately retrieved on the next turn and reinforce its own style.
        Explicit recall/search paths still include assistant turns so that a
        user can ask what Shinku said earlier.
        """
        return store_queries.recent_events(
            scope=scope,
            limit=limit,
            now=now,
            max_age_seconds=max_age_seconds,
            layers=layers,
            topic_batch_id=topic_batch_id,
            include_assistant=include_assistant,
            deps=self._store_queries_dependencies(),
        )

    def confirm_assistant_delivery(
        self, *, scope: str, source_id: str, expected_text: str, delivered_at: int,
        status: str = "sent", requester_name: str = "", proof_id: str = "",
        content_complete: bool = False,
    ) -> bool:
        """Attach a transport receipt to an existing, active assistant source.

        No new memory or raw speech is copied. Dry runs/total failures never
        call this; partial sends prove occurrence, not the complete body.
        """
        return store_delivery.confirm_assistant_delivery(
            scope=scope,
            source_id=source_id,
            expected_text=expected_text,
            delivered_at=delivered_at,
            status=status,
            requester_name=requester_name,
            proof_id=proof_id,
            content_complete=content_complete,
            deps=self._store_delivery_dependencies(),
        )

    def delivered_group_activity(
        self, *, start_at: int, end_at: int, now: int, scope: str = "",
        self_id: str = "", limit: int = 8,
    ) -> list[MemoryRecord]:
        """Candidates are ONLY the bot's receipt-backed group replies.

        The service must authenticate private cross-group readers. No user
        messages, observations, summaries, facts, or private turns are selected.
        """
        return store_delivery.delivered_group_activity(
            start_at=start_at,
            end_at=end_at,
            now=now,
            scope=scope,
            self_id=self_id,
            limit=limit,
            deps=self._store_delivery_dependencies(),
        )

    def recent_assistant_events(
        self, *, scope: str, now: int, max_age_seconds: int = 1800, limit: int = 48
    ) -> list[MemoryRecord]:
        """Fetch Shinku's own active QQ replies for same-scope fact checking.

        This is deliberately separate from ``recent_events``: generated replies
        must not become general conversational or voice examples.
        """
        return store_queries.recent_assistant_events(
            scope=scope,
            now=now,
            max_age_seconds=max_age_seconds,
            limit=limit,
            deps=self._store_queries_dependencies(),
        )

    def timeline(
        self,
        *,
        scope: str,
        limit: int = 8,
        now: int | None = None,
        layers: set[str] | tuple[str, ...] | None = None,
        include_facts: bool = True,
        start_at: int | None = None,
        end_at: int | None = None,
    ) -> list[MemoryRecord]:
        """Return a bounded read-only timeline for explicit memory inspection.

        This is deliberately separate from search: a user asking to see
        memory may not provide terms that overlap the stored text.  The
        timeline therefore uses recency and provenance, while the normal
        routing path continues to use term-based retrieval.
        """
        return store_queries.timeline(
            scope=scope,
            limit=limit,
            now=now,
            layers=layers,
            include_facts=include_facts,
            start_at=start_at,
            end_at=end_at,
            deps=self._store_queries_dependencies(),
        )

    def latest_observation(
        self,
        *,
        scope: str,
        before_created_at: int | None = None,
    ) -> MemoryRecord | None:
        return store_queries.latest_observation(
            scope=scope,
            before_created_at=before_created_at,
            deps=self._store_queries_dependencies(),
        )

    def observation_for_message(self, *, scope: str, message_id: str) -> MemoryRecord | None:
        return store_queries.observation_for_message(
            scope=scope,
            message_id=message_id,
            deps=self._store_queries_dependencies(),
        )

    def rollup_daily_summaries(
        self,
        *,
        scope: str,
        now: int | None = None,
        max_lines: int = 60,
        source_layers: set[str] | tuple[str, ...] = ("observation", "episodic"),
    ) -> int:
        """Create compact local summaries for completed conversation days.

        This is intentionally deterministic: it does not call a model and it
        never replaces the underlying observation evidence.
        """
        return store_maintenance.rollup_daily_summaries(
            scope=scope,
            now=now,
            max_lines=max_lines,
            source_layers=source_layers,
            deps=self._store_maintenance_dependencies(),
        )

    def _event_record(self, row: sqlite3.Row, *, score: float) -> MemoryRecord:
        return store_queries.event_record(
            row,
            score=score,
            deps=self._store_queries_dependencies(),
        )

    def _score_event(
        self,
        row: sqlite3.Row,
        terms: list[str],
        now: int,
        *,
        topic_batch_id: str = "",
        sender_id: str = "",
    ) -> MemoryRecord:
        return store_queries.score_event(
            row,
            terms,
            now,
            topic_batch_id=topic_batch_id,
            sender_id=sender_id,
            deps=self._store_queries_dependencies(),
        )

    def _score_fact(self, row: sqlite3.Row, terms: list[str], now: int) -> MemoryRecord:
        return store_queries.score_fact(
            row,
            terms,
            now,
            deps=self._store_queries_dependencies(),
        )

    def _fact_record(self, row: sqlite3.Row, *, score: float) -> MemoryRecord:
        return store_queries.fact_record(
            row,
            score=score,
            deps=self._store_queries_dependencies(),
        )

    def expire(self, *, now: int | None = None) -> int:
        return store_maintenance.expire(
            now=now,
            deps=self._store_maintenance_dependencies(),
        )

    def forget(
        self,
        *,
        query: str,
        scope: str,
        purge: bool = False,
        now: int | None = None,
        record_id: str = "",
    ) -> tuple[int, int]:
        return store_maintenance.forget(
            query=query,
            scope=scope,
            purge=purge,
            now=now,
            record_id=record_id,
            deps=self._store_maintenance_dependencies(),
        )

    def reset_all(self) -> dict[str, int]:
        """Clear every memory record while keeping the database schema.

        This is intentionally a low-level destructive operation.  The private
        command layer is responsible for authorization and explicit confirmation.
        """
        return store_maintenance.reset_all(
            deps=self._store_maintenance_dependencies(),
        )

    def stats(self) -> dict[str, int]:
        return store_maintenance.stats(
            deps=self._store_maintenance_dependencies(),
        )

    @staticmethod
    def _loads(value: Any) -> dict[str, Any]:
        try:
            parsed = json.loads(str(value or "{}"))
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}

    def _store_records_dependencies(self) -> MemoryStoreRecordsDependencies:
        return MemoryStoreRecordsDependencies(
            connect=lambda *args, **kwargs: self._connect(*args, **kwargs),
            event_record=lambda *args, **kwargs: self._event_record(*args, **kwargs),
            json=lambda *args, **kwargs: self._json(*args, **kwargs),
            loads=lambda *args, **kwargs: self._loads(*args, **kwargs),
            lock=lambda: self._lock,
            lifecycle=lambda: self.lifecycle,
        )

    def _store_queries_dependencies(self) -> MemoryStoreQueriesDependencies:
        return MemoryStoreQueriesDependencies(
            connect=lambda *args, **kwargs: self._connect(*args, **kwargs),
            embedding_provider=lambda: self._embedding_provider,
            event_record=lambda *args, **kwargs: self._event_record(*args, **kwargs),
            fact_record=lambda *args, **kwargs: self._fact_record(*args, **kwargs),
            loads=lambda *args, **kwargs: self._loads(*args, **kwargs),
            lock=lambda: self._lock,
            score_event=lambda *args, **kwargs: self._score_event(*args, **kwargs),
            score_fact=lambda *args, **kwargs: self._score_fact(*args, **kwargs),
            semantic_candidates=lambda *args, **kwargs: self._semantic_candidates(*args, **kwargs),
            terms=lambda *args, **kwargs: self._terms(*args, **kwargs),
            backfill_embeddings=lambda *args, **kwargs: self.backfill_embeddings(*args, **kwargs),
            event_scan_limit=lambda: self.event_scan_limit,
            expire=lambda *args, **kwargs: self.expire(*args, **kwargs),
            fact_scan_limit=lambda: self.fact_scan_limit,
            max_query_terms=lambda: self.max_query_terms,
            max_results=lambda: self.max_results,
            search_index=lambda: self.search_index,
            semantic_backfill_batch_size=lambda: self.semantic_backfill_batch_size,
            semantic_enabled=lambda: self.semantic_enabled,
            set_search_keys_last_status=lambda value: self._set_memory_search_keys_last_status(value),
            term_pattern=lambda: _TERM_RE,
            search_keys_available=lambda: getattr(self, "search_keys_enabled", False),
        )

    def _store_semantic_dependencies(self) -> MemoryStoreSemanticDependencies:
        return MemoryStoreSemanticDependencies(
            connect=lambda *args, **kwargs: self._connect(*args, **kwargs),
            discard_cached_text=lambda *args, **kwargs: self._discard_cached_text(*args, **kwargs),
            embedding_allowed=lambda *args, **kwargs: self._embedding_allowed(*args, **kwargs),
            embedding_lock=lambda: self._embedding_lock,
            embedding_provider=lambda: self._embedding_provider,
            embedding_worker=lambda: self._embedding_worker,
            embedding_worker_stop=lambda: self._embedding_worker_stop,
            hybrid_record=lambda *args, **kwargs: self._hybrid_record(*args, **kwargs),
            l1_semantic_cache=lambda: self._l1_semantic_cache,
            lock=lambda: self._lock,
            score_event=lambda *args, **kwargs: self._score_event(*args, **kwargs),
            score_fact=lambda *args, **kwargs: self._score_fact(*args, **kwargs),
            semantic_cleanup_done=lambda: self._semantic_cleanup_done,
            backfill_embeddings=lambda *args, **kwargs: self.backfill_embeddings(*args, **kwargs),
            embedding_error=lambda: self.embedding_error,
            embedding_model_key=lambda: self.embedding_model_key,
            embedding_status=lambda: self.embedding_status,
            event_scan_limit=lambda: self.event_scan_limit,
            fact_scan_limit=lambda: self.fact_scan_limit,
            lifecycle=lambda: self.lifecycle,
            semantic_backfill_batch_size=lambda: self.semantic_backfill_batch_size,
            semantic_enabled=lambda: self.semantic_enabled,
            semantic_min_score=lambda: self.semantic_min_score,
            set_embedding_worker=lambda value: self._set_memory_embedding_worker(value),
            set_semantic_cleanup_done=lambda value: self._set_memory_semantic_cleanup_done(value),
        )

    def _store_delivery_dependencies(self) -> MemoryStoreDeliveryDependencies:
        return MemoryStoreDeliveryDependencies(
            connect=lambda *args, **kwargs: self._connect(*args, **kwargs),
            event_record=lambda *args, **kwargs: self._event_record(*args, **kwargs),
            loads=lambda *args, **kwargs: self._loads(*args, **kwargs),
            lock=lambda: self._lock,
            lifecycle=lambda: self.lifecycle,
        )

    def _store_maintenance_dependencies(self) -> MemoryStoreMaintenanceDependencies:
        return MemoryStoreMaintenanceDependencies(
            clear_derived_caches=lambda *args, **kwargs: self._clear_derived_caches(*args, **kwargs),
            connect=lambda *args, **kwargs: self._connect(*args, **kwargs),
            l1_semantic_cache=lambda: self._l1_semantic_cache,
            loads=lambda *args, **kwargs: self._loads(*args, **kwargs),
            lock=lambda: self._lock,
            terms=lambda *args, **kwargs: self._terms(*args, **kwargs),
            lifecycle=lambda: self.lifecycle,
            record_event=lambda *args, **kwargs: self.record_event(*args, **kwargs),
            set_l1_semantic_cache=lambda value: self._set_memory_l1_semantic_cache(value),
        )

    def _set_memory_embedding_worker(self, value: threading.Thread | None) -> None:
        self._embedding_worker = value

    def _set_memory_l1_semantic_cache(
        self,
        value: dict[tuple[str, str, str, str], list[float]],
    ) -> None:
        self._l1_semantic_cache = value

    def _set_memory_semantic_cleanup_done(self, value: bool) -> None:
        self._semantic_cleanup_done = value

    def _set_memory_search_keys_last_status(self, value: dict[str, Any]) -> None:
        self.search_keys_last_status = value
