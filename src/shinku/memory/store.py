"""Small SQLite store for the independent Shinku memory layers."""

from __future__ import annotations

import json
import hashlib
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import MemoryRecord
from .lifecycle import MemoryLifecycle, SCHEMA as LIFECYCLE_SCHEMA
from .lifecycle_policy import LifecyclePolicy
from .semantic import cosine_similarity, pack_vector, unpack_vector

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
        try:
            yield connection
            connection.commit()
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

        model_key = self.embedding_model_key
        with self._lock, self._connect() as conn:
            total = int(
                conn.execute(
                    "SELECT COUNT(*) FROM memory_embeddings WHERE model_key = ?",
                    (model_key,),
                ).fetchone()[0]
            ) if model_key else 0
            active = int(
                conn.execute(
                    """SELECT COUNT(*) FROM memory_embeddings AS e
                       WHERE e.model_key = ? AND (
                         (e.record_type = 'event' AND EXISTS (
                           SELECT 1 FROM memory_events m
                           WHERE m.event_id = e.record_id AND m.status = 'active'
                         )) OR
                         (e.record_type = 'fact' AND EXISTS (
                           SELECT 1 FROM memory_facts f
                           WHERE f.fact_id = e.record_id AND f.status = 'active'
                         ))
                       )""",
                    (model_key,),
                ).fetchone()[0]
            ) if model_key else 0
        provider = self._embedding_provider
        return {
            "enabled": bool(self.semantic_enabled),
            "status": self.embedding_status,
            "error": self.embedding_error,
            "provider": str(getattr(provider, "name", "") or "") if provider else "",
            "model": str(getattr(provider, "version", "") or "") if provider else "",
            "model_key": model_key,
            "dimension": int(getattr(provider, "dimension", 0) or 0) if provider else 0,
            "vectors": total,
            "active_vectors": active,
            "worker_running": bool(self._embedding_worker and self._embedding_worker.is_alive()),
        }

    def start_embedding_worker(self, *, interval_seconds: float = 3.0) -> None:
        """Warm the local index in the background without blocking QQ turns."""

        if not self.semantic_enabled or self._embedding_provider is None:
            return
        if self._embedding_worker and self._embedding_worker.is_alive():
            return
        self._embedding_worker_stop.clear()
        interval = max(0.5, min(60.0, float(interval_seconds or 3.0)))

        def run() -> None:
            while not self._embedding_worker_stop.is_set():
                try:
                    self.backfill_embeddings(limit=self.semantic_backfill_batch_size)
                except Exception:
                    # Keyword retrieval remains the safety net.  A transient
                    # model/DB failure must not terminate the backend worker.
                    pass
                self._embedding_worker_stop.wait(interval)

        self._embedding_worker = threading.Thread(
            target=run,
            name="shinku-memory-embedding-indexer",
            daemon=True,
        )
        self._embedding_worker.start()

    def stop_embedding_worker(self) -> None:
        self._embedding_worker_stop.set()

    def backfill_embeddings(self, *, limit: int | None = None, scope: str | None = None) -> int:
        """Embed active records missing from the current model collection.

        The operation is idempotent and model-versioned.  It is deliberately
        separate from ``record_event`` so passive QQ observation never waits
        for a sentence-transformer model.
        """

        if not self.semantic_enabled or self._embedding_provider is None:
            return 0
        model_key = self.embedding_model_key
        if not model_key:
            return 0
        if not self._semantic_cleanup_done:
            with self._lock, self._connect() as conn:
                conn.execute(
                    """DELETE FROM memory_embeddings
                       WHERE model_key = ? AND record_type = 'event'
                         AND record_id IN (
                           SELECT event_id FROM memory_events
                           WHERE trim(text) IN ('[图片附件]', '[未解析消息]')
                         )""",
                    (model_key,),
                )
            self._semantic_cleanup_done = True
        wanted = max(1, min(1000, int(limit or self.semantic_backfill_batch_size)))
        event_scope_clause = " AND m.scope = ?" if scope else ""
        fact_scope_clause = " AND f.scope = ?" if scope else ""
        scope_values: tuple[Any, ...] = (str(scope),) if scope else ()
        scan_limit = max(wanted, min(10000, wanted * 8))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""SELECT 'event' AS record_type, m.event_id AS record_id,
                          m.scope, m.text, m.created_at,
                          COALESCE(e.content_hash, '') AS indexed_hash,
                          e.embedding_id AS embedding_id
                    FROM memory_events m
                    LEFT JOIN memory_embeddings e
                      ON e.record_type = 'event' AND e.record_id = m.event_id
                     AND e.model_key = ?
                   WHERE m.status = 'active'
                     AND m.source_id NOT LIKE 'owner:agent-task-detail:%'{event_scope_clause}
                   ORDER BY m.created_at DESC LIMIT ?""",
                (model_key, *scope_values, scan_limit),
            ).fetchall()
            fact_rows = conn.execute(
                f"""SELECT 'fact' AS record_type, f.fact_id AS record_id,
                          f.scope, f.text, f.updated_at AS created_at,
                          COALESCE(e.content_hash, '') AS indexed_hash,
                          e.embedding_id AS embedding_id
                    FROM memory_facts f
                    LEFT JOIN memory_embeddings e
                      ON e.record_type = 'fact' AND e.record_id = f.fact_id
                     AND e.model_key = ?
                   WHERE f.status = 'active'{fact_scope_clause}
                   ORDER BY f.updated_at DESC LIMIT ?""",
                (model_key, *scope_values, scan_limit),
            ).fetchall()
            rows = [*rows, *fact_rows]
        fresh_rows = []
        for row in rows:
            body = str(row["text"] or "")
            with self._lock, self._connect() as conn:
                if not self._embedding_allowed(conn, str(row["record_type"]), str(row["record_id"])):
                    continue
            if body.strip() in {"[图片附件]", "[未解析消息]"}:
                continue
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if not row["embedding_id"] or str(row["indexed_hash"] or "") != digest:
                fresh_rows.append(row)
        rows = fresh_rows[:wanted]
        if not rows:
            return 0
        texts = [str(row["text"] or "")[:6000] for row in rows]
        try:
            with self._embedding_lock:
                vectors = self._embedding_provider.embed_texts(texts)
        except Exception:
            return 0
        dimension = int(getattr(self._embedding_provider, "dimension", 0) or 0)
        stamp = int(time.time())
        written = 0
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for row, vector in zip(rows, vectors):
                values = [float(value) for value in (vector or [])]
                if not values:
                    continue
                width = len(values)
                if dimension and width != dimension:
                    continue
                body = str(row["text"] or "")
                content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
                # Encoding ran outside the transaction. A forget, correction
                # or raw reclamation during that time revokes this snapshot.
                current = self.lifecycle._record(conn, str(row["record_type"]), str(row["record_id"]))
                if current is None or current["status"] != "active" or str(current["scope"]) != str(row["scope"]) or hashlib.sha256(str(current["text"]).encode("utf-8")).hexdigest() != content_hash:
                    self._discard_cached_text(body)
                    continue
                if not self._embedding_allowed(conn, str(row["record_type"]), str(row["record_id"])):
                    continue
                conn.execute(
                    """INSERT INTO memory_embeddings
                       (embedding_id, record_type, record_id, scope, model_key,
                        content_hash, dimension, vector_blob, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(record_type, record_id, model_key) DO UPDATE SET
                         scope=excluded.scope,
                         content_hash=excluded.content_hash,
                         dimension=excluded.dimension,
                         vector_blob=excluded.vector_blob,
                         updated_at=excluded.updated_at""",
                    (
                        uuid.uuid4().hex,
                        str(row["record_type"]),
                        str(row["record_id"]),
                        str(row["scope"]),
                        model_key,
                        content_hash,
                        width,
                        pack_vector(values),
                        int(row["created_at"] or stamp),
                        stamp,
                    ),
                )
                written += 1
        return written

    def _embedding_allowed(self, conn, record_type: str, record_id: str) -> bool:
        policy = self.lifecycle.policy
        if policy.mode != "active" or not policy.defer_l1_embeddings:
            return True
        state = conn.execute("SELECT tier,pinned FROM memory_lifecycle WHERE record_type=? AND record_id=?", (record_type, record_id)).fetchone()
        if state is None:
            row = self.lifecycle._record(conn, record_type, record_id)
            if not self.lifecycle._eligible(row, record_type):
                return True  # Unmanaged summaries/temporary layers keep their own indexing policy.
        return record_type == "fact" or (state is not None and (state["tier"] >= 2 or state["pinned"]))

    @staticmethod
    def _json(value: Any) -> str:
        try:
            return json.dumps(value if isinstance(value, dict) else {}, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return "{}"

    def _terms(self, query: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for term in _TERM_RE.findall(str(query or "").lower()):
            if any("\u4e00" <= char <= "\u9fff" for char in term):
                pieces = [term] if len(term) <= 4 else []
                pieces.extend(term[index : index + 2] for index in range(len(term) - 1))
            else:
                pieces = [term]
            for piece in pieces:
                if piece not in seen:
                    seen.add(piece)
                    terms.append(piece)
        return terms[: self.max_query_terms]

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
        text = str(content or "").strip()
        if not text or not source_id:
            return False
        stamp = int(created_at or time.time())
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO memory_turns
                    (turn_id, conversation_id, scope, role, content, created_at, source_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (uuid.uuid4().hex, str(conversation_id), str(scope), str(role or "user"), text, stamp, str(source_id), self._json(metadata)),
            )
            return cursor.rowcount > 0

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

        data = dict(decision or {})
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO memory_routing_audit
                   (audit_id, conversation_id, scope, created_at, history_mode,
                    need_retrieval, used_model, confidence, reason,
                    retrieved_count, injected_chars, decision_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex,
                    str(conversation_id),
                    str(scope),
                    int(created_at),
                    str(data.get("history_mode") or "none"),
                    1 if bool(data.get("need_retrieval")) else 0,
                    1 if bool(data.get("used_model")) else 0,
                    max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0))),
                    str(data.get("reason") or "")[:240],
                    max(0, int(retrieved_count or 0)),
                    max(0, int(injected_chars or 0)),
                    self._json(data),
                ),
            )
            return cursor.rowcount > 0

    def recent_routing_audits(self, *, scope: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        params: list[Any] = []
        clause = ""
        if scope:
            clause = " WHERE scope = ?"
            params.append(str(scope))
        params.append(max(1, min(200, int(limit or 20))))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_routing_audit{clause} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "audit_id": str(row["audit_id"] or ""),
                    "conversation_id": str(row["conversation_id"] or ""),
                    "scope": str(row["scope"] or ""),
                    "created_at": int(row["created_at"] or 0),
                    "history_mode": str(row["history_mode"] or "none"),
                    "need_retrieval": bool(row["need_retrieval"]),
                    "used_model": bool(row["used_model"]),
                    "confidence": float(row["confidence"] or 0.0),
                    "reason": str(row["reason"] or ""),
                    "retrieved_count": int(row["retrieved_count"] or 0),
                    "injected_chars": int(row["injected_chars"] or 0),
                    "decision": self._loads(row["decision_json"]),
                }
            )
        return result

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
        body = str(text or "").strip()
        if not body or not source_id:
            return False
        stamp = int(created_at or time.time())
        with self._lock, self._connect() as conn:
            if self.lifecycle.revoked(conn, scope=str(scope), source_id=str(source_id), text=body):
                return False
            meta = metadata if isinstance(metadata, dict) else {}
            parents = meta.get("source_record_ids", [])
            if not isinstance(parents, list) or len(parents) > 128:
                return False
            hashes = meta.get("source_content_hashes", {})
            if not isinstance(hashes, dict):
                return False
            for parent_id in parents:
                try:
                    parent_type, parent_bare = self.lifecycle._identity(str(parent_id))
                except ValueError:
                    return False
                parent = self.lifecycle._record(conn, parent_type, parent_bare)
                if parent is None or parent["status"] != "active" or str(parent["scope"]) != str(scope):
                    return False
                expected = hashes.get(str(parent_id))
                if expected and hashlib.sha256(str(parent["text"]).encode("utf-8")).hexdigest() != expected:
                    return False
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO memory_events
                    (event_id, conversation_id, scope, text, layer, created_at,
                     importance, confidence, expires_at, source_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    str(conversation_id),
                    str(scope),
                    body,
                    str(layer or "episodic"),
                    stamp,
                    max(0.0, min(1.0, float(importance))),
                    max(0.0, min(1.0, float(confidence))),
                    max(0, int(expires_at or 0)),
                    str(source_id),
                    self._json(metadata),
                ),
            )
            if cursor.rowcount > 0 and self.lifecycle.policy.enabled:
                row = conn.execute("SELECT * FROM memory_events WHERE source_id=?", (str(source_id),)).fetchone()
                self.lifecycle._enroll(conn, "event", row, int(self.lifecycle.wall_time()))
            return cursor.rowcount > 0

    def event_by_source_id(self, *, source_id: str, scope: str) -> MemoryRecord | None:
        """Read one active event by its stable source key inside one scope."""

        source = str(source_id or "").strip()
        selected_scope = str(scope or "").strip()
        if not source or not selected_scope:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_events WHERE source_id = ? AND scope = ? AND status = 'active' LIMIT 1",
                (source, selected_scope),
            ).fetchone()
        return self._event_record(row, score=1.0) if row is not None else None

    def prune_agent_task_details(self, *, scope: str, max_records: int) -> int:
        """Forget the oldest active detail archives past the configured cap."""

        selected_scope = str(scope or "").strip()
        cap = max(1, int(max_records or 1))
        if not selected_scope:
            return 0
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT event_id FROM memory_events "
                "WHERE scope = ? AND status = 'active' AND layer = 'episodic' "
                "AND source_id LIKE 'owner:agent-task-detail:%' "
                "ORDER BY created_at DESC, rowid DESC",
                (selected_scope,),
            ).fetchall()
            forgotten_ids = [str(row["event_id"]) for row in rows[cap:]]
            if not forgotten_ids:
                return 0
            conn.executemany(
                "UPDATE memory_events SET status = 'forgotten' WHERE event_id = ?",
                [(item,) for item in forgotten_ids],
            )
            conn.executemany(
                "DELETE FROM memory_embeddings WHERE record_type = 'event' AND record_id = ?",
                [(item,) for item in forgotten_ids],
            )
            return len(forgotten_ids)

    def promote_scope_to_owner(self, *, source_scope: str, owner_scope: str = "private:master") -> dict[str, int]:
        """Copy an already isolated private scope into the owner library.

        This is intentionally limited to a private scope whose identity is
        configured as the owner.  It never promotes group spectators, and the
        deterministic source prefix makes startup migration idempotent.
        """

        source_scope = str(source_scope or "").strip()
        owner_scope = str(owner_scope or "").strip()
        if not source_scope or not owner_scope or source_scope == owner_scope:
            return {"events": 0, "facts": 0}
        copied_events = 0
        copied_facts = 0
        with self._lock, self._connect() as conn:
            event_rows = conn.execute(
                "SELECT * FROM memory_events WHERE scope = ? AND status = 'active'",
                (source_scope,),
            ).fetchall()
            for row in event_rows:
                source_id = f"owner:import:{row['source_id']}"
                if self.lifecycle.revoked(conn, scope=owner_scope, source_id=source_id, text=row["text"]):
                    continue
                metadata = self._loads(row["metadata_json"])
                metadata.update({"owner_memory": True, "owner_source_scope": source_scope, "owner_import": True})
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_events
                        (event_id, conversation_id, scope, text, layer, created_at,
                         last_used_at, importance, confidence, expires_at, status,
                         source_id, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                    """,
                    (
                        uuid.uuid5(uuid.NAMESPACE_URL, source_id).hex,
                        "master",
                        owner_scope,
                        str(row["text"] or ""),
                        str(row["layer"] or "episodic"),
                        int(row["created_at"] or 0),
                        int(row["last_used_at"] or 0),
                        float(row["importance"] or 0.0),
                        float(row["confidence"] or 0.0),
                        int(row["expires_at"] or 0),
                        source_id,
                        self._json(metadata),
                    ),
                )
                copied_events += max(0, int(cursor.rowcount))
            fact_rows = conn.execute(
                "SELECT * FROM memory_facts WHERE scope = ? AND status = 'active'",
                (source_scope,),
            ).fetchall()
            for row in fact_rows:
                key = str(row["fact_key"] or "").strip()
                if not key:
                    continue
                source_id = f"owner:import:{row['source_id']}"
                if self.lifecycle.revoked(conn, scope=owner_scope, source_id=source_id, text=row["text"]):
                    continue
                prior = conn.execute("SELECT status,updated_at FROM memory_facts WHERE scope=? AND fact_key=?", (owner_scope, key)).fetchone()
                if prior is not None and (prior["status"] != "active" or prior["updated_at"] >= row["updated_at"]):
                    continue
                cursor = conn.execute(
                    """
                    INSERT INTO memory_facts
                        (fact_id, conversation_id, scope, fact_key, text, created_at,
                         updated_at, last_used_at, importance, confidence, status,
                         expires_at, source_id, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
                    ON CONFLICT(scope, fact_key) DO UPDATE SET
                        text=excluded.text,
                        conversation_id=excluded.conversation_id,
                        updated_at=excluded.updated_at,
                        last_used_at=excluded.last_used_at,
                        importance=excluded.importance,
                        confidence=excluded.confidence,
                        status='active',
                        expires_at=excluded.expires_at,
                        source_id=excluded.source_id,
                        metadata_json=excluded.metadata_json
                    """,
                    (
                        uuid.uuid5(uuid.NAMESPACE_URL, f"owner:fact:{source_scope}:{key}").hex,
                        "master",
                        owner_scope,
                        key,
                        str(row["text"] or ""),
                        int(row["created_at"] or 0),
                        int(row["updated_at"] or 0),
                        int(row["last_used_at"] or 0),
                        float(row["importance"] or 0.0),
                        float(row["confidence"] or 0.0),
                        int(row["expires_at"] or 0),
                        f"owner:import:{row['source_id']}",
                        row["metadata_json"],
                    ),
                )
                copied_facts += max(0, int(cursor.rowcount))
        return {"events": copied_events, "facts": copied_facts}

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
        stamp = int(now or time.time())
        with self._lock, self._connect() as conn:
            if self.lifecycle.revoked(conn, scope=str(scope), source_id=str(source_id), text=str(text).strip()):
                return
            conn.execute(
                """
                INSERT INTO memory_facts
                    (fact_id, conversation_id, scope, fact_key, text, created_at, updated_at,
                     importance, confidence, expires_at, source_id, metadata_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
                ON CONFLICT(scope, fact_key) DO UPDATE SET
                    text=excluded.text,
                    conversation_id=excluded.conversation_id,
                    updated_at=excluded.updated_at,
                    importance=excluded.importance,
                    confidence=excluded.confidence,
                    expires_at=excluded.expires_at,
                    source_id=excluded.source_id,
                    metadata_json=excluded.metadata_json,
                    status='active'
                """,
                (
                    uuid.uuid4().hex,
                    str(conversation_id),
                    str(scope),
                    str(fact_key),
                    str(text).strip(),
                    stamp,
                    stamp,
                    max(0.0, min(1.0, float(importance))),
                    max(0.0, min(1.0, float(confidence))),
                    max(0, int(expires_at or 0)),
                    str(source_id),
                    self._json(metadata or {}),
                ),
            )

            if self.lifecycle.policy.enabled:
                row = conn.execute("SELECT * FROM memory_facts WHERE scope=? AND fact_key=?", (str(scope), str(fact_key))).fetchone()
                self.lifecycle._enroll(conn, "fact", row, stamp)

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

        query_text = str(query or "").strip()
        terms = self._terms(query_text)
        if not terms and not (self.semantic_enabled and query_text):
            return []
        stamp = int(now or time.time())
        self.expire(now=stamp)
        selected_layers = tuple(dict.fromkeys(str(item).strip() for item in (layers or ()) if str(item).strip()))
        layer_clause = ""
        layer_values: list[str] = []
        if selected_layers:
            layer_clause = " AND layer IN (" + ",".join("?" for _ in selected_layers) + ")"
            layer_values = list(selected_layers)
        time_clause = ""
        time_values: list[int] = []
        if start_at is not None:
            time_clause += " AND created_at >= ?"
            time_values.append(int(start_at))
        if end_at is not None:
            time_clause += " AND created_at < ?"
            time_values.append(int(end_at))
        with self._lock, self._connect() as conn:
            event_rows = conn.execute(
                f"SELECT * FROM memory_events WHERE scope = ? AND status = 'active' "
                f"AND source_id NOT LIKE 'owner:agent-task-detail:%'{layer_clause}{time_clause} "
                f"ORDER BY created_at DESC LIMIT {self.event_scan_limit}",
                (str(scope), *layer_values, *time_values),
            ).fetchall()
            fact_rows = () if selected_layers or start_at is not None or end_at is not None else conn.execute(
                f"SELECT * FROM memory_facts WHERE scope = ? AND status = 'active' ORDER BY updated_at DESC LIMIT {self.fact_scan_limit}",
                (str(scope),),
            ).fetchall()
        candidates: list[MemoryRecord] = []
        for row in event_rows:
            if require_term_overlap and terms:
                body = str(row["text"] or "").lower()
                if not any(term in body for term in terms):
                    continue
            candidates.append(
                self._score_event(
                    row,
                    terms,
                    stamp,
                    topic_batch_id=topic_batch_id,
                    sender_id=sender_id,
                )
            )
        for row in fact_rows:
            candidates.append(self._score_fact(row, terms, stamp))
        candidates = [item for item in candidates if item.score > 0.0]

        # Exact/date-bounded recall stays lexical by design.  Semantic search
        # is a second candidate source for ordinary retrieve queries; it never
        # gets to replace the time and scope guards above.
        if self.semantic_enabled and self._embedding_provider is not None and not require_term_overlap and query_text:
            self.backfill_embeddings(limit=self.semantic_backfill_batch_size, scope=str(scope))
            candidates.extend(
                self._semantic_candidates(
                    query=query_text,
                    terms=terms,
                    scope=str(scope),
                    selected_layers=selected_layers,
                    start_at=start_at,
                    end_at=end_at,
                    topic_batch_id=topic_batch_id,
                    sender_id=sender_id,
                    now=stamp,
                )
            )

        # The same row can arrive from lexical and semantic retrieval.  Keep
        # the better score, and retain the provenance marker in metadata so
        # the debug hub can explain why a memory was selected.
        merged: dict[str, MemoryRecord] = {}
        for item in candidates:
            previous = merged.get(item.record_id)
            is_hybrid = item.metadata.get("retrieval_method") == "hybrid"
            previous_is_hybrid = previous is not None and previous.metadata.get("retrieval_method") == "hybrid"
            if previous is None or is_hybrid or (not previous_is_hybrid and item.score > previous.score):
                merged[item.record_id] = item
        candidates = list(merged.values())
        if getattr(self, "search_keys_enabled", False) and self.search_index is not None:
            try:
                self.search_keys_last_status = self.search_index.rebuild(scope=str(scope), now=stamp, limit=self.event_scan_limit, dry_run=False)
                matches = self.search_index.matches(records=candidates, query=query_text, scope=str(scope), now=stamp)
            except Exception as exc:
                matches = {}  # Source search still works if the derived index is unavailable.
                self.search_keys_last_status = {"scope": str(scope), "fallback": "source_search", "error_type": type(exc).__name__, "remote_calls": 0}
            candidates = [replace(item, score=item.score + min(0.30, 0.15 * len(matches[item.record_id])),
                          metadata={**item.metadata, "search_hints": list(matches[item.record_id]),
                                    "search_generator": "local-topic-hints-v1"})
                          if item.record_id in matches else item for item in candidates]
            # Bind ALL returned sources, not just boosted hits, to this reading version.
            fresh_candidates = []
            fresh_sources = self.search_index.read_sources(requests=tuple(
                (item.record_id, str(scope), "") for item in candidates), now=stamp)
            for item, fresh in zip(candidates, fresh_sources):
                if (fresh is not None and fresh.text == item.text and fresh.source_id == item.source_id
                        and fresh.created_at == item.created_at and fresh.layer == item.layer):
                    # Never overwrite current provenance with an older candidate.
                    diagnostics = {key: value for key, value in item.metadata.items()
                                   if key in {"search_hints", "search_generator", "retrieval_method",
                                              "semantic_similarity", "lexical_score"}}
                    fresh_candidates.append(replace(fresh, score=item.score, metadata={**fresh.metadata, **diagnostics}))
            candidates = fresh_candidates
        candidates.sort(key=lambda item: (item.score, item.importance, item.created_at), reverse=True)
        selected = candidates[: max(1, min(self.max_results, int(limit or 5)))]
        if selected:
            with self._lock, self._connect() as conn:
                ids = [item.record_id for item in selected]
                conn.executemany(
                    "UPDATE memory_events SET last_used_at = ? WHERE event_id = ?",
                    [(stamp, item.removeprefix("event:")) for item in ids if item.startswith("event:")],
                )
                conn.executemany(
                    "UPDATE memory_facts SET last_used_at = ? WHERE fact_id = ?",
                    [(stamp, item.record_id.removeprefix("fact:")) for item in selected if item.record_id.startswith("fact:")],
                )
        return selected

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

        model_key = self.embedding_model_key
        if not model_key:
            return []
        try:
            with self._embedding_lock:
                query_vector = self._embedding_provider.embed_text(query)
        except Exception:
            return []
        if not query_vector:
            return []
        layer_clause = ""
        layer_values: list[Any] = []
        if selected_layers:
            layer_clause = " AND m.layer IN (" + ",".join("?" for _ in selected_layers) + ")"
            layer_values = list(selected_layers)
        time_clause = ""
        time_values: list[Any] = []
        if start_at is not None:
            time_clause += " AND m.created_at >= ?"
            time_values.append(int(start_at))
        if end_at is not None:
            time_clause += " AND m.created_at < ?"
            time_values.append(int(end_at))
        with self._lock, self._connect() as conn:
            event_rows = conn.execute(
                f"""SELECT m.*, e.vector_blob, e.dimension,e.content_hash AS indexed_hash
                    FROM memory_events m
                    JOIN memory_embeddings e
                      ON e.record_type = 'event' AND e.record_id = m.event_id
                     AND e.model_key = ?
                   WHERE m.scope = ? AND m.status = 'active'
                     AND m.source_id NOT LIKE 'owner:agent-task-detail:%'
                     AND trim(m.text) NOT IN ('[图片附件]', '[未解析消息]'){layer_clause}{time_clause}
                   ORDER BY m.created_at DESC LIMIT ?""",
                (model_key, scope, *layer_values, *time_values, self.event_scan_limit),
            ).fetchall()
            fact_rows: tuple[Any, ...] = ()
            if not selected_layers and start_at is None and end_at is None:
                fact_rows = tuple(
                    conn.execute(
                        """SELECT f.*, e.vector_blob, e.dimension,e.content_hash AS indexed_hash
                           FROM memory_facts f
                           JOIN memory_embeddings e
                             ON e.record_type = 'fact' AND e.record_id = f.fact_id
                            AND e.model_key = ?
                          WHERE f.scope = ? AND f.status = 'active'
                          ORDER BY f.updated_at DESC LIMIT ?""",
                        (model_key, scope, self.fact_scan_limit),
                    ).fetchall()
                )
            l1_rows = []
            if self.lifecycle.policy.mode == "active" and self.lifecycle.policy.defer_l1_embeddings:
                l1_rows = conn.execute(f"""SELECT m.* FROM memory_events m JOIN memory_lifecycle l
                  ON l.record_type='event' AND l.record_id=m.event_id
                  WHERE m.scope=? AND m.status='active' AND l.tier=1 AND l.pinned=0
                  AND trim(m.text) NOT IN ('[图片附件]','[未解析消息]'){layer_clause}{time_clause}
                  ORDER BY m.created_at DESC LIMIT 64""", (scope, *layer_values, *time_values)).fetchall()
        semantic: list[MemoryRecord] = []
        for row in event_rows:
            with self._lock, self._connect() as conn:
                if not self._embedding_allowed(conn, "event", str(row["event_id"])):
                    continue
            if hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest() != row["indexed_hash"]:
                continue
            vector = unpack_vector(row["vector_blob"], dimension=int(row["dimension"] or 0))
            similarity = cosine_similarity(query_vector, vector)
            if similarity < self.semantic_min_score:
                continue
            lexical = self._score_event(row, terms, now, topic_batch_id=topic_batch_id, sender_id=sender_id)
            semantic.append(self._hybrid_record(lexical, similarity, terms=terms))
        for row in fact_rows:
            if hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest() != row["indexed_hash"]:
                continue
            vector = unpack_vector(row["vector_blob"], dimension=int(row["dimension"] or 0))
            similarity = cosine_similarity(query_vector, vector)
            if similarity < self.semantic_min_score:
                continue
            lexical = self._score_fact(row, terms, now)
            semantic.append(self._hybrid_record(lexical, similarity, terms=terms))
        # Batch the bounded recent L1 window. One encode per row made a cold
        # 64-record recall disproportionately expensive on CPU.
        missing = []
        with self._lock:
            for row in l1_rows:
                content_hash = hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
                key = (scope, str(row["event_id"]), model_key, content_hash)
                if key not in self._l1_semantic_cache:
                    missing.append((row, key, content_hash))
        if missing:
            try:
                with self._embedding_lock:
                    fresh_vectors = self._embedding_provider.embed_texts([str(row["text"])[:6000] for row, _, _ in missing])
            except Exception:
                fresh_vectors = []
            with self._lock, self._connect() as conn:
                for (row, key, content_hash), vector in zip(missing, fresh_vectors):
                    current = self.lifecycle._record(conn, "event", str(row["event_id"]))
                    if current is None or current["status"] != "active" or hashlib.sha256(str(current["text"]).encode("utf-8")).hexdigest() != content_hash:
                        self._discard_cached_text(str(row["text"]))
                        continue
                    self._l1_semantic_cache[key] = list(vector)
                while len(self._l1_semantic_cache) > 128:
                    self._l1_semantic_cache.pop(next(iter(self._l1_semantic_cache)))
        for row in l1_rows:
            content_hash = hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
            key = (scope, str(row["event_id"]), model_key, content_hash)
            with self._lock:
                vector = self._l1_semantic_cache.get(key)
            if vector is None:
                continue
            similarity = cosine_similarity(query_vector, vector)
            if similarity >= self.semantic_min_score:
                lexical = self._score_event(row, terms, now, topic_batch_id=topic_batch_id, sender_id=sender_id)
                match = self._hybrid_record(lexical, similarity, terms=terms)
                semantic.append(replace(match, metadata={**match.metadata, "semantic_cache": "ephemeral_l1"}))
        return semantic

    @staticmethod
    def _hybrid_record(record: MemoryRecord, similarity: float, *, terms: list[str]) -> MemoryRecord:
        """Blend normalized lexical evidence with semantic similarity."""

        lexical_ceiling = max(1.0, len(terms) * 0.55 + 0.45)
        lexical_score = max(0.0, min(1.0, record.score / lexical_ceiling))
        prior = max(0.0, min(1.0, record.importance * max(record.confidence, 0.5)))
        score = max(0.0, min(1.0, similarity * 0.58 + lexical_score * 0.27 + prior * 0.15))
        metadata = dict(record.metadata)
        metadata["retrieval_method"] = "hybrid"
        metadata["semantic_similarity"] = round(float(similarity), 6)
        metadata["lexical_score"] = round(float(lexical_score), 6)
        return replace(record, score=score, metadata=metadata)

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

        center = int(center_at)
        radius = max(1, int(window_seconds or 600))
        wanted = max(1, min(self.max_results, int(limit or 6)))
        selected_layers = tuple(dict.fromkeys(str(item).strip() for item in layers if str(item).strip()))
        if not selected_layers:
            return []
        placeholders = ",".join("?" for _ in selected_layers)
        lower = center - radius if start_at is None else max(center - radius, int(start_at))
        upper = center + radius if end_at is None else min(center + radius, int(end_at) - 1)
        if upper < lower:
            return []
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM memory_events
                    WHERE scope = ? AND status = 'active'
                      AND source_id NOT LIKE 'owner:agent-task-detail:%'
                      AND layer IN ({placeholders})
                      AND created_at >= ? AND created_at <= ?
                    ORDER BY ABS(created_at - ?) ASC, created_at DESC, event_id DESC
                    LIMIT ?""",
                (str(scope), *selected_layers, lower, upper, center, wanted),
            ).fetchall()
        records = [
            self._event_record(
                row,
                score=max(0.0, 1.0 - abs(center - int(row["created_at"] or center)) / max(1, radius) * 0.25),
            )
            for row in rows
        ]
        records.sort(key=lambda item: (item.created_at, item.record_id), reverse=True)
        return records

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

        stamp = int(now or time.time())
        age = max(1, int(max_age_seconds or 900))
        selected_layers = tuple(dict.fromkeys(str(item).strip() for item in layers if str(item).strip()))
        if not selected_layers:
            return []
        placeholders = ",".join("?" for _ in selected_layers)
        wanted = max(1, min(self.max_results, int(limit or 6)))
        # Fetch a bounded overscan because role filtering happens after JSON
        # metadata decoding.  This keeps recent user/observation messages
        # available even when several assistant turns are newest.
        scan = max(wanted, min(self.max_results * 8, wanted * 8))
        params: list[Any] = [str(scope), *selected_layers, stamp - age, scan]
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM memory_events
                    WHERE scope = ? AND status = 'active'
                      AND source_id NOT LIKE 'owner:agent-task-detail:%'
                      AND layer IN ({placeholders}) AND created_at >= ?
                    ORDER BY created_at DESC LIMIT ?""",
                params,
            ).fetchall()
        records: list[MemoryRecord] = []
        for row in rows:
            metadata = self._loads(row["metadata_json"])
            if not include_assistant and str(metadata.get("role") or "").strip().lower() == "assistant":
                continue
            same_topic = bool(topic_batch_id and metadata.get("topic_batch_id") == topic_batch_id)
            age_ratio = max(0.0, min(1.0, (stamp - int(row["created_at"] or stamp)) / age))
            records.append(self._event_record(row, score=(1.0 - age_ratio * 0.35) + (0.25 if same_topic else 0.0)))
            if len(records) >= wanted:
                break
        return records

    def confirm_assistant_delivery(
        self, *, scope: str, source_id: str, expected_text: str, delivered_at: int,
        status: str = "sent", requester_name: str = "", proof_id: str = "",
        content_complete: bool = False,
    ) -> bool:
        """Attach a transport receipt to an existing, active assistant source.

        No new memory or raw speech is copied. Dry runs/total failures never
        call this; partial sends prove occurrence, not the complete body.
        """
        if not scope.startswith("group:") or status not in {"sent", "partial"}:
            return False
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM memory_events WHERE scope=? AND source_id=? AND status='active' AND layer='episodic'",
                               (scope, source_id)).fetchone()
            if row is None or row["text"] != expected_text or int(row["created_at"]) > delivered_at:
                return False
            meta = self._loads(row["metadata_json"])
            if (meta.get("role") != "assistant" or meta.get("source") != "qq_turn"
                    or not meta.get("self_id") or meta.get("sender_id") != meta.get("self_id")
                    or not source_id.startswith(f"turn:{row['conversation_id']}:")
                    or not source_id.endswith(":assistant")
                    or self.lifecycle.revoked(conn, scope=scope, source_id=source_id, text=row["text"])):
                return False
            if meta.get("delivery_status") == "sent" and meta.get("delivery_proof") == "qq_transport":
                return True
            meta.update({"delivery_status": status, "delivered_at": int(delivered_at),
                         "delivery_content_complete": bool(content_complete and status == "sent"),
                         "delivery_proof": "qq_transport", "delivery_proof_id": str(proof_id)[:160]})
            if requester_name:
                meta["requester_name"] = str(requester_name).strip()[:80]
            encoded = json.dumps(meta, ensure_ascii=False)
            conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?", (encoded, row["event_id"]))
            conn.execute("UPDATE memory_turns SET metadata_json=? WHERE scope=? AND source_id=? AND role='assistant'",
                         (encoded, scope, source_id))
            return True

    def delivered_group_activity(
        self, *, start_at: int, end_at: int, now: int, scope: str = "",
        self_id: str = "", limit: int = 8,
    ) -> list[MemoryRecord]:
        """Candidates are ONLY the bot's receipt-backed group replies.

        The service must authenticate private cross-group readers. No user
        messages, observations, summaries, facts, or private turns are selected.
        """
        wanted = max(1, min(12, int(limit)))
        scope_clause = " AND scope=?" if scope else ""
        bot_clause = " AND json_extract(metadata_json,'$.self_id')=?" if self_id else ""
        params = [int(start_at), int(end_at), int(now), *([scope] if scope else []),
                  *([self_id] if self_id else []), min(96, wanted * 8)]
        with self._lock, self._connect() as conn:
            rows = conn.execute("""SELECT * FROM memory_events
                WHERE scope LIKE 'group:%' AND status='active' AND layer='episodic'
                  AND json_extract(metadata_json,'$.role')='assistant'
                  AND json_extract(metadata_json,'$.source')='qq_turn'
                  AND json_extract(metadata_json,'$.delivery_proof')='qq_transport'
                  AND json_extract(metadata_json,'$.delivery_status') IN ('sent','partial')
                  AND json_extract(metadata_json,'$.delivered_at') BETWEEN ? AND ?
                  AND (expires_at=0 OR expires_at>?)
                  AND json_extract(metadata_json,'$.self_id') IS NOT NULL
                  AND json_extract(metadata_json,'$.self_id') != ''
                  AND json_extract(metadata_json,'$.sender_id')=json_extract(metadata_json,'$.self_id')
                """ + scope_clause + bot_clause + " ORDER BY json_extract(metadata_json,'$.delivered_at') DESC,source_id DESC LIMIT ?", params).fetchall()
            return [self._event_record(row, score=1.0) for row in rows
                    if not self.lifecycle.revoked(conn, scope=row["scope"], source_id=row["source_id"], text=row["text"])][:wanted]

    def recent_assistant_events(
        self, *, scope: str, now: int, max_age_seconds: int = 1800, limit: int = 48
    ) -> list[MemoryRecord]:
        """Fetch Shinku's own active QQ replies for same-scope fact checking.

        This is deliberately separate from ``recent_events``: generated replies
        must not become general conversational or voice examples.
        """

        wanted = max(1, min(100, int(limit)))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM memory_events
                   WHERE scope = ? AND status = 'active' AND layer = 'episodic'
                     AND created_at BETWEEN ? AND ?
                     AND json_extract(metadata_json, '$.role') = 'assistant'
                   ORDER BY created_at DESC LIMIT ?""",
                (str(scope), int(now) - max(1, int(max_age_seconds)), int(now), wanted),
            ).fetchall()
        return [
            self._event_record(row, score=0.0)
            for row in rows
            if str(self._loads(row["metadata_json"]).get("role") or "").lower() == "assistant"
        ]

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

        stamp = int(now or time.time())
        self.expire(now=stamp)
        wanted = max(1, min(self.max_results, int(limit or 8)))
        selected_layers = tuple(dict.fromkeys(str(item).strip() for item in (layers or ()) if str(item).strip()))
        layer_clause = ""
        layer_values: list[str] = []
        if selected_layers:
            layer_clause = " AND layer IN (" + ",".join("?" for _ in selected_layers) + ")"
            layer_values = list(selected_layers)
        time_clause = ""
        time_values: list[int] = []
        if start_at is not None:
            time_clause += " AND created_at >= ?"
            time_values.append(int(start_at))
        if end_at is not None:
            time_clause += " AND created_at < ?"
            time_values.append(int(end_at))
        scan = max(wanted * 4, wanted)
        with self._lock, self._connect() as conn:
            event_rows = conn.execute(
                f"""SELECT * FROM memory_events
                    WHERE scope = ? AND status = 'active'
                      AND source_id NOT LIKE 'owner:agent-task-detail:%'{layer_clause}{time_clause}
                    ORDER BY created_at DESC, event_id DESC LIMIT ?""",
                (str(scope), *layer_values, *time_values, scan),
            ).fetchall()
            fact_rows = ()
            if include_facts and start_at is None and end_at is None:
                fact_rows = conn.execute(
                    """SELECT * FROM memory_facts
                       WHERE scope = ? AND status = 'active'
                       ORDER BY updated_at DESC, fact_id DESC LIMIT ?""",
                    (str(scope), scan),
                ).fetchall()
        records = [
            self._event_record(
                row,
                score=max(0.0, 1.0 - max(0, stamp - int(row["created_at"] or stamp)) / 31536000.0),
            )
            for row in event_rows
        ]
        records.extend(self._fact_record(row, score=0.8) for row in fact_rows)
        records.sort(key=lambda item: (item.created_at, item.importance, item.confidence), reverse=True)
        return records[:wanted]

    def latest_observation(self, *, scope: str, before_created_at: int | None = None) -> MemoryRecord | None:
        params: list[Any] = [str(scope)]
        clause = ""
        if before_created_at is not None:
            clause = " AND created_at <= ?"
            params.append(int(before_created_at))
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"""SELECT * FROM memory_events
                    WHERE scope = ? AND status = 'active' AND layer = 'observation'{clause}
                    ORDER BY created_at DESC, event_id DESC LIMIT 1""",
                params,
            ).fetchone()
        return self._event_record(row, score=1.0) if row is not None else None

    def observation_for_message(self, *, scope: str, message_id: str) -> MemoryRecord | None:
        wanted = str(message_id or "").strip()
        if not wanted:
            return None
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM memory_events
                   WHERE scope = ? AND status = 'active' AND layer = 'observation'
                   ORDER BY created_at DESC LIMIT ?""",
                (str(scope), self.event_scan_limit),
            ).fetchall()
        for row in rows:
            metadata = self._loads(row["metadata_json"])
            if str(metadata.get("message_id") or "").strip() == wanted:
                return self._event_record(row, score=1.0)
        return None

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

        stamp = int(now or time.time())
        current_day = time.strftime("%Y-%m-%d", time.localtime(stamp))
        layers = tuple(dict.fromkeys(str(item).strip() for item in source_layers if str(item).strip()))
        if not layers:
            return 0
        layer_clause = " AND layer IN (" + ",".join("?" for _ in layers) + ")"
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"""SELECT * FROM memory_events
                   WHERE scope = ? AND status = 'active'
                     AND source_id NOT LIKE 'owner:agent-task-detail:%'{layer_clause}
                   ORDER BY created_at ASC""",
                (str(scope), *layers),
            ).fetchall()
        grouped: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            day = time.strftime("%Y-%m-%d", time.localtime(int(row["created_at"] or stamp)))
            if day < current_day:
                grouped.setdefault(day, []).append(row)
        created = 0
        for day, day_rows in grouped.items():
            lines: list[str] = []
            included_rows = day_rows[: max(1, min(100, int(max_lines or 60)))]
            for row in included_rows:
                metadata = self._loads(row["metadata_json"])
                role = str(metadata.get("role") or "").strip().lower()
                sender = str(
                    metadata.get("sender_name")
                    or metadata.get("sender_id")
                    or ("真红" if role == "assistant" else "用户" if role == "user" else "群成员")
                ).strip()
                text = str(row["text"] or "").replace("\n", " ").strip()
                if text:
                    lines.append(f"{sender}: {text[:160]}")
            if not lines:
                continue
            scope_label = (
                "群聊摘要" if str(scope).startswith("group:")
                else "私聊摘要" if str(scope).startswith("private:")
                else "会话摘要"
            )
            summary = f"{day} {scope_label}（{len(day_rows)} 条记录）\n" + "\n".join(lines)
            if self.record_event(
                conversation_id=str(day_rows[0]["conversation_id"] or ""),
                scope=scope,
                text=summary[:12000],
                source_id=f"summary:{scope}:{day}",
                layer="summary",
                created_at=int(day_rows[-1]["created_at"] or stamp),
                importance=0.62,
                confidence=0.72,
                expires_at=0,
                metadata={"summary_day": day, "source": "local_daily_rollup", "observation_count": len(day_rows), "source_record_ids": [f"event:{row['event_id']}" for row in included_rows], "source_content_hashes": {f"event:{row['event_id']}": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest() for row in included_rows}},
            ):
                created += 1
        return created

    def _event_record(self, row: sqlite3.Row, *, score: float) -> MemoryRecord:
        return MemoryRecord(
            record_id=f"event:{row['event_id']}",
            layer=str(row["layer"] or "episodic"),
            scope=str(row["scope"] or ""),
            text=str(row["text"] or ""),
            created_at=int(row["created_at"] or 0),
            score=float(score),
            importance=float(row["importance"] or 0.0),
            confidence=float(row["confidence"] or 0.0),
            source_id=str(row["source_id"] or ""),
            metadata=self._loads(row["metadata_json"]),
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
        text = str(row["text"] or "")
        metadata = self._loads(row["metadata_json"])
        overlap = sum(1 for term in terms if term in text.lower())
        age_days = max(0.0, (now - int(row["created_at"] or now)) / 86400.0)
        decay = 0.5 ** (age_days / 30.0)
        score = overlap * 0.55 + float(row["importance"] or 0.0) * decay * 0.45
        if topic_batch_id and metadata.get("topic_batch_id") == topic_batch_id:
            score += 0.35
        if sender_id and str(metadata.get("sender_id") or "") == str(sender_id):
            score += 0.10
        return self._event_record(row, score=score)

    def _score_fact(self, row: sqlite3.Row, terms: list[str], now: int) -> MemoryRecord:
        text = str(row["text"] or "")
        overlap = sum(1 for term in terms if term in text.lower())
        score = overlap * 0.7 + float(row["importance"] or 0.0) * float(row["confidence"] or 0.0) * 0.3
        return self._fact_record(row, score=score)

    def _fact_record(self, row: sqlite3.Row, *, score: float) -> MemoryRecord:
        text = str(row["text"] or "")
        return MemoryRecord(
            record_id=f"fact:{row['fact_id']}",
            layer="fact",
            scope=str(row["scope"] or ""),
            text=text,
            created_at=int(row["updated_at"] or 0),
            score=score,
            importance=float(row["importance"] or 0.0),
            confidence=float(row["confidence"] or 0.0),
            source_id=str(row["source_id"] or ""),
            metadata={**(self._loads(row["metadata_json"]) if "metadata_json" in row.keys() else {}),
                      "fact_key": str(row["fact_key"] or "")},
        )

    def expire(self, *, now: int | None = None) -> int:
        stamp = int(now or time.time())
        with self._lock, self._connect() as conn:
            lifecycle_clause = " AND NOT (layer IN ('observation','episodic','legacy_raw','legacy_lesson') AND source_id NOT LIKE 'owner:agent-task-%')" if self.lifecycle.policy.mode == "active" else ""
            cursor = conn.execute(
                f"""
                UPDATE memory_events SET status='expired'
                WHERE status='active' AND expires_at > 0 AND expires_at <= ?{lifecycle_clause}
                """,
                (stamp,),
            )
            facts = conn.execute("UPDATE memory_facts SET status='expired' WHERE status='active' AND expires_at>0 AND expires_at<=?", (stamp,))
            return int(cursor.rowcount or 0) + int(facts.rowcount or 0)

    def forget(self, *, query: str, scope: str, purge: bool = False, now: int | None = None, record_id: str = "") -> tuple[int, int]:
        terms = self._terms(query)
        exact_kind, exact_bare = self.lifecycle._identity(record_id) if record_id else ("", "")
        if not terms and not record_id:
            return 0, 0
        stamp = int(now or time.time())
        forgotten = 0
        purged = 0
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT event_id,text,layer,metadata_json,source_id FROM memory_events WHERE scope=? AND status IN ('active','expired')",
                (str(scope),),
            ).fetchall()
            fact_rows = conn.execute(
                "SELECT fact_id,text,source_id FROM memory_facts WHERE scope=? AND status IN ('active','expired')",
                (str(scope),),
            ).fetchall()
            # A summary/retired event may no longer contain the forgotten
            # term. Its separately retained scoped source turn proves linkage.
            turn_matches = []
            if terms and not record_id:
                turn_matches = [row for row in conn.execute("SELECT source_id,content FROM memory_turns WHERE scope=?", (str(scope),))
                                if all(term in str(row["content"]).lower() for term in terms)]
                for turn in turn_matches:
                    self.lifecycle.revoke(conn, scope=str(scope), source_id=turn["source_id"], text=turn["content"], now=stamp)
            source_ids = {str(row["source_id"]) for row in turn_matches}
            event_ids = [str(row["event_id"]) for row in rows if
                         exact_kind == "event" and row["event_id"] == exact_bare] if record_id else [
                         str(row["event_id"]) for row in rows if all(term in str(row["text"]).lower() for term in terms) or row["source_id"] in source_ids]
            fact_ids = [str(row["fact_id"]) for row in fact_rows if exact_kind == "fact" and row["fact_id"] == exact_bare] if record_id else [
                        str(row["fact_id"]) for row in fact_rows if all(term in str(row["text"]).lower() for term in terms) or row["source_id"] in source_ids]
            matched_event_ids = set(event_ids)
            matched_task_ids = {
                str(metadata.get("task_id") or "")
                for row in rows
                if str(row["event_id"]) in matched_event_ids
                for metadata in (self._loads(row["metadata_json"]),)
                if metadata.get("record_type") in {"agent_task_summary", "agent_task_detail"}
                and str(metadata.get("task_id") or "").strip()
            }
            if matched_task_ids:
                linked_ids = {
                    str(row["event_id"])
                    for row in rows
                    for metadata in (self._loads(row["metadata_json"]),)
                    if metadata.get("record_type") in {"agent_task_summary", "agent_task_detail"}
                    and str(metadata.get("task_id") or "") in matched_task_ids
                }
                event_ids = list(dict.fromkeys((*event_ids, *linked_ids)))
            # Source-linked summaries may omit the original query. Remove
            # those derivatives by provenance, not by similarity or scope mixing.
            while True:
                known = set(event_ids)
                known_records = {f"event:{item}" for item in event_ids} | {f"fact:{item}" for item in fact_ids}
                linked = [str(row["event_id"]) for row in rows
                          for metadata in (self._loads(row["metadata_json"]),)
                          for parents in (metadata.get("source_record_ids", []),)
                          if (isinstance(parents, list) and known_records.intersection(str(item) for item in parents))
                          or (known_records and metadata.get("source") == "local_daily_rollup" and not parents)]
                event_ids = list(dict.fromkeys((*event_ids, *linked)))
                if set(event_ids) == known:
                    break
            for record_type, selected_ids, selected_rows, pk in (("event", event_ids, rows, "event_id"), ("fact", fact_ids, fact_rows, "fact_id")):
                for row in selected_rows:
                    rid = str(row[pk])
                    if rid not in selected_ids:
                        continue
                    state = conn.execute("SELECT summary_source_hash FROM memory_lifecycle WHERE record_type=? AND record_id=?", (record_type, rid)).fetchone()
                    self.lifecycle.revoke(conn, scope=str(scope), source_id=row["source_id"], text=row["text"], now=stamp, original_hash=state["summary_source_hash"] if state else "")
                    conn.execute("DELETE FROM memory_lifecycle WHERE record_type=? AND record_id=?", (record_type, rid))
                    conn.execute("DELETE FROM memory_lifecycle_evidence WHERE record_type=? AND record_id=?", (record_type, rid))
                    conn.execute("DELETE FROM memory_embeddings WHERE record_type=? AND record_id=?", (record_type, rid))
                    if record_type == "event":
                        self._l1_semantic_cache = {key: value for key, value in self._l1_semantic_cache.items() if key[1] != rid}
            if purge:
                conn.executemany("DELETE FROM memory_events WHERE event_id = ?", [(item,) for item in event_ids])
                conn.executemany("DELETE FROM memory_facts WHERE fact_id = ?", [(item,) for item in fact_ids])
                conn.executemany("DELETE FROM memory_embeddings WHERE record_type = 'event' AND record_id = ?", [(item,) for item in event_ids])
                conn.executemany("DELETE FROM memory_embeddings WHERE record_type = 'fact' AND record_id = ?", [(item,) for item in fact_ids])
                purged = len(event_ids) + len(fact_ids)
            else:
                conn.executemany("UPDATE memory_events SET status='forgotten' WHERE event_id = ?", [(item,) for item in event_ids])
                conn.executemany("UPDATE memory_facts SET status='forgotten', updated_at=? WHERE fact_id = ?", [(stamp, item) for item in fact_ids])
                forgotten = len(event_ids) + len(fact_ids)
            conn.executemany(
                "INSERT INTO memory_forget_log (forget_id, scope, query, record_id, layer, created_at, action) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (uuid.uuid4().hex, str(scope), "sha256:" + hashlib.sha256(str(query).encode("utf-8")).hexdigest(), item, "event", stamp, "purge" if purge else "forget")
                    for item in event_ids
                ]
                + [
                    (uuid.uuid4().hex, str(scope), "sha256:" + hashlib.sha256(str(query).encode("utf-8")).hexdigest(), item, "fact", stamp, "purge" if purge else "forget")
                    for item in fact_ids
                ],
            )
        with self._lock:
            self._clear_derived_caches(scope=str(scope), record_ids={*(f"event:{item}" for item in event_ids), *(f"fact:{item}" for item in fact_ids)})
        return forgotten, purged

    def reset_all(self) -> dict[str, int]:
        """Clear every memory record while keeping the database schema.

        This is intentionally a low-level destructive operation.  The private
        command layer is responsible for authorization and explicit confirmation.
        """

        tables = (
            "memory_turns",
            "memory_events",
            "memory_facts",
            "memory_embeddings",
            "memory_forget_log",
            "memory_routing_audit",
        )
        with self._lock, self._connect() as conn:
            self._l1_semantic_cache.clear()
            counts = {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in tables
            }
            stamp = int(time.time())
            for table in ("memory_events", "memory_facts"):
                for row in conn.execute(f"SELECT scope,source_id,text FROM {table}").fetchall():
                    self.lifecycle.revoke(conn, scope=row["scope"], source_id=row["source_id"], text=row["text"], now=stamp)
            # Compact revocations intentionally survive a reset, so importing
            # an old backup cannot undo the user's explicit forgetting.
            for table in ("memory_lifecycle", "memory_lifecycle_evidence", "memory_lifecycle_audit", "memory_lifecycle_activity", "memory_lifecycle_scopes"):
                conn.execute(f"DELETE FROM {table}")
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
        with self._lock:
            self._clear_derived_caches(scope="", record_ids=set())
        return counts

    def stats(self) -> dict[str, int]:
        with self._lock, self._connect() as conn:
            return {
                "turns": int(conn.execute("SELECT COUNT(*) FROM memory_turns").fetchone()[0]),
                "active_events": int(conn.execute("SELECT COUNT(*) FROM memory_events WHERE status='active'").fetchone()[0]),
                "active_facts": int(conn.execute("SELECT COUNT(*) FROM memory_facts WHERE status='active'").fetchone()[0]),
                "forgotten": int(conn.execute("SELECT COUNT(*) FROM memory_forget_log").fetchone()[0]),
                "routing_audits": int(conn.execute("SELECT COUNT(*) FROM memory_routing_audit").fetchone()[0]),
                "embedding_vectors": int(conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0]),
            }

    @staticmethod
    def _loads(value: Any) -> dict[str, Any]:
        try:
            parsed = json.loads(str(value or "{}"))
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
