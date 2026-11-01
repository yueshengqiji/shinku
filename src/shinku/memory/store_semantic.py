"""Store responsibilities: semantic."""

from __future__ import annotations

import threading
from typing import Any

from .models import MemoryRecord

from .store_models import MemoryStoreSemanticDependencies


from . import store_embedding_index, store_semantic_query
from .store_semantic_models import MemoryEmbeddingBackfill, MemorySemanticQuery


def semantic_status(*, deps: MemoryStoreSemanticDependencies) -> dict[str, Any]:
    """Return safe diagnostics for the active local semantic index."""

    model_key = deps.embedding_model_key()
    with deps.lock(), deps.connect() as conn:
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
    provider = deps.embedding_provider()
    return {
        "enabled": bool(deps.semantic_enabled()),
        "status": deps.embedding_status(),
        "error": deps.embedding_error(),
        "provider": str(getattr(provider, "name", "") or "") if provider else "",
        "model": str(getattr(provider, "version", "") or "") if provider else "",
        "model_key": model_key,
        "dimension": int(getattr(provider, "dimension", 0) or 0) if provider else 0,
        "vectors": total,
        "active_vectors": active,
        "worker_running": bool(deps.embedding_worker() and deps.embedding_worker().is_alive()),
    }


def start_embedding_worker(
    *,
    interval_seconds: float=3.0,
    deps: MemoryStoreSemanticDependencies,
) -> None:
    """Warm the local index in the background without blocking QQ turns."""

    if not deps.semantic_enabled() or deps.embedding_provider() is None:
        return
    if deps.embedding_worker() and deps.embedding_worker().is_alive():
        return
    deps.embedding_worker_stop().clear()
    interval = max(0.5, min(60.0, float(interval_seconds or 3.0)))

    def run() -> None:
        while not deps.embedding_worker_stop().is_set():
            try:
                # L1 延后的是前台建索引的成本；后台在原批次预算内补齐有效来源。
                # 向量只派生自原文，写入前仍复验撤销、作用域、正文和模型版本。
                deps.backfill_embeddings(limit=deps.semantic_backfill_batch_size(), include_deferred_l1=True)
            except Exception:
                # Keyword retrieval remains the safety net.  A transient
                # model/DB failure must not terminate the backend worker.
                pass
            deps.embedding_worker_stop().wait(interval)

    deps.set_embedding_worker(threading.Thread(target=run, name='shinku-memory-embedding-indexer', daemon=True))
    deps.embedding_worker().start()


def stop_embedding_worker(*, deps: MemoryStoreSemanticDependencies) -> None:
    deps.embedding_worker_stop().set()


def backfill_embeddings(
    *,
    limit: int | None=None,
    scope: str | None=None,
    include_deferred_l1: bool=False,
    deps: MemoryStoreSemanticDependencies,
) -> int:
    """Embed active records missing from the current model collection.

        The operation is idempotent and model-versioned.  It is deliberately
        separate from ``record_event`` so passive QQ observation never waits
        for a sentence-transformer model.
        """

    q = MemoryEmbeddingBackfill(
        limit=limit,
        scope=scope,
        include_deferred_l1=include_deferred_l1,
    )
    if not store_embedding_index.prepare_backfill(q, deps):
        return 0
    if not store_embedding_index.read_backfill_sources(q, deps):
        return 0
    if not store_embedding_index.encode_backfill_sources(q, deps):
        return 0
    return store_embedding_index.write_backfill_vectors(q, deps)


def embedding_allowed(
    conn,
    record_type: str,
    record_id: str,
    *,
    include_deferred_l1: bool = False,
    deps: MemoryStoreSemanticDependencies,
) -> bool:
    policy = deps.lifecycle().policy
    if include_deferred_l1 or policy.mode != "active" or not policy.defer_l1_embeddings:
        return True
    state = conn.execute("SELECT tier,pinned FROM memory_lifecycle WHERE record_type=? AND record_id=?", (record_type, record_id)).fetchone()
    if state is None:
        row = deps.lifecycle()._record(conn, record_type, record_id)
        if not deps.lifecycle()._eligible(row, record_type):
            return True  # Unmanaged summaries/temporary layers keep their own indexing policy.
    return record_type == "fact" or (state is not None and (state["tier"] >= 2 or state["pinned"]))


def semantic_candidates(
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
    deps: MemoryStoreSemanticDependencies,
) -> list[MemoryRecord]:
    """Return vector candidates already present in the local SQLite index."""

    q = MemorySemanticQuery(
        query=query,
        terms=terms,
        scope=scope,
        selected_layers=selected_layers,
        start_at=start_at,
        end_at=end_at,
        topic_batch_id=topic_batch_id,
        sender_id=sender_id,
        now=now,
    )
    if not store_semantic_query.prepare_semantic_query(q, deps):
        return []
    store_semantic_query.read_semantic_sources(q, deps)
    store_semantic_query.score_persistent_vectors(q, deps)
    store_semantic_query.fill_l1_query_cache(q, deps)
    return store_semantic_query.score_l1_vectors(q, deps)
