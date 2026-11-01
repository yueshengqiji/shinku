"""Store responsibilities: semantic."""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any

from .semantic import pack_vector

from .store_models import MemoryStoreSemanticDependencies


from .store_semantic_models import MemoryEmbeddingBackfill


def prepare_backfill(q: MemoryEmbeddingBackfill, deps: MemoryStoreSemanticDependencies) -> bool:
    if not deps.semantic_enabled() or deps.embedding_provider() is None:
        return False
    q.model_key = deps.embedding_model_key()
    if not q.model_key:
        return False
    if not deps.semantic_cleanup_done():
        with deps.lock(), deps.connect() as conn:
            conn.execute(
                """DELETE FROM memory_embeddings
                       WHERE model_key = ? AND record_type = 'event'
                         AND record_id IN (
                           SELECT event_id FROM memory_events
                           WHERE trim(text) IN ('[图片附件]', '[未解析消息]')
                         )""",
                (q.model_key,),
            )
        deps.set_semantic_cleanup_done(True)
    return True


def read_backfill_sources(
    q: MemoryEmbeddingBackfill,
    deps: MemoryStoreSemanticDependencies,
) -> bool:
    q.wanted = max(1, min(1000, int(q.limit or deps.semantic_backfill_batch_size())))
    event_scope_clause = " AND m.scope = ?" if q.scope else ""
    fact_scope_clause = " AND f.scope = ?" if q.scope else ""
    scope_values: tuple[Any, ...] = (str(q.scope),) if q.scope else ()
    # The batch bounds encoding, not the age of records that can be repaired.
    # Iterate bounded fetches through the source snapshot: an indexed or
    # ineligible recent window must not permanently hide older gaps.
    fetch_size = max(q.wanted, min(10000, q.wanted * 8))
    sources = (
        (f"""SELECT 'event' AS record_type, m.event_id AS record_id,
                          m.scope, m.text, m.created_at, m.source_id,
                          COALESCE(e.content_hash, '') AS indexed_hash,
                          e.embedding_id AS embedding_id
                    FROM memory_events m
                    LEFT JOIN memory_embeddings e
                      ON e.record_type = 'event' AND e.record_id = m.event_id
                     AND e.model_key = ?
                   WHERE m.status = 'active'
                     AND m.source_id NOT LIKE 'owner:agent-task-detail:%'{event_scope_clause}
                   ORDER BY m.created_at DESC, m.event_id DESC""",
         (q.model_key, *scope_values)),
        (f"""SELECT 'fact' AS record_type, f.fact_id AS record_id,
                          f.scope, f.text, f.updated_at AS created_at, f.source_id,
                          COALESCE(e.content_hash, '') AS indexed_hash,
                          e.embedding_id AS embedding_id
                    FROM memory_facts f
                    LEFT JOIN memory_embeddings e
                      ON e.record_type = 'fact' AND e.record_id = f.fact_id
                     AND e.model_key = ?
                   WHERE f.status = 'active'{fact_scope_clause}
                   ORDER BY f.updated_at DESC, f.fact_id DESC""",
         (q.model_key, *scope_values)),
    )
    fresh_rows = []
    with deps.lock(), deps.connect() as conn:
        for sql, values in sources:
            cursor = conn.execute(sql, values)
            while len(fresh_rows) < q.wanted:
                rows = cursor.fetchmany(fetch_size)
                if not rows:
                    break
                for row in rows:
                    body = str(row["text"] or "")
                    if body.strip() in {"[图片附件]", "[未解析消息]"}:
                        continue
                    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
                    if row["embedding_id"] and str(row["indexed_hash"] or "") == digest:
                        continue
                    if not deps.embedding_allowed(conn, str(row["record_type"]), str(row["record_id"]),
                                                  include_deferred_l1=q.include_deferred_l1):
                        continue
                    fresh_rows.append(row)
                    if len(fresh_rows) == q.wanted:
                        break
            cursor.close()
            if len(fresh_rows) == q.wanted:
                break
    q.rows = fresh_rows
    return bool(q.rows)


def encode_backfill_sources(
    q: MemoryEmbeddingBackfill,
    deps: MemoryStoreSemanticDependencies,
) -> bool:
    texts = [str(row["text"] or "")[:6000] for row in q.rows]
    try:
        with deps.embedding_lock():
            q.vectors = deps.embedding_provider().embed_texts(texts)
    except Exception:
        return False
    q.dimension = int(getattr(deps.embedding_provider(), "dimension", 0) or 0)
    q.stamp = int(time.time())
    q.written = 0
    return True


def write_backfill_vectors(
    q: MemoryEmbeddingBackfill,
    deps: MemoryStoreSemanticDependencies,
) -> int:
    with deps.lock(), deps.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        # 编码可能跨越配置切换；旧任务不能把新模型向量写进旧 collection。
        if deps.embedding_model_key() != q.model_key:
            return 0
        for row, vector in zip(q.rows, q.vectors):
            values = [float(value) for value in (vector or [])]
            if not values:
                continue
            width = len(values)
            if q.dimension and width != q.dimension:
                continue
            body = str(row["text"] or "")
            content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
            # Encoding ran outside the transaction. A forget, correction
            # or raw reclamation during that time revokes this snapshot.
            current = deps.lifecycle()._record(conn, str(row["record_type"]), str(row["record_id"]))
            if (current is None or current["status"] != "active"
                    or str(current["scope"]) != str(row["scope"])
                    or str(current["source_id"]) != str(row["source_id"])
                    or hashlib.sha256(str(current["text"]).encode("utf-8")).hexdigest() != content_hash):
                deps.discard_cached_text(body)
                continue
            if not deps.embedding_allowed(conn, str(row["record_type"]), str(row["record_id"]),
                                          include_deferred_l1=q.include_deferred_l1):
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
                    q.model_key,
                    content_hash,
                    width,
                    pack_vector(values),
                    int(row["created_at"] or q.stamp),
                    q.stamp,
                ),
            )
            q.written += 1
    return q.written
