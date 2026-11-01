"""持久化记忆语义向量查询与评分引擎。

职责与上下游：
- 职责：从底层 SQLite 载入已向量化的事件（event）与事实（fact），
  计算与查询文本的余弦相似度，并结合词法与时间衰减计算混合权重，筛选出最高优先级的语义候选。
- 上游：由 `MemoryStore.search_semantic` 驱动并执行。
- 下游：向调用方输出已通过哈希核验与最新来源撤销检查的 `MemoryRecord` 候选列表。

设计原因与关键边界：
1. 全范围评分（Global Scoring）：
   - 先比较当前作用域内全部有效向量，再限制保留的候选数量。
   - 若在相似度评分前按时间截断，较早的记忆即使已有向量，也无法被召回；
2. 来源撤销防并发穿透（Revocation Re-check）：
   - 扫描与评分期间其他写入者可能执行遗忘（forget）或更正来源。
   - 在评分完成后、最终返回前，必须再次核对来源状态和正文，避免旧快照重新进入回答。
"""

from __future__ import annotations

import hashlib
import heapq
from itertools import chain
from dataclasses import replace
from typing import Any

from .models import MemoryRecord
from .semantic import cosine_similarity
from .semantic_scan import cosine_scores

from .store_models import MemoryStoreSemanticDependencies


from .store_semantic_models import MemorySemanticQuery
from .store_relevance import query_relevance
from . import retrieval_trace


def prepare_semantic_query(q: MemorySemanticQuery, deps: MemoryStoreSemanticDependencies) -> bool:
    q.model_key = deps.embedding_model_key()
    if not q.model_key:
        return False
    try:
        with deps.embedding_lock():
            q.query_vector = deps.embedding_provider().embed_text(q.query)
    except Exception:
        return False
    if not q.query_vector:
        return False
    return True


def read_semantic_sources(q: MemorySemanticQuery, deps: MemoryStoreSemanticDependencies) -> None:
    layer_clause = ""
    layer_values: list[Any] = []
    if q.selected_layers:
        layer_clause = " AND m.layer IN (" + ",".join("?" for _ in q.selected_layers) + ")"
        layer_values = list(q.selected_layers)
    time_clause = ""
    time_values: list[Any] = []
    if q.start_at is not None:
        time_clause += " AND m.created_at >= ?"
        time_values.append(int(q.start_at))
    if q.end_at is not None:
        time_clause += " AND m.created_at < ?"
        time_values.append(int(q.end_at))
    def batches(sql, values, record_type):
        # Cursor and lifecycle reads share one bounded snapshot. No recency
        # cutoff precedes similarity scoring, and no corpus matrix is retained.
        with deps.lock(), deps.connect() as conn:
            cursor = conn.execute(sql, values)
            while rows := cursor.fetchmany(256):
                eligible = []
                for row in rows:
                    if hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest() != row["indexed_hash"]:
                        continue
                    if not deps.embedding_allowed(conn, record_type, str(row[record_type + "_id"]),
                                                  include_deferred_l1=True):
                        continue
                    eligible.append(row)
                for row, similarity in zip(eligible, cosine_scores(q.query_vector, eligible)):
                    if similarity is not None and similarity >= deps.semantic_min_score():
                        yield row, similarity
            cursor.close()

    q.event_rows = batches(
        f"""SELECT m.*, e.vector_blob, e.dimension,e.content_hash AS indexed_hash
                FROM memory_events m
                JOIN memory_embeddings e
                  ON e.record_type = 'event' AND e.record_id = m.event_id
                 AND e.model_key = ?
               WHERE m.scope = ? AND m.status = 'active'
                 AND m.source_id NOT LIKE 'owner:agent-task-detail:%'
                 AND trim(m.text) NOT IN ('[图片附件]', '[未解析消息]'){layer_clause}{time_clause}
               ORDER BY m.created_at DESC, m.event_id DESC""",
        (q.model_key, q.scope, *layer_values, *time_values), "event",
    )
    q.fact_rows = ()
    if not q.selected_layers and q.start_at is None and q.end_at is None:
        q.fact_rows = batches(
            """SELECT f.*, e.vector_blob, e.dimension,e.content_hash AS indexed_hash
                   FROM memory_facts f
                   JOIN memory_embeddings e
                     ON e.record_type = 'fact' AND e.record_id = f.fact_id
                    AND e.model_key = ?
                  WHERE f.scope = ? AND f.status = 'active'
                  ORDER BY f.updated_at DESC, f.fact_id DESC""",
            (q.model_key, q.scope), "fact",
        )
    with deps.lock(), deps.connect() as conn:
        q.l1_rows = []
        if deps.lifecycle().policy.mode == "active" and deps.lifecycle().policy.defer_l1_embeddings:
            cursor = conn.execute(f"""SELECT m.*, e.content_hash AS indexed_hash
                  FROM memory_events m JOIN memory_lifecycle l
                  ON l.record_type='event' AND l.record_id=m.event_id
                  LEFT JOIN memory_embeddings e ON e.record_type='event' AND e.record_id=m.event_id
                  AND e.model_key=?
                  WHERE m.scope=? AND m.status='active' AND l.tier=1 AND l.pinned=0
                  AND m.source_id NOT LIKE 'owner:agent-task-detail:%'
                  AND trim(m.text) NOT IN ('[图片附件]','[未解析消息]'){layer_clause}{time_clause}
                  ORDER BY m.created_at DESC, m.event_id DESC""",
                  (q.model_key, q.scope, *layer_values, *time_values))
            # 后台已补齐的 L1 走全范围持久向量评分，不重复临时编码。
            # 前台只补最近 64 个索引缺口；第 65 个是诊断哨兵，尚未补齐不等于不相关。
            try:
                for row in cursor:
                    if row["indexed_hash"] == hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest():
                        continue
                    q.l1_rows.append(row)
                    if len(q.l1_rows) == 65:
                        break
            finally:
                cursor.close()
            truncated = len(q.l1_rows) > 64
            q.l1_rows = q.l1_rows[:64]
            if retrieval_trace.enabled():
                retrieval_trace.observe("store.l1_candidates",
                    tuple(("event:" + str(row["event_id"]), str(row["scope"])) for row in q.l1_rows),
                    state="pending_background_l1", vector_truncated=truncated, limit=64)


def score_persistent_vectors(q: MemorySemanticQuery, deps: MemoryStoreSemanticDependencies) -> None:
    # 全范围评分：先比较当前作用域内全部有效向量，再限制保留的候选数量。
    # 若在相似度评分前按时间截断，较早的记忆即使已有向量，也无法被召回。
    # 排序时采用词法与向量融合的混合评分，避免单一指标偏倚。
    budget = max(1, deps.event_scan_limit() + deps.fact_scan_limit())
    best = []
    features = query_relevance(q.query, q.terms)
    rows = chain((("event", row, similarity) for row, similarity in q.event_rows),
                 (("fact", row, similarity) for row, similarity in q.fact_rows))
    try:
        for serial, (kind, row, similarity) in enumerate(rows):
            if kind == "event":
                lexical = deps.score_event(row, q.terms, q.now, topic_batch_id=q.topic_batch_id, sender_id=q.sender_id)
            else:
                lexical = deps.score_fact(row, q.terms, q.now)
            match = deps.hybrid_record(lexical, similarity, terms=q.terms, features=features)
            entry = (match.score, match.importance, match.created_at, -serial, kind, match)
            if len(best) < budget:
                heapq.heappush(best, entry)
            elif entry[:4] > best[0][:4]:
                heapq.heapreplace(best, entry)
    finally:
        for source in (q.event_rows, q.fact_rows):
            close = getattr(source, "close", None)
            if close is not None:
                close()
    q.semantic = []
    # 释放游标快照后，核验留存候选的有效性。
    # 扫描与评分期间其他写入者可能遗忘（forget）或更正来源。
    # 返回前再次核对来源状态和正文，避免旧快照重新进入回答。
    with deps.lock(), deps.connect() as conn:
        for *_, kind, match in sorted(best, key=lambda item: item[:4], reverse=True):
            bare_id = match.record_id.split(":", 1)[1]
            current = deps.lifecycle()._record(conn, kind, bare_id)
            if (current is None or current["status"] != "active"
                    or str(current["scope"]) != q.scope or str(current["text"]) != match.text
                    or str(current["source_id"] or "") != match.source_id
                    or (kind == "event" and q.selected_layers and current["layer"] not in q.selected_layers)
                    or (q.start_at is not None and int(current["created_at"]) < q.start_at)
                    or (q.end_at is not None and int(current["created_at"]) >= q.end_at)
                    or not deps.embedding_allowed(conn, kind, bare_id, include_deferred_l1=True)):
                continue
            q.semantic.append(match)


def fill_l1_query_cache(q: MemorySemanticQuery, deps: MemoryStoreSemanticDependencies) -> None:
    missing = []
    with deps.lock():
        for row in q.l1_rows:
            content_hash = hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
            key = (q.scope, str(row["event_id"]), q.model_key, content_hash)
            if key not in deps.l1_semantic_cache():
                missing.append((row, key, content_hash))
    if missing:
        try:
            with deps.embedding_lock():
                fresh_vectors = deps.embedding_provider().embed_texts([str(row["text"])[:6000] for row, _, _ in missing])
        except Exception:
            fresh_vectors = []
        with deps.lock(), deps.connect() as conn:
            for (row, key, content_hash), vector in zip(missing, fresh_vectors):
                current = deps.lifecycle()._record(conn, "event", str(row["event_id"]))
                if current is None or current["status"] != "active" or hashlib.sha256(str(current["text"]).encode("utf-8")).hexdigest() != content_hash:
                    deps.discard_cached_text(str(row["text"]))
                    continue
                deps.l1_semantic_cache()[key] = list(vector)
            while len(deps.l1_semantic_cache()) > 128:
                deps.l1_semantic_cache().pop(next(iter(deps.l1_semantic_cache())))


def score_l1_vectors(
    q: MemorySemanticQuery,
    deps: MemoryStoreSemanticDependencies,
) -> list[MemoryRecord]:
    features = query_relevance(q.query, q.terms)
    for row in q.l1_rows:
        content_hash = hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest()
        key = (q.scope, str(row["event_id"]), q.model_key, content_hash)
        with deps.lock():
            vector = deps.l1_semantic_cache().get(key)
        if vector is None:
            continue
        similarity = cosine_similarity(q.query_vector, vector)
        if similarity >= deps.semantic_min_score():
            lexical = deps.score_event(row, q.terms, q.now, topic_batch_id=q.topic_batch_id, sender_id=q.sender_id)
            match = deps.hybrid_record(lexical, similarity, terms=q.terms, features=features)
            q.semantic.append(replace(match, metadata={**match.metadata, "semantic_cache": "ephemeral_l1"}))
    return q.semantic
