"""Store responsibilities: queries."""

from __future__ import annotations

import time

from .models import MemoryRecord

from .store_models import MemoryStoreQueriesDependencies


from .store_search_models import MemoryStoreSearch
from .store_relevance import query_relevance, attributed_search_text


def prepare_search(q: MemoryStoreSearch, deps: MemoryStoreQueriesDependencies) -> bool:
    q.query_text = str(q.query or "").strip()
    q.terms = deps.terms(q.query_text)
    if not q.terms and not (deps.semantic_enabled() and q.query_text):
        return False
    q.stamp = int(q.now or time.time())
    deps.expire(now=q.stamp)
    q.selected_layers = tuple(dict.fromkeys(str(item).strip() for item in (q.layers or ()) if str(item).strip()))
    return True


def collect_lexical_candidates(q: MemoryStoreSearch, deps: MemoryStoreQueriesDependencies) -> None:
    layer_clause = ""
    layer_values: list[str] = []
    if q.selected_layers:
        layer_clause = " AND layer IN (" + ",".join("?" for _ in q.selected_layers) + ")"
        layer_values = list(q.selected_layers)
    time_clause = ""
    time_values: list[int] = []
    if q.start_at is not None:
        time_clause += " AND created_at >= ?"
        time_values.append(int(q.start_at))
    if q.end_at is not None:
        time_clause += " AND created_at < ?"
        time_values.append(int(q.end_at))
    with deps.lock(), deps.connect() as conn:
        event_rows = conn.execute(
            f"SELECT * FROM memory_events WHERE scope = ? AND status = 'active' "
            f"AND source_id NOT LIKE 'owner:agent-task-detail:%'{layer_clause}{time_clause} "
            f"ORDER BY created_at DESC LIMIT {deps.event_scan_limit()}",
            (str(q.scope), *layer_values, *time_values),
        ).fetchall()
        fact_rows = () if q.selected_layers or q.start_at is not None or q.end_at is not None else conn.execute(
            f"SELECT * FROM memory_facts WHERE scope = ? AND status = 'active' ORDER BY updated_at DESC LIMIT {deps.fact_scan_limit()}",
            (str(q.scope),),
        ).fetchall()
        # 在相关性过滤与精确通道扩充之前判断覆盖；候选少不等于原始窗口读全。
        q.lexical_window_complete = (len(event_rows) < deps.event_scan_limit()
                                     and len(fact_rows) < deps.fact_scan_limit())
        # 精确来源不依赖向量覆盖，也不受最近窗口限制。SQL 只做参数化子串
        # 预筛；Python 先核对标识符边界再截断，防止 R70 噪声挤掉旧 R7。
        # 仍使用当前 scope、状态、层和日期；不开启新的跨会话读取权限。
        features = query_relevance(q.query_text, q.terms)
        exact_enabled = bool(features.anchors and deps.semantic_enabled() and deps.embedding_provider() is not None
                             and not q.require_term_overlap)
        if exact_enabled:
            clauses = " OR ".join("instr(lower(text), ?) > 0" for _ in features.anchors)
            values = tuple(anchor.text for anchor in features.anchors)
            def exact_rows(sql, params, limit):
                matches = []
                cursor = conn.execute(sql, params)
                try:
                    for row in cursor:
                        if features.scores(str(row["text"]))[1] > 0:
                            matches.append(row)
                            if len(matches) >= limit:
                                break
                finally:
                    cursor.close()
                return matches
            event_rows.extend(exact_rows(
                f"SELECT * FROM memory_events WHERE scope=? AND status='active' "
                f"AND source_id NOT LIKE 'owner:agent-task-detail:%'{layer_clause}{time_clause} "
                f"AND ({clauses}) ORDER BY created_at DESC, event_id DESC",
                (str(q.scope), *layer_values, *time_values, *values), deps.event_scan_limit()))
            if not q.selected_layers and q.start_at is None and q.end_at is None:
                fact_rows.extend(exact_rows(
                    f"SELECT * FROM memory_facts WHERE scope=? AND status='active' "
                    f"AND ({clauses}) ORDER BY updated_at DESC, fact_id DESC",
                    (str(q.scope), *values), deps.fact_scan_limit()))
    q.candidates: list[MemoryRecord] = []
    for row in event_rows:
        if q.require_term_overlap and q.terms:
            body = attributed_search_text(str(row["text"] or ""), deps.loads(row["metadata_json"])).lower()
            if not any(term in body for term in q.terms):
                continue
        q.candidates.append(
            deps.score_event(
                row,
                q.terms,
                q.stamp,
                topic_batch_id=q.topic_batch_id,
                sender_id=q.sender_id,
            )
        )
    for row in fact_rows:
        q.candidates.append(deps.score_fact(row, q.terms, q.stamp))
    q.candidates = [item for item in q.candidates if item.score > 0.0
                    or (exact_enabled and features.scores(item.text)[1] > 0)]


def collect_semantic_candidates(q: MemoryStoreSearch, deps: MemoryStoreQueriesDependencies) -> None:
    if deps.semantic_enabled() and deps.embedding_provider() is not None and not q.require_term_overlap and q.query_text:
        deps.backfill_embeddings(limit=deps.semantic_backfill_batch_size(), scope=str(q.scope))
        q.candidates.extend(
            deps.semantic_candidates(
                query=q.query_text,
                terms=q.terms,
                scope=str(q.scope),
                selected_layers=q.selected_layers,
                start_at=q.start_at,
                end_at=q.end_at,
                topic_batch_id=q.topic_batch_id,
                sender_id=q.sender_id,
                now=q.stamp,
            )
        )


def mark_search_usage(q: MemoryStoreSearch, deps: MemoryStoreQueriesDependencies) -> None:
    if q.selected:
        with deps.lock(), deps.connect() as conn:
            ids = [item.record_id for item in q.selected]
            conn.executemany(
                "UPDATE memory_events SET last_used_at = ? WHERE event_id = ?",
                [(q.stamp, item.removeprefix("event:")) for item in ids if item.startswith("event:")],
            )
            conn.executemany(
                "UPDATE memory_facts SET last_used_at = ? WHERE fact_id = ?",
                [(q.stamp, item.record_id.removeprefix("fact:")) for item in q.selected if item.record_id.startswith("fact:")],
            )
