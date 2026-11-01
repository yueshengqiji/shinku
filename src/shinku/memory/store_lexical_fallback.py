"""近期窗口未读全时补找文字来源，保留原评分和所有范围约束。

上游为 Store.search 的候选阶段。只读取同一个 SQLite 来源库，不建立索引、
修改生命周期或制造语义分数。近期候选保留；补充候选由原排名器去重排序。
SQLite 进度回调限制补查工作，超时退回已有候选，诊断显式标记覆盖不完整。
"""
from __future__ import annotations

import sqlite3
import time

from . import retrieval_trace
from .store_models import MemoryStoreQueriesDependencies
from .store_relevance import query_relevance
from .store_search_models import MemoryStoreSearch


def collect_lexical_fallback(q: MemoryStoreSearch, deps: MemoryStoreQueriesDependencies,
                             *, semantic_count: int) -> None:
    """补查前先按正文相关性过滤，再限制数量，避免旧来源被近期无关消息挤掉。"""
    if q.lexical_window_complete:
        retrieval_trace.observe("store.lexical_fallback_skipped", reason="scoped_sources_fit_window")
        return
    terms = query_relevance(q.query_text, q.terms).terms
    # 任意向量命中不代表目标已经索引，也不代表文字匹配通过了向量门槛。
    # 两路独立提供候选；后续仍用原评分合并，不给缺少向量的记录伪造分数。
    if not terms:
        retrieval_trace.observe("store.lexical_fallback_skipped", reason="no_content_terms")
        return
    clauses = ["scope = ?", "status = 'active'", "source_id NOT LIKE 'owner:agent-task-detail:%'"]
    values: list[object] = [str(q.scope)]
    if q.selected_layers:
        clauses.append("layer IN (" + ",".join("?" for _ in q.selected_layers) + ")")
        values.extend(q.selected_layers)
    if q.start_at is not None:
        clauses.append("created_at >= ?")
        values.append(int(q.start_at))
    if q.end_at is not None:
        clauses.append("created_at < ?")
        values.append(int(q.end_at))
    overlap = " + ".join("CASE WHEN instr(lower(text), ?) > 0 THEN 1 ELSE 0 END" for _ in terms)
    match = " OR ".join("instr(lower(text), ?) > 0" for _ in terms)
    tables = [("event", "memory_events", "event_id", "created_at", clauses, values, deps.event_scan_limit())]
    if not q.selected_layers and q.start_at is None and q.end_at is None:
        tables.append(("fact", "memory_facts", "fact_id", "updated_at",
                       ["scope = ?", "status = 'active'"], [str(q.scope)], deps.fact_scan_limit()))
    extra = []
    interrupted = truncated = False
    deadline = time.monotonic() + 0.1
    with deps.lock(), deps.connect() as conn:
        # 此连接只属于本次读阶段；finally 清除回调，不影响后续原排序/来源读取。
        # 检查间隔是 VM 指令数，不声称每条 SQL 操作都有硬实时终止保证。
        conn.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        try:
            for kind, table, key, stamp, predicates, params, limit in tables:
                if time.monotonic() >= deadline:
                    interrupted = True
                    break
                rows = conn.execute(
                    f"SELECT *, ({overlap}) AS query_overlap FROM {table} "
                    f"WHERE {' AND '.join(predicates)} AND ({match}) "
                    f"ORDER BY query_overlap DESC, {stamp} DESC, {key} DESC LIMIT ?",
                    (*terms, *params, *terms, limit + 1),
                ).fetchall()
                truncated |= len(rows) > limit
                for row in rows[:limit]:
                    record = (deps.score_event(row, q.terms, q.stamp, topic_batch_id=q.topic_batch_id,
                                               sender_id=q.sender_id) if kind == "event"
                              else deps.score_fact(row, q.terms, q.stamp))
                    extra.append(record)
        except sqlite3.OperationalError as exc:
            if getattr(exc, "sqlite_errorcode", None) != sqlite3.SQLITE_INTERRUPT:
                raise
            interrupted = True
        finally:
            conn.set_progress_handler(None, 0)
    q.candidates.extend(extra)
    retrieval_trace.observe("store.lexical_fallback", extra,
                            reason=("sql_budget_exhausted" if interrupted else
                                    "independent_lexical_channel" if semantic_count else "no_semantic_candidates"),
                            lexical_truncated=truncated or interrupted)
