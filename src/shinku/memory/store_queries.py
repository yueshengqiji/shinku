"""Store responsibilities: queries."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import replace
from typing import Any

from .models import MemoryRecord

from .store_models import MemoryStoreQueriesDependencies


from . import store_search_ranking, store_search_sources
from . import retrieval_trace
from .store_lexical_fallback import collect_lexical_fallback
from .store_search_models import MemoryStoreSearch


def terms(query: str, *, deps: MemoryStoreQueriesDependencies) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for term in deps.term_pattern().findall(str(query or "").lower()):
        if any("\u4e00" <= char <= "\u9fff" for char in term):
            pieces = [term] if len(term) <= 4 else []
            pieces.extend(term[index : index + 2] for index in range(len(term) - 1))
        else:
            pieces = [term]
        for piece in pieces:
            if piece not in seen:
                seen.add(piece)
                terms.append(piece)
    return terms[: deps.max_query_terms()]


def search(
    *,
    query: str,
    scope: str,
    limit: int=5,
    now: int | None=None,
    layers: set[str] | tuple[str, ...] | None=None,
    topic_batch_id: str='',
    sender_id: str='',
    start_at: int | None=None,
    end_at: int | None=None,
    require_term_overlap: bool=False,
    deps: MemoryStoreQueriesDependencies,
) -> list[MemoryRecord]:
    """Search one scope, optionally bounded to a calendar window.

        Normal retrieval keeps the small recency prior used by the original
        memory contract.  Date-specific recall can opt into
        ``require_term_overlap`` so a query such as "前天我问你的那件事"
        does not simply return the newest records from that day when its
        actual topic is present farther down the timeline.
        """

    q = MemoryStoreSearch(
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
    )
    retrieval_trace.query("store.request", str(query or ""), scope=scope, limit=limit,
                          semantic_enabled=deps.semantic_enabled(), require_term_overlap=require_term_overlap)
    if not store_search_sources.prepare_search(q, deps):
        retrieval_trace.observe("store.skipped", reason="empty_query")
        return []
    store_search_sources.collect_lexical_candidates(q, deps)
    retrieval_trace.observe("store.lexical", q.candidates)
    lexical_count = len(q.candidates)
    store_search_sources.collect_semantic_candidates(q, deps)
    if retrieval_trace.enabled():
        retrieval_trace.observe("store.semantic", q.candidates[lexical_count:])
    collect_lexical_fallback(q, deps, semantic_count=len(q.candidates) - lexical_count)
    store_search_ranking.rank_search_candidates(q, deps)
    retrieval_trace.observe("store.ranked", q.candidates)
    retrieval_trace.observe("store.selected", q.selected, limit=q.limit)
    store_search_sources.mark_search_usage(q, deps)
    return q.selected


def nearby_events(
    *,
    scope: str,
    center_at: int,
    limit: int=6,
    window_seconds: int=600,
    layers: set[str] | tuple[str, ...]=('observation', 'episodic'),
    start_at: int | None=None,
    end_at: int | None=None,
    deps: MemoryStoreQueriesDependencies,
) -> list[MemoryRecord]:
    """Return a small chronological neighborhood around one hit.

        A date question often matches the user's question but not the
        assistant's answer.  Pulling the adjacent turn keeps that answer
        available without injecting an entire day's transcript.
        """

    center = int(center_at)
    radius = max(1, int(window_seconds or 600))
    wanted = max(1, min(deps.max_results(), int(limit or 6)))
    selected_layers = tuple(dict.fromkeys(str(item).strip() for item in layers if str(item).strip()))
    if not selected_layers:
        return []
    placeholders = ",".join("?" for _ in selected_layers)
    lower = center - radius if start_at is None else max(center - radius, int(start_at))
    upper = center + radius if end_at is None else min(center + radius, int(end_at) - 1)
    if upper < lower:
        return []
    with deps.lock(), deps.connect() as conn:
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
        deps.event_record(
            row,
            score=max(0.0, 1.0 - abs(center - int(row["created_at"] or center)) / max(1, radius) * 0.25),
        )
        for row in rows
    ]
    records.sort(key=lambda item: (item.created_at, item.record_id), reverse=True)
    return records


def recent_events(
    *,
    scope: str,
    limit: int=6,
    now: int | None=None,
    max_age_seconds: int=900,
    layers: set[str] | tuple[str, ...]=('observation', 'episodic'),
    topic_batch_id: str='',
    include_assistant: bool=False,
    deps: MemoryStoreQueriesDependencies,
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
    wanted = max(1, min(deps.max_results(), int(limit or 6)))
    # Fetch a bounded overscan because role filtering happens after JSON
    # metadata decoding.  This keeps recent user/observation messages
    # available even when several assistant turns are newest.
    scan = max(wanted, min(deps.max_results() * 8, wanted * 8))
    params: list[Any] = [str(scope), *selected_layers, stamp - age, scan]
    with deps.lock(), deps.connect() as conn:
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
        metadata = deps.loads(row["metadata_json"])
        if not include_assistant and str(metadata.get("role") or "").strip().lower() == "assistant":
            continue
        same_topic = bool(topic_batch_id and metadata.get("topic_batch_id") == topic_batch_id)
        age_ratio = max(0.0, min(1.0, (stamp - int(row["created_at"] or stamp)) / age))
        records.append(deps.event_record(row, score=(1.0 - age_ratio * 0.35) + (0.25 if same_topic else 0.0)))
        if len(records) >= wanted:
            break
    return records


def recent_assistant_events(
    *,
    scope: str,
    now: int,
    max_age_seconds: int=1800,
    limit: int=48,
    deps: MemoryStoreQueriesDependencies,
) -> list[MemoryRecord]:
    """Fetch Shinku's own active QQ replies for same-scope fact checking.

        This is deliberately separate from ``recent_events``: generated replies
        must not become general conversational or voice examples.
        """

    wanted = max(1, min(100, int(limit)))
    with deps.lock(), deps.connect() as conn:
        rows = conn.execute(
            """SELECT * FROM memory_events
                   WHERE scope = ? AND status = 'active' AND layer = 'episodic'
                     AND created_at BETWEEN ? AND ?
                     AND json_extract(metadata_json, '$.role') = 'assistant'
                   ORDER BY created_at DESC LIMIT ?""",
            (str(scope), int(now) - max(1, int(max_age_seconds)), int(now), wanted),
        ).fetchall()
    return [
        deps.event_record(row, score=0.0)
        for row in rows
        if str(deps.loads(row["metadata_json"]).get("role") or "").lower() == "assistant"
    ]


def timeline(
    *,
    scope: str,
    limit: int=8,
    now: int | None=None,
    layers: set[str] | tuple[str, ...] | None=None,
    include_facts: bool=True,
    start_at: int | None=None,
    end_at: int | None=None,
    deps: MemoryStoreQueriesDependencies,
) -> list[MemoryRecord]:
    """Return a bounded read-only timeline for explicit memory inspection.

        This is deliberately separate from search: a user asking to see
        memory may not provide terms that overlap the stored text.  The
        timeline therefore uses recency and provenance, while the normal
        routing path continues to use term-based retrieval.
        """

    stamp = int(now or time.time())
    deps.expire(now=stamp)
    wanted = max(1, min(deps.max_results(), int(limit or 8)))
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
    with deps.lock(), deps.connect() as conn:
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
        deps.event_record(
            row,
            score=max(0.0, 1.0 - max(0, stamp - int(row["created_at"] or stamp)) / 31536000.0),
        )
        for row in event_rows
    ]
    records.extend(deps.fact_record(row, score=0.8) for row in fact_rows)
    records.sort(key=lambda item: (item.created_at, item.importance, item.confidence), reverse=True)
    return records[:wanted]


def latest_observation(
    *,
    scope: str,
    before_created_at: int | None=None,
    deps: MemoryStoreQueriesDependencies,
) -> MemoryRecord | None:
    params: list[Any] = [str(scope)]
    clause = ""
    if before_created_at is not None:
        clause = " AND created_at <= ?"
        params.append(int(before_created_at))
    with deps.lock(), deps.connect() as conn:
        row = conn.execute(
            f"""SELECT * FROM memory_events
                    WHERE scope = ? AND status = 'active' AND layer = 'observation'{clause}
                    ORDER BY created_at DESC, event_id DESC LIMIT 1""",
            params,
        ).fetchone()
    return deps.event_record(row, score=1.0) if row is not None else None


def observation_for_message(
    *,
    scope: str,
    message_id: str,
    deps: MemoryStoreQueriesDependencies,
) -> MemoryRecord | None:
    wanted = str(message_id or "").strip()
    if not wanted:
        return None
    with deps.lock(), deps.connect() as conn:
        rows = conn.execute(
            """SELECT * FROM memory_events
                   WHERE scope = ? AND status = 'active' AND layer = 'observation'
                   ORDER BY created_at DESC LIMIT ?""",
            (str(scope), deps.event_scan_limit()),
        ).fetchall()
    for row in rows:
        metadata = deps.loads(row["metadata_json"])
        if str(metadata.get("message_id") or "").strip() == wanted:
            return deps.event_record(row, score=1.0)
    return None


def event_record(
    row: sqlite3.Row,
    *,
    score: float,
    deps: MemoryStoreQueriesDependencies,
) -> MemoryRecord:
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
        metadata=deps.loads(row["metadata_json"]),
    )


def score_event(
    row: sqlite3.Row,
    terms: list[str],
    now: int,
    *,
    topic_batch_id: str='',
    sender_id: str='',
    deps: MemoryStoreQueriesDependencies,
) -> MemoryRecord:
    text = str(row["text"] or "")
    metadata = deps.loads(row["metadata_json"])
    from .store_relevance import attributed_search_text
    search_text = attributed_search_text(text, metadata).lower()
    overlap = sum(1 for term in terms if term in search_text)
    age_days = max(0.0, (now - int(row["created_at"] or now)) / 86400.0)
    decay = 0.5 ** (age_days / 30.0)
    score = overlap * 0.55 + float(row["importance"] or 0.0) * decay * 0.45
    if topic_batch_id and metadata.get("topic_batch_id") == topic_batch_id:
        score += 0.35
    if sender_id and str(metadata.get("sender_id") or "") == str(sender_id):
        score += 0.10
    record = deps.event_record(row, score=score)
    # 保留显式词法路径的旧分数；混排只读取这些重新计算的辅助特征，
    # 不把重要性和时间再次当成词法命中，也不信任持久化 metadata 的同名值。
    return replace(record, metadata={**record.metadata, "ranking_auxiliary": {
        "importance": max(0.0, min(1.0, record.importance * record.confidence)),
        "recency": decay,
        "topic": float(bool(topic_batch_id and metadata.get("topic_batch_id") == topic_batch_id)),
        "sender": float(bool(sender_id and str(metadata.get("sender_id") or "") == str(sender_id))),
    }})


def score_fact(
    row: sqlite3.Row,
    terms: list[str],
    now: int,
    *,
    deps: MemoryStoreQueriesDependencies,
) -> MemoryRecord:
    text = str(row["text"] or "")
    overlap = sum(1 for term in terms if term in text.lower())
    score = overlap * 0.7 + float(row["importance"] or 0.0) * float(row["confidence"] or 0.0) * 0.3
    record = deps.fact_record(row, score=score)
    age_days = max(0.0, (now - int(row["updated_at"] or now)) / 86400.0)
    return replace(record, metadata={**record.metadata, "ranking_auxiliary": {
        "importance": max(0.0, min(1.0, record.importance * record.confidence)),
        "recency": 0.5 ** (age_days / 30.0), "topic": 0.0, "sender": 0.0,
    }})


def fact_record(
    row: sqlite3.Row,
    *,
    score: float,
    deps: MemoryStoreQueriesDependencies,
) -> MemoryRecord:
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
        metadata={**(deps.loads(row["metadata_json"]) if "metadata_json" in row.keys() else {}),
                  "fact_key": str(row["fact_key"] or "")},
    )
