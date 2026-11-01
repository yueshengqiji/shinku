"""Service responsibilities: queries."""

from __future__ import annotations

from typing import Any


from .service_models import MemoryServiceQueriesDependencies


from .service_query_models import MemoryInspectionQuery, MemoryTurnQuery


def select_turn_records(q: MemoryTurnQuery, deps: MemoryServiceQueriesDependencies) -> None:
    q.topic_batch_id = str(q.effective_meta.get("topic_batch_id") or "").strip()
    q.fallback = ""
    from .participant_memory import recall_participant_events
    participant_records = recall_participant_events(deps.store(), deps.policy(),
        query=q.user_message, scopes=q.history_scopes, mode=q.decision.history_mode,
        now=q.now, limit=deps.policy().retrieval_limit,
        date_window=deps.request_date_window(q.user_message, now=q.now))
    if participant_records is not None:
        q.records = participant_records
        q.fallback = "participant_sources" if q.records else "participant_sources_empty"
        return
    if q.natural_records is not None:
        q.records = q.natural_records
        # 偏好继续使用经确认的当前来源；普通短续问才补一轮有依据的查询。
        if q.history_need.get("kind") == "followup" and q.history_need.get("topic") not in {"diet", "drink", "environment"}:
            from .query_context import search_context
            q.records = search_context(deps.store(), query=q.user_message, scopes=q.history_scopes,
                now=q.now, limit=deps.policy().retrieval_limit, anchors=q.records)
    elif q.decision.history_mode == 'recent':
        q.records = tuple(
            deps.store().recent_events(
                scope=q.scope,
                limit=deps.policy().recent_limit,
                now=q.now,
                max_age_seconds=deps.policy().recent_window_seconds,
                topic_batch_id=q.topic_batch_id,
                # Recent context is a continuity snapshot.  Do not feed
                # generated Shinku replies back as voice examples; that
                # creates a self-reinforcing loop for cold/defensive turns.
                # Explicit retrieve/summary requests still include them.
                include_assistant=False,
            )
        )
    elif q.decision.history_mode == 'summary':
        _select_turn_summary(q, deps)
    else:
        _select_turn_retrieval(q, deps)


def _select_turn_summary(q: MemoryTurnQuery, deps: MemoryServiceQueriesDependencies) -> None:
    for memory_scope in q.history_scopes:
        deps.store().rollup_daily_summaries(scope=memory_scope, now=q.now)
    date_window = deps.request_date_window(q.decision.query or q.user_message, now=q.now)
    if date_window is not None:
        start_at, end_at = date_window
        _select_dated_turn_summary(q, start_at, end_at, deps)
    else:
        _select_undated_turn_summary(q, deps)
    if not q.records and (
        date_window is None
        or not deps.precise_day_request(q.decision.query or q.user_message)
    ):
        evidence_rows = []
        for memory_scope in q.history_scopes:
            evidence_rows.extend(
                deps.store().search(
                    query=q.decision.query or q.user_message,
                    scope=memory_scope,
                    limit=deps.policy().retrieval_limit,
                    now=q.now,
                    topic_batch_id=q.topic_batch_id,
                    sender_id=q.sender_id,
                )
            )
        q.records = deps.merge_by_source(evidence_rows, (), limit=deps.policy().retrieval_limit)
        q.fallback = "summary_to_evidence"


def _select_dated_turn_summary(
    q: MemoryTurnQuery,
    start_at: int,
    end_at: int,
    deps: MemoryServiceQueriesDependencies,
) -> None:
    recall_query = deps.recall_query(q.decision.query or q.user_message)
    focused_all, neighbors_all = _collect_dated_matches(q, start_at, end_at, recall_query, deps)
    if focused_all:
        q.records = deps.merge_by_source(focused_all, neighbors_all, limit=deps.policy().retrieval_limit)
        q.fallback = "date_query_match"
    else:
        _collect_dated_timeline(q, start_at, end_at, deps)
        q.fallback = "date_timeline" if q.records else "date_timeline_empty"


def _collect_dated_matches(
    q: MemoryTurnQuery,
    start_at: int,
    end_at: int,
    recall_query: str,
    deps: MemoryServiceQueriesDependencies,
) -> tuple[list[Any], list[Any]]:
    focused_all: list[Any] = []
    neighbors_all: list[Any] = []
    for memory_scope in q.history_scopes:
        focused = deps.store().search(
                query=recall_query,
                scope=memory_scope,
                limit=deps.policy().retrieval_limit,
                now=q.now,
                # Date-scoped recall remains lexical and bounded;
                # owner retrieval must not cross the requested day.
                layers=(
                    "observation",
                    "episodic",
                    "legacy_raw",
                    "legacy_semantic",
                    "legacy_lesson",
                    "legacy_state",
                ),
                start_at=start_at,
                end_at=end_at,
                require_term_overlap=True,
            )
        focused_all.extend(focused)
        for match in focused[: deps.policy().retrieval_limit]:
            neighbors_all.extend(
                deps.store().nearby_events(
                    scope=memory_scope,
                    center_at=match.created_at,
                    limit=4,
                    window_seconds=deps.policy().topic_window_seconds,
                    start_at=start_at,
                    end_at=end_at,
                )
            )
    return focused_all, neighbors_all


def _collect_dated_timeline(
    q: MemoryTurnQuery,
    start_at: int,
    end_at: int,
    deps: MemoryServiceQueriesDependencies,
) -> None:
    summary_rows: list[Any] = []
    for memory_scope in q.history_scopes:
        summary_rows.extend(
            deps.store().timeline(
                scope=memory_scope,
                limit=deps.policy().retrieval_limit,
                now=q.now,
                layers=("summary", "legacy_summary", "daily_summary"),
                include_facts=False,
                start_at=start_at,
                end_at=end_at,
            )
        )
    q.records = deps.merge_by_source(summary_rows, (), limit=deps.policy().retrieval_limit)
    if not q.records:
        evidence_rows: list[Any] = []
        for memory_scope in q.history_scopes:
            evidence_rows.extend(
                deps.store().timeline(
                    scope=memory_scope,
                    limit=deps.policy().retrieval_limit,
                    now=q.now,
                    layers=(
                        "observation",
                        "episodic",
                        "legacy_raw",
                        "legacy_semantic",
                        "legacy_lesson",
                        "legacy_state",
                    ),
                    include_facts=False,
                    start_at=start_at,
                    end_at=end_at,
                )
            )
        q.records = deps.merge_by_source(evidence_rows, (), limit=deps.policy().retrieval_limit)


def _select_undated_turn_summary(
    q: MemoryTurnQuery,
    deps: MemoryServiceQueriesDependencies,
) -> None:
    summary_rows: list[Any] = []
    for memory_scope in q.history_scopes:
        summary_rows.extend(
            deps.store().search(
                query=f"{q.decision.query or q.user_message} {q.decision.time_range}".strip(),
                scope=memory_scope,
                limit=deps.policy().retrieval_limit,
                now=q.now,
                layers={"summary", "legacy_summary", "daily_summary"},
                topic_batch_id=q.topic_batch_id,
                sender_id=q.sender_id,
            )
        )
    q.records = deps.merge_by_source(summary_rows, (), limit=deps.policy().retrieval_limit)


def _select_turn_retrieval(q: MemoryTurnQuery, deps: MemoryServiceQueriesDependencies) -> None:
    if deps.policy().history_needs_enabled:
        # 复用同一已授权范围和结果预算；日期/摘要分支不经这条扩展路径。
        from .query_context import search_context
        q.records = search_context(deps.store(), query=q.decision.query or q.user_message,
            scopes=q.history_scopes, now=q.now, limit=deps.policy().retrieval_limit,
            topic_batch_id=q.topic_batch_id, sender_id=q.sender_id)
        return
    evidence_rows: list[Any] = []
    for memory_scope in q.history_scopes:
        evidence_rows.extend(
            deps.store().search(
                query=q.decision.query or q.user_message,
                scope=memory_scope,
                limit=deps.policy().retrieval_limit,
                now=q.now,
                topic_batch_id=q.topic_batch_id,
                sender_id=q.sender_id,
            )
        )
    q.records = deps.merge_by_source(evidence_rows, (), limit=deps.policy().retrieval_limit)


def select_inspection_records(
    q: MemoryInspectionQuery,
    deps: MemoryServiceQueriesDependencies,
) -> None:
    from .participant_memory import recall_participant_events
    participant_records = recall_participant_events(deps.store(), deps.policy(),
        query=q.clean_query, scopes=q.history_scopes, mode=q.selected_mode,
        now=q.now, limit=q.wanted,
        date_window=deps.request_date_window(q.clean_query, now=q.now))
    if participant_records is not None:
        q.records = participant_records
        q.fallback = "participant_sources" if q.records else "participant_sources_empty"
        return
    if q.selected_mode == 'recent' or not q.clean_query and q.selected_mode == 'retrieve':
        q.records = tuple(
            deps.store().timeline(
                scope=q.scope,
                limit=q.wanted,
                now=q.now,
                layers=("observation", "episodic", "summary"),
                include_facts=True,
            )
        )
        if q.selected_mode == "retrieve" and not q.clean_query:
            q.fallback = "retrieve_without_query_to_recent"
            q.selected_mode = "recent"
    elif q.selected_mode == 'summary':
        _select_inspection_summary(q, deps)
    else:
        _select_inspection_retrieval(q, deps)


def _select_inspection_summary(
    q: MemoryInspectionQuery,
    deps: MemoryServiceQueriesDependencies,
) -> None:
    for memory_scope in q.history_scopes:
        deps.store().rollup_daily_summaries(scope=memory_scope, now=q.now)
    date_window = deps.request_date_window(q.clean_query, now=q.now)
    if date_window is not None:
        start_at, end_at = date_window
        summary_rows: list[Any] = []
        for memory_scope in q.history_scopes:
            summary_rows.extend(
                deps.store().timeline(
                    scope=memory_scope,
                    limit=q.wanted,
                    now=q.now,
                    layers=("summary", "legacy_summary", "daily_summary"),
                    include_facts=False,
                    start_at=start_at,
                    end_at=end_at,
                )
            )
        q.records = deps.merge_by_source(summary_rows, (), limit=q.wanted)
        if not q.records:
            evidence_rows: list[Any] = []
            for memory_scope in q.history_scopes:
                evidence_rows.extend(
                    deps.store().timeline(
                        scope=memory_scope,
                        limit=q.wanted,
                        now=q.now,
                        layers=("observation", "episodic"),
                        include_facts=False,
                        start_at=start_at,
                        end_at=end_at,
                    )
                )
            q.records = deps.merge_by_source(evidence_rows, (), limit=q.wanted)
        q.fallback = "date_timeline" if q.records else "date_timeline_empty"
    else:
        summary_rows = []
        for memory_scope in q.history_scopes:
            summary_rows.extend(
                deps.store().search(
                    query=q.clean_query,
                    scope=memory_scope,
                    limit=q.wanted,
                    now=q.now,
                    layers={"summary", "legacy_summary", "daily_summary"},
                )
            )
        q.records = deps.merge_by_source(summary_rows, (), limit=q.wanted)
    if not q.records and (date_window is None or not deps.precise_day_request(q.clean_query)):
        timeline_rows: list[Any] = []
        for memory_scope in q.history_scopes:
            timeline_rows.extend(
                deps.store().timeline(
                    scope=memory_scope,
                    limit=q.wanted,
                    now=q.now,
                    layers=("summary", "observation", "episodic"),
                    include_facts=False,
                )
            )
        q.records = deps.merge_by_source(timeline_rows, (), limit=q.wanted)
        q.fallback = "summary_to_timeline"


def _select_inspection_retrieval(
    q: MemoryInspectionQuery,
    deps: MemoryServiceQueriesDependencies,
) -> None:
    if deps.policy().history_needs_enabled:
        # 显式工具读取与自动检索共用桥接，工作台仍按调用者和 scope 隔离。
        from .query_context import search_context
        anchors = ()
        if deps.history_workbench() is not None:
            need, cached = deps.history_workbench().consider(scope=q.scope, requester=q.requester_id,
                query=q.clean_query, allowed_scopes=q.history_scopes, now=q.now)
            if need is not None and need.kind == "followup":
                anchors = cached
        q.records = search_context(deps.store(), query=q.clean_query, scopes=q.history_scopes,
            now=q.now, limit=q.wanted, anchors=anchors)
        if not q.records:
            q.fallback = "retrieve_empty"
        return
    evidence_rows: list[Any] = []
    for memory_scope in q.history_scopes:
        evidence_rows.extend(
            deps.store().search(
                query=q.clean_query,
                scope=memory_scope,
                limit=q.wanted,
                now=q.now,
            )
        )
    q.records = deps.merge_by_source(evidence_rows, (), limit=q.wanted)
    if not q.records:
        q.fallback = "retrieve_empty"
