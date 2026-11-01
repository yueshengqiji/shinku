"""Service responsibilities: queries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import MemoryContext

from .service_models import MemoryServiceQueriesDependencies


from . import service_history_selection, service_query_sources
from . import retrieval_trace
from .service_query_models import MemoryInspectionQuery, MemoryTurnQuery


def prepare_turn(
    *,
    conversation_id: str,
    conversation_type: str,
    user_message: str,
    recent_context: str='',
    message_meta: Mapping[str, Any] | None=None,
    deps: MemoryServiceQueriesDependencies,
) -> MemoryContext:
    retrieval_trace.query("service.request", user_message, mode="automatic")
    q = MemoryTurnQuery(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        user_message=user_message,
        recent_context=recent_context,
        message_meta=message_meta,
    )
    early = service_query_sources.begin_turn(q, deps)
    if early is not None:
        retrieval_trace.observe("service.special", early.records, reason=early.decision.reason)
        return early
    early = service_query_sources.route_turn(q, deps)
    retrieval_trace.observe("service.route", mode=q.decision.history_mode, reason=q.decision.reason)
    if early is not None:
        return early
    service_history_selection.select_turn_records(q, deps)
    retrieval_trace.observe("service.selected", q.records)
    service_query_sources.refresh_turn_sources(q, deps)
    retrieval_trace.observe("service.refreshed", q.records)
    service_query_sources.render_turn_sources(q, deps)
    return service_query_sources.finish_turn_context(q, deps)


def inspect(
    *,
    conversation_id: str,
    conversation_type: str,
    requester_id: str='',
    query: str='',
    mode: str='recent',
    limit: int=8,
    message_id: str='',
    user_text: str='',
    self_id: str='',
    deps: MemoryServiceQueriesDependencies,
) -> dict[str, Any]:
    """Read memory for an explicit user/tool request.

        Automatic routing and explicit inspection are intentionally separate:
        this method never changes persona context and never writes facts.  It
        only reads the current conversation scope and returns a bounded,
        provenance-preserving result for the Agent to summarize.
        """

    retrieval_trace.query("inspection.request", query, mode=mode, limit=limit)
    q = MemoryInspectionQuery(
        conversation_id=conversation_id,
        conversation_type=conversation_type,
        requester_id=requester_id,
        query=query,
        mode=mode,
        limit=limit,
        message_id=message_id,
        user_text=user_text,
        self_id=self_id,
    )
    early = service_query_sources.begin_inspection(q, deps)
    if early is not None:
        return early
    service_history_selection.select_inspection_records(q, deps)
    retrieval_trace.observe("inspection.selected", q.records)
    service_query_sources.refresh_inspection_sources(q, deps)
    retrieval_trace.observe("inspection.refreshed", q.records)
    return service_query_sources.finish_inspection(q, deps)
