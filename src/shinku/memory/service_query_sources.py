"""Service responsibilities: queries."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from .models import MemoryContext, MemoryDecision, MemoryRecord
from .self_activity import is_self_group_recall

from .service_models import MemoryServiceQueriesDependencies


from .service_query_models import MemoryInspectionQuery, MemoryTurnQuery


def begin_turn(q: MemoryTurnQuery, deps: MemoryServiceQueriesDependencies) -> MemoryContext | None:
    q.now = int(deps.clock())
    deps.store().expire(now=q.now)
    q.scope = deps.scope_for(q.conversation_id, q.conversation_type)
    q.effective_meta = dict(q.message_meta or {})
    if (not (deps.policy().projection.enabled and deps.policy().projection.retrieval_strategy == "model")
            and is_self_group_recall(q.user_message, aliases=deps.character.aliases)):
        requester = str(q.effective_meta.get("sender_id") or (q.conversation_id if not q.scope.startswith("group:") else ""))
        result = deps.inspect_self_group_activity(conversation_id=q.conversation_id,
            conversation_type=q.conversation_type, requester_id=requester, query=q.user_message,
            self_id=str(q.effective_meta.get("self_id") or ""), limit=deps.policy().retrieval_limit)
        q.decision = MemoryDecision(True, memory_type="self_activity", scope="bot_group_activity",
            query=q.user_message[:deps.policy().max_query_chars], confidence=1.0,
            reason="self_group_activity", history_mode="retrieve", local_signals=("self_group_recall",))
        q.records = tuple(MemoryRecord(record_id=row["record_id"], layer="episodic", scope=row["scope"],
            text=row["text"], created_at=row["created_at"], source_id=row["source_id"], metadata=row["metadata"]) for row in result["records"])
        context = MemoryContext(decision=q.decision, records=q.records, rendered=result["rendered"],
            retrieved_count=result["count"], injected_chars=len(result["rendered"]),
            audit={"self_activity": {key: result[key] for key in ("access", "count", "coverage", "absence_means", "time_window")},
                   "retrieval_scopes": result["retrieval_scopes"]})
        deps.record_audit(conversation_id=q.conversation_id, scope=q.scope, now=q.now, context=context)
        return context
    return None


def route_turn(q: MemoryTurnQuery, deps: MemoryServiceQueriesDependencies) -> MemoryContext | None:
    q.message_id = str(q.effective_meta.get("message_id") or "").strip()
    if q.message_id:
        observed = deps.store().observation_for_message(scope=q.scope, message_id=q.message_id)
        if observed is not None:
            # The passive observer is the source of truth for reply links,
            # media markers and the local topic batch.
            q.effective_meta = {**observed.metadata, **q.effective_meta}
    latest = deps.store().latest_observation(scope=q.scope)
    topic_state: dict[str, Any] = {}
    if latest is not None:
        topic_state = {
            "message_id": latest.metadata.get("message_id", ""),
            "created_at": latest.created_at,
            "topic_batch_id": latest.metadata.get("topic_batch_id", ""),
            **latest.metadata,
        }
    model_retrieval = (deps.policy().projection.enabled
                       and deps.policy().projection.retrieval_strategy == "model")
    if model_retrieval:
        # The main conversation already sees the complete current turn and
        # adjacent dialogue. Its scoped read tool carries any semantic query.
        q.decision = MemoryDecision(False, history_mode="none", reason="main_model_retrieval")
    else:
        q.decision = deps.router().decide(
            user_message=q.user_message,
            recent_context=q.recent_context,
            conversation_type=q.conversation_type,
            message_meta=q.effective_meta,
            topic_state=topic_state,
        )
    q.owner_request = deps.is_master_request(
        conversation_id=q.conversation_id,
        conversation_type=q.conversation_type,
        message_meta=q.effective_meta,
    )
    q.sender_id = str(q.effective_meta.get("sender_id") or
        (q.conversation_id if str(q.conversation_type).lower() != "group" else "")).strip()
    from .workbench import route_history_need
    if not model_retrieval:
        q.decision, q.natural_records, q.history_need = deps.route_history_need(route_history_need, scope=q.scope,
            requester=q.sender_id, query=q.user_message, owner_request=q.owner_request, decision=q.decision,
            now=q.now, recent_context=q.recent_context, message_meta=q.effective_meta)
    q.history_scopes = deps.history_scopes(
        q.scope,
        owner_request=q.owner_request,
        mode=q.decision.history_mode,
    )
    q.recent_dialogue = deps.private_recent_dialogue(
        scope=q.scope, conversation_id=q.conversation_id, conversation_type=q.conversation_type,
        requester=q.sender_id, self_id=str(q.effective_meta.get("self_id") or "").strip(), now=q.now,
    )
    q.dialogue_audit = {"private_dialogue_count": int(bool(q.recent_dialogue)),
                      "private_dialogue_chars": len(q.recent_dialogue)}
    if q.decision.history_mode == "none":
        context = MemoryContext(
            decision=q.decision,
            recent_dialogue=q.recent_dialogue,
            audit={
                "history_mode": "none",
                "retrieved_count": 0,
                "injected_chars": 0,
                "retrieval_scopes": [q.scope],
                **q.dialogue_audit,
                **({"history_need": q.history_need} if q.history_need else {}),
            },
        )
        deps.record_audit(
            conversation_id=q.conversation_id,
            scope=q.scope,
            now=q.now,
            context=context,
        )
        return context
    return None


def refresh_turn_sources(q: MemoryTurnQuery, deps: MemoryServiceQueriesDependencies) -> None:
    if q.decision.history_mode != "recent":
        if q.natural_records is None:
            q.records = deps.current_records(q.records, scopes=q.history_scopes, subject=q.sender_id,
                query=q.decision.query or q.user_message, now=q.now, limit=deps.policy().retrieval_limit)
            q.records = deps.evidence_records(q.records, query=q.decision.query or q.user_message, scopes=q.history_scopes,
                requester=q.sender_id, now=q.now, limit=deps.policy().retrieval_limit)
        if deps.source_reader() is not None:
            if q.natural_records is None:
                q.records = deps.source_reader().read_rounds(q.records, now=q.now, limit=deps.policy().retrieval_limit)
            else:
                # Natural preferences are exact current sources, not an
                # opportunity to append old assistant advice/reply examples.
                fresh = deps.source_reader().read_sources(requests=tuple((row.record_id, row.scope,
                    row.metadata.get("source_version", "")) for row in q.records), now=q.now)
                q.records = tuple(replace(current, score=old.score, metadata={**current.metadata,
                    **{key: value for key, value in old.metadata.items() if key in {"current_understanding", "query_bridge"}}})
                    for old, current in zip(q.records, fresh) if current is not None)


def render_turn_sources(q: MemoryTurnQuery, deps: MemoryServiceQueriesDependencies) -> None:
    # 工作台只能记住实际展示的来源句柄。一般检索也可承接，不再限于少数项目名。
    workbench = deps.history_workbench() if deps.policy().history_needs_enabled else None
    history_delivered = [] if workbench is not None else None
    q.rendered, q.evidence_package = deps.render_evidence(q.records, query=q.decision.query or q.user_message,
        scopes=q.history_scopes, requester=q.sender_id, mode=q.decision.history_mode, now=q.now, limit=deps.policy().retrieval_limit,
        history_need=q.history_need if q.natural_records is not None else None, delivered_records=history_delivered)
    if workbench is not None and q.decision.history_mode == "retrieve":
        workbench.put(scope=q.scope, requester=q.sender_id, topic=(q.history_need or {}).get("topic", "source"),
            records=history_delivered, allowed_scopes=q.history_scopes, now=q.now)
    q.own_evidence: tuple[Any, ...] = ()
    if q.decision.history_mode == "recent" and "group" in str(q.conversation_type).lower():
        q.own_evidence = deps.own_utterance_evidence(
            scope=q.scope, now=q.now, user_message=q.user_message, recent_records=q.records,
        )
        if q.own_evidence:
            q.rendered += "\n" + deps.render_own_utterances(q.own_evidence)


def finish_turn_context(
    q: MemoryTurnQuery,
    deps: MemoryServiceQueriesDependencies,
) -> MemoryContext:
    context = MemoryContext(
        decision=q.decision,
        records=q.records,
        rendered=q.rendered,
        recent_dialogue=q.recent_dialogue,
        retrieved_count=len(q.records),
        injected_chars=len(q.rendered),
        audit={
            "history_mode": q.decision.history_mode,
            "retrieved_count": len(q.records),
            "injected_chars": len(q.rendered),
            "scope": q.scope,
            "retrieval_scopes": list(q.history_scopes),
            "owner_memory": q.owner_request,
            "fallback": q.fallback,
            "topic_batch_id": q.topic_batch_id,
            "own_utterance_count": len(q.own_evidence),
            **q.dialogue_audit,
            **({"history_need": q.history_need} if q.history_need else {}),
            **({"evidence_package": q.evidence_package} if q.evidence_package is not None else {}),
        },
    )
    deps.record_audit(
        conversation_id=q.conversation_id,
        scope=q.scope,
        now=q.now,
        context=context,
    )
    deps.bind_usage_context(
        scope=q.scope,
        message_id=q.message_id,
        requester_id=str(q.effective_meta.get("sender_id") or (q.conversation_id if str(q.conversation_type).lower() != "group" else "")),
        records=q.records,
        user_text=q.user_message,
    )
    return context


def begin_inspection(
    q: MemoryInspectionQuery,
    deps: MemoryServiceQueriesDependencies,
) -> dict[str, Any] | None:
    if is_self_group_recall(q.query or q.user_text, aliases=deps.character.aliases):
        return deps.inspect_self_group_activity(conversation_id=q.conversation_id,
            conversation_type=q.conversation_type, requester_id=q.requester_id,
            query=q.query or q.user_text, self_id=q.self_id, limit=q.limit)
    q.now = int(deps.clock())
    deps.store().expire(now=q.now)
    q.scope = deps.scope_for(q.conversation_id, q.conversation_type)
    q.selected_mode = str(q.mode or "recent").strip().lower()
    if q.selected_mode not in {"recent", "retrieve", "summary"}:
        q.selected_mode = "recent"
    q.clean_query = str(q.query or "").strip()[: deps.policy().max_query_chars]
    q.wanted = max(1, min(deps.policy().max_results, int(q.limit or 8)))
    q.owner_request = deps.is_master_request(
        conversation_id=q.conversation_id,
        conversation_type=q.conversation_type,
        message_meta={"sender_id": q.requester_id},
    )
    q.history_scopes = deps.history_scopes(
        q.scope,
        owner_request=q.owner_request,
        mode=q.selected_mode,
    )
    q.records: tuple[Any, ...]
    q.fallback = ""
    return None


def refresh_inspection_sources(
    q: MemoryInspectionQuery,
    deps: MemoryServiceQueriesDependencies,
) -> None:
    if q.selected_mode != "recent":
        q.records = deps.current_records(q.records, scopes=q.history_scopes, subject=q.requester_id,
            query=q.clean_query, now=q.now, limit=q.wanted)
        q.records = deps.evidence_records(q.records, query=q.clean_query, scopes=q.history_scopes,
            requester=q.requester_id, now=q.now, limit=q.wanted)
        if deps.source_reader() is not None:
            q.records = deps.source_reader().read_rounds(q.records, now=q.now, limit=q.wanted)


def finish_inspection(
    q: MemoryInspectionQuery,
    deps: MemoryServiceQueriesDependencies,
) -> dict[str, Any]:
    workbench = deps.history_workbench() if deps.policy().history_needs_enabled else None
    delivered = [] if workbench is not None else None
    q.rendered, q.evidence_package = deps.render_evidence(q.records, query=q.clean_query, scopes=q.history_scopes,
        requester=q.requester_id, mode=q.selected_mode, now=q.now, limit=q.wanted, delivered_records=delivered)
    if workbench is not None:
        if q.selected_mode == "retrieve":
            workbench.put(scope=q.scope, requester=q.requester_id, topic="source",
                records=delivered, allowed_scopes=q.history_scopes, now=q.now)
        else:
            workbench.clear(scope=q.scope, requester=q.requester_id)
    if not q.rendered:
        q.rendered = ""
    inspection = {
        "scope": q.scope,
        "retrieval_scopes": list(q.history_scopes),
        "owner_memory": q.owner_request,
        "mode": q.selected_mode,
        "query": q.clean_query,
        "read_at": {"unix": q.now, "local_iso": datetime.fromtimestamp(q.now, timezone.utc).astimezone().isoformat()},
        "count": len(q.records),
        "fallback": q.fallback,
        "records": [record.as_dict() for record in q.records],
        "rendered": q.rendered,
        **({"evidence_package": q.evidence_package} if q.evidence_package is not None else {}),
    }
    deps.bind_usage_context(scope=q.scope, message_id=q.message_id, requester_id=q.requester_id, records=q.records, user_text=q.user_text)
    return {
        **inspection,
        "followup_context": (
            f"已查看当前会话记忆（作用域：{q.scope}，模式：{q.selected_mode}，"
            f"结果：{len(q.records)} 条）。\n{q.rendered or '没有足够的可核实内容。'}"
        ),
    }
