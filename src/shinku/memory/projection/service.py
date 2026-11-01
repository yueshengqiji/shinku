"""MemoryService authorization adapter; callers cannot nominate extra scopes."""
from collections.abc import Mapping

from .contracts import ProjectionRequest
from ..self_activity import needs_group_activity_context


def project_context(service, *, conversation_id, conversation_type, user_message,
                    message_meta, memory_context, flow_reason, continuity_handles=None):
    if service.projection is None or not service.policy.projection.enabled:
        return None
    meta = message_meta if isinstance(message_meta, Mapping) else {}
    scope = service.scope_for(conversation_id, conversation_type)
    requester = str(meta.get("sender_id") or (conversation_id if scope.startswith("private:") else ""))
    if not requester:
        return None
    memory = memory_context if isinstance(memory_context, Mapping) else {}
    decision = memory.get("decision", {})
    mode = str(decision.get("history_mode") or "none") if isinstance(decision, Mapping) else "none"
    routed_query = decision.get("query", "") if isinstance(decision, Mapping) else ""
    routed_query = routed_query.strip() if isinstance(routed_query, str) else ""
    # Reject malformed/oversized hints as a whole; a clipped query may lose a
    # negation or target. Existing host authorization still determines scopes.
    if mode not in {"retrieve", "summary"} or len(routed_query) > 2000:
        routed_query = ""
    owner = service._is_master_request(conversation_id=conversation_id,
        conversation_type=conversation_type, message_meta=meta)
    model_retrieval = service.policy.projection.retrieval_strategy == "model"
    # 目录和既读锚点不受旧语言分类约束；仅为认证主人私聊自动扩展范围，其他调用沿用原授权。
    scope_mode = "retrieve" if model_retrieval and owner and scope.startswith("private:") else mode
    scopes = service._history_scopes(scope, owner_request=owner, mode=scope_mode)
    # Recompute access on every call; a locator retained by a previous prompt
    # must never expand this turn's authorized scopes or retain stale content.
    continuity = None if continuity_handles is None else tuple(
        (str(rid), bound_scope, "") for rid, bound_scope, _ in continuity_handles[:64] if bound_scope in scopes)
    handles = []
    for row in memory.get("records", [])[:64]:
        if not isinstance(row, Mapping) or row.get("scope") not in scopes:
            continue
        row_meta = row.get("metadata", {})
        handles.append((str(row.get("record_id") or ""), row["scope"], str(row_meta.get("source_version") or "")))
    now = int(service.clock())
    # A local recent/chat classification must not erase an explicit calendar
    # qualifier before archive discovery. This does not add any read scopes.
    window = service._request_date_window(user_message, now=now) if (
        mode in {"recent", "retrieve", "summary"} or needs_group_activity_context(user_message)) else None
    reply_ids = meta.get("reply_message_ids", [])
    reply_ids = list(reply_ids) if isinstance(reply_ids, (list,tuple)) else []
    if meta.get("reply_to_message_id"):
        reply_ids.append(str(meta["reply_to_message_id"]))
    try:
        request = ProjectionRequest(scope=scope, allowed_scopes=scopes, requester=requester,
            self_id=str(meta.get("self_id") or ""), query=user_message, now=now,
            message_id=str(meta.get("message_id") or ""), history_mode=mode,
            seed_handles=tuple(handles), reply_message_ids=tuple(reply_ids[:8]), flow_reason=flow_reason,
            start_at=window[0] if window else None, end_at=window[1] if window else None,
            continuity_handles=continuity, retrieval_query=routed_query,
            participant_ids=service.policy.owner_participant_ids if owner and "private:master" in scopes else ())
        view = service.projection.project(request)
        service.projection_status = view.diagnostics if view else {"state":"disabled"}
        return view.as_dict() if view else None
    except Exception as exc:
        service.projection_status = {"state":"failed", "error_type":type(exc).__name__}
        return None
