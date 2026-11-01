"""QQ consumes canonical results and keeps its existing sender/memory hooks."""
from __future__ import annotations

from shinku.agent_runtime.results import build_query_result, export_task_result
from shinku.conversation import ConversationScope
from shinku.result_expression.facts import fact_blocks, template
from .agent_call import scope_for_turn
from .reply_models import QQAgentReply


def current_guard(bridge, turn, context, source, trace_id):
    scope, _ = scope_for_turn(turn)
    authority = context.get("agent_authority_revision", bridge.command_state.authority_revision())
    def check():
        if scope_for_turn(turn)[0] != scope:
            return "expression_scope_mismatch"
        if not bridge.command_state.responses_enabled():
            return "responses_disabled"
        if bridge.command_state.authority_revision() != authority:
            return "agent_permission_changed"
        if not bridge.command_state.is_agent_enabled():
            return "agent_disabled"
        if bridge._owner_gate_enabled() and scope.requester not in bridge.master_user_ids:
            return "requester_unauthorized"
        task_id = source.get("task_id")
        if task_id:
            task = bridge.task_store.get(task_id, requester=scope.requester, session=scope.session)
            if not task or (task.get("execution_attempt", 0), task["revision"]) != (source["attempt"], source["version"]):
                return "stale_result"
            if task.get("cancel_requested") and source["execution"]["status"] != "canceled":
                return "task_canceled"
            stored = task.get("execution_result")
            if stored and stored.get("result_hash") != source["result_hash"]:
                return "result_state_conflict"
            if not stored and export_task_result(task).get("result_hash") != source["result_hash"]:
                return "result_state_conflict"
            work_id = task.get("execution_work_id")
            if work_id:
                work = bridge.task_coordinator.get(work_id, profile_user_id=scope.requester, session_id=scope.session)
                if not work or work["attempt"] != task.get("execution_work_attempt"):
                    return "stale_execution_attempt"
                if work.get("cancel_requested") and source["execution"]["status"] != "canceled":
                    return "task_canceled"
                recovered_commit = bool(bridge.task_store.result_events and
                    bridge.task_store.result_events.committed_after_restart(task, work))
                if source["execution"]["status"] == "completed" and work["status"] in {"failed", "canceled", "blocked"} and not recovered_commit:
                    return "execution_state_conflict"
        elif not source.get("operation_id") or source["operation_id"] != turn.primary_message_id:
            return "expression_context_binding_invalid"
        elif trace_id:
            snapshot = bridge.conversation_runtime.snapshot(ConversationScope(turn.conversation_type, turn.conversation_id, scope.requester))
            operation = next((row for row in snapshot["recent_turns"] if row["turn_id"] == trace_id), None)
            if not operation or operation["status"] != "running" or operation["lease_until"] <= bridge.conversation_runtime.clock():
                return "expression_context_expired"
        return ""
    return scope, check


def publish_expression(bridge, turn, source, *, context, trace_id, status, reason, envelopes, remember=None,
                       prepared=None, lease_guard=None):
    scope, guard = current_guard(bridge, turn, context, source, trace_id)
    if lease_guard is not None:
        host_guard = guard
        guard = lambda: host_guard() or lease_guard()
    service = bridge.result_expression
    record = prepared if prepared is not None else service.prepare(source, scope=scope, check_current=guard)
    context["agent_execution_result" if source.get("task_id") else "agent_query_result"] = source
    context["agent_result_expression"] = record
    bridge._trace(trace_id, "result_expression", status=record["status"],
        result_hash=source["result_hash"], result_version=source["version"],
        expression_id=record.get("expression_id", ""), fallback_reason=record.get("fallback_reason", ""))
    if not service.reserve_publication(record, scope=scope, check_current=guard):
        return QQAgentReply(status, reason=record.get("fallback_reason") or "expression_publication_rejected", delivery_status="suppressed")
    if remember is not None:
        remember(template(fact_blocks(source)))
    # Memory is auxiliary and can take time; recheck after that hook too.
    if guard():
        service.record_delivery(record, scope=scope, receipt={"status": "suppressed", "sent": False})
        return QQAgentReply(status, reason="expression_publication_stale", delivery_status="suppressed")
    if source["execution"]["status"] == "unverified" and status == "completed":
        status = "partial"
    try:
        reply = bridge._deliver_text(turn, record["text"], agent_status=status, reason=reason,
            record_memory=not bridge._is_poke_turn(turn),
            track_chat_flow=not bridge._is_poke_turn(turn) and not context.get("agent_async_execution"),
            trace_id=trace_id, tool_envelopes=envelopes,
            autonomous_authorized=bool(context.get("qq_chat_model_decision_valid")),
            expression_id=record["expression_id"], result_hash=source["result_hash"],
            publication_guard=guard)
    except Exception:
        # The existing delivery boundary normally converts transport errors.
        # An unexpected host failure can have unknown external effects.
        reply = QQAgentReply(status, reason="expression_delivery_unknown", delivery_status="uncertain")
    receipt = {"status": reply.delivery_status, "sent": reply.sent,
               "message_ids": list(reply.delivery.message_ids) if reply.delivery is not None else []}
    try:
        record = service.record_delivery(record, scope=scope, receipt=receipt)
        context["agent_result_expression"] = record
    except Exception:
        bridge._trace(trace_id, "result_expression_receipt", status="unknown")
    context["agent_result_expression_delivery"] = receipt
    return reply


def finish_task_expression(bridge, turn, result, *, context, task_id, trace_id=""):
    scope, _ = scope_for_turn(turn)
    try:
        task = bridge.task_store.get(task_id, requester=scope.requester, session=scope.session)
    except Exception:
        return QQAgentReply(result.status, reason="result_event_store_unavailable", delivery_status="suppressed")
    if not task or not task.get("execution_result"):
        return QQAgentReply(result.status, reason="expression_result_unavailable", delivery_status="suppressed")
    source = task["execution_result"]
    expected = context.get("agent_task_state")
    if (not isinstance(expected, dict) or expected.get("task_id") != task_id
            or (expected.get("execution_attempt", 0), expected.get("revision")) != (source["attempt"], source["version"])):
        return QQAgentReply(result.status, reason="stale_result", delivery_status="suppressed")
    def remember(outcome):
        bridge._record_agent_task_memory(turn, task_id=task_id, status=result.status, outcome=outcome,
            reason=result.reason, rounds=result.rounds, tool_envelopes=result.tool_envelopes,
            transcript=result.transcript, trace_id=trace_id)
    from shinku.result_expression.events import ResultEventConsumer
    events = bridge.result_expression.archive.events
    try:
        event = events.enqueue(source, scope=scope)
    except Exception:
        return QQAgentReply(result.status, reason="result_event_store_unavailable", delivery_status="suppressed")
    _, guard = current_guard(bridge, turn, context, source, trace_id)
    project_task_experience(bridge, turn, task_id=task_id, guard=guard, context=context, trace_id=trace_id)
    def publish(current, prepared, lease_guard):
        reply = publish_expression(bridge, turn, current, context=context, trace_id=trace_id, status=result.status,
            reason=result.reason, envelopes=result.tool_envelopes, remember=remember,
            prepared=prepared, lease_guard=lease_guard)
        receipt = context.get("agent_result_expression_delivery", {"status": reply.delivery_status, "sent": reply.sent})
        if "agent_result_expression_delivery" in context:
            stored = bridge.result_expression.archive.get(prepared["expression_id"], scope=scope)
            if stored and stored["delivery"]["status"] != "not_sent":
                receipt = stored["delivery"]
        return receipt, reply
    consumed = ResultEventConsumer(bridge.result_expression).consume(event["event_id"],
        scope=scope, check_current=guard, publish=publish)
    context["agent_result_event"] = consumed.get("event")
    project_task_experience(bridge, turn, task_id=task_id, guard=guard, context=context, trace_id=trace_id)
    return consumed.get("value") or QQAgentReply(result.status,
        reason=consumed.get("error") or "result_event_already_consumed", delivery_status="suppressed")


def project_task_experience(bridge, turn, *, task_id, guard, context, trace_id=""):
    """Independent memory consumer; never affects execution/publication status."""
    projector = getattr(bridge, "task_experience", None)
    if projector is None:
        return
    scope, _ = scope_for_turn(turn)
    try:
        result = projector.project(task_id=task_id, scope=scope, still_authorized=lambda: guard() == "",
            metadata={"shinku_involved": True})
    except Exception:
        result = {"status": "rejected", "reason": "experience_store_unavailable"}
    context["agent_task_experience"] = result
    bridge._trace(trace_id, "agent_task_experience", **result)


def replay_task_expression(bridge, turn, *, task_id, context, trace_id=""):
    """Host-authorized replay with an explicitly resolved original QQ route.

    No tool or model execution; original routing/authority must be supplied by
    the host. Startup does not automatically invoke this operation.
    """
    from shinku.agent.loop import AgentRunResult
    scope, _ = scope_for_turn(turn)
    task = bridge.task_store.get(task_id, requester=scope.requester, session=scope.session)
    if not task or not task.get("execution_result"):
        return QQAgentReply("blocked", reason="replay_result_unavailable", delivery_status="suppressed")
    if turn.primary_message_id not in {r.get("message_id") for r in task.get("requests", [])}:
        return QQAgentReply("blocked", reason="replay_route_unconfirmed", delivery_status="suppressed")
    context["agent_task_state"] = task
    source = task["execution_result"]
    result = AgentRunResult(status=source["execution"]["status"], reason=source["execution"].get("reason", ""),
        final_text="", completion=task.get("completion", {}), rounds=0)
    return finish_task_expression(bridge, turn, result, context=context, task_id=task_id, trace_id=trace_id)


def finish_query_expression(bridge, turn, text, *, context, status, reason, route, envelopes,
                            trace_id="", answer_evidence=None, source_guard=None):
    source = build_query_result(context, turn, envelopes, route=route, status=status, reason=reason,
                               text=text, answer_evidence=answer_evidence)
    if source is None:
        return QQAgentReply(status, reason="expression_scope_missing", delivery_status="suppressed")
    # 复用原表达/发布守卫执行历史来源回查；生成期间撤销后不发布旧原文或旧草稿。
    return publish_expression(bridge, turn, source, context=context, trace_id=trace_id,
                              status=status, reason=reason, envelopes=envelopes,
                              lease_guard=source_guard)
