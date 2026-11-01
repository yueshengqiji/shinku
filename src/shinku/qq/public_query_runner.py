"""Bounded anonymous source calls, evidence checks and final delivery."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, current_task_budget
from shinku.agent.loop import AgentDecision, AgentState
from shinku.agent.reply_body import is_empty_reply
from shinku.conversation import AnswerEvidence
from shinku.tools.execution import execute_invocation
from shinku.tools.invocation import ToolInvocation
from shinku.tools.registry import ToolRegistry

from .public_query_evidence import PublicQueryEvidence
from .public_query_feedback import PublicQueryFeedback, argument_fields, rejected_followup
from .public_query_model import request_public_query_decision
from .query_models import PUBLIC_QUERY_MAX_TOOLS, QQPublicQueryRun, QQQueryDependencies
from .reply_models import QQAgentReply
from .turns import QQTurn


def run_public_query(
    turn: QQTurn,
    decision: AgentDecision,
    *,
    context: dict[str, Any],
    initial_transcript: list[dict[str, Any]],
    trace_id: str = "",
    deps: QQQueryDependencies,
) -> QQAgentReply:
    if deps.existing_query is not None:
        existing = deps.existing_query(turn, context, trace_id)
        if existing is not None:
            return existing
    run = _prepare_query(turn, decision, context, initial_transcript, trace_id, deps)
    for _ in range(PUBLIC_QUERY_MAX_TOOLS):
        budget = current_task_budget()
        batch = _admit_batch(run, budget, deps)
        if batch is None:
            break
        if not _execute_batch(run, *batch, budget, deps):
            break
        proceed, next_decision, final_only = request_public_query_decision(run, budget, deps)
        if not proceed or not _accept_next_decision(run, next_decision, final_only, budget, deps):
            break
    return _finish_query(run, deps)


def _prepare_query(
    turn: QQTurn,
    decision: AgentDecision,
    context: dict[str, Any],
    initial_transcript: list[dict[str, Any]],
    trace_id: str,
    deps: QQQueryDependencies,
) -> QQPublicQueryRun:
    deps.model.effects.conversation_step(trace_id, phase="query", route="public_query", evidence=AnswerEvidence("public", "unverified"))
    handlers = deps.public_handlers(context)
    if deps.foreground_policy is not None:
        budget = current_task_budget()
        context.setdefault("agent_query_turn_started", deps.foreground_policy.clock() -
            (budget.snapshot()["active_seconds"] if budget is not None else 0))
    authority = context.get("agent_authority_revision", deps.routes.command_state.authority_revision())
    state = AgentState(context=context, transcript=list(initial_transcript))
    parts = deps.model.build_prompt(state, include_agent_control=False)
    dialogue = [{"role": str(item.get("role")), "content": str(item.get("content") or "")}
        for item in initial_transcript if isinstance(item, Mapping)
        and item.get("role") in {"user", "assistant"} and str(item.get("content") or "").strip()]
    current_message = dialogue.pop()["content"] if dialogue and dialogue[-1]["role"] == "user" else turn.combined_text
    feedback = []
    envelopes = []
    seen = set()
    source_read = False
    schema_repaired = False
    model_outcome = ""
    goal_status = "unverified"
    search_calls = 0
    from shinku.agent.request_scope import RequestScope
    request_scope = context.get("agent_request_scope")
    goal = "\n".join(request_scope.user_messages()) if isinstance(request_scope, RequestScope) else turn.combined_text
    evidence = PublicQueryEvidence(goal=goal, context=context, handlers=handlers,
        runtime=deps.model.runtime, system_prompt=deps.model.system_prompt_for(turn), prompt_cache_key=deps.model.prompt_cache_key)
    diagnostics = PublicQueryFeedback()
    rejection = None
    model_error = ""
    safe_specs = ToolRegistry(handlers).native_specs(allowed_tool_names=set(handlers)) if handlers else []
    text = "这次没有取得足够的来源资料，还不能确认答案。"
    outcome = "failed"
    reason = "lightweight_public_query_fast_track"
    deps.model.effects.trace(trace_id, "chat_decision", kind="fast_public_query", max_tools=PUBLIC_QUERY_MAX_TOOLS)
    return QQPublicQueryRun(
        turn=turn,
        decision=decision,
        context=context,
        trace_id=trace_id,
        handlers=handlers,
        authority=authority,
        parts=parts,
        dialogue=dialogue,
        current_message=current_message,
        feedback=feedback,
        envelopes=envelopes,
        seen=seen,
        source_read=source_read,
        schema_repaired=schema_repaired,
        model_outcome=model_outcome,
        goal_status=goal_status,
        search_calls=search_calls,
        goal=goal,
        evidence=evidence,
        diagnostics=diagnostics,
        rejection=rejection,
        model_error=model_error,
        safe_specs=safe_specs,
        text=text,
        outcome=outcome,
        reason=reason,
        started=deps.foreground_policy.clock() if deps.foreground_policy is not None else time.monotonic(),
    )


def _admit_batch(
    run: QQPublicQueryRun,
    budget: TaskBudget | None,
    deps: QQQueryDependencies,
) -> tuple[Sequence[ToolInvocation], list[str]] | None:
    if (deps.routes.command_state.authority_revision() != run.authority
            or not deps.routes.command_state.is_agent_enabled() or not deps.routes.command_state.responses_enabled()
            or (budget and budget.stop_reason())):
        run.reason = "public_query_authority_or_budget_stopped"
        run.text = "查询已停止，没有继续调用工具或执行新操作。"
        run.outcome = "partial" if run.envelopes else "blocked"
        return None
    if not deps.is_public_query(run.decision, context=run.context, handlers=run.handlers):
        run.reason = "public_query_scope_guard"
        run.rejection = rejected_followup(run.decision, safe_names=set(run.handlers), registered_names=set(deps.routes.registry_handlers()))
        deps.model.effects.trace(run.trace_id, "fast_public_query_rejected", **run.rejection)
        run.text = "这次公开查询未能完成，我已停止后续执行。"
        run.outcome = "partial" if run.envelopes else "blocked"
        return None
    calls = run.decision.invocations or ((run.decision.invocation,) if run.decision.invocation else ())
    keys = [json.dumps([call.name, call.arguments], ensure_ascii=False, sort_keys=True, default=str) for call in calls]
    if len(run.envelopes) + len(calls) > PUBLIC_QUERY_MAX_TOOLS or any(key in run.seen for key in keys) or len(set(keys)) != len(keys):
        run.reason = "public_query_tool_limit"
        run.text = "这次简短查询还没有得到足够的核实结果，我没有重复搜索或继续扩展操作。"
        run.outcome = "partial" if run.envelopes else "failed"
        return None
    return calls, keys


def _execute_batch(
    run: QQPublicQueryRun,
    calls: Sequence[ToolInvocation],
    keys: list[str],
    budget: TaskBudget | None,
    deps: QQQueryDependencies,
) -> bool:
    for call, key in zip(calls, keys):
        # Recheck after each potentially slow tool, not just each batch.
        if (deps.routes.command_state.authority_revision() != run.authority
                or not deps.routes.command_state.is_agent_enabled()
                or not deps.routes.command_state.responses_enabled() or (budget and budget.stop_reason())):
            run.reason = "public_query_authority_or_budget_stopped"
            break
        if getattr(run.handlers[call.name], "agent_research_candidate_field", ""):
            # Only one reformulation/alternate-source tool call. A
            # relevant unread candidate is not a reason to search again.
            if run.search_calls >= 2 or run.search_calls and not run.evidence.allow_another_search():
                run.reason = "public_query_search_not_justified"
                deps.model.effects.trace(run.trace_id, "fast_public_query_search_guard", status="blocked", calls=run.search_calls)
                break
            run.search_calls += 1
            if (deps.routes.command_state.authority_revision() != run.authority or not deps.routes.command_state.is_agent_enabled()
                    or not deps.routes.command_state.responses_enabled() or (budget and budget.stop_reason())):
                run.reason = "public_query_authority_or_budget_stopped"
                break
        if budget:
            try:
                budget.use_tool()
            except TaskBudgetExceeded:
                run.reason = "public_query_authority_or_budget_stopped"
                break
        started = time.perf_counter()
        tool_context = {**run.context, "agent_permission_mode": deps.routes.command_state.permission_mode()}
        if deps.routes.tool_decision_mode == "task":
            # Read limits and source receipts belong to this whole query.
            # A fresh dict loses the first adapter's receipts/counts and can
            # either reject a valid candidate or reset subsequent limits.
            run.context["agent_permission_mode"] = deps.routes.command_state.permission_mode()
            tool_context = run.context
        _, envelope = execute_invocation(call, handlers=run.handlers,
            context=tool_context,
            policy=deps.routes.policy_for_permission_mode(deps.routes.command_state.permission_mode()))
        run.seen.add(key)
        run.calls.append(call)
        run.envelopes.append(envelope)
        run.evidence.observe(call, envelope, run.handlers[call.name])
        error = run.diagnostics.observe(call, envelope, run.handlers[call.name])
        if error:
            # Add host-verified structured causes to the existing result;
            # never copy raw arguments/exceptions into the trace or reply.
            envelope.data = {**(envelope.data or {}), "public_query_error": error}
            envelope.model_feedback = "[宿主核实的失败原因] " + json.dumps(error, ensure_ascii=False) + "\n" + str(envelope.model_feedback or "")
        updates = envelope.data.get("state_updates", {})
        content_field = getattr(run.handlers[call.name], "agent_research_content_field", "")
        if (envelope.status == "ok" and isinstance(updates, Mapping)
                and updates.get("site_access_ok") is True
                and content_field and str(updates.get(content_field) or "").strip()):
            run.source_read = True
        raw = str(envelope.model_feedback or "")
        run.feedback.append(f"[外部资料 tool={call.name} status={envelope.status}]\n" + raw[:6000]
            + ("\n[回执超过6000字符，末尾已截断；不得当作全文已读]" if len(raw) > 6000 else ""))
        deps.model.effects.trace(run.trace_id, "fast_public_query_tool", tool=call.name, status=envelope.status,
            tool_count=len(run.envelopes), duration_ms=round((time.perf_counter() - started) * 1000, 2),
            argument_fields=argument_fields(call), error=error)
    return _tools_can_continue(run, budget, deps)


def _tools_can_continue(
    run: QQPublicQueryRun,
    budget: TaskBudget | None,
    deps: QQQueryDependencies,
) -> bool:
    if run.reason == "public_query_authority_or_budget_stopped":
        run.text, run.outcome = "查询已停止，没有继续调用工具或执行新操作。", "partial"
        return False
    if run.reason == "public_query_search_not_justified":
        run.text, run.outcome = "已有候选尚未核实，未继续重复搜索；这次查询还没有形成有正文依据的答案。", "partial"
        return False
    # Never start another model request after permissions/budget changed.
    if (deps.routes.command_state.authority_revision() != run.authority
            or not deps.routes.command_state.is_agent_enabled() or not deps.routes.command_state.responses_enabled()
            or (budget and budget.stop_reason())):
        run.reason, run.text, run.outcome = "public_query_authority_or_budget_stopped", "查询已停止，没有继续调用工具或执行新操作。", "partial"
        return False
    return True


def _accept_next_decision(
    run: QQPublicQueryRun,
    next_decision: AgentDecision | None,
    final_only: bool,
    budget: TaskBudget | None,
    deps: QQQueryDependencies,
) -> bool:
    if next_decision is None:
        run.reason = "public_query_invalid_decision"
        run.text = "模型没有返回可用的最终答复或工具决定，已停止查询，不能确认完整答案。"
        return False
    if next_decision.kind in {"final", "wait"}:
        run.text = str(next_decision.text or "").strip()
        if is_empty_reply(run.text):
            run.text, run.outcome, run.reason = "没有取得可核实的完整答案。", "failed", "public_query_empty_answer"
        else:
            run.outcome = "waiting_user" if next_decision.kind == "wait" else next_decision.outcome
            if run.outcome not in {"completed", "partial", "failed", "blocked", "waiting_user"}:
                run.outcome = "partial"
            run.model_outcome = run.outcome
            if not run.source_read:
                run.outcome = "partial"
                run.text = "尚未核实来源正文，这次还不能确认答案；没有将模型生成的未核实信息当作查询结果。"
            elif (deps.routes.command_state.authority_revision() == run.authority and deps.routes.command_state.is_agent_enabled()
                    and deps.routes.command_state.responses_enabled() and not (budget and budget.stop_reason())):
                run.text, run.outcome, run.goal_status, review_reason = run.evidence.finish(run.text, run.outcome,
                    admitted=lambda: deps.routes.command_state.authority_revision() == run.authority
                        and deps.routes.command_state.is_agent_enabled() and deps.routes.command_state.responses_enabled()
                        and not (budget and budget.stop_reason()))
                if review_reason:
                    run.reason = review_reason
            else:
                run.reason, run.outcome, run.text = "public_query_authority_or_budget_stopped", "partial", "查询已停止，没有完成后续答案核验。"
            if not any(envelope.status == "ok" for envelope in run.envelopes):
                run.text, run.outcome = "这次查询的工具没有成功取得来源资料，还不能确认答案。", "failed"
            run.turn = deps.response_turn(run.turn, next_decision.reply_routing)
        return False
    if final_only:
        run.reason, run.outcome = "public_query_tool_limit", "partial"
        run.text = "这次查询已到简短查询的调用上限，还没有形成完整答案；没有继续执行其他操作。"
        return False
    # The first choice must explicitly say none. Missing metadata on a
    # later raw native call cannot create/continue a task in this runner.
    run.decision = next_decision if next_decision.task_link is not None else replace(next_decision, task_link={"mode": "none"})
    return True


def _finish_query(run: QQPublicQueryRun, deps: QQQueryDependencies) -> QQAgentReply:
    if run.outcome == "partial" and not any(envelope.status == "ok" for envelope in run.envelopes):
        run.outcome = "failed"
    if run.outcome != "completed":
        run.text = run.diagnostics.explain(run.text, reason=run.reason, source_read=run.source_read,
                                   rejection=run.rejection, model_error=run.model_error)
    if deps.continue_query is not None:
        continued = deps.continue_query(run)
        if continued is not None:
            return continued
    deps.model.effects.trace(run.trace_id, "fast_public_query_finished", tool_count=len(run.envelopes),
                source_read=run.source_read, outcome=run.outcome, model_outcome=run.model_outcome,
                goal_status=run.goal_status, verified=run.evidence.corrected,
                extra_model_calls=run.evidence.ledger.review_calls + run.evidence.ledger.answer_review_calls + run.evidence.correction_calls, reason=run.reason,
                error_codes=[error["code"] for error in run.diagnostics.errors],
                rejection_code=run.rejection["code"] if run.rejection else "")
    is_poke = deps.is_poke(run.turn)
    deps.model.effects.conversation_step(run.trace_id, phase="verification",
        evidence=AnswerEvidence.from_source_ledger(run.evidence.ledger, kind="public", query_count=len(run.envelopes)))
    if deps.express_query is not None:
        return deps.express_query(run.turn, run.text, context=run.context, status=run.outcome, reason=run.reason,
            route="public_query", envelopes=run.envelopes, trace_id=run.trace_id,
            answer_evidence=AnswerEvidence.from_source_ledger(run.evidence.ledger, kind="public", query_count=len(run.envelopes)))
    reply = deps.model.effects.deliver(run.turn, run.text, agent_status=run.outcome, reason=run.reason,
        record_memory=not is_poke, track_chat_flow=not is_poke, trace_id=run.trace_id,
        tool_envelopes=run.envelopes, autonomous_authorized=bool(run.context.get("qq_chat_model_decision_valid")))
    from shinku.agent_runtime.results import capture_query_result
    capture_query_result(run.context, run.turn, reply, run.envelopes, route="public_query",
        answer_evidence=AnswerEvidence.from_source_ledger(run.evidence.ledger, kind="public", query_count=len(run.envelopes)))
    return reply
