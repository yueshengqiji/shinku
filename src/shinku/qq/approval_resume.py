"""Adjudicate one pending approval against its original task and scope."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from shinku.agent.approval import ConsentResolver
from shinku.agent.budget import current_task_budget
from shinku.agent.pending import PendingApprovalRecord, PendingApprovalStore
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tasks.handoff import TaskHandoff

from .commands import QQCommandState
from .reply_models import QQAgentReply
from .turn_models import QQPreparedTurn, QQTurnEffects
from .turns import QQTurn


@dataclass(frozen=True)
class QQApprovalResumeDependencies:
    approval_store: PendingApprovalStore
    approval_session: Callable[[QQTurn], str]
    can_silence_autonomous: Callable[[QQTurn], bool]
    owner_gate_enabled: Callable[[], bool]
    async_enabled: Callable[[], bool]
    command_state: QQCommandState
    task_coordinator: AgentTaskCoordinator | None
    approval_ttl_seconds: float
    consent_resolver: ConsentResolver
    slice_session: Callable[[str], Mapping[str, Any] | None]
    discard_slice: Callable[[str], None]
    request_scope: Callable[..., Any]
    enqueue_agent_task: Callable[..., QQAgentReply]
    run_agent: Callable[..., Any]
    finish_result: Callable[..., QQAgentReply]
    effects: QQTurnEffects
    recovery_check: Callable[[dict], None] | None = None


def resume_qq_approval(
    prepared_turn: QQPreparedTurn,
    *,
    deps: QQApprovalResumeDependencies,
) -> QQAgentReply | None:
    turn = prepared_turn.turn
    trace_id = prepared_turn.trace_id
    try:
        # Passive group chatter is not a response to an approval prompt.
        # Preserve the pending task and let the ordinary model decide
        # whether to speak; do not emit owner/scope templates first.
        pending = (
            None if deps.can_silence_autonomous(turn)
            else deps.approval_store.get(deps.approval_session(turn))
        )
    except Exception as exc:
        deps.effects.trace(trace_id, "approval", status="store_error", error=type(exc).__name__)
        return QQAgentReply("failed", reason=f"approval_store_error:{type(exc).__name__}")
    if pending is None:
        return None
    reply = _validate_pending_turn(prepared_turn, pending, deps)
    if reply is not None:
        return reply
    previous_work = pending.context.get("agent_work_id")
    reply = _check_pending_task(prepared_turn, pending, previous_work, deps)
    if reply is not None:
        return reply
    if time.time() - pending.created_at > deps.approval_ttl_seconds:
        return _delete_pending(prepared_turn, deps)
    decision = deps.consent_resolver.resolve(
        request=pending.request,
        user_message=turn.combined_text,
    )
    budget = current_task_budget()
    if budget and budget.stop_reason():
        return deps.effects.deliver(turn, deps.effects.terminal_failure_text(budget.reason),
                                  agent_status="blocked", reason=budget.reason, trace_id=trace_id)
    deps.effects.trace(trace_id, "approval", status="resolved", verdict=decision.verdict)
    if decision.verdict == "approve":
        return _resume_approved(prepared_turn, pending, deps)
    if decision.verdict == "deny":
        return _deny_pending(prepared_turn, previous_work, deps)
    deps.effects.trace(trace_id, "approval", status="waiting", verdict="unclear")
    return deps.effects.deliver(
        turn,
        "我还没判断出你是不是要执行这一步。要执行刚才那项工具吗？",
        agent_status="waiting_user",
        reason="tool_approval_unclear",
        trace_id=trace_id,
    )


def _validate_pending_turn(
    prepared_turn: QQPreparedTurn,
    pending: PendingApprovalRecord,
    deps: QQApprovalResumeDependencies,
) -> QQAgentReply | None:
    turn = prepared_turn.turn
    trace_id = prepared_turn.trace_id
    is_poke = prepared_turn.is_poke
    audience_context = prepared_turn.audience_context
    requester = prepared_turn.requester
    session = prepared_turn.session
    if deps.owner_gate_enabled() and not bool(audience_context.get("is_master")):
        deps.effects.trace(trace_id, "approval", status="rejected", reason="tool_owner_only")
        return deps.effects.deliver(
            turn, deps.effects.non_master_tool_refusal, agent_status="rejected", reason="tool_owner_only",
            record_memory=not is_poke, track_chat_flow=not is_poke, trace_id=trace_id,
        )
    if deps.async_enabled():
        pending_scope = (pending.context.get("agent_requester"), pending.context.get("agent_session"))
        if not all(pending_scope):
            # Old ambiguous approvals are not executable grants. An
            # authorized user can start afresh after clearing one.
            deps.approval_store.delete(deps.approval_session(turn))
            return deps.effects.deliver(
                turn, "这条历史审批缺少可核验的请求者身份，未执行；请重新说明任务。",
                agent_status="blocked", reason="pending_approval_scope_missing", trace_id=trace_id,
            )
        if pending_scope != (requester, session):
            return deps.effects.deliver(
                turn, "这项待确认操作不属于你当前会话的任务，未执行；原审批仍保留。",
                agent_status="rejected", reason="pending_approval_scope_mismatch", trace_id=trace_id,
            )
    if not deps.command_state.is_agent_enabled():
        try:
            deps.approval_store.delete(deps.approval_session(turn))
        except Exception as exc:
            deps.effects.trace(trace_id, "approval", status="store_error", error=type(exc).__name__)
            return QQAgentReply("failed", reason=f"approval_store_error:{type(exc).__name__}")
        deps.effects.trace(trace_id, "approval", status="cancelled", reason="agent_disabled")
        return deps.effects.deliver(
            turn,
            "工具任务处理已经关闭，这一步没有执行；日常聊天不受影响。",
            agent_status="chat",
            reason="agent_disabled_pending_approval",
            trace_id=trace_id,
        )
    return None


def _check_pending_task(
    prepared_turn: QQPreparedTurn,
    pending: PendingApprovalRecord,
    previous_work: Any,
    deps: QQApprovalResumeDependencies,
) -> QQAgentReply | None:
    turn = prepared_turn.turn
    trace_id = prepared_turn.trace_id
    if deps.async_enabled() and pending.context.get("agent_sliced_execution"):
        live = deps.slice_session(str(previous_work or ""))
        if live is None:
            return deps.effects.deliver(turn, "这项任务的可执行检查点已不在当前进程，未重放待授权操作；请重新说明后续目标。",
                                      agent_status="blocked", reason="task_checkpoint_unavailable", trace_id=trace_id)
        budget_reason = live["budget"].stop_reason()
        if budget_reason:
            return deps.effects.deliver(turn, deps.effects.terminal_failure_text(budget_reason),
                                      agent_status="blocked", reason=budget_reason, trace_id=trace_id)
    if deps.async_enabled() and previous_work:
        previous = deps.task_coordinator.get(str(previous_work))
        if previous and previous.get("cancel_requested"):
            deps.approval_store.delete(deps.approval_session(turn))
            return deps.effects.deliver(turn, "那项后台任务已取消，这次没有执行待授权操作。",
                                      agent_status="canceled", reason="task_canceled", trace_id=trace_id)
    return None


def _delete_pending(
    prepared_turn: QQPreparedTurn,
    deps: QQApprovalResumeDependencies,
) -> QQAgentReply | None:
    turn = prepared_turn.turn
    trace_id = prepared_turn.trace_id
    try:
        deps.approval_store.delete(deps.approval_session(turn))
    except Exception as exc:
        deps.effects.trace(trace_id, "approval", status="store_error", error=type(exc).__name__)
        return QQAgentReply("failed", reason=f"approval_store_error:{type(exc).__name__}")
    return None


def _resume_approved(
    prepared_turn: QQPreparedTurn,
    pending: PendingApprovalRecord,
    deps: QQApprovalResumeDependencies,
) -> QQAgentReply:
    turn = prepared_turn.turn
    trace_id = prepared_turn.trace_id
    resumed_context, resumed_transcript = _approved_materials(prepared_turn, pending, deps)
    if deps.recovery_check is not None:
        from shinku.agent.task_state import TaskContractError
        try:
            deps.recovery_check(resumed_context)
        except TaskContractError as exc:
            return deps.effects.deliver(turn, "这项任务的旧回执、资料或剩余额度尚未通过恢复核对；待确认操作未执行，原审批仍保留。",
                agent_status="blocked", reason=str(exc), trace_id=trace_id)
    reply = _delete_pending(prepared_turn, deps)
    if reply is not None:
        return reply
    if deps.async_enabled():
        return deps.enqueue_agent_task(
            turn, task_id=pending.task_id, context=resumed_context,
            initial_transcript=resumed_transcript, trace_id=trace_id,
            approved_invocations=pending.request.invocations,
            handoff=TaskHandoff.resume(pending.task_id),
        )
    try:
        result = deps.run_agent(
            task_id=pending.task_id,
            context=resumed_context,
            initial_transcript=resumed_transcript,
            approved_invocations=pending.request.invocations,
            trace_id=trace_id,
        )
    except Exception as exc:
        deps.effects.trace(trace_id, "agent", status="error", error=type(exc).__name__)
        return QQAgentReply("failed", reason=f"bridge_error:{type(exc).__name__}")
    return deps.finish_result(
        turn,
        result,
        context=resumed_context,
        initial_transcript=resumed_transcript,
        task_id=pending.task_id,
        trace_id=trace_id,
    )


def _approved_materials(
    prepared_turn: QQPreparedTurn,
    pending: PendingApprovalRecord,
    deps: QQApprovalResumeDependencies,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    turn = prepared_turn.turn
    trace_id = prepared_turn.trace_id
    chat_flow = prepared_turn.chat_flow
    audience_context = prepared_turn.audience_context
    resumed_context = dict(pending.context)
    resumed_context["qq_turn"] = turn
    resumed_context["conversation_turn_id"] = trace_id
    deps.effects.conversation_step(trace_id, phase="preparation", route="agent", task_id=pending.task_id)
    resumed_context["approval_response"] = turn.combined_text
    resumed_context["chat_flow"] = chat_flow.as_dict() if chat_flow is not None else {}
    resumed_context["audience_context"] = audience_context
    resumed_context["agent_authority_revision"] = deps.command_state.authority_revision()
    resumed_context["agent_permission_mode"] = deps.command_state.permission_mode()
    resumed_context["agent_request_scope"] = deps.request_scope(
        turn, task=resumed_context.get("agent_task_state"),
    )
    resumed_transcript = list(pending.transcript)
    res_text = turn.combined_text
    first_msg = turn.messages[0] if turn.messages else None
    first_ts = getattr(first_msg, "timestamp", None)
    if first_ts is not None:
        try:
            ts_float = float(first_ts)
            if ts_float > 0:
                res_text = f"[{time.strftime('%H:%M', time.localtime(ts_float))}] {res_text}"
        except (TypeError, ValueError, OverflowError, OSError):
            pass
    resumed_transcript.append({"role": "user", "content": res_text})
    resumed_transcript.append({"role": "system", "content": "用户已通过自然语言授权执行待处理工具。"})
    return resumed_context, resumed_transcript


def _deny_pending(
    prepared_turn: QQPreparedTurn,
    previous_work: Any,
    deps: QQApprovalResumeDependencies,
) -> QQAgentReply:
    turn = prepared_turn.turn
    trace_id = prepared_turn.trace_id
    requester = prepared_turn.requester
    session = prepared_turn.session
    reply = _delete_pending(prepared_turn, deps)
    if reply is not None:
        return reply
    if deps.async_enabled() and previous_work:
        deps.discard_slice(str(previous_work))
        deps.task_coordinator.cancel(str(previous_work), profile_user_id=requester, session_id=session)
    return deps.effects.deliver(
        turn,
        "好，那就先不执行。",
        reason="tool_denied",
        trace_id=trace_id,
    )
