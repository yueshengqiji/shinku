"""Validated admission, shared checkpoints and cooperative QQ background slices."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from shinku.agent.budget import TaskBudget, bind_task_budget, current_task_budget
from shinku.agent.loop import AgentDecision, AgentRunResult
from shinku.agent.task_state import AgentTaskStore, TaskContractError
from shinku.llm.turn_usage import bind_qq_turn_usage
from shinku.tasks.coordinator import AgentTaskContinuation, AgentTaskCoordinator, AgentTaskOutcome
from shinku.tasks.handoff import TaskHandoff, TaskHandoffPhase
from shinku.tools.invocation import ToolInvocation

from .commands import QQCommandState
from .reply_models import QQAgentReply
from .trace import TurnTraceStore
from .turn_models import QQTurnEffects, _PreparedAgentTask
from .turns import QQTurn


@dataclass(frozen=True)
class QQBackgroundTaskDependencies:
    coordinator: Callable[[], AgentTaskCoordinator]
    command_state: Callable[[], QQCommandState]
    task_store: Callable[[], AgentTaskStore]
    trace_store: Callable[[], TurnTraceStore]
    budget_limits: Callable[[], Mapping[str, Any]]
    approval_ttl_seconds: Callable[[], float]
    task_scope: Callable[[QQTurn], tuple[str, str]]
    primary_request: Callable[[QQTurn], str]
    slice_lock: Any
    slice_sessions: dict[str, dict[str, Any]]
    commit_prepared_task: Callable[..., TaskHandoff]
    run_agent: Callable[..., AgentRunResult]
    finish_result: Callable[..., QQAgentReply]
    preparation_failure_reply: Callable[..., QQAgentReply]
    execution_failure_reply: Callable[..., QQAgentReply]
    conversation_finish: Callable[[str, QQAgentReply], None]
    queued_lease: Callable[[], float]
    effects: QQTurnEffects
    logger: logging.Logger
    cleanup_checkpoint: Callable[..., None] | None = None


@dataclass
class QQBackgroundWork:
    turn: QQTurn
    trace_id: str
    initial_transcript: list[dict[str, Any]]
    requester: str
    scope_session: str
    work_id: str
    work_attempt: int
    session: dict[str, Any]
    budget: TaskBudget
    authority_revision: int
    active_task_id: str
    decision: AgentDecision | None
    approved: Sequence[ToolInvocation] | None
    prepared_task: _PreparedAgentTask | None
    handoff_state: TaskHandoff
    queued_handoff: TaskHandoff
    prior_session: dict[str, Any] | None
    target: str
    context: dict[str, Any]


def enqueue_background_task(
    turn: QQTurn,
    *,
    task_id: str,
    context: dict[str, Any],
    initial_transcript: list[dict[str, Any]],
    first_decision: AgentDecision | None = None,
    approved_invocations: Sequence[ToolInvocation] | None = None,
    handoff: TaskHandoff | None = None,
    prepared_task: _PreparedAgentTask | None = None,
    trace_id: str = "",
    deps: QQBackgroundTaskDependencies,
) -> QQAgentReply:
    resolved = _resolve_handoff(turn, context, handoff, prepared_task, trace_id, deps)
    if isinstance(resolved, QQAgentReply):
        return resolved
    handoff, queued_handoff, requester, scope_session = resolved
    work = _prepare_work(
        turn, context, initial_transcript, first_decision, approved_invocations,
        handoff, prepared_task, queued_handoff, requester, scope_session, trace_id, deps,
    )

    def queued_cancel():
        _cancel_queued_work(work, deps)

    def handler(cancel_event):
        return _run_worker(work, cancel_event, handler, queued_cancel, deps)

    return _submit_work(work, handler, queued_cancel, deps)


def _resolve_handoff(
    turn: QQTurn,
    context: dict[str, Any],
    handoff: TaskHandoff | None,
    prepared_task: _PreparedAgentTask | None,
    trace_id: str,
    deps: QQBackgroundTaskDependencies,
) -> tuple[TaskHandoff, TaskHandoff, str, str] | QQAgentReply:
    requester, scope_session = deps.task_scope(turn)
    if requester == "unknown" or not turn.conversation_id or turn.conversation_type not in {"private", "group"}:
        return QQAgentReply("blocked", reason="background_task_scope_missing")
    if handoff is None and prepared_task is not None:
        handoff = prepared_task.handoff
    if handoff is None:
        state = context.get("agent_task_state")
        if not isinstance(state, Mapping) or not state.get("task_id"):
            return QQAgentReply("blocked", reason="background_task_handoff_missing")
        handoff = TaskHandoff.resume(str(state["task_id"]))
    if prepared_task is not None and prepared_task.handoff != handoff:
        return QQAgentReply("blocked", reason="background_task_handoff_mismatch")
    if prepared_task is None:
        state = context.get("agent_task_state")
        if not isinstance(state, Mapping) or str(state.get("task_id") or "") != handoff.target_task_id:
            return QQAgentReply("blocked", reason="background_task_handoff_uncommitted")
    target = handoff.target_task_id
    for record in deps.coordinator().list(profile_user_id=requester, session_id=scope_session, limit=200):
        if (str(record["metadata"].get("agent_task_id") or "").strip() == target
                and (record["status"] == "queued" or record["worker_active"])):
            return deps.effects.deliver(
                turn, f"这项任务仍在后台执行，任务号 {record['task_id']}；没有重复启动。",
                agent_status="waiting_background", reason="background_task_already_active",
                record_memory=False, track_chat_flow=False, trace_id=trace_id,
            )
    try:
        queued_handoff = handoff.advance(TaskHandoffPhase.QUEUED)
    except TaskContractError as exc:
        return deps.preparation_failure_reply(turn, exc, context=context, trace_id=trace_id)
    return handoff, queued_handoff, requester, scope_session


def _prepare_work(
    turn: QQTurn,
    context: dict[str, Any],
    initial_transcript: list[dict[str, Any]],
    first_decision: AgentDecision | None,
    approved_invocations: Sequence[ToolInvocation] | None,
    handoff: TaskHandoff,
    prepared_task: _PreparedAgentTask | None,
    queued_handoff: TaskHandoff,
    requester: str,
    scope_session: str,
    trace_id: str,
    deps: QQBackgroundTaskDependencies,
) -> QQBackgroundWork:
    target = handoff.target_task_id
    prior_work_id = str(context.get("agent_work_id") or "") if approved_invocations else ""
    with deps.slice_lock:
        prior_session = deps.slice_sessions.get(prior_work_id)
    query = context.get("agent_query_checkpoint")
    if query is not None:
        from shinku.agent_runtime.query_handoff import ForegroundQueryCheckpoint
        if not isinstance(query, ForegroundQueryCheckpoint) or query.task_id != target:
            raise TaskContractError("host_query_checkpoint_required")
    work_id = prior_work_id if prior_session else query.work_id if query is not None else f"work:{uuid.uuid4().hex}"
    prior_record = deps.coordinator().get(work_id) if prior_session else None
    work_attempt = int(prior_record["attempt"]) + 1 if prior_record else 1
    session = prior_session or {"budget": current_task_budget() or TaskBudget(**deps.budget_limits()), "checkpoint": None}
    budget = session["budget"]
    work_context = dict(context)
    authority_revision = work_context.get("agent_authority_revision", deps.command_state().authority_revision())
    work_context.update({"agent_work_id": work_id, "agent_work_attempt": work_attempt,
                         "agent_async_execution": True, "agent_sliced_execution": True})
    if prior_session and session["checkpoint"] is not None:
        session["checkpoint"].state.transcript = [dict(item) for item in initial_transcript]
    active_task_id = target
    decision = prepared_task.decision if prepared_task is not None else first_decision
    approved = approved_invocations
    handoff_state = handoff
    return QQBackgroundWork(
        turn=turn,
        trace_id=trace_id,
        initial_transcript=initial_transcript,
        requester=requester,
        scope_session=scope_session,
        work_id=work_id,
        work_attempt=work_attempt,
        session=session,
        budget=budget,
        authority_revision=authority_revision,
        active_task_id=active_task_id,
        decision=decision,
        approved=approved,
        prepared_task=prepared_task,
        handoff_state=handoff_state,
        queued_handoff=queued_handoff,
        prior_session=prior_session,
        target=target,
        context=work_context,
    )


def _cancel_queued_work(work: QQBackgroundWork, deps: QQBackgroundTaskDependencies) -> None:
    _cleanup_work(work, deps, "task_canceled")
    with deps.slice_lock:
        deps.slice_sessions.pop(work.work_id, None)
    task = work.context.get("agent_task_state")
    if isinstance(task, dict):
        task["status"] = "canceled"
        task["notes"]["blocker"] = "task_canceled"
        store = deps.task_store()
        if getattr(store, "result_events", None) is not None:
            from shinku.agent_runtime.results import export_task_result
            now = store.clock()
            task.pop("result_commit_pending", None)
            task.update(stop_reason="task_canceled", cancel_requested=True,
                completion={"accepted": False, "reason": "task_canceled"},
                completion_attempt=int(task.get("execution_attempt") or 0))
            task["execution_result"] = export_task_result({**task, "revision": task["revision"] + 1, "updated": now},
                status="canceled", reason="task_canceled")
            saved = store.save(task, now=now)
        else:
            saved = store.save(task)
        if isinstance(saved, dict):
            task.clear()
            task.update(saved)
    deps.trace_store().finish(work.trace_id, status="canceled", agent_status="canceled", reason="task_canceled")
    deps.conversation_finish(work.trace_id, QQAgentReply("canceled", reason="task_canceled"))


def _cleanup_work(work, deps, reason):
    checkpoint = work.session.get("checkpoint")
    if checkpoint is None or not callable(deps.cleanup_checkpoint):
        return
    try:
        deps.cleanup_checkpoint(checkpoint.state, reason)
    except Exception:
        task = work.context.get("agent_task_state")
        if isinstance(task, dict):
            task.setdefault("progress", {})["execution_cleanup"] = [{"status": "unknown"}]
        return
    task = work.context.get("agent_task_state")
    if isinstance(task, dict) and "execution_cleanup" in checkpoint.state.progress:
        task.setdefault("progress", {})["execution_cleanup"] = checkpoint.state.progress["execution_cleanup"]


def _interruption_reason(
    work: QQBackgroundWork,
    cancel_event: threading.Event,
    deps: QQBackgroundTaskDependencies,
) -> str:
    if cancel_event.is_set():
        return "task_canceled"
    if not deps.command_state().is_agent_enabled():
        return "agent_disabled"
    if not deps.command_state().responses_enabled():
        return "responses_disabled"
    if deps.command_state().permission_mode() != str(work.context.get("agent_permission_mode") or "traditional"):
        return "agent_permission_changed"
    if deps.command_state().authority_revision() != work.authority_revision:
        return "agent_authority_revoked"
    query = work.context.get("agent_query_checkpoint")
    if query is not None and query.stop_reason():
        return query.stop_reason()
    return work.budget.stop_reason()


def _run_worker(
    work: QQBackgroundWork,
    cancel_event: threading.Event,
    handler: Callable[..., Any],
    queued_cancel: Callable[[], None],
    deps: QQBackgroundTaskDependencies,
) -> AgentTaskOutcome | AgentTaskContinuation:
    deps.effects.conversation_step(work.trace_id, phase="execution", task_id=work.active_task_id)
    interrupt = lambda: _interruption_reason(work, cancel_event, deps)
    work.context["agent_task_interrupt"] = interrupt
    def usage(values):
        details = dict(values)
        details.pop("trace_id", None)
        deps.trace_store().stage(work.trace_id, "llm_usage", **details)
    try:
        reply = _execute_slice(work, interrupt, usage, handler, queued_cancel, deps)
        if isinstance(reply, AgentTaskContinuation):
            return reply
    except TaskContractError as exc:
        _cleanup_work(work, deps, "task_contract_error")
        reply = deps.preparation_failure_reply(work.turn, exc, context=work.context, trace_id=work.trace_id)
    except Exception as exc:
        _cleanup_work(work, deps, "task_handler_error")
        deps.logger.error("Background agent execution failed (trace_id=%s, task_id=%s, work_id=%s, type=%s)",
                     work.trace_id, work.active_task_id, work.work_id, type(exc).__name__)
        deps.effects.trace(work.trace_id, "background_agent", status="error", error=type(exc).__name__)
        reply = deps.execution_failure_reply(work.turn, exc, context=work.context, trace_id=work.trace_id)
    return _finish_work(work, reply, deps)


def _execute_slice(
    work: QQBackgroundWork,
    interrupt: Callable[[], str],
    usage: Callable[..., None],
    handler: Callable[..., Any],
    queued_cancel: Callable[[], None],
    deps: QQBackgroundTaskDependencies,
) -> QQAgentReply | AgentTaskContinuation:
    with deps.trace_store().observe(work.trace_id), bind_qq_turn_usage(work.trace_id, usage), bind_task_budget(work.budget):
        reason = interrupt()
        if reason:
            _cleanup_work(work, deps, reason)
            result = AgentRunResult(status="canceled" if reason == "task_canceled" else "blocked", rounds=0, reason=reason)
        else:
            if work.prepared_task is not None:
                work.active_task_id = work.prepared_task.handoff.target_task_id
                work.decision = work.prepared_task.decision
                work.handoff_state = deps.commit_prepared_task(
                    work.prepared_task, work.turn, work.context, work.trace_id,
                )
                work.prepared_task = None
            else:
                if not deps.coordinator().claim_target(work.work_id, attempt=work.work_attempt, agent_task_id=work.active_task_id):
                    raise TaskContractError("task_already_active")
            work.handoff_state = work.handoff_state.advance(TaskHandoffPhase.RUNNING)
            deps.effects.trace(work.trace_id, "task_handoff", phase=work.handoff_state.phase.value,
                        status="running", route=work.handoff_state.route,
                        target_task_id=work.handoff_state.target_task_id)
            result = deps.run_agent(
                task_id=work.active_task_id, context=work.context,
                initial_transcript=list(work.initial_transcript), first_decision=work.decision,
                approved_invocations=work.approved, trace_id=work.trace_id,
                checkpoint=work.session["checkpoint"],
            )
            work.decision, work.approved = None, None
            work.session["checkpoint"] = result.checkpoint
        # Cooperative cancellation may arrive at the last tool's
        # return or during a non-preemptible custom handler.
        reason = interrupt()
        if reason and result.status != "canceled":
            result = replace(result, status="canceled" if reason == "task_canceled" else "blocked",
                             reason=reason, final_text="", pending_approval=None)
        work.context["agent_budget_snapshot"] = work.budget.snapshot()
        if result.status == "yielded":
            deps.effects.trace(work.trace_id, "agent_slice_yield", rounds=result.rounds,
                        slices=result.checkpoint.slices, budget=work.budget.snapshot())
            return AgentTaskContinuation(handler, metadata={"budget": work.budget.snapshot(),
                "slices": result.checkpoint.slices, "rounds": result.rounds}, on_cancel=queued_cancel)
    # Execution time ends before presentation, memory indexing and delivery.
    # Expression has a separate explicit deadline and call/token accounting.
    work.context["agent_budget_snapshot"] = work.budget.snapshot()
    with deps.trace_store().observe(work.trace_id), bind_qq_turn_usage(work.trace_id, usage):
        return deps.finish_result(
            work.turn, result, context=work.context, initial_transcript=work.initial_transcript,
            task_id=work.active_task_id, trace_id=work.trace_id,
        )


def _finish_work(
    work: QQBackgroundWork,
    reply: QQAgentReply,
    deps: QQBackgroundTaskDependencies,
) -> AgentTaskOutcome:
    work.handoff_state = work.handoff_state.advance(TaskHandoffPhase.FINISHED)
    deps.effects.trace(work.trace_id, "task_handoff", phase=work.handoff_state.phase.value,
                status=reply.agent_status, route=work.handoff_state.route,
                target_task_id=work.handoff_state.target_task_id)
    deps.trace_store().finish(work.trace_id, reply=reply)
    deps.conversation_finish(work.trace_id, reply)
    if reply.agent_status != "waiting_user":
        with deps.slice_lock:
            deps.slice_sessions.pop(work.work_id, None)
    status = reply.agent_status if reply.agent_status in {"completed", "failed", "blocked", "canceled", "waiting_user"} else "blocked"
    return AgentTaskOutcome(status=status, summary=reply.final_text, error=reply.reason if status != "completed" else "",
        metadata={"agent_status": reply.agent_status, "agent_task_id": work.active_task_id,
                  "sent": reply.sent, "delivery_status": reply.delivery_status, "reason": reply.reason,
                  "budget": work.budget.snapshot()})


def _submit_work(
    work: QQBackgroundWork,
    handler: Callable[..., Any],
    queued_cancel: Callable[[], None],
    deps: QQBackgroundTaskDependencies,
) -> QQAgentReply:
    try:
        old_records = {item["task_id"]: item for item in deps.coordinator().list(limit=200)}
        with deps.slice_lock:
            # Drop canceled/expired approvals; never evict live runnable work.
            for key, value in list(deps.slice_sessions.items()):
                old = old_records.get(key)
                if old and not old["worker_active"] and (old["cancel_requested"] or old["status"] in {"failed", "canceled", "completed"}
                        or old["status"] == "waiting_user" and time.time() - old["updated_at"] > deps.approval_ttl_seconds()):
                    deps.slice_sessions.pop(key, None)
            deps.slice_sessions[work.work_id] = work.session
        deps.coordinator().submit(
            handler, task_id=work.work_id, goal=deps.primary_request(work.turn),
            profile_user_id=work.requester, session_id=work.scope_session, defer_start=True,
            allow_reopen=bool(work.prior_session),
            on_cancel=queued_cancel,
            metadata={"kind": "qq_agent", "retry_policy": "fresh_continuation_only", "trace_id": work.trace_id,
                      "agent_task_id": work.target, "target_claimed": False,
                      "message_id": work.turn.primary_message_id, "delivery_status": "not_sent",
                      **({"query_operation_id": work.context["agent_query_checkpoint"].operation_id}
                         if work.context.get("agent_query_checkpoint") is not None else {})},
        )
        work.handoff_state = work.queued_handoff
        if work.prepared_task is not None:
            work.prepared_task = replace(work.prepared_task, handoff=work.handoff_state)
        deps.effects.trace(work.trace_id, "task_handoff", phase=work.handoff_state.phase.value,
                    status="queued", route=work.handoff_state.route,
                    target_task_id=work.handoff_state.target_task_id)
    except Exception as exc:
        query = work.context.get("agent_query_checkpoint")
        if query is not None:
            # submit may have durably admitted the original key before an
            # acknowledgement failed. Inspect it; never make a second key.
            try:
                saved = deps.coordinator().get(work.work_id, profile_user_id=work.requester, session_id=work.scope_session)
            except Exception:
                return QQAgentReply("blocked", reason="query_handoff_confirmation_unknown", background_task_id=work.work_id)
            if (saved and saved["metadata"].get("query_operation_id") == query.operation_id
                    and saved["metadata"].get("agent_task_id") == work.target
                    and (saved["status"] == "queued" or saved["worker_active"])):
                if work.prepared_task is not None:
                    work.prepared_task = replace(work.prepared_task, handoff=work.queued_handoff)
                work.handoff_state = work.queued_handoff
                return QQAgentReply("waiting_background", delivery_status="queued", reason="agent_task_queued",
                    background_task_id=work.work_id)
        if not work.prior_session:
            with deps.slice_lock:
                deps.slice_sessions.pop(work.work_id, None)
        deps.effects.trace(work.trace_id, "background_admission", status="rejected", error=type(exc).__name__)
        return deps.effects.deliver(
            work.turn, "后台任务没有入队，可能是队列已满或处理器不可用；没有执行操作。",
            agent_status="blocked", reason="background_task_admission_failed",
            record_memory=False, track_chat_flow=False, trace_id=work.trace_id,
        )
    # Quiet handoff. No canned "working" chat message or completion claim.
    deps.effects.conversation_step(work.trace_id, phase="execution", task_id=work.target, queued=True,
        lease_seconds=deps.queued_lease())
    return QQAgentReply("waiting_background", delivery_status="queued", reason="agent_task_queued", background_task_id=work.work_id)
