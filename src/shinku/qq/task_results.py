"""QQ terminal task reports, pending approval persistence and publication routing."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any
from shinku.agent.budget import current_task_budget
from shinku.agent.loop import AgentRunResult
from shinku.agent.pending import PendingApprovalRecord
from shinku.agent.task_state import TaskContractError
from shinku.conversation import AnswerEvidence
from .reply_models import QQAgentReply
from .turns import QQTurn

from .task_lifecycle_models import QQTaskResultDependencies


def finish_result(
    turn: QQTurn,
    result: AgentRunResult,
    *,
    context: dict[str, Any],
    initial_transcript: list[dict[str, Any]],
    task_id: str,
    trace_id: str='',
    deps: QQTaskResultDependencies,
) -> QQAgentReply:
    final_text = str(result.final_text or "").strip()
    evidence = getattr(result, "answer_evidence", None)
    deps.conversation_step(trace_id, phase="verification", task_id=task_id,
        **({"evidence": evidence} if isinstance(evidence, AnswerEvidence) and evidence.kind != "none" else {}))
    turn = deps.response_turn(turn, context.get("qq_reply_routing"))
    task = context.get("agent_task_state")
    from shinku.agent.budget_policy import BUDGET_STOPS
    if result.reason in BUDGET_STOPS and isinstance(task, dict):
        from .budget_report import budget_stop_report
        final_text = budget_stop_report(task, result.reason, context.get("agent_work_id") or task_id)
    if not final_text and result.status in {"failed", "blocked", "partial", "waiting_background", "canceled"}:
        final_text = deps.terminal_failure_text(result.reason, completion=result.completion,
            contract=task.get("contract") if isinstance(task, Mapping) else context.get("agent_completion_contract"))
    if isinstance(task, dict):
        task.pop("result_commit_pending", None)
        task["last_report"] = final_text[:2000]
        task["status"] = result.status
        task["completion"] = dict(result.completion)
        task["completion_attempt"] = int(task.get("execution_attempt") or 0)
        task["stop_reason"] = str(result.reason or "")
        budget = current_task_budget()
        if context.get("agent_sliced_execution") or context.get("agent_recovery_enabled"):
            task["budget"] = budget.snapshot() if budget is not None else context.get("agent_budget_snapshot", task.get("budget", {}))
        if result.reason:
            task["notes"]["blocker"] = str(result.reason)[:2000]
        if deps.express_result is not None:
            from shinku.agent_runtime.results import export_task_result
            if isinstance(evidence, AnswerEvidence):
                task["result_answer_evidence"] = evidence.bound_to(final_text).as_dict()
            existing = deps.task_store().get(task["task_id"], requester=task["requester"], session=task["session"])
            if existing and existing["revision"] != task["revision"]:
                raise TaskContractError("task_state_conflict")
            same = export_task_result({**task, "updated": (existing or task).get("updated", 0)},
                status=result.status, reason=result.reason, report=final_text)
            if (existing and (existing.get("execution_result") or {}).get("result_hash") == same["result_hash"]):
                # Consuming the same terminal callback is not a new result
                # version and cannot buy another polish/delivery attempt.
                saved = existing
            else:
                now = deps.task_store().clock()
                task["execution_result"] = export_task_result({**task, "revision": task["revision"] + 1, "updated": now},
                    status=result.status, reason=result.reason, report=final_text)
                task["expression_finalized_result_hash"] = task["execution_result"]["result_hash"]
                saved = deps.task_store().save(task, now=now)
        else:
            saved = deps.task_store().save(task)
        task.clear()
        task.update(saved)
        deps.conversation_step(trace_id, phase="verification", task_id=task_id,
            task_attempt=int(task.get("execution_attempt") or 0), task_version=task["revision"])
    if result.pending_approval is not None:
        try:
            deps.approval_store().put(
                    deps.approval_session(turn),
                    PendingApprovalRecord(
                        request=result.pending_approval,
                        task_id=task_id,
                        context={key: value for key, value in context.items() if key != "agent_task_interrupt"},
                        # Keep the planner's exact proposed calls so the resumed
                        # turn has an auditable user -> assistant tool request ->
                        # approval -> execution chain.
                        transcript=tuple(dict(item) for item in result.transcript),
                        created_at=time.time(),
                    ),
            )
        except Exception as exc:
            deps.trace(trace_id, "approval", status="store_error", error=type(exc).__name__)
            return QQAgentReply("failed", reason=f"approval_store_error:{type(exc).__name__}")
        deps.trace(trace_id, "approval", status="pending", verdict="required")

    envelopes = getattr(result, "tool_envelopes", ())
    if not isinstance(task, dict):
        deps.remember_task_snapshot(turn, result, task_id=task_id, final_text=final_text)

    deps.trace(trace_id, "agent_completion", status=result.status, reason=result.reason, completion=getattr(result, "completion", {}))
    if deps.express_result is not None and result.status in {"completed", "partial", "failed", "blocked", "canceled", "unverified"}:
        return deps.express_result(turn, result, context=context, task_id=task_id, trace_id=trace_id)
    if context.get("agent_async_execution") and not deps.command_state().responses_enabled():
        return QQAgentReply(result.status, final_text=final_text, reason=result.reason or "responses_disabled", delivery_status="suppressed")
    if result.status not in {"completed", "waiting_user"} or not final_text:
        if result.status not in {"failed", "blocked", "partial", "waiting_background", "canceled"}:
            return QQAgentReply(result.status, final_text=final_text, reason=result.reason)
        final_text = final_text or deps.terminal_failure_text(result.reason)
        deps.record_agent_task_memory(
            turn,
            task_id=task_id,
            status=result.status,
            outcome=final_text,
            reason=result.reason,
            rounds=result.rounds,
            tool_envelopes=envelopes,
            transcript=result.transcript,
            trace_id=trace_id,
        )
        return deps.deliver_text(
            turn,
            final_text,
            agent_status=result.status,
            reason=result.reason,
            record_memory=not deps.is_poke_turn(turn),
            track_chat_flow=not deps.is_poke_turn(turn) and not context.get("agent_async_execution"),
            trace_id=trace_id,
            tool_envelopes=envelopes,
            autonomous_authorized=bool(context.get("qq_chat_model_decision_valid")),
        )

    if result.status == "completed":
        deps.record_agent_task_memory(
            turn,
            task_id=task_id,
            status=result.status,
            outcome=final_text,
            reason=result.reason,
            rounds=result.rounds,
            tool_envelopes=envelopes,
            transcript=result.transcript,
            trace_id=trace_id,
        )
    return deps.deliver_text(
        turn,
        final_text,
        agent_status=result.status,
        reason=result.reason,
        record_memory=not deps.is_poke_turn(turn),
        track_chat_flow=not deps.is_poke_turn(turn) and not context.get("agent_async_execution"),
        trace_id=trace_id,
        tool_envelopes=envelopes,
        autonomous_authorized=bool(context.get("qq_chat_model_decision_valid")),
    )


def terminal_failure_text(
    reason: str,
    *,
    completion: Mapping | None=None,
    contract: Mapping | None=None,
) -> str:
    reason = str(reason or "").strip()
    if reason == "completion_no_progress":
        from .failure import completion_failure
        return completion_failure(completion, contract).text
    if reason == "task_budget_snapshot_invalid":
        return "旧任务的预算记录不完整，无法安全继续；我已停止后续操作，原目标和执行记录保留。"
    if reason == "task_canceled":
        return "后台任务已在检查点停止后续执行；已经发生的操作不会自动撤销。"
    if reason == "agent_permission_changed":
        return "当前权限已改变，后台任务停止后续操作；已有记录保留，未确认全部完成。"
    if reason == "agent_authority_revoked":
        return "任务授权期间发生了开关或权限变更，旧后台任务已停止后续操作；重新开启不会自动恢复它。"
    if reason == "agent_disabled":
        return "工具任务处理已关闭，后台任务停止后续操作；已有记录保留。"
    if reason in {"task_budget_low", "task_model_budget_exhausted", "task_token_budget_exhausted", "task_tool_budget_exhausted", "task_time_budget_exhausted"}:
        return "整项任务的总预算已达到上限，已停止后续执行；原目标、剩余条件和执行证据保留。这次尚未全部完成，需要你决定是否追加预算后继续。"
    if reason == "no_progress":
        return "我刚才一直停在只读查看，没有推进到实际操作或验证。先停在这里；这次任务还没有完成。"
    if reason == "max_rounds":
        return "这轮已经到达任务规划上限，我没能完成后续核验和整理；不会把未完成的结果说成成功。"
    if reason == "repeated_tool_call":
        return "工具调用重复了几次，结果没有继续推进，我先停下了。这次任务还没完成。"
    return "这轮没能完成，我先如实停在这里，没有把未完成的事说成成功。"


def task_preparation_failure_reply(
    turn: QQTurn,
    exc: TaskContractError,
    *,
    context: Mapping[str, Any],
    trace_id: str,
    deps: QQTaskResultDependencies,
) -> QQAgentReply:
    from .failure import task_preparation_failure
    memory = context.get("memory_context")
    recalled = int(memory.get("retrieved_count") or 0) if isinstance(memory, Mapping) else 0
    report = task_preparation_failure(str(exc), recalled_candidates=recalled)
    deps.trace(trace_id, "task_contract", status="rejected", error=str(exc), field_path=exc.field_path,
                phase=report.phase, failure_kind=report.category)
    budget = current_task_budget()
    reason = budget.stop_reason() if budget is not None else ""
    return deps.deliver_text(turn, deps.terminal_failure_text(reason) if reason else report.text,
        agent_status="blocked" if reason else report.status, reason=reason or f"task_contract_error:{exc}",
        record_memory=False, track_chat_flow=False, trace_id=trace_id)


def execution_failure_reply(
    turn: QQTurn,
    exc: Exception,
    *,
    context: Mapping[str, Any],
    trace_id: str,
    deps: QQTaskResultDependencies,
) -> QQAgentReply:
    from .failure import execution_failure
    task = context.get("agent_task_state")
    tool_count = len(task.get("tool_calls", ())) if isinstance(task, Mapping) else 0
    report = execution_failure(type(exc).__name__, tool_count=tool_count)
    deps.trace(trace_id, "execution_failure", error=report.code, phase=report.phase,
                failure_kind=report.category, tool_count=tool_count)
    budget = current_task_budget()
    reason = budget.stop_reason() if budget is not None else ""
    return deps.deliver_text(turn, deps.terminal_failure_text(reason) if reason else report.text,
        agent_status="blocked" if reason else report.status, reason=reason or f"agent_error:{report.code}",
        record_memory=False, track_chat_flow=False, trace_id=trace_id)
