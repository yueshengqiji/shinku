"""QQ executor checkpoint and task evidence persistence."""

from __future__ import annotations

import copy
from typing import Any
from shinku.agent.budget import current_task_budget
from shinku.agent.loop import AgentState

from .task_lifecycle_models import QQTaskCheckpointDependencies


def checkpoint_task(
    context: dict[str, Any],
    state: AgentState,
    status: str,
    *,
    deps: QQTaskCheckpointDependencies,
) -> None:
    if context.get("agent_work_id"):
        inflight = state.inflight_tool_calls
        deps.task_coordinator().update_progress(
            str(context["agent_work_id"]), attempt=int(context["agent_work_attempt"]),
            metadata={"rounds": state.rounds, "tool_calls": len(state.tool_calls), "last_stage": status,
                      "budget": current_task_budget().snapshot() if current_task_budget() else {},
                      "current_tool": str(inflight[0].get("name") or "") if inflight else ""},
        )
    task = context.get("agent_task_state")
    if not isinstance(task, dict):
        deps.conversation_step(str(context.get("conversation_turn_id") or ""),
            phase="planning" if status == "planning" else "execution", queued=status == "yielded",
            task_id=state.task_id, lease_seconds=deps.conversation_queued_lease() if status == "yielded" else None)
        return
    from shinku.agent_runtime.result_events import TERMINAL
    deferred = getattr(deps.task_store(), "result_events", None) is not None and status in TERMINAL
    task["status"] = "running" if status == "planning" or deferred else "queued" if status == "yielded" else status
    if deferred:
        task["result_commit_pending"] = status
    else:
        task.pop("result_commit_pending", None)
    task["tool_calls"] = copy.deepcopy(state.tool_calls)
    if status != "planning":
        task["inflight"] = copy.deepcopy(state.inflight_tool_calls)
    # Only executor records enter the evidence ledger. Model notes cannot
    # insert arbitrary verified_effects, status=ok, or authorization grants.
    for record in state.transcript:
        if record.get("role") != "tool":
            continue
        key = (record.get("invocation_id"), record.get("name"), record.get("content"))
        bucket = "evidence" if record.get("status") == "ok" else "failed_attempts"
        item = {"id": record.get("invocation_id", ""), "tool": record.get("name", ""),
                "status": record.get("status", "unknown"),
                "summary": str(record.get("content") or "")[:1400], "source": "executor"}
        if record.get("research_kind"):
            item["research_stage"] = record["research_kind"]
            item["candidate_is_not_answer_evidence"] = record["research_kind"] == "candidate"
        if not any((old.get("id"), old.get("tool"), old.get("summary")) ==
                   (key[0] or "", key[1] or "", str(key[2] or "")[:1400]) for old in task[bucket]):
            task[bucket].append(item)
        task[bucket] = task[bucket][-32:]
    task["completion"] = copy.deepcopy(state.completion)
    task["completion_attempt"] = int(task.get("execution_attempt") or 0)
    budget = current_task_budget()
    if budget is not None and (context.get("agent_sliced_execution") or context.get("agent_recovery_enabled")):
        task["budget"] = budget.snapshot()
    task["progress"] = copy.deepcopy(state.progress)
    if state.research.enabled:
        task["research"] = state.research.snapshot()
    if state.local_answer.enabled:
        task["local_answer"] = state.local_answer.snapshot()
    saved = deps.task_store().save(task)
    task.clear()
    task.update(saved)
    deps.conversation_step(str(context.get("conversation_turn_id") or ""),
        phase="planning" if status == "planning" else "execution", queued=status == "yielded",
        task_id=state.task_id, task_attempt=int(task.get("execution_attempt") or 0), task_version=task["revision"],
        lease_seconds=deps.conversation_queued_lease() if status == "yielded" else None)
