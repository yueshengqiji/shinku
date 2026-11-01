"""QQ task archive indexing and scoped compatibility snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from shinku.agent.loop import AgentRunResult
from shinku.agent.task_state import new_task
from .turns import QQTurn

from .task_lifecycle_models import QQTaskMemoryDependencies


def record_agent_task_memory(
    turn: QQTurn,
    *,
    task_id: str,
    status: str,
    outcome: str,
    reason: str,
    rounds: int,
    tool_envelopes: Sequence[Any],
    transcript: Sequence[Mapping[str, Any]]=(),
    trace_id: str='',
    deps: QQTaskMemoryDependencies,
) -> None:
    """Index terminal Agent work in the owner's long-term memory only."""

    if deps.memory_service() is None or not tool_envelopes:
        return
    if getattr(getattr(deps.memory_service(), "store", None), "task_experience", None) is not None:
        # The opt-in host projects committed results and delivery receipts.
        # This pre-send prose hook must not create a competing legacy card.
        return
    metadata = deps.memory_message_meta(turn)
    metadata["shinku_involved"] = True
    invocation_names: dict[str, str] = {}
    for entry in tuple(transcript or ()):
        if not isinstance(entry, Mapping):
            continue
        calls = entry.get("tool_calls")
        if not isinstance(calls, (list, tuple)):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            call_id = str(call.get("id") or "").strip()
            call_name = str(call.get("name") or "").strip()
            if call_id and call_name:
                invocation_names[call_id] = call_name
    tool_outcomes: list[dict[str, str]] = []
    for item in tuple(tool_envelopes)[:32]:
        data = getattr(item, "data", {})
        data = data if isinstance(data, Mapping) else {}
        invocation_id = str(getattr(item, "invocation_id", "") or "").strip()
        name = str(
            data.get("tool_type")
            or data.get("tool")
            or invocation_names.get(invocation_id)
            or getattr(item, "name", "")
            or getattr(item, "tool", "")
            or ""
        ).strip()
        if name:
            tool_outcomes.append(
                {"name": name, "status": str(getattr(item, "status", "unknown") or "unknown")}
            )
    if not tool_outcomes:
        return
    try:
        stored_summary = deps.memory_service().record_agent_task_summary(
            conversation_id=turn.conversation_id,
            conversation_type=turn.conversation_type,
            requester_id=deps.effective_turn_sender_id(turn),
            task_id=task_id,
            goal=turn.combined_text,
            status=status,
            outcome=outcome,
            reason=reason,
            rounds=rounds,
            tool_outcomes=tool_outcomes,
            metadata=metadata,
        )
        stored_detail = deps.memory_service().record_agent_task_detail(
            conversation_id=turn.conversation_id,
            conversation_type=turn.conversation_type,
            requester_id=deps.effective_turn_sender_id(turn),
            task_id=task_id,
            status=status,
            rounds=rounds,
            steps=tool_outcomes,
            metadata=metadata,
        )
        deps.trace(
            trace_id,
            "agent_task_memory",
            status="stored" if stored_summary else "skipped_or_duplicate",
            detail_status="stored" if stored_detail else "skipped",
            tool_count=len(tool_outcomes),
        )
    except Exception as exc:
        # Memory indexing is auxiliary; never turn a delivered Agent result
        # into a failure because the summary could not be saved.
        deps.trace(trace_id, "agent_task_memory", status="error", error=type(exc).__name__)


def remember_task_snapshot(
    turn: QQTurn,
    result: AgentRunResult,
    *,
    task_id: str,
    final_text: str,
    now: float | None=None,
    deps: QQTaskMemoryDependencies,
) -> None:
    """Compatibility adapter; retain failed/partial records as task evidence."""
    requester, session = deps.task_scope(turn)
    task = deps.task_store().get(task_id, requester=requester, session=session, now=now) or new_task(task_id=task_id, requester=requester, session=session,
                    request=deps.primary_task_request(turn), message_id=turn.primary_message_id, contract=None)
    task["status"] = result.status
    task["last_report"] = str(final_text or "")[:2000]
    envelopes = tuple(getattr(result, "tool_envelopes", ()) or ())
    for item in envelopes:
        data = getattr(item, "data", {})
        data = data if isinstance(data, Mapping) else {}
        name = str(data.get("tool_type") or data.get("tool") or getattr(item, "tool", "") or "").strip()
        bucket = "evidence" if getattr(item, "status", "") == "ok" else "failed_attempts"
        task[bucket].append({"tool": name, "status": getattr(item, "status", "unknown"),
                             "summary": str(getattr(item, "model_feedback", ""))[:1400], "source": "executor"})
    deps.task_store().save(task, now=now)


def task_snapshot_for_turn(
    turn: QQTurn,
    *,
    now: float | None=None,
    deps: QQTaskMemoryDependencies,
) -> Mapping[str, Any] | None:
    requester, session = deps.task_scope(turn)
    candidates = deps.task_store().candidates(requester=requester, session=session, limit=1, now=now)
    if not candidates:
        return None
    task = candidates[0]
    return {**task, "tools_used": list(dict.fromkeys(item["tool"] for item in task["evidence"])),
            "final_text": task["last_report"], "timestamp": task["updated"]}
