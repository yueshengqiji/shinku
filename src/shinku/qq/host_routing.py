"""Resolve task mechanics at the host; model intent never grants authority."""

from collections.abc import Mapping
from dataclasses import replace

from shinku.agent.loop import AgentDecision


def resolve_host_route(decision, *, context, handlers, observation, public_query):
    """Canonicalize an uncommitted selection without changing calls/contracts.

    `query` describes the whole goal, not merely a read as its first step.
    Missing intent defaults to a tracked task. Legacy explicit new/continue
    routes remain supported; only host policies can grant a query fast path.
    """
    if decision.kind != "tool":
        return decision
    link = decision.task_link
    if link is not None and not isinstance(link, Mapping):
        return decision  # Let the normal validation/repair boundary reject it.
    link = dict(link or {})
    mode = link.get("mode")
    if mode == "continue":
        if str(link.get("task_id") or "").strip():
            return decision  # Scoped ledger lookup still checks the target.
        candidates = context.get("agent_task_candidates", ())
        ids = list(dict.fromkeys(str(row.get("task_id") or "").strip()
            for row in candidates if isinstance(row, Mapping) and row.get("task_id")))
        if len(ids) == 1:
            return replace(decision, task_link={"mode": "continue", "task_id": ids[0]})
        return AgentDecision.wait(
            "你想继续哪项任务？请指明之前的目标。" if ids else "当前会话没有可关联的任务，请说明要继续处理的目标。",
            reason="task_continuation_ambiguous" if ids else "task_continuation_missing",
        )
    if mode == "none" and decision.task_contract is not None:
        return decision  # An explicit conflicting contract is not silently relinked.
    if mode == "new":
        return decision
    if mode not in (None, "none", "auto"):
        return decision
    # A provider-native history read is already a bounded host capability.
    if not link and observation(decision, context=context):
        return decision
    query = mode == "none" or mode == "auto" and link.get("intent") in {"query", "public_query"}
    if query and decision.task_contract is None:
        calls = decision.invocations or ((decision.invocation,) if decision.invocation else ())
        history = bool(calls) and all(getattr(handlers.get(c.name), "agent_history_answer_source", False) for c in calls)
        # Keep a date lookup within a whole-goal public query, when requested.
        tracks = ["public_query", "perception"] if link.get("track") == "public_query" or link.get("intent") == "public_query" else ["history" if history else "perception", "public_query"]
        for track in tracks:
            candidate = replace(decision, task_link={"mode": "none", "track": track})
            admits = public_query if track == "public_query" else observation
            if admits(candidate, context=context):
                return candidate
    return replace(decision, task_link={"mode": "new"})
