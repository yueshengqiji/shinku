"""Admit a model-selected request and hand its prepared task to execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from shinku.agent.task_state import TaskContractError
from shinku.conversation import AnswerEvidence
from shinku.tasks.handoff import TaskHandoff, TaskHandoffPhase

from .commands import QQCommandState
from .host_routing import resolve_host_route
from .reply_models import QQAgentReply
from .turn_models import QQTurnDecision, QQTurnEffects, _PreparedAgentTask
from .turns import QQTurn


@dataclass(frozen=True)
class QQTaskAdmissionDependencies:
    registry_handlers: Callable[[], Mapping[str, Any]]
    is_master_turn: Callable[[QQTurn], bool]
    command_state: QQCommandState
    owner_gate_enabled: Callable[[], bool]
    is_lightweight_perception: Callable[..., bool]
    handle_lightweight_perception: Callable[..., QQAgentReply]
    is_lightweight_public_query: Callable[..., bool]
    handle_lightweight_public_query: Callable[..., QQAgentReply]
    contract_enabled: Callable[[], bool]
    preparation_failure_reply: Callable[..., QQAgentReply]
    prepare_agent_task: Callable[..., _PreparedAgentTask]
    execution_failure_reply: Callable[..., QQAgentReply]
    async_enabled: Callable[[], bool]
    enqueue_agent_task: Callable[..., QQAgentReply]
    commit_prepared_agent_task: Callable[..., TaskHandoff]
    run_agent: Callable[..., Any]
    finish_result: Callable[..., QQAgentReply]
    effects: QQTurnEffects


def dispatch_agent_request(
    selection: QQTurnDecision,
    *,
    deps: QQTaskAdmissionDependencies,
) -> QQAgentReply:
    decision = resolve_host_route(selection.decision, context=selection.context,
        handlers=deps.registry_handlers(), observation=deps.is_lightweight_perception,
        public_query=deps.is_lightweight_public_query)
    selection = replace(selection, decision=decision)
    if decision.kind == "wait":
        return deps.effects.deliver(selection.turn, decision.text, agent_status="waiting_user",
            reason=decision.reason, trace_id=selection.trace_id)
    requested_names = _check_agent_route(selection, deps)
    if isinstance(requested_names, QQAgentReply):
        return requested_names
    prepared = _prepare_selected_task(selection, requested_names, deps)
    if isinstance(prepared, QQAgentReply):
        return prepared
    return _execute_selected_task(selection, prepared, deps)


def _check_agent_route(
    selection: QQTurnDecision,
    deps: QQTaskAdmissionDependencies,
) -> set[str] | QQAgentReply:
    turn = selection.turn
    trace_id = selection.trace_id
    is_poke = selection.is_poke
    context = selection.context
    initial_transcript = selection.initial_transcript
    decision = selection.decision
    available_tool_names = selection.available_tool_names
    invocations = decision.invocations or ((decision.invocation,) if decision.invocation is not None else ())
    requested_names = {item.name for item in invocations}
    history_requested = any(getattr(deps.registry_handlers().get(name), "agent_history_answer_source", False)
                            for name in requested_names)
    deps.effects.conversation_step(trace_id, phase="preparation", route="history" if history_requested else "agent",
        evidence=AnswerEvidence("history", "unverified") if history_requested else AnswerEvidence())
    if (decision.kind == "tool" and (not available_tool_names or not requested_names.issubset(available_tool_names))) or (
        not deps.is_master_turn(turn) or not deps.command_state.is_agent_enabled()
    ):
        audience = context.get("audience_context") if isinstance(context, Mapping) else None
        is_non_master = (
            deps.owner_gate_enabled()
            and isinstance(audience, Mapping)
            and not bool(audience.get("is_master"))
        )
        refusal = (
            deps.effects.non_master_tool_refusal
            if is_non_master
            else "这项任务需要调用工具，不过工具任务处理现在没有开启或当前能力不可用。普通聊天不受影响。"
        )
        deps.effects.trace(
            trace_id,
            "chat_decision",
            kind="tool_rejected",
            requested_tool_count=len(requested_names),
        )
        return deps.effects.deliver(
            turn,
            refusal,
            agent_status="blocked",
            reason="tool_unavailable_or_unauthorized",
            record_memory=not is_poke,
            track_chat_flow=not is_poke,
            trace_id=trace_id,
        )

    if deps.is_lightweight_perception(decision, context=context):
        return deps.handle_lightweight_perception(
            turn,
            decision,
            context=context,
            initial_transcript=initial_transcript,
            trace_id=trace_id,
        )

    if deps.is_lightweight_public_query(decision, context=context):
        return deps.handle_lightweight_public_query(
            turn, decision, context=context,
            initial_transcript=initial_transcript, trace_id=trace_id,
        )
    return requested_names


def _prepare_selected_task(
    selection: QQTurnDecision,
    requested_names: set[str],
    deps: QQTaskAdmissionDependencies,
) -> _PreparedAgentTask | QQAgentReply:
    turn = selection.turn
    trace_id = selection.trace_id
    context = selection.context
    task_id = selection.task_id
    decision = selection.decision
    try:
        handoff = TaskHandoff.select(
            decision.task_link,
            host_task_id=task_id,
            strict=deps.contract_enabled(),
            allow_missing=True,
        )
    except TaskContractError as handoff_error:
        return deps.preparation_failure_reply(
            turn, handoff_error, context=context, trace_id=trace_id,
        )

    deps.effects.trace(
        trace_id,
        "task_handoff",
        phase=handoff.phase.value,
        status="selected",
        route=handoff.route,
        target_task_id=handoff.target_task_id,
        requested_tool_count=len(requested_names),
    )
    try:
        prepared = deps.prepare_agent_task(
            turn, decision, context, task_id, trace_id, handoff=handoff,
        )
    except TaskContractError as exc:
        return deps.preparation_failure_reply(turn, exc, context=context, trace_id=trace_id)
    except Exception as exc:  # noqa: BLE001 - convert unexpected preparation errors into a safe reply.
        deps.effects.trace(trace_id, "task_preparation", status="error", error=type(exc).__name__)
        return deps.execution_failure_reply(turn, exc, context=context, trace_id=trace_id)
    return prepared


def _execute_selected_task(
    selection: QQTurnDecision,
    prepared: _PreparedAgentTask,
    deps: QQTaskAdmissionDependencies,
) -> QQAgentReply:
    turn = selection.turn
    trace_id = selection.trace_id
    context = selection.context
    initial_transcript = selection.initial_transcript
    if deps.async_enabled():
        return deps.enqueue_agent_task(
            turn,
            task_id=prepared.handoff.target_task_id,
            context=context,
            initial_transcript=initial_transcript,
            first_decision=prepared.decision,
            handoff=prepared.handoff,
            prepared_task=prepared,
            trace_id=trace_id,
        )
    try:
        handoff = deps.commit_prepared_agent_task(prepared, turn, context, trace_id)
        handoff = handoff.advance(TaskHandoffPhase.RUNNING)
        deps.effects.trace(trace_id, "task_handoff", phase=handoff.phase.value, status="running",
                    route=handoff.route, target_task_id=handoff.target_task_id)
        result = deps.run_agent(
            task_id=handoff.target_task_id,
            context=context,
            initial_transcript=initial_transcript,
            first_decision=prepared.decision,
            trace_id=trace_id,
        )
    except TaskContractError as exc:
        return deps.preparation_failure_reply(turn, exc, context=context, trace_id=trace_id)
    except Exception as exc:  # noqa: BLE001 - convert unexpected execution errors into a safe reply.
        deps.effects.trace(trace_id, "agent", status="error", error=type(exc).__name__)
        return deps.execution_failure_reply(turn, exc, context=context, trace_id=trace_id)
    handoff = handoff.advance(TaskHandoffPhase.FINISHED)
    deps.effects.trace(trace_id, "task_handoff", phase=handoff.phase.value, status=result.status,
                route=handoff.route, target_task_id=handoff.target_task_id)
    return deps.finish_result(
        turn,
        result,
        context=context,
        initial_transcript=initial_transcript,
        task_id=handoff.target_task_id,
        trace_id=trace_id,
    )
