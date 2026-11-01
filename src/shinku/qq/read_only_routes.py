"""Host admission policy for anonymous public reads and system/history observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shinku.agent.loop import AgentDecision
from shinku.agent.verification import SmartCompletionVerifier
from shinku.conversation.routing import ObservationRoutePolicy

from .query_models import PUBLIC_QUERY_MAX_TOOLS, QQReadOnlyRouteDependencies
from .turns import QQTurn


def public_query_handlers(
    context: Mapping[str, Any],
    *,
    deps: QQReadOnlyRouteDependencies,
) -> dict[str, Any]:
    """Opt-in anonymous variants; original name/owner/profile remain gates."""
    allowed = deps.tool_names(context)
    policy = deps.policy_for_permission_mode(deps.command_state.permission_mode())
    result = {}
    handlers = deps.registry_handlers()
    if deps.tool_decision_mode == "task":
        from .interaction_tools import audited_handlers
        handlers = audited_handlers(handlers, deps.read_capabilities or {}, route="public_query")
    for name, handler in handlers.items():
        if (allowed and name not in allowed) or not policy.accepts(name) or policy.requires_approval(name):
            continue
        if getattr(handler, "agent_fast_public_query", None) is False:
            continue
        factory = getattr(handler, "public_query_variant", None) if deps.tool_decision_mode != "task" else None
        if callable(factory):
            try:
                handler = factory()
            except Exception:
                continue
        if getattr(handler, "agent_fast_public_query", None) is not True:
            continue
        if (str(getattr(handler, "agent_permission_class", "")).casefold() != "read"
                or str(getattr(handler, "agent_progress_class", "")).casefold() != "observe"
                or getattr(handler, "tool_type", name) != name):
            continue
        effects = {str(effect).strip().casefold() for effect in
            (getattr(handler, "agent_completion_effects", ()) or ())}
        if effects:
            continue
        result[name] = handler
    return result


def is_lightweight_public_query(
    decision: AgentDecision,
    *,
    context: Any = None,
    handlers: Mapping[str, Any] | None = None,
    deps: QQReadOnlyRouteDependencies,
) -> bool:
    if (decision.kind != "tool" or decision.task_contract is not None
            or not isinstance(decision.task_link, Mapping)
            or decision.task_link.get("mode") != "none"
            or decision.task_link.get("track") not in (None, "public_query")
            or not deps.command_state.is_agent_enabled()):
        return False
    context = context if isinstance(context, Mapping) else {}
    turn = context.get("qq_turn")
    if isinstance(turn, QQTurn):
        if not deps.is_master_turn(turn):
            return False
        if any(effects & {"write", "delete", "send", "process_stop", "reminder"} for effects in
                SmartCompletionVerifier._explicit_target_roles([deps.primary_request(turn)]).values()):
            return False
    calls = decision.invocations or ((decision.invocation,) if decision.invocation else ())
    if not calls or len(calls) > PUBLIC_QUERY_MAX_TOOLS:
        return False
    safe = deps.public_handlers(context) if handlers is None else handlers
    policy = deps.policy_for_permission_mode(deps.command_state.permission_mode())
    allowed = deps.tool_names(context)
    if deps.tool_decision_mode == "task":
        capabilities = deps.read_capabilities or {}
        if any(call.name not in capabilities or capabilities[call.name].route != "public_query"
               or not capabilities[call.name].admits(call.arguments, context) for call in calls):
            return False
    return all(call.name in safe and (not allowed or call.name in allowed)
               and policy.accepts(call.name) and not policy.requires_approval(call.name) for call in calls)


def is_lightweight_perception(
    decision: AgentDecision,
    *,
    context: Any = None,
    deps: QQReadOnlyRouteDependencies,
) -> bool:
    """Compatibility adapter; observation admission has one host policy."""
    turn = context.get("qq_turn") if isinstance(context, Mapping) else None
    mode = str(
        (context.get("agent_permission_mode") if isinstance(context, Mapping) else None)
        or deps.command_state.permission_mode()
    )
    handlers = deps.registry_handlers()
    if deps.tool_decision_mode == "task":
        from .interaction_tools import audited_handlers
        handlers = audited_handlers(handlers, deps.read_capabilities or {}, route="perception")
        calls = decision.invocations or ((decision.invocation,) if decision.invocation else ())
        if not calls or any(call.name not in handlers
            or not handlers[call.name].capability.admits(call.arguments, context) for call in calls):
            return False
    return ObservationRoutePolicy.admits(
        decision, handlers=handlers, policy=deps.policy_for_permission_mode(mode),
        strict=deps.contract_enabled(), asynchronous=deps.async_enabled(),
        action_goal=isinstance(turn, QQTurn) and SmartCompletionVerifier.request_requires_action(
            deps.primary_request(turn)),
    )
