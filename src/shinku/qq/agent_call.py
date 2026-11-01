"""QQ host adapters call the Agent boundary; execution receives no persona."""

from __future__ import annotations

import copy
from dataclasses import replace

from shinku.agent.budget import bind_task_budget, current_task_budget
from shinku.agent.loop import AgentDecision
from shinku.agent.request_scope import RequestScope
from shinku.agent_runtime import AgentBoundaryError, AgentScope, AgentService, ExecutionGrant
from shinku.agent_runtime.context import execution_context
from shinku.agent_runtime.results import export_task_result


def scope_for_turn(turn):
    primary = next((message for message in turn.messages
                    if str(getattr(message, "message_id", "")) == turn.primary_message_id), None)
    if primary is None and len(turn.messages) == 1 and not getattr(turn.messages[0], "message_id", ""):
        primary = turn.messages[0]
    if primary is None or not getattr(primary, "sender_id", ""):
        raise AgentBoundaryError("host_requester_missing")
    return AgentScope(str(primary.sender_id), f"{turn.conversation_type.casefold()}:{turn.conversation_id}"), primary


class QQAgentCaller:
    """Only the transport-neutral port is visible to the interaction caller."""
    def __init__(self, port_factory):
        self.port_factory = port_factory

    def submit(self, turn, request):
        scope, primary = scope_for_turn(turn)
        if request.original_request != str(getattr(primary, "text", "") or "").strip():
            raise AgentBoundaryError("original_request_mismatch")
        return self.port_factory(scope).submit(request)

    def status(self, turn, task_id):
        return self.port_factory(scope_for_turn(turn)[0]).status(task_id)

    def result(self, turn, task_id):
        return self.port_factory(scope_for_turn(turn)[0]).result(task_id)

    def cancel(self, turn, handle):
        return self.port_factory(scope_for_turn(turn)[0]).cancel(handle)

    def resume(self, turn, handle, *, request_key):
        return self.port_factory(scope_for_turn(turn)[0]).resume(handle, request_key=request_key)

    def recovery(self, turn, handle):
        return self.port_factory(scope_for_turn(turn)[0]).recovery(handle)

    def supplement(self, turn, handle, request):
        return self.port_factory(scope_for_turn(turn)[0]).supplement(handle, request)


class QQExecutionAdapter:
    """Execute an already-admitted QQ task inside its existing worker/slice.

    This avoids submitting a second job into a single-worker coordinator or
    introducing a second business ledger. Independent submit/status APIs and
    this host admission adapter reuse the same executor, store and coordinator.
    """
    def __init__(self, executor, task_store, coordinator):
        self.executor, self.task_store, self.coordinator = executor, task_store, coordinator

    def run_from_bridge(self, bridge, *, task_id, context, tool_names, first_decision=None,
                        approved_invocation=None, approved_invocations=None, checkpoint=None):
        commands = bridge.command_state
        revision = context.get("agent_authority_revision", commands.authority_revision())
        mode = str(context.get("agent_permission_mode") or commands.permission_mode())
        policy = bridge._policy_for_permission_mode(mode)
        allowed = {name for name in tool_names if policy.accepts(name)}
        query = context.get("agent_query_checkpoint")
        original_query = context.get("agent_task_state", {}).get("foreground_query")
        if original_query:
            from shinku.agent_runtime.query_handoff import ForegroundQueryCheckpoint
            if not isinstance(query, ForegroundQueryCheckpoint):
                raise AgentBoundaryError("query_live_checkpoint_unavailable")
            allowed &= set(original_query["allowed_tools"]) & set(query.handlers)
        tool_context = dict(context.get("agent_execution_tool_context") or {})
        if query is not None:
            tool_context.update(query.read_context)
            tool_context["agent_query_checkpoint"] = query
        grant = ExecutionGrant(str(revision), frozenset(allowed), policy=policy, permission_mode=mode,
            budget_limits=bridge.task_budget_limits,
            tool_context=tool_context,
            still_authorized=lambda: commands.is_agent_enabled() and not commands.is_stopped()
                and commands.authority_revision() == revision and not (query is not None and query.stop_reason()))
        approved = tuple(approved_invocations or ()) or ((approved_invocation,) if approved_invocation else None)
        return self.run_admitted(task_id=task_id, host_context=context, grant=grant,
            observer=lambda state, status: bridge._checkpoint_task(context, state, status),
            checkpoint=checkpoint, first_decision=first_decision, approved_invocations=approved,
            completion_gate=bridge.completion_gate, approval_gate=bridge._approval_gate)

    def run_admitted(self, *, task_id, host_context, grant, observer, checkpoint=None,
                     first_decision=None, approved_invocations=None, completion_gate=None, approval_gate=None):
        scope = host_context.get("agent_request_scope")
        task = host_context.get("agent_task_state")
        if not isinstance(scope, RequestScope) or not isinstance(task, dict):
            raise AgentBoundaryError("host_admitted_task_required")
        stored = self.task_store.get(task_id, requester=scope.requester, session=scope.session)
        if not stored or stored["revision"] != task.get("revision") or stored["task_id"] != task_id:
            raise AgentBoundaryError("task_version_conflict")
        if stored.get("tool_decision_owner") == "execution" and first_decision is not None and (
                first_decision.invocations or first_decision.invocation):
            raise AgentBoundaryError("chat_first_execution_call_rejected")
        work_id = str(host_context.get("agent_work_id") or "")
        worker_attempt = int(host_context.get("agent_work_attempt") or 1)
        if work_id:
            record = self.coordinator.get(work_id, profile_user_id=scope.requester, session_id=scope.session)
            if (not record or record["attempt"] != worker_attempt or not record.get("worker_active")
                    or record.get("metadata", {}).get("agent_task_id") != task_id):
                raise AgentBoundaryError("stale_execution_attempt")
        if checkpoint is not None:
            previous = checkpoint.state.context
            if (checkpoint.state.task_id != task_id or previous.get("agent_work_id") != work_id
                    or previous.get("agent_work_attempt") != worker_attempt):
                raise AgentBoundaryError("checkpoint_scope_conflict")
        else:
            if task.get("execution_work_id") != work_id or not task.get("execution_attempt"):
                task["execution_attempt"] = int(task.get("execution_attempt") or 0) + 1
            task["execution_work_id"] = work_id
            task["execution_work_attempt"] = worker_attempt if work_id else 0
            task["host_grant"] = {"reference": grant.authorization_ref,
                "allowed_tools": sorted(grant.allowed_tools), "permission_mode": grant.permission_mode}
            saved = self.task_store.save(task)
            task.clear()
            task.update(saved)
        budget = current_task_budget()
        if budget is None:
            budget = grant.budget()
            budget.absorb_prior(task.get("budget"))
        context = execution_context(task, grant, attempt=worker_attempt, work_id=work_id,
                                   interrupt=host_context.get("agent_task_interrupt"))
        context["agent_recovery_resume"] = bool(host_context.get("agent_recovery_resume")
            or host_context.get("agent_budget_resume"))
        active_materials = host_context.get("agent_material_session")
        if active_materials is not None:
            from shinku.agent_runtime.materials import TaskMaterialSession
            active_materials = TaskMaterialSession(active_materials.store, task, still_authorized=grant.still_authorized)
            context["agent_material_session"] = active_materials
            host_context["agent_material_session"] = active_materials
        # Chat may nominate an admitted first tool, but its prose/notes and
        # persona decision metadata do not become execution prompt history.
        if first_decision is not None:
            calls = first_decision.invocations or ((first_decision.invocation,) if first_decision.invocation else ())
            first_decision = AgentDecision.tools(calls) if calls else None
        sliced = bool(host_context.get("agent_sliced_execution"))
        with bind_task_budget(budget):
            while True:
                result = self.executor.run(task_id=task_id, context=context, grant=grant, observer=observer,
                    checkpoint=checkpoint, first_decision=first_decision, approved_invocations=approved_invocations,
                    completion_gate=completion_gate, approval_gate=approval_gate)
                if result.status != "yielded" or sliced:
                    break
                checkpoint, first_decision, approved_invocations = result.checkpoint, None, None
        task["budget"] = budget.snapshot()
        if result.status != "yielded":
            task.pop("result_commit_pending", None)
            task.update(completion=copy.deepcopy(result.completion),
                        completion_attempt=int(task.get("execution_attempt") or 0), status=result.status,
                        stop_reason=result.reason, last_report=str(result.final_text or "")[:2000])
            now = self.task_store.clock()
            payload = {**task, "revision": task["revision"] + 1, "updated": now}
            task["execution_result"] = export_task_result(payload, status=result.status,
                                                        reason=result.reason, report=result.final_text)
            saved = self.task_store.save(task, now=now)
            task.clear()
            task.update(saved)
            host_context["agent_execution_result"] = copy.deepcopy(task["execution_result"])
            if task["execution_result"]["execution"]["status"] == "unverified":
                result = replace(result, status="unverified", reason="completion_receipt_missing")
        return result


def caller_for_bridge(bridge, adapter):
    def authorize(scope, task, operation):
        if bridge._owner_gate_enabled() and scope.requester not in bridge.master_user_ids:
            return None
        commands = bridge.command_state
        if operation != "cancel" and (not commands.is_agent_enabled() or commands.is_stopped()):
            return None
        revision, mode = commands.authority_revision(), commands.permission_mode()
        policy = bridge._policy_for_permission_mode(mode)
        allowed = bridge._registered_tool_names() if bridge.allowed_tool_names is None else set(bridge.allowed_tool_names)
        allowed = {name for name in allowed if policy.accepts(name)}
        return ExecutionGrant(str(revision), frozenset(allowed), policy=policy, permission_mode=mode,
            budget_limits=bridge.task_budget_limits,
            still_authorized=lambda: operation == "cancel" or (
                commands.is_agent_enabled() and not commands.is_stopped() and commands.authority_revision() == revision),
            resume_safe=False)
    from .task_status import reader_for_bridge
    service = AgentService(executor=adapter.executor, authorize=authorize,
                           task_store=adapter.task_store, coordinator=adapter.coordinator,
                           task_view_reader=reader_for_bridge(bridge), material_store=bridge.material_store,
                           recovery_enabled=bridge.recovery_enabled)
    return QQAgentCaller(service.bind)
