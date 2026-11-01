"""Agent API over the existing business ledger and execution coordinator.

Only checkpoints, budgets and host callbacks live in the process-local map.
Goals, versions, cumulative usage and results remain in AgentTaskStore. No
handler is deserialized or automatically replayed after a process restart.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shinku.agent.budget import bind_task_budget
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.tasks.coordinator import AgentTaskContinuation, AgentTaskCoordinator, AgentTaskOutcome

from .context import execution_context
from .contracts import AgentBoundaryError, AgentRequest, AgentTurnRequest, AgentScope, AgentSupplement, TaskHandle, bounded_text, canonical_hash
from .execution import AgentExecutor, ExecutionGrant
from .results import export_task_result
from .task_view import TaskViewReader


@dataclass
class _Session:
    scope: AgentScope
    attempt: int
    grant: ExecutionGrant
    budget: Any
    context: dict
    checkpoint: Any = None
    executor: Any = None


class AgentService:
    def __init__(self, *, executor: AgentExecutor,
                 authorize: Callable[[AgentScope, dict, str], ExecutionGrant | None],
                 task_store: AgentTaskStore, coordinator: AgentTaskCoordinator, task_view_reader=None, material_store=None,
                 recovery_enabled=False, result_exporter=None):
        self.executor, self.authorize = executor, authorize
        self.task_store, self.coordinator = task_store, coordinator
        self._lock = threading.RLock()
        self._sessions: dict[str, _Session] = {}
        self.task_views = task_view_reader or TaskViewReader(task_store, coordinator)
        if material_store is not None and material_store.store is not task_store:
            raise AgentBoundaryError("material_store_task_boundary_mismatch")
        self.materials = material_store
        self.recovery_enabled = bool(recovery_enabled)
        # 跨角色接力由宿主保留下游规范回执；模型不能选择导出器或改写验收事实。
        if result_exporter is not None and not callable(result_exporter):
            raise ValueError("host_result_exporter_required")
        self.result_exporter = result_exporter or export_task_result
        # 只有本机独立宿主显式绑定，调用者/模型不能用payload打开无准备器入口。
        self.conversation_scope = None
        self.conversation_reply_delivery = False
        # 仅显式认证的跨进程宿主安装此入口，消息payload不能开启。
        self.conversation_authorizer = None

    def accepts_conversation(self, scope):
        local = self.conversation_scope is not None and scope.requester == self.conversation_scope[0] and scope.session.startswith(self.conversation_scope[1])
        return local or bool(self.conversation_authorizer and self.conversation_authorizer(scope))

    def bind(self, scope: AgentScope) -> AgentClient:
        """Trusted host bootstrap. A remote transport must authenticate here."""
        if not isinstance(scope, AgentScope):
            raise AgentBoundaryError("host_scope_required")
        return AgentClient(self, scope)

    @staticmethod
    def _handle(task):
        return TaskHandle(task["task_id"], task["execution_attempt"], task["revision"])

    def _get(self, scope, task_id):
        try:
            task = self.task_store.get(task_id, requester=scope.requester, session=scope.session)
        except Exception as exc:
            raise AgentBoundaryError("task_store_unavailable") from exc
        if not task:
            raise AgentBoundaryError("task_not_found")
        return task

    def _grant(self, scope, task, operation):
        try:
            grant = self.authorize(scope, copy.deepcopy(task), operation)
        except Exception as exc:
            raise AgentBoundaryError("host_authorization_unavailable") from exc
        if not isinstance(grant, ExecutionGrant) or not grant.still_authorized():
            raise AgentBoundaryError("host_authorization_denied")
        grant.budget()  # Reject malformed host limits before recording a task.
        return grant

    def _write(self, scope, task_id, attempt, update):
        with self._lock:
            task = self._get(scope, task_id)
            if task.get("execution_attempt") != attempt:
                raise AgentBoundaryError("stale_execution_attempt")
            task["updated"] = self.task_store.clock()
            update(task)
            return self.task_store.save(task, now=task["updated"])

    def _submit(self, scope, request):
        if not isinstance(request, (AgentRequest, AgentTurnRequest)):
            raise AgentBoundaryError("typed_request_required")
        if isinstance(request, AgentTurnRequest) and not self.accepts_conversation(scope):
            raise AgentBoundaryError('conversation_host_required')
        payload = request.payload()
        task_id = "agent:" + canonical_hash([scope.requester, scope.session, request.request_key])[:32]
        digest = canonical_hash(payload)
        with self._lock:
            previous = self.task_store.get(task_id, requester=scope.requester, session=scope.session)
            if previous:
                if previous.get("request_digest") != digest:
                    raise AgentBoundaryError("idempotency_key_conflict")
                return self._handle(previous)
            task = new_task(task_id=task_id, requester=scope.requester, session=scope.session,
                request=payload["original_request"], message_id=payload["request_key"], contract=payload["contract"])
            task["requests"][0]["text"] = payload["original_request"]
            task.update({"request_digest": digest, "confirmed_constraints": payload["constraints"],
                "source_data": payload["sources"], "execution_attempt": 1, "status": "queued",
                "execution_owner": "agent_service"})
            if isinstance(request, AgentTurnRequest):
                task['input_mode'] = 'conversation'
            if payload.get("history_context"):
                task["history_context"] = payload["history_context"]
            grant = self._grant(scope, task, "submit")
            task["host_grant"] = {"reference": grant.authorization_ref,
                "allowed_tools": sorted(grant.allowed_tools), "permission_mode": grant.permission_mode,
                "policy": {"allowed": sorted(grant.policy.allowed), "blocked": sorted(grant.policy.blocked),
                    "risk_by_tool": dict(grant.policy.risk_by_tool), "default_risk": grant.policy.default_risk,
                    "approval_required_by_risk": sorted(grant.policy.approval_required_by_risk)}}
            task["budget"] = grant.budget().snapshot()
            saved = self.task_store.save(task)
            if self.materials is not None:
                from .materials import MaterialSource
                sources = [MaterialSource(row["id"], "caller_data", row["content"], required=True)
                           for row in payload["sources"]]
                if payload.get("history_context"):
                    # 历史可回读但不是必需证据；附件/工具读取的硬条件沿用原有门禁。
                    sources.append(MaterialSource("conversation-history", "related_message",
                        payload["history_context"], required=False))
                saved["material_package"] = self.materials.package(saved, [*sources, *grant.material_sources],
                    authorization_ref=grant.authorization_ref, gaps=grant.material_gaps)
                saved = self.task_store.save(saved)
            try:
                self._queue(scope, saved, grant)
            except Exception:
                self._write(scope, task_id, 1, lambda t: t.update(status="blocked", stop_reason="task_admission_failed"))
                raise
            return self._handle(saved)

    def _queue(self, scope, task, grant, *, checkpoint=None, allow_reopen=False):
        task_id, attempt = task["task_id"], task["execution_attempt"]
        budget = grant.budget()
        budget.absorb_prior(task.get("budget"))
        session = _Session(scope, attempt, grant, budget, {})
        # 可信宿主可按会话派生执行器；在派发前冻结，后续分片不读取可变全局模型。
        selector = getattr(self.executor, 'for_task', None)
        session.executor = selector(scope, task) if callable(selector) else self.executor
        if callable(selector) and not isinstance(session.executor, AgentExecutor):
            raise AgentBoundaryError('host_executor_required')
        session.checkpoint = checkpoint
        session.context = execution_context(task, grant, attempt=attempt, work_id=task_id)
        if task.get('input_mode') == 'conversation':
            if not self.accepts_conversation(scope):
                raise AgentBoundaryError('conversation_host_required')
            from .conversation import ConversationTurn
            session.context['_agent_conversation_turn'] = ConversationTurn(self, scope, task_id, attempt, grant)
        session.context["agent_recovery_pending"] = bool(self.recovery_enabled and task.get("resume_request"))
        if self.materials is not None:
            from .materials import TaskMaterialSession
            session.context["agent_material_session"] = TaskMaterialSession(self.materials, task,
                still_authorized=grant.still_authorized)
        self._sessions[task_id] = session
        self.coordinator.submit(lambda event: self._run_slice(task_id, session, event),
            task_id=task_id, goal=(task.get('contract') or {}).get('goal') or task['requests'][0]['text'], profile_user_id=scope.requester,
            session_id=scope.session, defer_start=True, allow_reopen=allow_reopen,
            metadata={"agent_task_id": task_id, "retry_policy": "fresh_continuation_only"},
            on_cancel=lambda: self._queued_cancel(task_id, session))
        if not self.coordinator.dispatch(task_id, profile_user_id=scope.requester, session_id=scope.session):
            raise AgentBoundaryError("task_dispatch_failed")

    def _queued_cancel(self, task_id, session):
        if session.checkpoint is not None:
            from shinku.agent.execution_sessions import stop_execution_sessions
            stop_execution_sessions(session.checkpoint.state, self.executor.tool_host.registry.view(), "task_canceled")
        # Coordinator cancellation is the authoritative stop receipt. Do not
        # acquire the service lock from a coordinator-owned callback: its
        # continuation-finalizer can run while the coordinator lock is held.

    def _observe(self, task_id, session, state, status):
        from .result_events import TERMINAL
        # 所有宿主都等正式结果封存后再发布终态，不能只在启用outbox时防提前轮询。
        deferred = status in TERMINAL
        def update(task):
            task.update(status="running" if status == "planning" or deferred else "queued" if status == "yielded" else status,
                tool_calls=copy.deepcopy(state.tool_calls), inflight=copy.deepcopy(state.inflight_tool_calls),
                completion=copy.deepcopy(state.completion), completion_attempt=session.attempt,
                progress=copy.deepcopy(state.progress),
                budget=session.budget.snapshot(), notes=copy.deepcopy(session.context["agent_task_state"].get("notes", {})))
            for key in ("material_package", "material_information", "stage_strategy"):
                if key in session.context["agent_task_state"]:
                    task[key] = copy.deepcopy(session.context["agent_task_state"][key])
            if deferred:
                task["result_commit_pending"] = status
            else:
                task.pop("result_commit_pending", None)
        saved = self._write(session.scope, task_id, session.attempt, update)
        ledger = session.context["agent_task_state"]
        ledger.clear()
        ledger.update(saved)
        self.coordinator.update_progress(task_id, attempt=session.attempt,
            metadata={"rounds": state.rounds, "tool_calls": len(state.tool_calls), "last_stage": status,
                      "budget": session.budget.snapshot()})

    def _run_slice(self, task_id, session, event):
        def interrupt():
            if event.is_set():
                return "task_canceled"
            task = self._get(session.scope, task_id)
            if task["execution_attempt"] != session.attempt:
                return "stale_execution_attempt"
            return "task_canceled" if task.get("cancel_requested") else ""
        session.context["agent_task_interrupt"] = interrupt
        with bind_task_budget(session.budget):
            recovery_error = ""
            if session.context.pop("agent_recovery_pending", False):
                from .recovery import require_continuable
                try:
                    task = self._get(session.scope, task_id)
                    require_continuable(self._review_recovery(session.scope, task, session.grant))
                except AgentBoundaryError as exc:
                    recovery_error = exc.code
            if recovery_error:
                from shinku.agent.loop import AgentRunResult
                result = AgentRunResult("blocked", 0, reason=recovery_error,
                    completion={"accepted": False, "reason": recovery_error, "missing": [recovery_error]})
            else:
                result = session.executor.run(task_id=task_id, context=session.context, grant=session.grant,
                    checkpoint=session.checkpoint,
                    observer=lambda state, status: self._observe(task_id, session, state, status))
        session.checkpoint = result.checkpoint
        if result.status == "yielded" and not event.is_set():
            return AgentTaskContinuation(lambda e: self._run_slice(task_id, session, e),
                metadata={"budget": session.budget.snapshot()}, on_cancel=lambda: self._queued_cancel(task_id, session))
        status = "canceled" if event.is_set() else result.status

        def update(task):
            task.pop("result_commit_pending", None)
            task.update(status=status, stop_reason="task_canceled" if event.is_set() else result.reason,
                completion=copy.deepcopy(result.completion), completion_attempt=session.attempt,
                budget=session.budget.snapshot())
            for key in ("material_package", "material_information", "stage_strategy"):
                if key in session.context["agent_task_state"]:
                    task[key] = copy.deepcopy(session.context["agent_task_state"][key])
            # Bind the export to the revision and timestamp that will be saved.
            export = {**task, "revision": task["revision"] + 1}
            task['execution_result_version'] = export['revision']
            task["execution_result"] = self.result_exporter(export, status=status,
                reason=task["stop_reason"], report=result.final_text)
        saved = self._write(session.scope, task_id, session.attempt, update)
        exposed = saved["execution_result"]["execution"]["status"]
        workflow = exposed if exposed in {"completed", "failed", "canceled", "blocked", "waiting_user"} else "blocked"
        return AgentTaskOutcome(workflow, error=saved["stop_reason"],
            metadata={"agent_status": exposed, "budget": session.budget.snapshot()})

    def _status(self, scope, task_id):
        with self._lock:
            self._get(scope, task_id)  # Preserve the port's explicit scope/not-found errors.
            return self.task_views.get(scope, task_id)

    def _check_handle(self, task, handle):
        if not isinstance(handle, TaskHandle) or (task.get("execution_attempt", 0), task.get("revision", 0)) != (handle.attempt, handle.version):
            raise AgentBoundaryError("task_version_conflict")

    def _cancel(self, scope, handle):
        with self._lock:
            task = self._get(scope, handle.task_id)
            self._check_handle(task, handle)
            if task["status"] in {"completed", "failed", "canceled"} and task.get("execution_result"):
                raise AgentBoundaryError("task_already_" + task["status"])
            self._grant(scope, task, "cancel")
            response = self.coordinator.cancel(task.get("execution_work_id") or handle.task_id,
                                               profile_user_id=scope.requester, session_id=scope.session)
            if not response.get("ok"):
                raise AgentBoundaryError(response.get("reason") or "cancel_rejected")
            if not task.get("execution_work_id"):
                current = self._get(scope, handle.task_id)
                current["cancel_requested"] = True
                if response["task"]["status"] == "canceled" and not response["task"].get("worker_active"):
                    # waiting_user 已退出 worker，协调器不再保留 on_cancel。
                    # 页面为人工接续而保留，但用户取消后仍须按原任务归属释放；
                    # 不启动新执行、不重放动作，清理失败只记未知。
                    session = self._sessions.get(handle.task_id)
                    if session is not None and session.attempt == handle.attempt and session.checkpoint is not None:
                        from shinku.agent.execution_sessions import stop_execution_sessions
                        stop_execution_sessions(session.checkpoint.state, self.executor.tool_host.registry.view(), 'task_canceled')
                        current['progress'] = copy.deepcopy(session.checkpoint.state.progress)
                    current.pop("result_commit_pending", None)
                    current.update(status="canceled", stop_reason="task_canceled")
                    if self.task_store.result_events is not None:
                        now = self.task_store.clock()
                        current["updated"] = now
                        current["completion"] = {"accepted": False, "reason": "task_canceled"}
                        current["completion_attempt"] = current["execution_attempt"]
                        current['execution_result_version'] = current['revision'] + 1
                        current["execution_result"] = export_task_result({**current, "revision": current["revision"] + 1},
                            status="canceled", reason="task_canceled")
                        self.task_store.save(current, now=now)
                    else:
                        self.task_store.save(current)
                else:
                    self.task_store.save(current)
            # The existing QQ worker observes its coordinator event and writes
            # its own ledger checkpoint. Do not race its optimistic revision.
            state = self._status(scope, handle.task_id)
            return {"cancel_requested": True, "stopped": state["status"] == "canceled" and not state["worker_active"],
                    "files_rolled_back": False, "task": state}

    def _resume(self, scope, handle, request_key):
        bounded_text(request_key, 256, "invalid_request_key")
        with self._lock:
            task = self._get(scope, handle.task_id)
            replay = task.get("resume_request", {})
            if replay == {"key": request_key, "attempt": handle.attempt, "version": handle.version}:
                return self._handle(task)
            self._check_handle(task, handle)
            if task.get("execution_work_id"):
                # Existing QQ handoffs retain their own admission/approval
                # resume flow; do not resubmit its worker or executable data.
                raise AgentBoundaryError("resume_via_host_required")
            record = self.coordinator.get(handle.task_id, profile_user_id=scope.requester, session_id=scope.session)
            if record and (record.get("worker_active") or record["status"] in {"queued", "running"}):
                raise AgentBoundaryError("task_already_active")
            restarted = bool(self.recovery_enabled and record and record.get("metadata", {}).get("recovered_after_restart"))
            if task["status"] not in {"failed", "blocked", "waiting_user", "partial", "canceled"} and not restarted:
                raise AgentBoundaryError("task_not_resumable")
            grant = self._grant(scope, task, "resume")
            if not grant.resume_safe or (not self.recovery_enabled and (task.get("inflight") or task.get("uncertain_attempts"))):
                raise AgentBoundaryError("resume_effects_unconfirmed")
            old = task.get("host_grant")
            if not isinstance(old, dict) or "allowed_tools" not in old or "permission_mode" not in old:
                raise AgentBoundaryError("resume_legacy_grant_unavailable")
            modes = {"read": 0, "readonly": 0, "ask": 1, "traditional": 1, "confirm": 1, "full": 2, "ultra": 3}
            if (not set(grant.allowed_tools) <= set(old["allowed_tools"])
                    or modes.get(grant.permission_mode, 99) > modes.get(old["permission_mode"], -1)):
                raise AgentBoundaryError("resume_permission_expansion")
            old_policy = old.get("policy", {})
            if old_policy:
                for name in grant.allowed_tools:
                    formerly_allowed = name not in old_policy["blocked"] and (
                        not old_policy["allowed"] or name in old_policy["allowed"])
                    former_risk = old_policy["risk_by_tool"].get(name, old_policy["default_risk"])
                    if grant.policy.accepts(name) and (not formerly_allowed or (
                        former_risk in old_policy["approval_required_by_risk"] and not grant.policy.requires_approval(name))):
                        raise AgentBoundaryError("resume_permission_expansion")
            fresh = grant.budget().snapshot()
            if not isinstance(task.get("budget"), dict) or any(k not in task["budget"] for k in
                    ("max_requests", "max_tokens", "max_tools", "max_active_seconds")):
                raise AgentBoundaryError("resume_budget_unavailable")
            # null表示无累计上限。有限旧授权不能借恢复自动扩大为持续模式。
            if any(task["budget"][k] is not None and (fresh[k] is None or fresh[k] > task["budget"][k])
                   for k in ("max_requests", "max_tokens", "max_tools", "max_active_seconds")):
                raise AgentBoundaryError("resume_budget_expansion")
            previous = self._sessions.get(handle.task_id)
            if not self.recovery_enabled and previous is None and any(
                set((row.get("data") or {}).get("state_updates", {}).get("verified_effects", []))
                & {"write", "delete", "send", "process_stop"} for row in task.get("tool_calls", [])):
                raise AgentBoundaryError("resume_checkpoint_unavailable")
            if self.recovery_enabled:
                from .recovery import record_recovery, require_continuable
                review = self._review_recovery(scope, task, grant)
                require_continuable(review)
                record_recovery(task, review, request_key=request_key,
                    reason="service_restart" if restarted or previous is None else "explicit_continuation")
            from .continuation_context import retain_resume_report
            retain_resume_report(self, scope, task, grant)
            task.update(execution_attempt=handle.attempt + 1, status="queued", cancel_requested=False,
                resume_request={"key": request_key, "attempt": handle.attempt, "version": handle.version})
            task.pop("execution_result", None)
            task.pop('execution_result_version', None)
            saved = self.task_store.save(task)
            try:
                self._queue(scope, saved, grant,
                    checkpoint=previous.checkpoint if previous and not self.recovery_enabled else None,
                    allow_reopen=record is not None)
            except Exception:
                self._write(scope, handle.task_id, handle.attempt + 1,
                    lambda t: t.update(status="blocked", stop_reason="recovery_admission_failed"))
                raise
            return self._handle(saved)

    def _review_recovery(self, scope, task, grant):
        from .recovery import TaskRecoveryGate
        previous = self._sessions.get(task["task_id"])
        context = execution_context(task, grant, attempt=task["execution_attempt"], work_id=task["task_id"])
        budget = grant.budget()
        budget.absorb_prior(task.get("budget"))
        reviewed = {**task, "budget": budget.snapshot()}
        handlers = self.executor.tool_host.registry.view()
        if task.get('input_mode') == 'conversation':
            from .conversation import ConversationTurn
            turn = ConversationTurn(self, scope, task['task_id'], task['execution_attempt'], grant)
            context['_agent_conversation_turn'] = turn
            host, _, _ = turn.tool_host(self.executor.tool_host, grant.allowed_tools, grant.policy)
            handlers = host.registry.view()
        return TaskRecoveryGate(self.task_store, handlers,
            material_store=self.materials).review(reviewed, context=context, still_authorized=grant.still_authorized,
                mode="same_process" if previous is not None else "restart")

    def _recovery(self, scope, handle):
        with self._lock:
            task = self._get(scope, handle.task_id)
            self._check_handle(task, handle)
            grant = self._grant(scope, task, "resume")
            record = self.coordinator.get(task.get("execution_work_id") or task["task_id"],
                profile_user_id=scope.requester, session_id=scope.session)
            if record and (record.get("worker_active") or record["status"] in {"queued", "running"}):
                raise AgentBoundaryError("task_already_active")
            return self._review_recovery(scope, task, grant)

    def _result(self, scope, task_id):
        with self._lock:
            task = self._get(scope, task_id)
            result = task.get("execution_result")
            if result:
                if canonical_hash({k: v for k, v in result.items() if k != "result_hash"}) != result.get("result_hash"):
                    raise AgentBoundaryError("result_integrity_failed")
                if result.get("attempt") != task["execution_attempt"]:
                    raise AgentBoundaryError("result_attempt_conflict")
                version = result.get('version')
                # 补充资料只改变输入修订，旧结果/事件必须保持原版本可读；
                # 新结果绑定生成时版本。旧库无此标记时保留有界历史兼容，
                # 消费者的当前版本/禁止重发门仍独立执行，历史可读不是重新验收。
                if (result.get('task_id') != task_id or result.get('scope') != {
                        'requester': scope.requester, 'session': scope.session}
                        or type(version) is not int or not 1 <= version <= task['revision']
                        or version != task.get('execution_result_version', version)):
                    raise AgentBoundaryError('result_binding_conflict')
                if task["status"] != result["execution"]["status"] and not (
                    task["status"] == "completed" and result["execution"]["status"] == "unverified") and not (
                    task['status'] == 'partial' and task.get('stop_reason') == 'task_input_revised'
                    and version < task['revision'] and task.get('input_revision_base_result') == result['result_hash']):
                    raise AgentBoundaryError("result_state_conflict")
                return copy.deepcopy(result)
            view = self._status(scope, task_id)
            return export_task_result(task, status=view["status"],
                reason=view["reason"] or "execution_result_unavailable", available=False)

    def _supplement(self, scope, handle, request, *, allow_terminal=False):
        if self.materials is None or not isinstance(request, AgentSupplement):
            raise AgentBoundaryError("typed_material_supplement_required")
        payload = request.payload()
        digest = canonical_hash(payload)
        with self._lock:
            task = self._get(scope, handle.task_id)
            prior = task.get("material_supplements", [])
            repeated = next((row for row in prior if row["key"] == request.request_key), None)
            if repeated is not None:
                if repeated["digest"] != digest:
                    raise AgentBoundaryError("material_supplement_key_conflict")
                return self._handle(task)
            self._check_handle(task, handle)
            allowed = {"waiting_user", "partial", "blocked", "failed"}
            if allow_terminal:
                allowed |= {"canceled", "completed"}
            if task["status"] not in allowed or len(prior) >= 16:
                raise AgentBoundaryError("material_supplement_not_allowed")
            work = self.coordinator.get(task.get("execution_work_id") or handle.task_id,
                profile_user_id=scope.requester, session_id=scope.session)
            if work and (work.get("worker_active") or work["status"] in {"queued", "running"}):
                raise AgentBoundaryError("task_already_active")
            candidate = {**task, "material_supplement_request": payload}
            grant = self._grant(scope, candidate, "supplement")
            from .materials import MaterialSource, TaskMaterialSession
            sources = list(grant.material_sources)
            if request.clarification:
                sources.append(MaterialSource("clarification:" + request.request_key, "clarification", request.clarification,
                    provenance={"message_id": "supplement:" + request.request_key, "sender_id": scope.requester,
                                "relation": "caller_clarification"}))
                task["requests"].append({"message_id": "supplement:" + request.request_key, "text": request.clarification})
            constraints = list(dict.fromkeys([*task.get("confirmed_constraints", []), *payload["constraints"]]))
            if len(constraints) > 16:
                raise AgentBoundaryError("execution_input_oversized")
            # 显式修订可重开已结束任务的输入，但旧结果和证据保留为旧版本。
            # 不创建执行轮次；后续resume仍核对权限、预算和真实副作用。
            if allow_terminal and task['status'] == 'completed':
                task.update(status='partial', stop_reason='task_input_revised',
                    input_revision_base_result=task.get('execution_result', {}).get('result_hash'))
                task['completion'] = {'accepted': False, 'reason': 'task_input_revised',
                    'satisfied': [], 'missing': [r['id'] for r in (task.get('contract') or {}).get('requirements', [])]}
            task["confirmed_constraints"] = constraints
            old = copy.deepcopy(task.get("material_package", {}))
            source_ids = {source.source_id for source in sources}
            replaced = [row["reference"] for row in old.get("items", []) if row["source_id"] in source_ids]
            old["items"] = [row for row in old.get("items", []) if row["source_id"] not in source_ids]
            context_supplied = any(row.kind in {"related_message", "clarification"} and row.complete for row in sources)
            gaps = [row for row in old.get("gaps", []) if row["source_id"] not in source_ids
                    and not (row["source_id"] == "related_context" and context_supplied)]
            task["material_package"] = self.materials.package(task, sources,
                authorization_ref=grant.authorization_ref, previous=old, gaps=[*gaps, *grant.material_gaps])
            task["material_package"]["superseded_references"] = replaced
            task["material_supplements"] = [*prior, {"key": request.request_key, "digest": digest,
                "attempt": handle.attempt, "version": handle.version}]
            saved = self.task_store.save(task)
            active = TaskMaterialSession(self.materials, saved, still_authorized=grant.still_authorized)
            active.checkpoint()
            saved = self.task_store.save(saved)
            # Sources/clarification change only this task's input revision.
            # Old canonical result and event identities remain untouched;
            # continuing requires the existing separately authorized resume.
            return self._handle(saved)


class AgentClient:
    def __init__(self, service: AgentService, scope: AgentScope):
        self._service, self._scope = service, scope

    def submit(self, request):
        return self._service._submit(self._scope, request)

    def status(self, task_id):
        return self._service._status(self._scope, task_id)

    def cancel(self, handle):
        return self._service._cancel(self._scope, handle)

    def resume(self, handle, *, request_key):
        return self._service._resume(self._scope, handle, request_key)

    def recovery(self, handle):
        return self._service._recovery(self._scope, handle)

    def result(self, task_id):
        return self._service._result(self._scope, task_id)

    def question(self, task_id):
        from .questions import pending_question
        return pending_question(self._service, self._scope, task_id)

    def answer_question(self, handle, *, question_id, text):
        from .questions import answer_question
        return answer_question(self._service, self._scope, handle, question_id, text)

    def supplement(self, handle, request):
        return self._service._supplement(self._scope, handle, request)
