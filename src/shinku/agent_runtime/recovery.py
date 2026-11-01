"""Bounded read-only recovery checks over the existing task/receipt ledger.

This is an admission gate, not a scheduler or a serialized command replayer.
Unknown mutating calls require an external review; a caller's confirmation or
model prose cannot manufacture the missing executor receipt.
"""
from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping

from shinku.agent.execution_sessions import execution_event
from .contracts import AgentBoundaryError, AgentScope, canonical_hash

MUTATIONS = frozenset({"write", "delete", "send", "process_stop", "command_run", "reminder"})
MAX_RECOVERIES = 16


def repeatable_read(handler):
    effects = getattr(handler, "agent_completion_effects", ()) or ()
    if isinstance(effects, str):
        effects = (effects,)
    return bool(handler is not None and getattr(handler, "agent_permission_class", "") == "read"
        and isinstance(effects, (tuple, list, set, frozenset)) and not MUTATIONS.intersection(effects))


def _budget_reason(snapshot):
    if not isinstance(snapshot, Mapping):
        return "recovery_budget_unavailable"
    # null容量只接受宿主明确保存的持续模式；计数、错误及执行证据仍逐项核对。
    metered = snapshot.get("limit_mode") == "metered" and all(
        k in snapshot and snapshot[k] is None for k in ("max_requests", "max_tokens", "max_tools", "max_active_seconds"))
    for used, limit in (("requests", "max_requests"), ("tokens", "max_tokens"),
                        ("tools", "max_tools"), ("active_seconds", "max_active_seconds")):
        value, cap = snapshot.get(used), snapshot.get(limit)
        if (type(value) not in (int, float) or not math.isfinite(value) or value < 0
                or not metered and (type(cap) not in (int, float) or not math.isfinite(cap) or cap <= 0)):
            return "recovery_budget_invalid"
        if not metered and value >= cap:
            return "recovery_budget_insufficient"
    return "recovery_budget_insufficient" if snapshot.get("reason") else ""


def _effect_reason(call, handler, context):
    """区分执行失败与执行去向不明；已认证终态仅保留，不重放命令。"""
    if handler is None:
        return "recovery_executor_unavailable"
    if repeatable_read(handler):
        return ""
    if call.get("executed") is False:
        return ""
    failed_execution = call.get('status') == 'error' and call.get('executed') is True
    if call.get("status") != "ok" and not failed_execution:
        return "recovery_effect_unknown"
    data = call.get("data") or {}
    receipt = data.get("state_updates", {}) if isinstance(data, Mapping) else None
    if not isinstance(receipt, Mapping):
        return "recovery_receipt_unavailable"
    effects = receipt.get("verified_effects", [])
    if not isinstance(effects, (list, tuple)) or any(not isinstance(v, str) for v in effects):
        return "recovery_receipt_invalid"
    effects = set(effects)
    session = receipt.get("execution_session")
    if failed_execution and not session:
        return 'recovery_effect_unknown'
    if session:
        if not isinstance(session, Mapping):
            return "recovery_receipt_invalid"
        if session.get("status") == "running":
            return "recovery_process_requires_poll"
        binding = call.get("execution_binding")
        task = context.get("agent_task_state", {})
        if not isinstance(binding, Mapping):
            return "recovery_legacy_process_binding_unavailable"
        if (any(binding.get(key) != task.get(key) for key in ("task_id", "requester", "session"))
                or type(binding.get("attempt")) is not int or not 0 < binding["attempt"] <= task.get("execution_attempt", 0)
                or (binding.get("work_id") and (type(binding.get("work_attempt")) is not int or binding["work_attempt"] < 1))):
            return "recovery_process_binding_invalid"
        process_context = {**context, "agent_task_state": {**task, "execution_attempt": binding["attempt"]},
            "agent_work_id": binding.get("work_id", ""), "agent_work_attempt": binding.get("work_attempt", 0)}
        if execution_event(handler, receipt, process_context) is None:
            return "recovery_process_identity_unavailable"
        # A scoped executor can certify a past exit, never imply a process
        # survived a restart. Command side effects remain non-repeatable.
        if session.get("status") not in {"completed", "failed", "timed_out", "cancelled", "interrupted"}:
            return "recovery_process_state_unknown"
        if failed_execution and session.get('status') == 'completed':
            return 'recovery_receipt_invalid'
        # 失败退出也可能留下副作用。这里只允许接续规划；未知会话仍拦截，
        # 已认证的非零退出不升级为成功，不清除旧错误或重跑原命令。
    elif "command_run" in effects or "command_run" in (getattr(handler, "agent_completion_effects", ()) or ()):
        return "recovery_process_identity_unavailable"
    if effects & {"send", "process_stop", "reminder"}:
        # These adapters must expose a current, read-only receipt check.
        return "recovery_external_effect_requires_review"
    if effects & {"write", "delete"}:
        verify = getattr(handler, "agent_verify_result", None)
        try:
            refreshed = verify(receipt=copy.deepcopy(receipt), context=context) if callable(verify) else None
        except Exception:
            refreshed = None
        if not isinstance(refreshed, Mapping):
            return "recovery_effect_unconfirmed"
        artifacts = receipt.get("artifact_receipts")
        if artifacts:
            actual = refreshed.get("artifact_receipts", [])
            # Match every bound artifact, not an arbitrary true subset.
            if (not isinstance(artifacts, (list, tuple)) or not isinstance(actual, (list, tuple))
                    or len(actual) != len(artifacts) or any(
                    not isinstance(row, Mapping) or not isinstance(old, Mapping)
                    or not old.get("sha256") or not old.get("path")
                    or row.get("result_valid") is not True or row.get("path") != old.get("path")
                    or row.get("sha256") != old.get("sha256") for row, old in zip(actual, artifacts))):
                return "recovery_effect_unconfirmed"
        elif refreshed.get("result_valid") is not True:
            return "recovery_effect_unconfirmed"
    elif not session:
        # Missing metadata is not evidence that a tool was harmless.
        return "recovery_effect_class_unknown"
    return ""


class TaskRecoveryGate:
    def __init__(self, store, handlers, *, material_store=None, max_checks=256, max_seconds=10, clock=time.monotonic):
        self.store, self.handlers, self.materials = store, handlers, material_store
        self.max_checks = max(1, min(2048, int(max_checks)))
        self.max_seconds = max(0.01, min(60, float(max_seconds)))
        self.clock = clock

    def review(self, task, *, context, still_authorized, mode="restart", check_budget=True):
        deadline = self.clock() + self.max_seconds
        if not still_authorized():
            raise AgentBoundaryError("host_authorization_denied")
        current = self.store.get(task["task_id"], requester=task["requester"], session=task["session"])
        if current is None:
            raise AgentBoundaryError("recovery_task_expired_or_unavailable")
        if (current.get("revision"), current.get("execution_attempt")) != (
                task.get("revision"), task.get("execution_attempt")):
            raise AgentBoundaryError("task_version_conflict")
        reasons, checks = [], []
        result = current.get("execution_result")
        if result and current.get("status") == "completed":
            from shinku.result_expression.facts import validate_result
            if (validate_result(result, AgentScope(current["requester"], current["session"]))
                    or (result.get("attempt"), result.get("version")) != (
                        current.get("execution_attempt"), current["revision"])
                    or result.get("execution", {}).get("status") != "completed"
                    or result.get("verification", {}).get("accepted") is not True
                    or current.get("completion", {}).get("accepted") is not True
                    or current.get("completion_attempt") != current.get("execution_attempt")):
                reasons.append("recovery_result_binding_invalid")
            else:
                return {"schema_version": 1, "status": "delivery_only", "reason": "execution_already_committed",
                    "task_id": current["task_id"], "attempt": current["execution_attempt"], "version": current["revision"],
                    "delivery": self.store.result_events.delivery_view(current,
                        scope=AgentScope(current["requester"], current["session"])) if self.store.result_events else None,
                    "automatic_send": False, "automatic_execution": False}
        from .conversation import conversation_turn
        pending_turn = (current.get('input_mode') == 'conversation' and current.get('contract') is None
                        and conversation_turn(context) is not None)
        if not current.get("execution_attempt") or not current.get("contract") and not pending_turn:
            reasons.append("recovery_legacy_identity_unavailable")
        if len(current.get("recovery_history", [])) >= MAX_RECOVERIES:
            reasons.append("recovery_attempt_limit")
        if len(current.get("recovery_uncertain_reads", [])) + len(current.get("inflight", [])) + len(current.get("uncertain_attempts", [])) > 2048:
            reasons.append("recovery_check_limit")
        if check_budget:
            reason = _budget_reason(task.get("budget"))
            if reason:
                reasons.append(reason)
        package = current.get("material_package")
        if package:
            if self.materials is None:
                reasons.append("recovery_material_store_unavailable")
            else:
                if any(g.get("required", True) for g in package.get("gaps", [])) or package.get("omitted_required_gap_records", package.get("omitted_gap_records", 0)):
                    reasons.append("recovery_required_material_missing")
                for item in package.get("items", []):
                    if self.clock() >= deadline:
                        reasons.append("recovery_check_deadline")
                        break
                    try:
                        # 显式恢复先校验保留材料；尚未重新排队的取消任务仍禁止普通读取。
                        check_material = getattr(self.materials, "check_retained_for_recovery", self.materials.resolve)
                        check_material(item["reference"], task=current, still_authorized=still_authorized)
                        if item.get("required", True) and item.get("complete") is not True:
                            reasons.append("recovery_required_material_incomplete")
                    except AgentBoundaryError as exc:
                        from .reply_delivery import delivery_turn
                        if item.get('required', True) or delivery_turn(context) is None:
                            reasons.append(exc.code)
                        # 可选历史失效不妨碍当前任务恢复；后续读取仍重验引用和权限，
                        # 原材料缺口保留，不伪造完整读取或绕过必需资料门。
        calls = current.get("tool_calls", [])
        uncertain = [*current.get("inflight", []), *current.get("uncertain_attempts", [])]
        if not isinstance(calls, list) or any(not isinstance(call, Mapping) or not isinstance(call.get("name"), str)
                or not isinstance(call.get("arguments", {}), Mapping) for call in [*calls, *uncertain]):
            raise AgentBoundaryError("recovery_ledger_invalid")
        if len(calls) + len(uncertain) > self.max_checks:
            reasons.append("recovery_check_limit")
        else:
            latest_sessions = {}
            def session_key(row, session):
                # 同ID/命令也不能借用另一任务或另一尝试的终态回执。
                binding = row.get('execution_binding')
                if not isinstance(binding, Mapping):
                    return None
                return (row.get('name'), session.get('session_id'), session.get('command_sha256'),
                        *(binding.get(k) for k in ('task_id','requester','session','attempt','work_id','work_attempt')))
            for row in calls:
                data = row.get("data") or {}
                receipt = data.get("state_updates", {}) if isinstance(data, Mapping) else {}
                session = receipt.get("execution_session") if isinstance(receipt, Mapping) else None
                if isinstance(session, Mapping) and session.get("session_id") and session.get("command_sha256"):
                    key = session_key(row, session)
                    if key is not None:
                        latest_sessions[key] = row
            for call in [*calls, *uncertain]:
                if self.clock() >= deadline:
                    reasons.append("recovery_check_deadline")
                    break
                handler = self.handlers.get(call.get("name"))
                proof = call
                data = call.get("data") or {}
                receipt = data.get("state_updates", {}) if isinstance(data, Mapping) else {}
                session = receipt.get("execution_session") if isinstance(receipt, Mapping) else None
                if isinstance(session, Mapping) and call.get("status") in {'ok', 'error'} and call.get("executed") is not False:
                    proof = latest_sessions.get(session_key(call, session), call)
                from .recovery_observations import observe_process, changed_file_observation
                bound_context = {**context, "agent_task_state": current}
                proof, observation = observe_process(proof, handler, bound_context)
                reason = _effect_reason(proof, handler, bound_context)
                revalidation = changed_file_observation(proof, handler, bound_context) if reason == 'recovery_effect_unconfirmed' else None
                if revalidation:
                    reason = ''
                if reason:
                    reasons.append(reason)
                checks.append({"invocation_id": call.get("id", ""), "tool": call.get("name", ""),
                    "call_fingerprint": call.get("call_fingerprint", ""),
                    "receipt_invocation_id": proof.get("id", ""),
                    "status": "review_required" if reason else "preserve_effect_recheck_required" if revalidation else "repeatable_read" if repeatable_read(handler)
                        else "not_executed" if call.get("executed") is False else "preserve_effect",
                    "reason": reason,
                    **({'observation': observation} if observation else {}),
                    **({'revalidation': revalidation} if revalidation else {})})
        if self.clock() >= deadline:
            reasons.append("recovery_check_deadline")
        latest = self.store.get(current["task_id"], requester=current["requester"], session=current["session"])
        if not still_authorized() or latest is None or latest.get("revision") != current["revision"]:
            raise AgentBoundaryError("recovery_binding_changed")
        return {"schema_version": 1, "status": "review_required" if reasons else "continuable",
            "reason": reasons[0] if reasons else "", "reasons": list(dict.fromkeys(reasons)),
            "task_id": current["task_id"], "attempt": current["execution_attempt"], "version": current["revision"],
            "mode": mode, "checks": checks, "checks_hash": canonical_hash(checks),
            "automatic_execution": False, "automatic_send": False}


def require_continuable(review):
    if review["status"] != "continuable":
        raise AgentBoundaryError(review.get("reason") or "recovery_review_required")


def record_recovery(task, review, *, request_key, reason):
    if len(task.get("recovery_history", [])) >= MAX_RECOVERIES:
        raise AgentBoundaryError("recovery_attempt_limit")
    task["recovery_history"] = [*task.get("recovery_history", []), {
        "attempt": review["attempt"], "version": review["version"], "request_key": request_key,
        "prior_execution_status": task.get("status", "unknown"),
        "prior_commit_pending": task.get("result_commit_pending", ""),
        "original_stop_reason": task.get("stop_reason") or task.get("notes", {}).get("blocker") or "",
        "budget_stop_reason": task.get("budget", {}).get("reason", ""), "recovery_reason": reason,
        "mode": review["mode"], "checks_hash": review["checks_hash"]}]
    observations = [{'invocation_id': row['invocation_id'], **row['observation']}
                    for row in review.get('checks', []) if row.get('observation')]
    if observations:
        task.setdefault('notes', {})['recovery_process_observations'] = {
            'instruction': '以下为恢复时只读核对的进程终态。旧错误及原回执保留，不重新启动这些命令；终态不代表整个任务完成。',
            'observations': observations, 'completion_accepted': False}
        task['recovery_history'][-1]['process_observations'] = observations
    revalidation = [row['revalidation'] for row in review.get('checks', []) if row.get('revalidation')]
    if revalidation:
        task['recovery_history'][-1]['file_revalidations'] = revalidation
        # 进入接续上下文的提示不宣称当前产物通过；完成门仍使用原执行回执重验。
        task.setdefault('notes', {})['recovery_revalidation'] = {
            'instruction': '历史写入后的文件已有变化。保留当前文件，不重放旧写入；先读取当前版本并继续未完成工作，交付前重新验证。',
            'observations': revalidation, 'completion_accepted': False}
    # Unknown reads can be retried, but remain in the original audit ledger.
    task["recovery_uncertain_reads"] = [*task.get("recovery_uncertain_reads", []),
        *copy.deepcopy(task.get("inflight", [])), *copy.deepcopy(task.get("uncertain_attempts", []))]
    task["inflight"] = []
    task["uncertain_attempts"] = []
    task.pop("result_commit_pending", None)
