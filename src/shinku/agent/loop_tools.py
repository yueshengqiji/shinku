"""Exact approved-call restoration, tool-batch admission and authorization."""

from __future__ import annotations

import json
import hashlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from shinku.tools.execution import validate_invocation
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope

from .loop_models import AgentLoopDependencies, AgentLoopRun, LoopToolBatch
from .budget import current_task_budget
from .tool_closeout import close_repeated_tools


def _ledger_tool_limit() -> int:
    budget = current_task_budget()
    return 2048 if budget is not None and budget.managed else 128

if TYPE_CHECKING:
    from .loop import AgentDecision, AgentRunResult


def _restored_mutation_duplicate(run, invocation, deps):
    if not (isinstance(run.context, dict) and (run.context.get("agent_budget_resume") or run.context.get("agent_recovery_resume"))):
        return False
    handler = deps.handlers().get(invocation.name)
    effects = getattr(handler, "agent_completion_effects", ()) or ()
    effects = (effects,) if isinstance(effects, str) else effects
    if getattr(handler, "agent_permission_class", "") == "read" and not set(effects) & {
            "write", "delete", "send", "process_stop", "command_run", "reminder"}:
        return False
    fingerprint = hashlib.sha256(deps.call_signature(invocation).encode("utf-8")).hexdigest()
    return any(old.get("status") == "ok" and old.get("name") == invocation.name and (
        old.get("call_fingerprint") == fingerprint or (not old.get("call_fingerprint")
            and old.get("arguments") == dict(invocation.arguments)))
        for old in run.context.get("agent_restored_tool_calls", ()))


def execute_approved_calls(
    run: AgentLoopRun,
    approved_invocation: ToolInvocation | None,
    approved_invocations: Sequence[ToolInvocation] | None,
    note_execution: Callable[[ToolInvocation, ToolResultEnvelope], None],
    deps: AgentLoopDependencies,
) -> AgentRunResult | None:
    approved_calls = tuple(
        item
        for item in (
            approved_invocations
            if approved_invocations is not None
            else ((approved_invocation,) if approved_invocation is not None else ())
        )
        if isinstance(item, ToolInvocation)
    )

    # 用户已经在上一轮批准了确定的调用时，恢复这些调用本身，
    # 而不是让模型重新猜一次工具，避免重复调用或换错工具。
    if approved_calls:
        if any(_restored_mutation_duplicate(run, item, deps) for item in approved_calls):
            return deps.result(run.state, run.envelopes, status="blocked", reason="approved_effect_already_recorded")
        if len(approved_calls) + len(run.state.tool_calls) > _ledger_tool_limit():
            return deps.result(run.state, run.envelopes, status="blocked", reason="tool_budget_exhausted")
        if run.state.rounds >= run.round_end:
            return deps.result(run.state, run.envelopes, status="failed", reason="max_rounds")
        run.state.rounds += 1
        run.state.transcript.append(
            {
                "role": "assistant",
                "kind": "tool",
                "content": "执行用户刚刚明确授权的工具调用。",
                "tool_calls": [
                    {
                        "id": item.id,
                        "name": item.name,
                        "arguments": dict(item.arguments or {}),
                    }
                    for item in approved_calls
                ],
            }
        )
        for approved in approved_calls:
            signature = deps.call_signature(approved)
            run.signature_attempts[signature] = run.signature_attempts.get(signature, 0) + 1
            note_execution(
                approved,
                deps.execute_and_record(run.state, run.envelopes, approved, context=run.context, approved=True),
            )
    return None


def execute_tool_decision(
    run: AgentLoopRun,
    decision: AgentDecision,
    note_execution: Callable[[ToolInvocation, ToolResultEnvelope], None],
    deps: AgentLoopDependencies,
) -> AgentRunResult | None:
    batch = _prepare_tool_batch(run, decision, deps)
    if not isinstance(batch, LoopToolBatch):
        return batch
    for invocation in batch.invocations:
        result = _admit_invocation(run, batch, invocation, note_execution, deps)
        if result is not None:
            return result
    approval = _request_batch_approval(run, batch.ready_calls, deps)
    if approval is not None:
        return approval
    return _execute_ready_calls(run, batch.ready_calls, note_execution, deps)


def _prepare_tool_batch(
    run: AgentLoopRun,
    decision: AgentDecision,
    deps: AgentLoopDependencies,
) -> LoopToolBatch | AgentRunResult:
    invocations = decision.invocations or ((decision.invocation,) if decision.invocation is not None else ())
    if not invocations:
        return deps.result(run.state, run.envelopes, status="failed", reason="missing_invocation")

    # 记录助理发起的工具调用，保持完整的时序上下文 (user -> assistant -> tool)
    assistant_content = str(decision.text or "").strip()
    if not assistant_content:
        call_descs = [
            f"调用工具 `{inv.name}` 参数={json.dumps(inv.arguments or {}, ensure_ascii=False)}"
            for inv in invocations
        ]
        assistant_content = "；".join(call_descs)
    run.state.transcript.append(
        {
            "role": "assistant",
            "kind": "tool",
            "content": assistant_content,
            "tool_calls": [
                {
                    "id": inv.id,
                    "name": inv.name,
                    "arguments": dict(inv.arguments or {}),
                }
                for inv in invocations
            ],
        }
    )

    ready_calls: list[ToolInvocation] = []
    round_signatures: set[str] = set()
    discovery_slots = max(0, deps.max_source_discovery_calls() - run.state.source_discovery_calls)
    return LoopToolBatch(invocations, ready_calls, round_signatures, discovery_slots)


def _allow_source_discovery(
    run: AgentLoopRun,
    batch: LoopToolBatch,
    invocation: ToolInvocation,
    deps: AgentLoopDependencies,
) -> bool:
    if deps.is_source_discovery(invocation.name):
        read_required = run.state.research.enabled and run.state.research.read_before_search()
        suppressed = invocation.name in run.state.suppressed_tool_names
        if batch.discovery_slots == 0 or read_required or suppressed:
            message = ("已有值得阅读的关联候选；本次扩搜未执行，先读必要正文或说明候选无法访问。" if read_required
                       else "本轮发现已暂停或额度已用完；该调用未执行。仍可读取已有来源或如实总结资料缺口。")
            code = "source_read_required" if read_required else "source_discovery_budget_exhausted"
            envelope = ToolResultEnvelope(invocation.id, "error", message, {"code": code})
            run.envelopes.append(envelope)
            run.state.tool_calls.append({"name": invocation.name, "id": invocation.id, "arguments": dict(invocation.arguments), "status": "error", "executed": False, "data": envelope.data})
            run.state.transcript.append({"role": "tool", "name": invocation.name, "invocation_id": invocation.id, "status": "error", "content": message,
                                     "data": envelope.data})
            return False
        batch.discovery_slots -= 1
    return True


def _admit_invocation(
    run: AgentLoopRun,
    batch: LoopToolBatch,
    invocation: ToolInvocation,
    note_execution: Callable[[ToolInvocation, ToolResultEnvelope], None],
    deps: AgentLoopDependencies,
) -> AgentRunResult | None:
    if not _allow_source_discovery(run, batch, invocation, deps):
        return None
    signature = deps.call_signature(invocation)
    previous_success = run.successful_signatures.get(signature)
    restored_duplicate = _restored_mutation_duplicate(run, invocation, deps)
    read_after_change = bool(
        previous_success
        and previous_success[0] == "observe"
        and previous_success[1] < run.mutation_epoch
    )
    handler = deps.handlers().get(invocation.name)
    changed = getattr(handler, "agent_previous_observation_changed", None)
    if previous_success and previous_success[0] == "observe" and callable(changed):
        try:
            read_after_change = read_after_change or changed(invocation=invocation, context=run.context) is True
        except Exception:
            pass
    repeatable_read = bool(
        deps.progress_class(invocation) == "observe"
        and getattr(handler, "agent_repeatable_observation", False)
    )
    # Background observations may change independently of this
    # Agent's mutation epoch. Existing attempt/read budgets still apply.
    duplicate = restored_duplicate or signature in batch.round_signatures or bool(previous_success and not read_after_change and not repeatable_read)
    if duplicate:
        envelope = ToolResultEnvelope(
            invocation_id=invocation.id,
            status="error",
            model_feedback="<tool_use_error>相同工具调用已经执行过，请根据上一次结果收尾，不要改写命令绕过重复保护。需要用户操作时说明当前状态并等待用户。</tool_use_error>",
            data={"code": "duplicate_tool_call", "tool": invocation.name},
        )
        run.envelopes.append(envelope)
        run.state.last_envelope = envelope
        run.state.tool_calls.append(
            {"name": invocation.name, "id": invocation.id, "arguments": dict(invocation.arguments),
             "status": "error", "executed": False, "data": dict(envelope.data or {})}
        )
        run.state.transcript.append(
            {"role": "tool", "name": invocation.name, "status": "error", "content": envelope.model_feedback}
        )
        run.duplicate_rejections[signature] = run.duplicate_rejections.get(signature, 0) + 1
        if run.duplicate_rejections[signature] >= 2:
            return close_repeated_tools(run, deps)
        return None
    failure_key = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    if (run.signature_attempts.get(signature, 0) >= 3
            or run.state.progress.get("failure_attempts", {}).get(failure_key, 0) >= 3):
        return close_repeated_tools(run, deps)
    run.signature_attempts[signature] = run.signature_attempts.get(signature, 0) + 1
    batch.round_signatures.add(signature)
    validation = validate_invocation(
        invocation,
        deps.handlers(),
        policy=deps.policy(),
        context=run.context,
    )
    if validation.ok:
        progress_class = deps.progress_class(invocation)
        if progress_class == "observe":
            run.state.observation_calls += 1
        batch.ready_calls.append(invocation)
    else:
        note_execution(invocation, deps.execute_and_record(run.state, run.envelopes, invocation, context=run.context))
    return None


def _request_batch_approval(
    run: AgentLoopRun,
    ready_calls: list[ToolInvocation],
    deps: AgentLoopDependencies,
) -> AgentRunResult | None:
    # 先完成本批次的无副作用校验，再一次性请求授权，避免模型连续
    # 规划多个工具时出现“批准一个、漏掉其余”的半执行状态。
    if len(ready_calls) + len(run.state.tool_calls) > _ledger_tool_limit():
        return deps.result(run.state, run.envelopes, status="blocked", reason="tool_budget_exhausted",
                            final_text="本轮已达到工具调用预算，尚未完成的部分先保留，不继续执行。")
    # 默认策略对未知工具按 medium 处理；只有明确标为不需授权的
    # 低风险批次才自动执行。
    approval_calls = list(ready_calls)
    if deps.approval_gate() is None and deps.policy() is not None:
        approval_calls = [item for item in ready_calls if deps.policy().requires_approval(item.name)]
        if approval_calls:
            return deps.result(
                run.state,
                run.envelopes,
                status="failed",
                reason="approval_gate_unavailable",
                final_text="这项工具操作需要授权，但当前宿主没有可用的授权通道，因此我没有执行。",
            )
    if ready_calls and deps.approval_gate() is not None and deps.policy() is not None:
        if not any(deps.policy().requires_approval(item.name) for item in ready_calls):
            approval_calls = []
    if approval_calls and deps.approval_gate() is not None:
        try:
            try:
                # 混合批次只要含有一个需要授权的调用，就整体等待授权，避免
                # 用户批准前先执行同一批里的另一项副作用。
                approval = deps.approval_gate()(invocations=tuple(ready_calls), state=run.state, context=run.context)
            except TypeError:
                if len(ready_calls) != 1:
                    raise
                approval = deps.approval_gate()(invocation=ready_calls[0], state=run.state, context=run.context)
        except Exception as exc:
            return deps.result(run.state, run.envelopes, status="failed", reason=f"approval_gate_error:{type(exc).__name__}")
        if approval is not None:
            run.state.transcript.append(
                {
                    "role": "system",
                    "content": f"工具调用等待用户授权：{approval.question}",
                    "approval_id": approval.request_id,
                }
            )
            return deps.result(
                run.state,
                run.envelopes,
                status="waiting_user",
                final_text=approval.question,
                reason="tool_approval_required",
                pending_approval=approval,
            )
    return None


def _execute_ready_calls(
    run: AgentLoopRun,
    ready_calls: list[ToolInvocation],
    note_execution: Callable[[ToolInvocation, ToolResultEnvelope], None],
    deps: AgentLoopDependencies,
) -> AgentRunResult | None:
    for invocation in ready_calls:
        if run.state.finalization_only:
            break
        note_execution(invocation, deps.execute_and_record(run.state, run.envelopes, invocation, context=run.context))
        if run.state.progress.get("last_event", {}).get("waiting"):
            continue
        if run.progress_ledger.stalled == deps.max_consecutive_observations():
            run.state.transcript.append({"role": "system", "content": "连续调用没有增加证据或推进条件，请换策略或说明阻塞原因；不要为重置预算而做无关修改。"})
            run.state.observation_budget_rejections += 1
        if run.progress_ledger.stalled >= deps.max_consecutive_observations() + 2:
            return deps.result(run.state, run.envelopes, status="blocked", reason="no_progress",
                                final_text="连续尝试没有增加证据或推进任务，我先停在这里；这次还没有完成。")
    return None
