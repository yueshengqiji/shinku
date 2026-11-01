"""Research answer checks and completion gates with bounded rejection recovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .completion import CompletionVerdict
from .loop_models import AgentLoopDependencies, AgentLoopRun, LoopGateReview
from .local_analysis import drafting_local_analysis
from .answer_protocol import local_failure_report

if TYPE_CHECKING:
    from .loop import AgentDecision, AgentRunResult


def complete_agent_decision(
    run: AgentLoopRun,
    decision: AgentDecision,
    deps: AgentLoopDependencies,
) -> AgentRunResult | None:
    if not decision.text:
        run.state.transcript.append({"role": "system", "content": "完成回复不能为空，请补充实际结果或说明阻塞原因。"})
        return None
    # 工具数量与核验范围已保存在结构化事实/回执中。正文保持模型原文，
    # 避免宿主再拼接同一段说明；普通会话走直接交付，旧结构化入口仍走原验收。
    run.state.last_final_text = str(decision.text or "").strip()
    from shinku.agent_runtime.reply_delivery import delivery_turn
    research_verdict = (CompletionVerdict(True) if delivery_turn(run.context) is not None
                        else run.state.research.evaluate_answer(decision.text, decision.outcome))
    if not research_verdict.accepted:
        run.state.completion = research_verdict.as_dict()
        if run.state.finalization_only:
            return deps.research_partial(run.state, run.envelopes)
        if run.state.research.sources and (
            run.state.research.answer_review_calls >= 2
            or "invalid_semantic_review" in (research_verdict.reason or "")
            or (not deps.sliced() and run.state.rounds >= deps.max_rounds() - 1)
        ):
            return deps.result(
                run.state,
                run.envelopes,
                status="partial" if decision.outcome != "failed" else "failed",
                reason="research_evidence_partial_fallback",
                final_text=decision.text,
            )
        run.state.transcript.append({"role": "system", "content": research_verdict.reason})
        return None
    terminal_status = decision.outcome or "completed"
    terminal_reason = ""
    if terminal_status not in {"completed", "partial", "failed", "blocked", "waiting_background"}:
        run.state.transcript.append({"role": "system", "content": "任务结果状态无效，请使用 completed/partial/failed/blocked/waiting_background。"})
        return None
    if deps.completion_gate() is not None:
        reviewed = _review_completion_gate(run, decision, terminal_status, terminal_reason, deps)
        if reviewed.result is not None:
            return reviewed.result
        if not reviewed.accepted:
            return None
        terminal_status, terminal_reason = reviewed.status, reviewed.reason
    if terminal_status not in {"completed", "partial", "failed", "blocked", "waiting_background"}:
        return deps.result(run.state, run.envelopes, status="failed", reason="invalid_completion_status")
    return deps.result(run.state, run.envelopes, status=terminal_status, reason=terminal_reason, final_text=decision.text)


def _review_completion_gate(
    run: AgentLoopRun,
    decision: AgentDecision,
    terminal_status: str,
    terminal_reason: str,
    deps: AgentLoopDependencies,
) -> LoopGateReview:
    try:
        evaluator = getattr(deps.completion_gate(), "evaluate", None)
        gate_eval = (
            evaluator(run.state, decision.text, outcome=decision.outcome)
            if callable(evaluator)
            else deps.completion_gate()(run.state, decision.text)
        )
    except Exception as exc:
        return LoopGateReview(accepted=False, result=deps.result(
            run.state,
            run.envelopes,
            status="failed",
            reason=f"completion_gate_error:{type(exc).__name__}",
            final_text="完成核验暂时出错，我不能确认任务已经完成。",
        ))
    passed, reject_reason, terminal_status, terminal_reason = _normalize_gate_result(run, gate_eval, terminal_status, terminal_reason)
    if not passed:
        if not isinstance(gate_eval, CompletionVerdict):
            run.state.completion = CompletionVerdict(False, reason=reject_reason).as_dict()
        run.rejected_finals += 1
        if run.saved:
            run.saved.rejected_finals = run.rejected_finals
        review_exhausted = (run.state.local_answer.enabled
            and run.state.local_answer.answer_review_calls >= 2
            and not run.state.local_answer.evaluate_answer(decision.text, decision.outcome).accepted)
        if run.rejected_finals >= 3 or review_exhausted:
            # 没有审查额度时不让模型生成无法重新绑定的第三份正文。
            # 统一沿用原稿展示门，权限/来源失效仍不展示；任务验收继续失败。
            reason = "local_answer_review_exhausted" if review_exhausted else "completion_no_progress"
            from shinku.agent_runtime.conversation import conversation_turn
            if not run.state.local_answer.enabled and conversation_turn(run.context) is None:
                # 旧桥接器负责自己的失败文案；保留原核验回执，不替它交付被拒成功声明。
                return LoopGateReview(accepted=False, result=deps.result(
                    run.state, run.envelopes, status="failed", reason=reason))
            _, text = local_failure_report(run.state, deps.handlers(), reason)
            return LoopGateReview(accepted=False, result=deps.result(
                run.state, run.envelopes, status="failed", reason=reason, final_text=text))
        if run.state.finalization_only and not drafting_local_analysis(run.state):
            return LoopGateReview(accepted=False, result=deps.research_partial(run.state, run.envelopes))
        if reject_reason:
            msg = reject_reason if "完成验收未通过" in reject_reason else f"完成验收未通过：{reject_reason}"
        else:
            msg = "完成验收未通过，请检查任务结果后再汇报。"
        run.state.transcript.append({"role": "system", "content": msg})
        if isinstance(gate_eval, CompletionVerdict):
            run.state.transcript.append({"role": "system", "content":
                f"保留已满足条件 {list(gate_eval.satisfied)}；仅补核验缺失条件 {list(gate_eval.missing)}，不要重做全部任务。"})
        return LoopGateReview(accepted=False)
    return LoopGateReview(accepted=True, status=terminal_status, reason=terminal_reason)


def _normalize_gate_result(
    run: AgentLoopRun,
    gate_eval: Any,
    terminal_status: str,
    terminal_reason: str,
) -> tuple[bool, str, str, str]:
    passed = True
    reject_reason = ""
    if isinstance(gate_eval, CompletionVerdict):
        passed = gate_eval.accepted
        reject_reason = gate_eval.reason
        run.state.completion = gate_eval.as_dict()
        if passed:
            terminal_status = gate_eval.status
            terminal_reason = gate_eval.reason
    elif isinstance(gate_eval, tuple):
        passed = bool(gate_eval[0])
        reject_reason = str(gate_eval[1] or "").strip() if len(gate_eval) > 1 else ""
    elif isinstance(gate_eval, str):
        if gate_eval.strip():
            passed = False
            reject_reason = gate_eval.strip()
        else:
            passed = True
    elif gate_eval is False:
        passed = False
    elif gate_eval is not True:
        passed = False
        reject_reason = "完成验收器返回了未知格式，不能确认任务成功。"
    return passed, reject_reason, terminal_status, terminal_reason
