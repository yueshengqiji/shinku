"""Agent 执行循环流水线（AgentLoop Pipeline）。

职责与上下游：
- 职责：负责驱动 Agent 的核心迭代循环。协调“准备状态/恢复切片 -> 执行已审批工具 ->
  规划下一步决策 -> 终态评估 -> 用户交互挂起 -> 工具执行与回执记录”。
- 上游：由 `AgentLoop.run` 或独立执行适配器 `agent_call.py` 调用。
- 下游：调用 `loop_decisions.py`（决策规划）、`loop_tools.py`（工具执行与审批拦截）、
  `loop_completion.py`（完成度核验）及 `progress.py`（循环防卡死台账）。

状态归属与生命周期：
- `AgentState`: 持有当前任务的状态机、上下文变量、对话转录与工具调用历史；
- `AgentRunCheckpoint`: 在切片模式（`sliced`）下保存断点，供跨回合或异步 Worker 恢复；
- `ProgressLedger`: 记录执行指纹，防止重复调用相同工具或陷入死循环；
- `ToolResultEnvelope`: 承载工具执行的真实回执与副作用证据。

阅读入口：
- 核心循环入口：`run_agent_loop`
- 切片断点与上下文准备：`_prepare_loop`
- 轮数超限收尾：`_close_run`
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from typing import TYPE_CHECKING, Any

from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope

from .loop_completion import complete_agent_decision
from .loop_decisions import plan_next_decision, wait_for_user
from .loop_models import AgentLoopDependencies, AgentLoopRun
from .loop_progress import note_loop_execution
from .loop_tools import execute_approved_calls, execute_tool_decision
from .progress import ProgressLedger
from .execution_sessions import execution_event
from .local_analysis import drafting_local_analysis

if TYPE_CHECKING:
    from .loop import AgentDecision, AgentRunResult


def run_agent_loop(
    *,
    task_id: str = "",
    context: Any = None,
    initial_transcript: list[dict[str, Any]] | None = None,
    first_decision: AgentDecision | Mapping[str, Any] | None = None,
    approved_invocation: ToolInvocation | None = None,
    approved_invocations: Sequence[ToolInvocation] | None = None,
    deps: AgentLoopDependencies,
) -> AgentRunResult:
    """运行单次 Agent 规划与执行循环。

    循环时序与设计原因：
    1. 准备与切片恢复（`_prepare_loop`）：恢复检查点或初始化状态机，载入历史工具调用与失败计数；
    2. 优先执行已审批调用（`execute_approved_calls`）：若本轮是用户同意授权后的恢复切片，
       必须立即执行已暂存的工具调用，不可重新呼叫模型规划；
    3. 轮次循环控制：
       - `_resume_execution_session`: 轮询长耗时外部执行会话（无需调用大模型）；
       - `plan_next_decision`: 调用大模型生成决策（final / wait / tool）；
       - `final`: 进入 `complete_agent_decision`，必须经过核验器审查；
       - `wait`: 挂起并等待用户输入或授权；
       - `tool`: 执行工具调用并记录证据；
    4. 循环超限兜底（`_close_run`）：当达到 `round_end` 时，切片模式下保存断点退出（yielded），
       非切片模式下按未完成或部分完成诚实退出，严禁把超限包装为成功。
    """
    run = _prepare_loop(task_id, context, initial_transcript, first_decision, deps)

    def note_execution(invocation: ToolInvocation, envelope: ToolResultEnvelope) -> None:
        note_loop_execution(run, invocation, envelope, deps)

    # 优先执行此前挂起并由主人刚授权的工具调用
    approved_result = execute_approved_calls(run, approved_invocation, approved_invocations, note_execution, deps)
    if approved_result is not None:
        return approved_result
    while run.state.rounds < run.round_end or (not deps.sliced() and run.state.rounds == run.round_end and run.search_urls):
        resumed_session = _resume_execution_session(run, deps, note_execution)
        if resumed_session is True:
            continue
        if resumed_session is not False:
            return resumed_session
        planned = plan_next_decision(run, deps)
        if planned.result is not None:
            return planned.result
        decision = planned.decision
        if decision is None:
            continue
        if decision.kind == "final":
            result = complete_agent_decision(run, decision, deps)
            if result is not None:
                return result
            continue
        if decision.kind == "wait":
            return wait_for_user(run, decision, deps)
        result = execute_tool_decision(run, decision, note_execution, deps)
        if result is not None:
            return result
    return _close_run(run, deps)


def _prepare_loop(
    task_id: str,
    context: Any,
    initial_transcript: list[dict[str, Any]] | None,
    first_decision: AgentDecision | Mapping[str, Any] | None,
    deps: AgentLoopDependencies,
) -> AgentLoopRun:
    resumed = deps.checkpoint() is not None
    state = deps.checkpoint().state if resumed else deps.new_state(task_id=str(task_id or "").strip(), context=context)
    if resumed:
        context = context if context is not None else state.context
        state.context = context
    state.plain_text_completion_verifiable = bool(
        getattr(deps.completion_gate(), "supports_plain_text_completion_verification", False)
        and getattr(deps.completion_gate(), "enabled", False)
    )
    if not resumed:
        state.research.configure(deps.handlers(), context)
        state.local_answer.configure(context)
        state.history_answer.configure(context)
        state.transcript.extend(dict(item) for item in (initial_transcript or []) if isinstance(item, Mapping))
    if not resumed and isinstance(context, Mapping):
        restored = context.get("agent_restored_tool_calls", ())
        state.tool_calls.extend(dict(item) for item in restored if isinstance(item, Mapping))
    from shinku.agent_runtime.reply_delivery import prepare_state
    prepare_state(state)
    if deps.sliced() and not resumed:
        deps.set_checkpoint(deps.new_checkpoint(state))
    saved = deps.checkpoint()
    rejected_finals = saved.rejected_finals if saved else 0
    if saved:
        saved.slices += 1
    envelopes: list[ToolResultEnvelope] = saved.envelopes if saved else []
    deps.check_interruption(state, envelopes)
    if not resumed:
        # DB 恢复不含进程内账本；以已绑定回执重验，不重新执行读取或其他副作用。
        policy = deps.policy()
        readers = {name: handler for name, handler in deps.handlers().items()
                   if policy is None or (name not in policy.blocked and (not policy.allowed or name in policy.allowed))}
        state.local_answer.restore_reads(state.tool_calls, readers, context)
    signature_attempts: dict[str, int] = saved.signature_attempts if saved else {}
    successful_signatures: dict[str, tuple[str, int]] = saved.successful_signatures if saved else {}
    duplicate_rejections: dict[str, int] = saved.duplicate_rejections if saved else {}
    mutation_epoch = saved.mutation_epoch if saved else 0
    search_urls: set[str] = saved.search_urls if saved else set()
    search_stalled = saved.search_stalled if saved else 0
    prior_progress = context.get("agent_task_state", {}).get("progress", {}) if isinstance(context, Mapping) else {}
    if not resumed:
        # 旧原稿只恢复为资料，不恢复展示许可/审查结论；新请求和执行轮次继续隔离。
        drafts = prior_progress.get("answer_drafts", [])
        if isinstance(drafts, list) and drafts:
            state.progress["answer_drafts"] = copy.deepcopy(drafts)
        if prior_progress.get('advisory_plans'):
            state.progress['advisory_plans'] = copy.deepcopy(prior_progress['advisory_plans'])
        failures = prior_progress.get("failure_attempts", {})
        if isinstance(failures, Mapping):
            state.progress["failure_attempts"] = {key: value for key, value in list(failures.items())[:128]
                if isinstance(key, str) and len(key) == 64 and type(value) is int and 0 < value <= 3}
    progress_ledger = saved.progress_ledger if resumed else ProgressLedger(
        prior_progress.get("fingerprints", []), stalled=prior_progress.get("stalled", 0))
    if saved:
        saved.progress_ledger = progress_ledger
    round_end = state.rounds + deps.max_rounds() if deps.sliced() else deps.max_rounds()
    return AgentLoopRun(
        state=state,
        context=context,
        saved=saved,
        rejected_finals=rejected_finals,
        envelopes=envelopes,
        signature_attempts=signature_attempts,
        successful_signatures=successful_signatures,
        duplicate_rejections=duplicate_rejections,
        mutation_epoch=mutation_epoch,
        search_urls=search_urls,
        search_stalled=search_stalled,
        progress_ledger=progress_ledger,
        round_end=round_end,
        pending_first_decision=first_decision,
    )


def _close_run(run: AgentLoopRun, deps: AgentLoopDependencies) -> AgentRunResult:
    if run.search_urls:
        if deps.sliced() and not run.state.finalization_only:
            return deps.result(run.state, run.envelopes, status="yielded", reason="slice_complete")
        return deps.research_partial(run.state, run.envelopes)
    if deps.sliced() and (not run.state.finalization_only or drafting_local_analysis(run.state)):
        return deps.result(run.state, run.envelopes, status="yielded", reason="slice_complete")
    if run.state.last_final_text and getattr(run.state.research, "sources", None):
        return deps.result(
            run.state,
            run.envelopes,
            status="partial",
            reason="max_rounds_partial",
            final_text=run.state.last_final_text,
        )
    if run.state.observation_budget_rejections:
        final_text = "这轮已经到达规划上限，我仍没能完成后续操作或核验；所以不能算完成。"
    else:
        final_text = "这轮已经到达规划上限，我没能完成后续核验和整理；不会把未完成的结果说成成功。"
    return deps.result(run.state, run.envelopes, status="failed", reason="max_rounds", final_text=final_text)


def _resume_execution_session(run, deps, note_execution):
    """Poll only a session the executor started, without another model call.

    The handler may construct a poll, never arbitrary replacement work. Every
    poll still crosses the normal execution policy, budget and interrupt gate.
    """
    if run.pending_first_decision is not None or run.state.last_envelope is None:
        return False
    envelope = run.state.last_envelope
    receipt = (envelope.data or {}).get("state_updates", {})
    session = receipt.get("execution_session", {})
    if envelope.status != "ok" or session.get("status") != "running":
        return False
    name = (envelope.data or {}).get("tool_type", "")
    handler = deps.handlers().get(name)
    if execution_event(handler, receipt, run.context) is None:
        return False
    resume = getattr(handler, "agent_resume_session", None)
    if not callable(resume):
        return False
    try:
        invocation = resume(receipt=receipt, context=run.context)
    except Exception:
        return False
    if (not isinstance(invocation, ToolInvocation) or invocation.name != name
            or invocation.arguments.get("action") != "poll"
            or invocation.arguments.get("session_id") != session.get("session_id")
            or invocation.arguments.get("cursor", 0) != session.get("cursor", 0)):
        return False
    from .loop_tools import _ledger_tool_limit
    if len(run.state.tool_calls) >= _ledger_tool_limit():
        return deps.result(run.state, run.envelopes, status="blocked", reason="tool_budget_exhausted")
    run.state.rounds += 1
    run.state.transcript.append({"role": "assistant", "kind": "tool", "content": "读取已启动执行会话的后续结果。",
        "tool_calls": [{"id": invocation.id, "name": name, "arguments": dict(invocation.arguments)}]})
    signature = deps.call_signature(invocation)
    run.signature_attempts[signature] = run.signature_attempts.get(signature, 0) + 1
    note_execution(invocation, deps.execute_and_record(run.state, run.envelopes, invocation, context=run.context))
    return True
