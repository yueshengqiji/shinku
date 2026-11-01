"""Round admission, planner decisions and host-owned task-contract updates."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .loop_models import AgentLoopDependencies, AgentLoopRun, LoopPlanOutcome
from .budget import current_task_budget
from .local_analysis import prepare_local_analysis, drafting_local_analysis
from .answer_protocol import retain_answer_draft, draft_scope

if TYPE_CHECKING:
    from .loop import AgentDecision, AgentRunResult


def plan_next_decision(run: AgentLoopRun, deps: AgentLoopDependencies) -> LoopPlanOutcome:
    deps.check_interruption(run.state, run.envelopes)
    prepare_local_analysis(run, deps)
    budget = current_task_budget()
    if budget is not None and budget.managed:
        stage = budget.stage()
        if run.state.progress.get("budget_stage") != stage:
            run.state.progress["budget_stage"] = stage
            run.state.transcript.append({"role": "system", "content":
                ("本任务为持续执行模式，累计用量只统计、不设硬停止；按原目标执行与核验，"
                 "适时保存工作进度。上下文压缩、取消、权限和单次请求超时仍由宿主管理。"
                 if budget.metered_only else
                f"任务累计预算状态：{stage}；请求{budget.requests}/{budget.max_requests}，"
                f"token记账{budget.tokens}/{budget.max_tokens}，工具{budget.tools}/{budget.max_tools}。"
                "这是预算提醒，不是暂停或禁止工具的指令。按原任务继续必要的执行和核验，避免无必要扩搜；"
                "分段保存进度后会自动继续，无需请求用户续跑。更新task_notes.remaining，"
                "只有满足完成条件才能宣称全部完成，硬预算由宿主在每次调用前检查。")})
    if run.saved:
        run.saved.idle_plans = run.saved.idle_plans + 1 if run.state.rounds and len(run.state.tool_calls) == run.saved.last_tool_count else 0
        run.saved.last_tool_count = len(run.state.tool_calls)
        if run.saved.idle_plans >= 6:
            return LoopPlanOutcome(result=deps.result(run.state, run.envelopes, status="blocked", reason="no_progress"))
    if not deps.sliced() and run.state.rounds >= deps.max_rounds():
        run.state.finalization_only = True
    if run.state.finalization_only:
        message = ("本地只读分析已转入交付阶段。停止扩读，使用当前有效读取账本输出实际分析正文、依据和具体建议。"
                   "逐项覆盖原契约字段；推测与未核实部分明确说明。仅在所有条件满足时用completed，缺项用partial并逐字标明缺失字段。"
                   "精简结构，聚焦关键事实和建议。数字仅限原文直接支持或可复算的结论；建议优先顺序用文字表达，"
                   "不要虚构改进耗时、性能收益或数量，也不要为段落标签添加与事实无关的数字。"
                   "未要求穷举时不逐包罗列所有模块；用约两千字以内覆盖契约字段，优先给具体优化方向及依据。"
                   "事实只引用当前有效账本sources.text；未收录的目录列表、截断文件及旧工具输出仅作线索，"
                   "不要报告目录数量、文件大小或从截断历史报告提炼当前缺陷。历史缺陷没有当前源码核实就只列为待核实。"
                   if drafting_local_analysis(run.state) else
                   "本轮发现预算已到或连续两次未新增来源。本轮只能 final，不允许工具；依据已有资料给部分结果，注明尚未核实的信息及遗漏，不能宣称完整核验。")
        run.state.transcript.append({"role": "system", "content": message})
    elif not deps.sliced() and run.state.rounds >= deps.max_rounds() - 1:
        run.state.transcript.append({"role": "system", "content": "接近规划上限，请优先核验已有来源并整理回复，不要继续换同义关键词扩搜。按原始任务交付所需信息及来源；缺少的证据明确标注，不把搜索摘要当作已核实事实。"})
    run.state.rounds += 1
    original_draft_scope = draft_scope(run.state)
    if run.pending_first_decision is not None:
        # A chat-first router may already have asked the model whether a
        # tool is needed. Reuse that exact decision so a second planner
        # call cannot silently change the selected tool or arguments.
        raw_decision = run.pending_first_decision
        run.pending_first_decision = None
    else:
        try:
            raw_decision = deps.planner()(state=run.state)
        except Exception as exc:
            deps.check_interruption(run.state, run.envelopes)
            # 只保留宿主边界错误的协议码及栈位置，不把任意异常正文/局部变量
            # （可能含凭据、用户原文）写进任务。失败状态与权限检查不放宽。
            from shinku.agent_runtime.contracts import AgentBoundaryError
            import re
            code = exc.code if isinstance(exc, AgentBoundaryError) else ''
            code = code if isinstance(code, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,95}', code) else ''
            frames, frame = [], exc.__traceback__
            while frame is not None:
                frames.append({'module': frame.tb_frame.f_globals.get('__name__', ''),
                               'function': frame.tb_frame.f_code.co_name, 'line': frame.tb_lineno})
                frame = frame.tb_next
            run.state.progress['planner_failure'] = {'exception': type(exc).__name__,
                'code': code, 'stage': 'planning', 'frames': frames[-8:]}
            detail = code or type(exc).__name__
            if run.state.finalization_only:
                return LoopPlanOutcome(result=deps.research_partial(run.state, run.envelopes, "local_analysis_planner_error:" + detail))
            return LoopPlanOutcome(result=deps.result(run.state, run.envelopes, status="failed", reason=f"planner_error:{detail}"))
    deps.check_interruption(run.state, run.envelopes)
    if isinstance(raw_decision, dict):
        retain_answer_draft(run.state, raw_decision, **original_draft_scope)
    decision = deps.parse_decision(raw_decision)
    if decision is not None and decision.kind == "final":
        # 原模型响应已捕获交接字段；规范化对象的正文副本不能覆盖同轮元数据。
        retain_answer_draft(run.state, {"kind": "final", "text": decision.text, "task_outcome": decision.outcome},
                            capture_brief=False, **original_draft_scope)
    if decision is None:
        conversation_failure = run.state.progress.get('conversation_protocol_failure')
        if conversation_failure:
            report = ({'output_empty': '模型返回了空白内容，未提供可用答复或工具调用。',
                       'output_invalid': '模型返回内容无法解析为有效的任务决策。'}.get(conversation_failure, ''))
            if conversation_failure in {'call_failed', 'call_timeout'}:
                from shinku.llm.provider_failure import failure_explanation
                report = failure_explanation(run.state.progress.get('conversation_call_failure'), conversation_failure)
            if report:
                report += '原始返回和已完成的工具记录保留；本次未完成，请处理原因后再恢复任务。错误码：conversation_' + conversation_failure
            if run.state.local_answer.enabled and conversation_failure in {'call_timeout', 'call_failed', 'output_empty', 'output_invalid'}:
                # 同一循环的传输失败优先保留已绑定原稿；状态仍失败，不能升级为
                # 完成。原文/来源实时复核、请求与尝试隔离由既有展示门负责。
                from .answer_protocol import local_failure_report
                _, report = local_failure_report(run.state, deps.handlers(), 'conversation_' + conversation_failure)
                run.state.completion['status'] = 'failed'
            return LoopPlanOutcome(result=deps.result(run.state, run.envelopes, status='failed',
                reason='conversation_' + conversation_failure, final_text=report))
        failure = run.state.progress.get("answer_protocol_failure", "")
        if run.state.local_answer.enabled and failure in {"output_length", "output_filtered", "output_incomplete", "call_timeout", "call_failed", "output_empty", "output_invalid"}:
            return LoopPlanOutcome(result=deps.research_partial(run.state, run.envelopes, "local_analysis_" + failure))
        if failure in {'output_empty', 'output_invalid'}:
            return LoopPlanOutcome(result=deps.result(run.state, run.envelopes, status='failed', reason='model_' + failure,
                final_text='模型返回空白内容或无效的任务决策，原始记录保留，本次未完成。错误码：model_' + failure))
        if run.state.finalization_only:
            return LoopPlanOutcome(result=deps.research_partial(run.state, run.envelopes, "local_analysis_invalid_decision"))
        return LoopPlanOutcome(result=deps.result(run.state, run.envelopes, status="failed", reason="invalid_decision"))
    if run.state.finalization_only:
        if decision.kind != "final":
            return LoopPlanOutcome(result=deps.research_partial(run.state, run.envelopes, "local_analysis_nonfinal_decision"))
        # Research has its own evidence/coverage gate. Do not turn an
        # honest failed/no-answer closeout into an invalid empty partial.
        if not run.state.research.enabled and not drafting_local_analysis(run.state):
            decision = replace(decision, outcome="partial")
    if not _update_task_contract(run, decision, deps):
        return LoopPlanOutcome()
    return LoopPlanOutcome(decision=decision)


def _update_task_contract(
    run: AgentLoopRun,
    decision: AgentDecision,
    deps: AgentLoopDependencies,
) -> bool:
    if isinstance(run.context, dict) and isinstance(run.context.get("agent_task_state"), dict):
        from .task_state import TaskContractError, extend_contract, update_notes

        task = run.context["agent_task_state"]
        from shinku.agent_runtime.reply_delivery import delivery_turn
        direct = delivery_turn(run.context)
        try:
            if decision.task_contract is not None:
                old_contract = task.get("contract")
                from shinku.agent_runtime.conversation import conversation_turn
                turn = conversation_turn(run.context)
                if turn is not None:
                    # 同一循环内的契约提案走宿主校验/持久化；普通旧调用者仍必须
                    # 由宿主预先初始化契约，模型元数据不能自行开启会话入口。
                    turn.adopt(decision.task_contract)
                elif old_contract is None:
                    raise TaskContractError("contract_not_initialized_by_host")
                else:
                    task["contract"] = extend_contract(old_contract, decision.task_contract)
                if direct is None:
                    run.context["agent_completion_contract"] = task["contract"]
            update_notes(task, decision.task_notes)
        except (TaskContractError, ValueError) as exc:
            if direct is not None:
                from shinku.agent_runtime.contracts import AgentBoundaryError
                if isinstance(exc, AgentBoundaryError):
                    raise  # 绑定/撤权属于宿主拒绝，不能被导航格式容错吞掉。
                # 可选导航的格式错误不能吞掉已成形的答复；原模型输出仍完整留档。
                # 权限/绑定错误属于 AgentBoundaryError，不会被此处降级。
                run.state.progress['advisory_metadata_error'] = type(exc).__name__
                deps.observe_state(run.state, "planning")
                return True
            import json
            # final携带的契约与声明工具返回相同诊断，不能迫使模型盲猜目标。
            details = json.dumps({'field_path': getattr(exc, 'field_path', ''),
                'uncovered_targets': getattr(exc, 'uncovered_targets', [])}, ensure_ascii=False)
            run.state.transcript.append({"role": "system", "content":
                f"任务状态校验失败：{exc}；{details}；保留原始目标和条件，请修正内部字段。"})
            return False
        deps.observe_state(run.state, "planning")
    return True


def wait_for_user(
    run: AgentLoopRun,
    decision: AgentDecision,
    deps: AgentLoopDependencies,
) -> AgentRunResult:
    if not decision.text:
        return deps.result(run.state, run.envelopes, status="failed", reason="empty_wait_message")
    return deps.result(run.state, run.envelopes, status="waiting_user", final_text=decision.text, reason=decision.reason)
