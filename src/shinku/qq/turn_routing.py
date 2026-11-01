"""选择对话路由并维持宿主历史核验义务。

职责与上下游：
- 职责：根据用户输入、候选任务快照和 LLM 意图裁决结果，决定当前回合的分流路径。
  区分直接对话回复（直复/闲聊/问答）与多步规划任务（Agent 模式）；
  并在直接回复投递前强制执行注意力静默、未启动动作拦截（Action Promise）以及智能完成度核验（SmartCompletionVerifier）。
- 上游：由 `turn_pipeline.py` 的 `handle_qq_turn` 在完成准备和审批恢复后调用。
- 下游：若选择 Agent 路线，交给 `task_admission.py` 或 `task_interaction.py` 调度；
  若选择直复路线，由 `deliver_direct_chat` 审查后通过 `QQTurnEffects.deliver` 投递。

阅读入口：
- 路由总入口：`select_chat_route`
- 模型决策与异常回退：`_select_model_route`
- 注意力静默裁决：`_resolve_attention`
- 历史核验与直复审查：`_validate_direct_route`
- 直复交付与完成度闸门：`deliver_direct_chat`
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, replace
from typing import Any

from shinku.agent.budget import current_task_budget
from shinku.agent.loop import AgentDecision, AgentState
from shinku.agent.reply_body import finalize_reply, is_silence_marker
from shinku.agent.task_state import AgentTaskStore

from .reply_models import QQAgentReply
from .action_promises import has_unstarted_action_promise, unstarted_action_reply
from .query_grounding import QUERY_NOT_EXECUTED_REPLY, unverified_query_reason
from .turn_models import QQPreparedTurn, QQTurnDecision, QQTurnEffects
from .turns import QQTurn


@dataclass(frozen=True)
class QQChatRoutingDependencies:
    transcript: Callable[..., list[dict[str, Any]]]
    master_user_ids: Collection[str]
    owner_gate_enabled: Callable[[], bool]
    task_id: Callable[[QQTurn], str]
    task_scope: Callable[[QQTurn], tuple[str, str]]
    task_store: AgentTaskStore
    run_chat_decision: Callable[..., tuple[AgentDecision, set[str]]]
    can_silence_autonomous: Callable[[QQTurn], bool]
    response_turn: Callable[..., QQTurn]
    effects: QQTurnEffects
    validate_dialogue_sources: Callable[..., AgentDecision] | None = None


def select_chat_route(
    prepared_turn: QQPreparedTurn,
    *,
    deps: QQChatRoutingDependencies,
) -> QQTurnDecision | QQAgentReply:
    """执行对话路由决策。

    阶段顺序与设计原因：
    1. `_select_model_route`: 调用模型获得初步决策（闲聊 vs 工具调用），并处理超额预算与上下文阻断；
    2. `_resolve_attention`: 处理自主注意力静默标记（is_silence_marker），防止未被唤醒的主动插话打扰用户；
    3. `_validate_direct_route`: 宿主层审查。若涉及历史问答，必须验证证据来源，不可信赖模型自由作答。
    """
    selection = _select_model_route(prepared_turn, deps)
    if isinstance(selection, QQAgentReply):
        return selection
    selection = _resolve_attention(selection, deps)
    if isinstance(selection, QQAgentReply):
        return selection
    decision = _validate_direct_route(selection, deps)
    if isinstance(decision, QQAgentReply):
        return decision
    return replace(selection, decision=decision)


def _select_model_route(
    prepared_turn: QQPreparedTurn,
    deps: QQChatRoutingDependencies,
) -> QQTurnDecision | QQAgentReply:
    turn = prepared_turn.turn
    trace_id = prepared_turn.trace_id
    context = prepared_turn.context
    try:
        initial_transcript = deps.transcript(
            turn,
            master_user_ids=(
                deps.master_user_ids if deps.owner_gate_enabled() else ()
            ),
        )
        task_id = deps.task_id(turn)
        requester, session = deps.task_scope(turn)
        context["agent_task_candidates"] = deps.task_store.candidates(requester=requester, session=session)
        decision, available_tool_names = deps.run_chat_decision(
            task_id=task_id,
            context=context,
            initial_transcript=initial_transcript,
            trace_id=trace_id,
        )
    except Exception as exc:
        stale_reason = context.get("qq_autonomous_stale_reason") or turn.suppression_reason("reply")
        if stale_reason:
            deps.effects.trace(trace_id, "autonomous_relevance", status="suppressed", reason=stale_reason)
            return QQAgentReply("silent", reason=stale_reason, delivery_status="suppressed")
        deps.effects.trace(trace_id, "chat", status="error", error=type(exc).__name__)
        budget = current_task_budget()
        if deps.can_silence_autonomous(turn):
            failure_reason = budget.stop_reason() if budget is not None else ""
            return deps.effects.deliver(
                turn, deps.effects.direct_chat_fallback, agent_status="failed",
                reason=failure_reason or f"bridge_error:{type(exc).__name__}",
                trace_id=trace_id,
            )
        if budget and budget.stop_reason():
            if str(turn.conversation_type or "").strip().casefold() == "group":
                return deps.effects.deliver(
                    turn,
                    deps.effects.direct_chat_fallback,
                    agent_status="failed",
                    reason=f"group_chat_suppressed_{budget.reason}",
                    trace_id=trace_id,
                )
            return deps.effects.deliver(turn, deps.effects.terminal_failure_text(budget.reason),
                                      agent_status="blocked", reason=budget.reason, trace_id=trace_id)
        return QQAgentReply("failed", reason=f"bridge_error:{type(exc).__name__}")
    turn = context.get("qq_turn", turn)
    return QQTurnDecision(
        turn, trace_id, prepared_turn.is_poke, context, task_id, initial_transcript, decision, available_tool_names,
    )


def _resolve_attention(
    selection: QQTurnDecision,
    deps: QQChatRoutingDependencies,
) -> QQTurnDecision | QQAgentReply:
    suppressed = _suppressed_autonomous_reply(selection, deps)
    if suppressed is not None:
        return suppressed
    turn = selection.turn
    trace_id = selection.trace_id
    context = selection.context
    decision = selection.decision
    turn = deps.response_turn(turn, context.get("qq_reply_routing"))
    context["qq_turn"] = turn
    if context.get("qq_visual_gap") and decision.outcome == "failed":
        return deps.effects.deliver(turn, decision.text, agent_status="failed",
            reason=context["qq_visual_gap"], trace_id=trace_id, record_memory=False,
            track_chat_flow=False, autonomous_authorized=False)
    if deps.can_silence_autonomous(turn) and not context.get("qq_chat_model_decision_valid"):
        return deps.effects.deliver(
            turn, deps.effects.direct_chat_fallback, agent_status="failed",
            reason="chat_generation_failed", trace_id=trace_id,
        )

    if (
        decision.kind in {"final", "wait"}
        and is_silence_marker(decision.text)
    ):
        # 登记 Agent 的正常聊天可自行收尾；重新验证宿主身份及整批消息，不能扩散到真人。
        from .group_participation import is_peer_exchange
        peer_closed = context.get("qq_peer_silence_confirmed") == (turn.conversation_id, tuple(turn.message_ids))
        if peer_closed and is_peer_exchange(turn, context):
            deps.effects.trace(trace_id, "peer_exchange", status="silent")
            return QQAgentReply("silent", delivery_status="suppressed", reason="peer_exchange_closed")
        # 明确@/引用仍必须回答实质请求；只有本轮独立核验的告别闭环可提前静默结束。
        closed = context.get("qq_group_closure_confirmed") == (turn.conversation_id, tuple(turn.message_ids))
        if closed and turn.conversation_type == "group":
            deps.effects.trace(trace_id, "group_closure", status="suppressed")
            return QQAgentReply("silent", delivery_status="suppressed", reason="group_farewell_closed")
        if deps.can_silence_autonomous(turn):
            deps.effects.trace(trace_id, "attention_silence", status="suppressed", mode=turn.attention_mode)
            return QQAgentReply(
                "silent",
                delivery_status="suppressed",
                reason="attention_silence",
            )
        # The marker is a private control value, never user-facing text.
        # 除自主回合与已核验告别收尾外，保留明确请求的回复。
        # If the marker appears elsewhere, preserve a reply
        # instead of leaking the marker or silently dropping an explicit turn.
        deps.effects.trace(trace_id, "attention_silence", status="ignored_ineligible", mode=turn.attention_mode)
        decision = AgentDecision.final(deps.effects.direct_chat_fallback)
    return replace(selection, turn=turn, decision=decision)


def _suppressed_autonomous_reply(selection, deps) -> QQAgentReply | None:
    reason = selection.context.get("qq_autonomous_stale_reason") or selection.turn.suppression_reason("reply")
    if reason:
        deps.effects.trace(selection.trace_id, "autonomous_relevance", status="suppressed", reason=reason)
        return QQAgentReply("silent", reason=reason, delivery_status="suppressed")
    return None


def _validate_direct_route(
    selection: QQTurnDecision,
    deps: QQChatRoutingDependencies,
) -> AgentDecision | QQAgentReply:
    turn = selection.turn
    trace_id = selection.trace_id
    context = selection.context
    decision = selection.decision
    # 主生成已收到当前请求与未决历史状态；旧义务不再触发另一个路由裁判。
    # 权限、真实动作完成度及实际历史来源的新鲜度仍在原宿主出口校验。
    # 本地来源检查不进行语义改写；完成度、权限与送达门仍由下游宿主拥有。
    if deps.validate_dialogue_sources is not None:
        changed = bool(context.get("qq_dialogue_sources_changed"))
        decision = deps.validate_dialogue_sources(turn, decision, context=context)
        if changed:
            return deps.effects.deliver(turn, decision.text, agent_status="failed",
                reason="dialogue_sources_changed", trace_id=trace_id, record_memory=False, track_chat_flow=False)
    return decision


def deliver_direct_chat(
    selection: QQTurnDecision,
    *,
    deps: QQChatRoutingDependencies,
) -> QQAgentReply:
    """直接文本对话交付及安全闸门。

    设计原因与完成判定关键边界：
    1. 自主陈旧过滤（`_suppressed_autonomous_reply`）：若消息在队列中已经过时或被取消，抑制发送；
    2. 空头承诺拦截（`has_unstarted_action_promise`）：
       - 防止模型在未调用工具的情况下口头声称“我已经开始执行/马上去办”；
       - 一旦发现未启动承诺，强制拦截为 `blocked` 状态并提示真实进展；
    3. 完成度核验（`SmartCompletionVerifier`）：
       - 模型声称完成不等于真实完成！若文本包含任务完成性断言但缺乏工具执行证据，拒绝向用户输出肯定答复；
    4. 记忆与流控追踪：仅普通对话计入记忆沉淀与流控，戳一戳（is_poke）完全旁路。
    """
    turn = selection.turn
    trace_id = selection.trace_id
    is_poke = selection.is_poke
    context = selection.context
    initial_transcript = selection.initial_transcript
    decision = selection.decision
    suppressed = _suppressed_autonomous_reply(selection, deps)
    if suppressed is not None:
        return suppressed
    reply_text = finalize_reply(decision.text, deps.effects.direct_chat_fallback)
    # 关键边界：口头动作承诺拦截，严禁未调工具就谎称在办
    if has_unstarted_action_promise(reply_text):
        deps.effects.trace(trace_id, "chat_action_promise", status="blocked_at_delivery")
        decision = AgentDecision.final(unstarted_action_reply(context), outcome="blocked")
        reply_text = decision.text
    # 只拦截未执行工具的直复路径；真实只读结果由各查询出口按回执核验，不能混用。
    query_gap = unverified_query_reason(context, replace(decision, text=reply_text))
    if query_gap:
        deps.effects.trace(trace_id, "chat_query_grounding", status="blocked_at_delivery", reason=query_gap)
        decision = replace(AgentDecision.final(QUERY_NOT_EXECUTED_REPLY, outcome="blocked"), reason=query_gap)
        reply_text = decision.text
    # 关键边界：完成判定必须有独立执行证据，非等待状态下必须经核验器校验
    if decision.kind != "wait":
        from shinku.agent.verification import SmartCompletionVerifier

        verdict = SmartCompletionVerifier().evaluate(
            AgentState(context=context, transcript=list(initial_transcript)),
            reply_text, outcome=decision.outcome,
        )
        if not verdict.accepted:
            return deps.effects.deliver(
                turn, "这项操作还没有得到实际执行和核验，不能确认已经完成。",
                agent_status="failed", reason="direct_reply_unverified", trace_id=trace_id,
            )
    suppressed = _suppressed_autonomous_reply(selection, deps)
    if suppressed is not None:
        return suppressed
    if (context.get("qq_tool_protocol_failed") and not context.get("qq_chat_model_decision_valid")
            and decision.outcome == "failed" and reply_text == deps.effects.direct_chat_fallback):
        return deps.effects.deliver(turn,
            "工具调用生成失败，内部重试也未恢复；这次还没有执行操作。",
            agent_status="failed", reason="tool_protocol_recovery_failed", trace_id=trace_id)
    deps.effects.trace(trace_id, "chat_decision", kind="direct_reply")
    return deps.effects.deliver(
        turn,
        reply_text,
        agent_status="waiting_user" if decision.kind == "wait" else
            decision.outcome if decision.outcome in {"failed", "blocked", "partial", "waiting_background"} else "completed",
        reason=(decision.reason or "chat_clarification") if decision.kind == "wait" else
            "chat_generation_failed" if decision.outcome == "failed" and reply_text == deps.effects.direct_chat_fallback
            and not context.get("qq_chat_model_decision_valid") else
            decision.reason if context.get("agent_tool_decision_mode") == "task"
            and not context.get("qq_chat_model_decision_valid") and decision.reason else "direct_chat_reply",
        record_memory=not is_poke,
        track_chat_flow=not is_poke,
        trace_id=trace_id,
        autonomous_authorized=bool(context.get("qq_chat_model_decision_valid")),
    )
