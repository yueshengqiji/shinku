"""Explicit owner controls for inspecting and funding a paused task."""
from __future__ import annotations

from dataclasses import replace

from shinku.agent.budget_policy import grant_budget, PROFILES
from shinku.agent.loop import AgentDecision
from shinku.agent.task_state import TaskContractError
from .commands import parse_slash_command
from .reply_models import QQAgentReply


def route_budget_control(bridge, turn, trace_id):
    command = parse_slash_command(bridge._turn_command_text(turn))
    if command is None or command.name != "agent" or not command.args:
        return None
    action, *args = command.args
    if action not in {"budget", "resume"}:
        return None
    requester, session = bridge._task_scope(turn)
    if not bridge._is_master_turn(turn) or not bridge.command_interpreter._authorized(requester):
        return QQAgentReply("command", reason="slash_command_forbidden", delivery_status="suppressed")

    def reply(text, reason="task_budget_control"):
        return bridge._turn_effects().deliver(turn, text, agent_status="command", reason=reason,
            record_memory=False, track_chat_flow=False, trace_id=trace_id)

    if not bridge.adaptive_budget_enabled or not bridge.async_agent_enabled:
        return reply("当前未启用分档预算和后台续跑。", "adaptive_budget_disabled")
    if not args or len(args) > (2 if action == "resume" else 1):
        return reply("查看：/agent budget 任务号\n追加一份额度并续接：/agent resume 任务号 standard|heavy")
    target = args[0]
    record = bridge.task_coordinator.get(target, profile_user_id=requester, session_id=session)
    if record:
        target = record.get("metadata", {}).get("agent_task_id", "")
    task = bridge.task_store.get(target, requester=requester, session=session)
    if task is None:
        return reply("当前请求者与会话内没有找到这项任务。", "task_not_found_in_requester_session")
    budget = task.get("budget") or {}
    if action == "budget":
        return reply(f"任务 {target}：{task['status']}；档位 {budget.get('profile', 'legacy')}。\n"
            f"模型请求 {budget.get('requests', 0)}/{budget.get('max_requests', '?')}，"
            f"Token记账 {budget.get('tokens', 0)}/{budget.get('max_tokens', '?')}，"
            f"工具 {budget.get('tools', 0)}/{budget.get('max_tools', '?')}，"
            f"有效执行时间 {round(budget.get('active_seconds', 0))}/{budget.get('max_active_seconds', '?')} 秒。\n"
            f"已追加 {len(task.get('budget_allocation', {}).get('grants', []))} 次；"
            "Token记账包含历史输入，未报告用量的请求使用保守预留。")
    if not bridge.command_state.is_agent_enabled() or not bridge.command_state.responses_enabled():
        return reply("任务处理或回复已暂停，未追加额度。", "agent_disabled")
    profile = args[1] if len(args) == 2 else "standard"
    if profile not in PROFILES:
        return reply("额度档位可选 light、standard、heavy。", "invalid_budget_profile")
    for row in bridge.task_coordinator.list(profile_user_id=requester, session_id=session, limit=200):
        if row.get("metadata", {}).get("agent_task_id") == target and (row["worker_active"] or row["status"] == "queued"):
            return reply("这项任务仍在执行或排队，没有重复追加或启动。", "task_already_active")
    try:
        if getattr(bridge, "recovery_enabled", False):
            # Inspect before granting any credit; commit repeats the check
            # after worker claim with the optimistic task revision.
            bridge._check_task_recovery(task, {"qq_turn": turn}, check_budget=False)
            task["inflight"] = []
            task["uncertain_attempts"] = []
        # Detached changes: queue rejection, stale revision or worker claim
        # failure cannot persist a grant or clear the old stop reason.
        grant_budget(task, profile=profile, ceilings=bridge.task_budget_limits,
                     request_id=str(turn.primary_message_id), requester=requester)
        prepared_turn = bridge._prepare_qq_turn(turn, trace_id)
        context = prepared_turn.context
        decision = AgentDecision(kind="tool", task_link={"mode": "continue", "task_id": target})
        prepared = bridge._prepare_agent_task(turn, decision, context, bridge._task_id(turn), trace_id)
        if prepared.task.get("revision") != task.get("revision"):
            raise TaskContractError("task_state_conflict")
        prepared.task["budget_allocation"] = task["budget_allocation"]
        prepared.task["budget"] = task["budget"]
        prepared.task["notes"]["blocker"] = ""
        # A durable checkpoint restores receipts/conditions, never executes a
        # serialized pending call. A fresh planner sees the current authority.
        prepared = replace(prepared, decision=None)
        context["agent_budget_resume"] = True
        transcript = [{"role": "user", "content": bridge._primary_task_request(turn)},
            {"role": "system", "content": "用户明确追加预算并续接指定任务。根据原契约、已保存的实际回执与剩余条件继续；"
             "已完成的副作用不重复执行。旧正文已省略或状态不确定时先只读核验，不把旧审批当成当前授权。"}]
        return bridge._enqueue_agent_task(turn, task_id=target, context=context,
            initial_transcript=transcript, prepared_task=prepared, trace_id=trace_id)
    except (ValueError, TaskContractError) as exc:
        messages = {
            "task_not_budget_paused": "这项任务并非因预算暂停，未追加额度；请根据当前问题续接。",
            "budget_grant_already_applied": "这条追加请求已经处理过，没有重复增加额度。",
            "uncertain_execution_requires_review": "旧任务存在结果不确定的操作，需要先核对其结果，未重放操作或追加额度。",
            "budget_lifetime_limit": "这项任务已接近累计额度上限，未自动扩额。",
        }
        return reply(messages.get(str(exc), f"预算续接未启动：{exc}。"), str(exc))
