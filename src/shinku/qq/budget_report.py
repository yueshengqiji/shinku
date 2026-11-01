"""No extra model call is needed to retain a truthful budget stop report."""
from shinku.agent.budget import current_task_budget


def budget_stop_report(task: dict, reason: str, work_id: str) -> str:
    budget = current_task_budget()
    snapshot = budget.snapshot() if budget is not None else task.get("budget", {})
    soft = reason == "task_budget_low"
    if soft:
        headline = "旧任务因预算预留策略暂停，已保存进度"
    elif reason == "task_token_budget_exhausted":
        headline = "剩余任务预算不足以容纳下一次模型请求，已保存进度"
    else:
        headline = "任务预算已达到上限，已停止后续执行"
    goal = str((task.get("contract") or {}).get("goal") or "原任务")[:160]
    receipts = sum(row.get("status") == "ok" for row in task.get("tool_calls", []))
    remaining = task.get("notes", {}).get("remaining", [])
    pending = "；".join(str(item)[:100] for item in remaining[:4]) if isinstance(remaining, list) else ""
    rejected = snapshot.get("rejected_reservation", 0)
    estimate = (f"下一次请求预留 {rejected}，剩余 {max(0, snapshot.get('max_tokens', 0) - snapshot.get('tokens', 0))}。"
                if reason == "task_token_budget_exhausted" and type(rejected) is int and rejected > 0 else "")
    return (f"{headline}。"
            f"原任务：{goal}。\n"
            f"已保留 {receipts} 条成功工具回执；这不代表所有要求已完成。"
            + (f"待继续核验：{pending}。" if pending else "剩余要求将按原任务条件继续核验。")
            + f"\n累计请求 {snapshot.get('requests', 0)}/{snapshot.get('max_requests', '?')}，"
            f"Token记账 {snapshot.get('tokens', 0)}/{snapshot.get('max_tokens', '?')}。"
            + estimate
            + f"\n追加普通额度并续接：/agent resume {work_id} standard"
            "\n复杂任务可将 standard 改为 heavy。追加预算不会免除工具权限检查。")
