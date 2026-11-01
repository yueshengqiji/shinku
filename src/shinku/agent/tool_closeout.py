"""重复调用终止后的有界汇报；不再执行工具，也不授予完成回执。

上游为循环流水线的防重复保护，下游仍沿用原任务结果导出。这里只补交付
文字：宿主锁定 blocked，模型只能整理已有回执；权限、预算、取消和原文
记录不变。模型拒绝收尾或上游失败时使用明确的宿主兜底，不再次重试。
"""
from __future__ import annotations


def close_repeated_tools(run, deps):
    """最多追加一次无工具规划，保留原失败原因及所有执行事实。"""
    reason = "repeated_tool_call"
    deps.check_interruption(run.state, run.envelopes)
    # 必需资料失效时不能再向模型交付旧正文。沿用宿主材料重验，直接说明
    # 缺口；这也避免把权限/完整性错误变成一次新的模型准备或补救请求。
    context = run.context if isinstance(run.context, dict) else {}
    from shinku.agent_runtime.material_session import TaskMaterialSession
    materials = context.get("agent_material_session")
    if isinstance(materials, TaskMaterialSession) and materials.information()["required_status"] != "complete":
        run.state.completion = {**run.state.completion, "accepted": False,
                                "status": "blocked", "reason": reason}
        return deps.result(run.state, run.envelopes, status="blocked", reason=reason,
            final_text="必要任务资料已变化、失效或尚未完整取得，已停止重复尝试。"
                       "任务尚未完成，请补充有效资料后继续；原始记录保留。")
    previous_finalization = run.state.finalization_only
    run.state.finalization_only = True
    run.state.tool_closeout_reason = reason
    run.state.progress["tool_closeout"] = {"reason": reason, "attempts": 1}
    run.state.transcript.append({"role": "system", "content":
        "宿主已停止重复调用。本次只能整理已有回执，不允许任何新工具或重试；"
        "说明已观察到的状态、未完成部分和下一步。若需要用户扫码、认证或确认，明确交代。"
        "返回kind=final、task_outcome=blocked（或kind=wait），不得宣称整个任务已完成；"
        "没有证据的内容说未核实，不把命令退出成功当作目标成功。"})
    decision = None
    run.state.rounds += 1
    try:
        decision = deps.parse_decision(deps.planner()(state=run.state))
    except Exception:
        # 汇报请求失败不吞掉原始阻塞，也不触发第二次付费调用。
        pass
    finally:
        run.state.finalization_only = previous_finalization
        run.state.tool_closeout_reason = ""
    deps.check_interruption(run.state, run.envelopes)
    text = ""
    if decision is not None and (decision.kind == "wait" or
            decision.kind == "final" and decision.outcome in {"blocked", "partial", "failed"}):
        text = decision.text.strip()
    if not text:
        count = sum(call.get("executed") is True and call.get("status") == "ok"
                    for call in run.state.tool_calls)
        text = (f"已停止重复检查，任务尚未完成。已有 {count} 次成功执行的工具回执保留在记录中，"
                "但不能据此确认目标已达成。请查看工具记录中的实际状态；"
                "如需扫码或其他人工操作，完成后补充情况再继续。")
    # 即使模型提出 completed/修改契约，也不能把防循环汇报升级为成功。
    run.state.completion = {**run.state.completion, "accepted": False,
                            "status": "blocked", "reason": reason}
    return deps.result(run.state, run.envelopes, status="blocked", reason=reason,
                       final_text="任务尚未完成，已停止重复检查。\n\n" + text)
