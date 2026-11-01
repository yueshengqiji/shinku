"""只读本地分析的收尾条件；沿用循环、回执账本和最终验收，不另建调度器。

不同源码读取仍会产生新指纹，所以“没有进展”无法约束无限扩读。这里以宿主
探索额度转入答复阶段。额度只决定何时整理，绝不证明字段覆盖或目标完成。
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from .analysis_acceptance import ordinary_analysis
from .analysis_contract import plan_ids


def prepare_local_analysis(run, deps) -> None:
    """在每次规划前重算阶段，跨切片和 DB 恢复均使用实际累计回执。"""
    state = run.state
    if not state.local_answer.enabled:
        return
    context = state.context if isinstance(state.context, Mapping) else {}
    requirements = context.get("agent_completion_contract", {}).get("requirements", [])
    task = context.get("agent_task_state", {})
    binding = hashlib.sha256(json.dumps(task.get("requests", []), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    previous = state.progress.get("local_analysis") or task.get("progress", {}).get("local_analysis", {})
    base = previous.get("base_tool_count", 0)
    if type(base) is not int or not 0 <= base <= len(state.tool_calls):
        base = 0
    if previous.get("request_binding") and previous["request_binding"] != binding:
        # 只有宿主接纳的新用户输入才开启下一段探索；模型自改备注/契约不能重置额度。
        # 原始累计工具/模型预算和审查次数保留，旧语义结论不得覆盖新的问题。
        base = len(state.tool_calls)
        if previous.get("phase") == "drafting":
            state.finalization_only = False
        state.local_answer.answer_check = {}
    state.progress["local_analysis"] = {**previous, "request_binding": binding, "base_tool_count": base}
    # 写入、命令执行、研究及其他独立效果不受只读分析额度截断；新用户输入
    # 已在上方释放本阶段的收尾标志，模型自行追加条件不会借此重置额度。
    ordinary = ordinary_analysis(state)
    if not requirements or (not ordinary and any(r.get("effect") not in {"read", "local_answer"} for r in requirements)):
        if not state.finalization_only:
            state.progress["local_analysis"]["phase"] = "not_applicable"
        return
    if state.finalization_only:
        return
    state.progress["local_analysis"]["phase"] = "gathering"
    # 分析可能通过命令或桥接器取得资料；换工具不能绕过探索额度。
    # 计入调用不授予其本地证据资格，来源仍必须经 LocalAnswerLedger 重验。
    calls = [c for c in state.tool_calls[base:] if c.get("executed") is not False]
    minimum = max(r.get("checks", {}).get("min_sources", 1) for r in requirements if r["effect"] == "local_answer")
    plans = plan_ids(task, context.get("agent_completion_contract", {})) if ordinary else set()
    hard_reads = [r for r in requirements if r["effect"] == "read" and r["id"] not in plans]
    targets = sum(len(r.get("targets", [])) for r in hard_reads)
    # 这是完整工具批次之后的收尾阈值；已发出的原生批次须完整记录真实回执，
    # 可以略超阈值，但下一次规划不再开启扩读，不能丢弃尚未配对的工具结果。
    exploration_limit = max(24, minimum * 2, targets + 12)
    if ordinary:
        # 普通分析的引用数量属于质量建议，不能反过来强迫扩读；明确读取条件仍须全过。
        exploration_limit = max(12, targets + 6)
    valid = sum(r["valid"] for r in state.local_answer.sources.values())
    # 足量来源时，先用同一验收器确认明确要求的读取；上限到达则诚实整理缺口。
    ready = False
    gate = deps.completion_gate()
    verify = getattr(gate, "_verify_requirement", None)
    enough = max(1, min(3, targets)) if ordinary else max(12, minimum)
    if valid >= enough and callable(verify):
        successful = [c for c in state.tool_calls if c.get("status") == "ok" and c.get("executed") is not False]
        # 计划变软不等于只读一份就强制收尾。模型可自行成稿或调整计划；
        # 宿主仅在现有读取计划已完成或探索额度耗尽时强制转入成稿。
        ready = all(verify(state, r, successful) for r in requirements if r["effect"] == "read")
    if not ready and len(calls) < exploration_limit:
        return
    state.local_answer.refresh(deps.handlers(), context)
    state.finalization_only = True
    state.progress["local_analysis"] = {"phase": "drafting", "request_binding": binding, "base_tool_count": base,
        "exploration_calls": len(calls),
        "exploration_limit": exploration_limit, "valid_sources": sum(r["valid"] for r in state.local_answer.sources.values()),
        "reason": "evidence_ready" if ready else "exploration_limit"}
    deps.observe_state(state, "planning")


def drafting_local_analysis(state) -> bool:
    return state.progress.get("local_analysis", {}).get("phase") == "drafting"
