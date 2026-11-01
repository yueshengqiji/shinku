"""模型调查计划的宿主归属记录，不改变权限或工具事实。

ConversationTurn 提交提案，TaskStore 在事务内保存归属及完整修订历史；核验只读取与原
请求、条件字节相符的记录。显式宿主契约、用户指定目标、行动条件不可降级。
模型提交的 origin/notes 不参与本记录，未知旧记录继续按硬条件处理。
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

from .task_state import extend_contract, validate_contract


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def plan_ids(task, contract):
    """只返回本次请求中宿主登记、内容未变的可调整调查条件。"""
    record = task.get("analysis_plan_origin", {})
    if (not isinstance(record, Mapping) or task.get("input_mode") != "conversation"
            or record.get("request_binding") != _hash(task.get("requests", []))):
        return set()
    origins = record.get("requirements", {})
    if not isinstance(origins, Mapping):
        return set()
    return {r["id"] for r in contract.get("requirements", [])
            if origins.get(r["id"]) == _hash(r)}


def _explicit_target(row, task):
    from .verification import SmartCompletionVerifier as V
    targets = V._extract_targets("\n".join(r["text"] for r in task.get("requests", [])))
    return any(V._targets_match(a, b)
               for proposed in row.get("targets", []) for a in V._normalize_target_values(proposed)
               for requested in targets for b in V._normalize_target_values(requested))


def origin_record(task, contract):
    """从原请求重新约束归属；结构化入口及旧未登记条件不会变为模型计划。"""
    from .analysis_acceptance import AnalysisDeliveryPolicy
    old = task.get("contract") or {}
    previous = {r["id"] for r in old.get("requirements", [])}
    adjustable = plan_ids(task, old)
    ordinary = AnalysisDeliveryPolicy.for_task(task) is not None
    rows = {}
    if task.get("input_mode") == "conversation" and ordinary:
        for row in contract["requirements"]:
            if row["id"] in previous and row["id"] not in adjustable:
                continue
            if (row["effect"] in {"read", "local_answer", "command_run"}
                    and not _explicit_target(row, task)):
                rows[row["id"]] = _hash(row)
    return {"request_binding": _hash(task.get("requests", [])), "requirements": rows}


def merge_plan(task, proposal):
    """只放开已登记的计划步骤；目标、范围、硬条件仍沿用原追加校验。"""
    proposal = validate_contract(proposal)
    old = task.get("contract")
    if not old:
        return proposal
    from .analysis_acceptance import AnalysisDeliveryPolicy
    adjustable = plan_ids(task, old) if AnalysisDeliveryPolicy.for_task(task) else set()
    if not adjustable:
        return extend_contract(old, proposal)
    # 仅从比较视图移除旧计划，完整原件另存历史；写/删/发送条件永不降级。
    stable = copy.deepcopy(old)
    stable["requirements"] = [r for r in old["requirements"] if r["id"] not in adjustable]
    if stable["requirements"]:
        return extend_contract(stable, proposal)
    # 全部为调查计划时仍保留原目标、交付项和范围，不让重规划改换任务。
    comparison = copy.deepcopy(old)
    comparison["requirements"] = copy.deepcopy(proposal["requirements"])
    return extend_contract(comparison, proposal)
