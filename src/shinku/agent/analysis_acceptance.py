"""普通说明的宿主验收策略：交付可用正文，证据质量另行披露。

入口由独立运行时用原始人类请求建立；模型契约、笔记和来源不能启用策略。
完成核验器仍先检查完整读取、当前资源和操作回执，本模块不授予任何权限。
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from collections.abc import Mapping

from .completion import CompletionVerdict
from .request_scope import RequestScope
from .request_intent import assertion_text, positive_request_text, execution_request_text
from .analysis_contract import plan_ids


_STRICT = re.compile(r"精确|逐[条项].{0,20}(?:引用|核验|验证|证据)|"
                     r"每[条项个].{0,20}(?:引用|核验|验证|证据)|"
                     r"(?:至少|不少于|必须).{0,16}(?:来源|引用|证据)|"
                     r"\bexact\b|every.{0,20}citation", re.I)
# 统计/审计作为本轮谓词才启用严格审查；“解释统计模块”中的宾语不是计算要求。
# 分句、并列请求、极性和引文范围仍复用原请求投影，不增加说明性问法白名单。
_REVIEW_ACTION = re.compile(
    r"(?:^|[，,。；;\n]|并|然后|同时|再|请|帮我|替我)\s*"
    r"(?:(?:你|先|实际|严格|完整|全面|直接|给我|一下)\s*)*"
    r"(?:审计|统计|计算|算出|求和|总计|平均)|"
    r"(?:^|\band\b|\bthen\b|\bplease\b)\s*(?:audit|calculate|count|sum|average)\b", re.I)
_EXECUTED = re.compile(r"已(?:经)?(?:运行|执行|完成|通过)[^。；\n]{0,16}(?:测试|命令|脚本)|"
                       r"已(?:经)?部署|(?:测试|部署|命令|脚本)[^。；\n]{0,12}已(?:经)?(?:通过|成功|完成)|"
                       r"\bI (?:have )?(?:ran|executed|deployed)\b", re.I)
_OPERATION = re.compile(r"(?:^|[，。；\n]|并|然后|同时|顺便|再|后|请|帮我|替我)\s*"
                        r"(?:直接|实际|也|把.{0,20})?\s*(?:修改|写入|删除|发送|创建|执行|运行|部署|重启|修复|修好)|"
                        r"\b(?:and|then|also)\s+(?:run|execute|deploy|delete|send|write|modify|restart|repair|fix)\b", re.I)
_CURRENT_ACTION = re.compile(
    r"^(?:(?:我|我们|本轮|这次|刚才|现在)(?:已经|已|成功)|"
    r"(?:已经|已|成功)(?=将|把|对|完成|修改|保存|写入|创建|删除|移除|清理|清空|发送|推送|部署|运行|执行|通过|停止|结束|关闭)|"
    r"(?:文件|代码|配置|报告|测试|部署|补丁|命令|脚本)[^，。；\n]{0,40}(?:已|成功)|"
    r"(?:全部|所有)?(?:测试|回归)(?:均|都|全部)?通过|I\s+(?:have\s+)?(?:ran|executed|deployed))", re.I)


def _execution_claim_error(verifier, state, text):
    # 只核对明确的本轮行动声明，并按分句隔离主语；代码行为描述中的
    # “已审批调用，再……保存断点”不是宿主刚执行了保存。混合行动契约不走本分支。
    for clause in re.split(r"[，。；;：:\n]", assertion_text(text)):
        clause = re.sub(r"^\s*(?:[-*#]+\s*|\d+[.)、]\s*)?", "", clause)
        if not _CURRENT_ACTION.search(clause):
            continue
        from shinku.agent_memory.claims import verified_memory_clause
        if verified_memory_clause(state, clause):
            continue  # 仅该明确记忆子句已由本库签名回执复验，其他行动仍逐句核对。
        effects = verifier._check_effects(state, clause)
        if effects is not True:
            return effects
        if _EXECUTED.search(clause) or re.match(r"(?:全部|所有)?(?:测试|回归)(?:均|都|全部)?通过", clause):
            return "执行成功声明缺少匹配的实际执行回执"
    return ""


def _binding(task):
    return hashlib.sha256(json.dumps(task.get("requests", []), ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class AnalysisDeliveryPolicy:
    """仅宿主构造的进程内策略；恢复时从完整原请求重建，不信任持久化模型备注。"""
    request_binding: str

    @classmethod
    def for_task(cls, task):
        requests = task.get("requests", [])
        text = "\n".join(row.get("text", "") for row in requests)
        intent = positive_request_text(text)
        # 说明交付是默认策略，不靠“分析/解释”白名单。此策略不授权任何动作；
        # 原请求极性、独立动作和当前真实工具效果仍由后续硬门检查。
        from .verification import SmartCompletionVerifier
        return cls(_binding(task)) if text.strip() and not (
            _STRICT.search(assertion_text(intent))
            or _REVIEW_ACTION.search(assertion_text(intent))
            or _OPERATION.search(execution_request_text(text))
            or SmartCompletionVerifier.request_requires_action(text)) else None


def ordinary_analysis(state) -> bool:
    context = state.context if isinstance(state.context, Mapping) else {}
    policy = context.get("agent_analysis_delivery_policy")
    task = context.get("agent_task_state", {})
    if not isinstance(policy, AnalysisDeliveryPolicy) or policy.request_binding != _binding(task):
        return False
    scope = context.get("agent_request_scope")
    if not isinstance(scope, RequestScope) or scope.user_messages() != [r["text"] for r in task.get("requests", [])]:
        return False
    from .verification import SmartCompletionVerifier
    if any(SmartCompletionVerifier.request_requires_action(t) for t in scope.user_messages()):
        return False
    rows = context.get("agent_completion_contract", {}).get("requirements", [])
    plans = plan_ids(task, context.get("agent_completion_contract", {}))
    return bool(rows and any(r.get("effect") == "local_answer" for r in rows)
        and all(r.get("effect") in {"read", "local_answer"}
                or r.get("effect") == "command_run" and r["id"] in plans for r in rows)
        and not getattr(getattr(state, "research", None), "enabled", False)
        and not getattr(getattr(state, "history_answer", None), "enabled", False))


def analysis_delivery_verdict(verifier, state, text, outcome, assessment):
    """检查硬条件后接纳正文；不把未核验观点写成已核实事实或已执行效果。"""
    ledger = state.local_answer
    task = state.context.get("agent_task_state", {})
    plans = plan_ids(task, state.context.get("agent_completion_contract", {}))
    soft = [r for r in assessment.requirements if r["effect"] == "local_answer" or r["id"] in plans]
    soft_ids = {r["id"] for r in soft}
    hard_missing = [key for key in assessment.missing if key not in soft_ids]
    if hard_missing:
        return CompletionVerdict(False, reason="分析所需的实际读取或目标条件未满足",
            satisfied=tuple(assessment.satisfied), missing=tuple(hard_missing))
    # 通用命令不冒充只读来源。仅已登记的调查命令、当前真实成功回执可作为
    # 执行事实；未经确认的命令导致部分交付，写/删/发送等仍禁止质量降级。
    command_gaps = []
    for call in state.tool_calls:
        handler = verifier.result_handlers.get(call.get("name"))
        effects = set(verifier._receipt(call).get("verified_effects", []))
        declared = [r for r in assessment.requirements if r["id"] in plans and r["effect"] == "command_run"]
        capabilities = set(verifier.tool_capabilities.get(call.get("name"), {}).get("effects", ()))
        if handler is not None and "command_run" in capabilities and declared:
            if effects - {"command_run", "tool:" + call.get("name", "")}:
                return CompletionVerdict(False, reason="调查命令包含其他执行效果")
            digest = verifier._receipt(call).get("execution_session", {}).get("command_sha256")
            matched = [r for r in declared
                       if digest == hashlib.sha256(r["checks"]["command"].strip().encode()).hexdigest()
                       and verifier._verify_requirement(state, r, [call])]
            if not matched:
                command_gaps.append(call.get("id", call.get("name")))
            continue
        if (getattr(handler, "agent_permission_class", "") != "read"
                or effects - {"read", "tool:" + call.get("name", "")}):
            return CompletionVerdict(False, reason="普通分析包含非只读工具或执行效果，不能使用交付降级")
    claim_error = _execution_claim_error(verifier, state, text)
    if claim_error:
        return CompletionVerdict(False, reason=claim_error)
    valid = [r for r in ledger.sources.values() if r["valid"]]
    if any(not r["valid"] and r.get("text") for r in ledger.sources.values()):
        return CompletionVerdict(False, reason="已用于答复的来源变更或授权失效")
    if not valid:
        return CompletionVerdict(False, reason="分析缺少当前有效的完整读取来源")
    # 这是交付检查，不伪装成语义正确性证明；占位完成语不能代替实际分析。
    if len(text) > 10000 or re.fullmatch(r"(?:已完成|完成了|分析完成|已分析|好的|done|completed|ok)[。.!！\s]*", text, re.I):
        return CompletionVerdict(False, reason="尚未形成可交付的分析正文")
    check = ledger.answer_check
    bound = ledger.evaluate_answer(text, outcome).accepted
    verified = bool(bound and all(ledger.satisfies(r.get("checks", {})) for r in soft if r["effect"] == "local_answer"))
    diagnostic = "" if verified else check.get("binding_error") or (
        "answer_not_independently_reviewed" if not ledger.answer_review_calls else "answer_coverage_or_sources_unverified")
    advisory = tuple({"id": r["id"], "strict_satisfied": bool(bound and ledger.satisfies(r.get("checks", {})))
        if r["effect"] == "local_answer" else r["id"] in assessment.satisfied,
        "required_fields": r.get("checks", {}).get("required_fields", []),
        "reason": "" if verified else diagnostic} for r in soft)
    plan_gaps = [r["id"] for r in soft if r["effect"] != "local_answer" and r["id"] in assessment.missing]
    return CompletionVerdict(True, status="partial" if plan_gaps or command_gaps else outcome or "completed", reason="analysis_delivered",
        satisfied=tuple(assessment.satisfied), advisory=advisory,
        layers={"policy": "ordinary_analysis" if any(r["effect"] == "command_run" for r in soft) else "ordinary_readonly_analysis",
            "execution_results": "unverified" if plan_gaps or command_gaps else "passed",
            "plan_gaps": plan_gaps, "command_gaps": command_gaps,
            "quality_review": "performed" if ledger.answer_review_calls else "not_requested",
            "analysis_delivery": "delivered", "answer_evidence": "verified" if verified else "unverified",
            "valid_read_sources": len(valid), "diagnostic": diagnostic,
            "warning": ("部分辅助读取或命令未完成，按部分完成交付；具体错误保留在执行记录中。" if plan_gaps or command_gaps else "")
                + ("" if verified else "分析已交付，未经独立事实复核；请结合依据与限制判断。")})


def advisory_ids(completion):
    """导出/视图复用宿主分层回执，旧结果与严格任务不受影响。"""
    layers = completion.get("layers", {})
    if not isinstance(layers, Mapping) or layers.get("policy") not in {"ordinary_readonly_analysis", "ordinary_analysis"}:
        return set()
    return {r["id"] for r in completion.get("advisory", []) if isinstance(r, Mapping) and isinstance(r.get("id"), str)}
