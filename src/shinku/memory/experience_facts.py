"""Small historical task cards; neither raw work context nor current facts."""
from __future__ import annotations

import re
from dataclasses import dataclass

from shinku.agent_runtime.contracts import canonical_hash
from shinku.result_expression.facts import presentation_data


@dataclass(frozen=True)
class DeploymentEvidence:
    """Optional explicit host receipt. Absence never means deployed."""
    status: str
    reference: str
    version: int = 1

    def __post_init__(self):
        if (self.status not in {"not_deployed", "deployed", "failed", "unknown"}
                or not re.fullmatch(r"[a-zA-Z0-9_:.-]{1,128}", self.reference)
                or type(self.version) is not int or self.version < 1):
            raise ValueError("experience_deployment_evidence_invalid")


def excerpt(value, limit=300):
    from .service import MemoryService
    text = " ".join(str(presentation_data(str(value or ""), full=True)).split())
    text = MemoryService._TASK_SECRET_ASSIGNMENT_RE.sub(r"\1\2[已省略]", text)
    text = MemoryService._TASK_BEARER_RE.sub("Bearer [已省略]", text)
    return text[:limit], max(0, len(text) - limit)


def make_projection(source, event, delivery, source_state, deployment=None):
    """Input must already have been loaded/validated by the trusted projector."""
    state = source["execution"]["status"]
    from shinku.agent_runtime.reply_delivery import is_reply_delivery
    reply_only = is_reply_delivery(source["verification"])
    # 历史记忆只投影宿主事实；答复交付不能在写入记忆时升级为目标验收成功。
    verified = source["verification"].get("accepted") is True and state == "completed" and not reply_only
    goals = []
    omitted = 0
    requests = source.get("original_requests", [])
    for request in requests[:4]:
        text, lost = excerpt(request.get("text", ""))
        if text:
            goals.append(text)
        omitted += lost
    reason, lost = excerpt(source["execution"].get("reason", ""), 160)
    omitted += lost
    information = source.get("information", {})
    def count(value):
        return len(value) if isinstance(value, (list, dict)) else 0
    info = {"status": information.get("status", "unknown"),
        "required_status": information.get("required_status", "unknown"),
        "upstream": information.get("upstream_completeness", "unknown"),
        "projection_gaps": information.get("projection", {}).get("gap_count", 0),
        "export_gaps": count(information.get("export_gaps")),
        "input_gaps": count(information.get("gaps")),
        "snapshot_gaps": information.get("snapshot", {}).get("gap_count", 0),
        "source_status": source_state["status"], "source_reasons": source_state["reasons"]}
    artifacts = [{"reference": canonical_hash(row), "verification": row.get("verification", "unconfirmed")}
                 for row in source.get("artifacts", [])[:8] if isinstance(row, dict)]
    deploy = ({"status": deployment.status, "reference": deployment.reference, "version": deployment.version}
              if deployment is not None else {"status": "unknown", "reference": "", "version": 0})
    snapshot = {"schema_version": 1, "task_id": source["task_id"], "scope": source["scope"],
        "attempt": source["attempt"], "version": source["version"], "result_hash": source["result_hash"],
        "event_id": event["event_id"], "as_of": source.get("updated", 0),
        "goals": goals, "goal_excerpt_omitted_chars": omitted,
        "omitted_request_count": max(0, len(requests) - 4), "execution_status": state,
        "verified": verified, "reason": reason,
        "satisfied_count": count(source["verification"].get("satisfied")),
        "remaining_count": count(source.get("remaining")),
        "uncertainty_count": count(source.get("uncertainties")),
        "artifact_count": count(source.get("artifacts")), "artifact_references": artifacts,
        "receipt_count": count(source.get("evidence")),
        "delivery": delivery, "deployment": deploy, "information": info,
        "material_binding": source_state["binding_hash"],
        "historical_snapshot": True, "is_current_project_fact": False,
        "raw_detail_unavailable": True}
    if reply_only:
        snapshot["delivery_policy"] = source["verification"]["layers"]["policy"]
    label = {"completed": "已完成，目标验收通过" if verified else "完成声明未验收",
        "failed": "失败", "partial": "部分完成，整项尚未验收",
        "blocked": "受阻，整项尚未完成", "canceled": "已取消；不证明副作用已撤销",
        "unverified": "未确认，不能宣称成功"}[state]
    if reply_only and state == "completed":
        label = "答复已交付；整项目标未独立验收"
    sent_label = {"sent": "已确认送达", "partial": "只确认部分送达",
        "failed": "发送失败", "uncertain": "送达未知", "dry_run": "仅测试，未实际发送",
        "suppressed": "发送被抑制", "not_sent": "尚未发送"}[delivery["status"]]
    lines = ["Agent任务经历（规范结果与宿主回执的历史快照；不证明当前项目状态）：",
        "目标摘录：" + (" / ".join(goals) or "未保留目标摘要"), "执行：" + label,
        f"验收：{'通过' if verified else '未通过或未确认'}；满足项 {snapshot['satisfied_count']}；未完成项 {snapshot['remaining_count']}",
        "交付：" + sent_label,
        "上线：" + {"deployed": "有独立部署回执", "not_deployed": "宿主确认未上线",
                   "failed": "部署失败", "unknown": "未验收或未知"}[deploy["status"]],
        f"资料完整性：{info['status']}；必要资料：{info['required_status']}；来源当前可用性：{info['source_status']}",
        f"限制：不确定项 {snapshot['uncertainty_count']}；来源缺口 {len(source_state['reasons'])}；裁剪缺口 {info['projection_gaps'] + info['export_gaps'] + info['snapshot_gaps']}",
        "本卡仅保留必要事实和安全引用，未复制原始代码、日志、资料或模型推理；完整条件须核对规范结果。"]
    if reason:
        lines.append("停止原因摘录：" + reason)
    if omitted or snapshot["omitted_request_count"]:
        lines.append("目标/原因只保留摘录，不能当作完整资料。")
    if source_state["reasons"]:
        lines.append("来源限制：" + "、".join(source_state["reasons"]))
    text = "\n".join(lines)
    snapshot["projection_hash"] = canonical_hash(snapshot)
    return snapshot, text
