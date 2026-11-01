"""One classification for synchronous and background failures, no raw prose."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class FailureReport:
    status: str
    category: str
    phase: str
    code: str
    text: str


def task_preparation_failure(code: str, *, recalled_candidates: int = 0) -> FailureReport:
    routing_errors = {
        "task_handoff_requires_new_or_continue",
        "task_handoff_changed_after_selection",
        "task_handoff_continue_requires_task_id",
        "task_handoff_host_id_missing",
        "task_handoff_resume_requires_task_id",
        "task_handoff_changed_selected_tools",
        "invalid_task_link",
    }
    if code in routing_errors:
        return FailureReport("blocked", "invalid_handoff_route", "routing", code,
            "工具调用的任务信息未通过内部校验，我还没有执行操作；这次请求尚未完成。")
    conflicts = {"task_already_active", "task_not_found_in_requester_session", "task_scope_changed",
                 "contract_requirement_removed_or_changed", "contract_deliverable_removed"}
    if code in conflicts:
        return FailureReport("blocked", "task_conflict", "task_preparation", code,
            "这轮任务关联与已有任务不一致，我没有执行新操作，也没有改掉原来的条件；这次还没完成。")
    if recalled_candidates:
        text = "我读到了一些相关记忆，但后续历史查询在内部准备阶段失败了，还没核对到你问的内容；不是需要你重新授权。"
    else:
        text = "这轮任务在内部准备阶段失败了，还没有执行新操作；不是你的权限不足，我也不能确认已经完成。"
    return FailureReport("failed", "internal_error", "task_preparation", code, text)


def execution_failure(code: str, *, tool_count: int = 0) -> FailureReport:
    stage = "execution" if tool_count else "planning"
    progress = "已有执行记录保留，但后续步骤没有完成" if tool_count else "还没有执行操作"
    return FailureReport("failed", "internal_error", stage, code,
        "任务发生内部错误，" + progress + "；我不能确认任务完成。")


def completion_failure(completion: Mapping | None, contract: Mapping | None = None) -> FailureReport:
    """Explain trusted receipt gaps without echoing raw errors, paths or secrets."""
    completion = completion if isinstance(completion, Mapping) else {}
    contract = contract if isinstance(contract, Mapping) else {}
    satisfied = completion.get("satisfied", ())
    missing = completion.get("missing", ())
    satisfied = satisfied if isinstance(satisfied, (list, tuple)) else ()
    missing = missing if isinstance(missing, (list, tuple)) else ()
    labels = {"browser_access": "页面访问与读取", "read": "文件读取", "write": "文件写入",
              "delete": "文件删除", "send": "发送回执", "research_answer": "资料与回答核验",
              "local_answer": "本地资料与回答核验", "history_answer": "历史记录与回答核验",
              "document_translation": "文档翻译与产物核验", "reminder": "提醒设置",
              "process_stop": "进程停止"}
    requirements = contract.get("requirements", ())
    requirements = requirements if isinstance(requirements, (list, tuple)) else ()
    kinds = list(dict.fromkeys(labels.get(str(row.get("effect", "")), "其他任务要求")
        for row in requirements if isinstance(row, Mapping) and row.get("id") in missing))
    if missing:
        details = f"（{'、'.join(kinds[:4])}）" if kinds else ""
        text = (f"完成核验连续未通过，已停止重复尝试。已核验 {len(satisfied)} 项要求，"
                f"仍有 {len(missing)} 项缺少实际结果证据{details}。")
    else:
        text = "完成核验连续未通过，已有结果不足以确认任务完成，已停止重复尝试。"
    return FailureReport("failed", "completion_unverified", "verification", "completion_no_progress", text)
