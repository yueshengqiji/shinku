"""One classification for synchronous and background failures, no raw prose."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureReport:
    status: str
    category: str
    phase: str
    code: str
    text: str


def task_preparation_failure(code: str, *, recalled_candidates: int = 0) -> FailureReport:
    if code == "task_handoff_requires_new_or_continue":
        return FailureReport("blocked", "invalid_handoff_route", "routing", code,
            "这项工具调用没有进入有效的任务流程，我没有执行操作；请重新发起，让我按可追踪任务处理。")
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
