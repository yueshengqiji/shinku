"""Completion verifier facade; contract checks, effect receipts and target parsing are separate responsibilities: verification_effects."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from .execution_sessions import execution_event
from .loop import AgentState
from .request_intent import assertion_text, execution_request_text
from .verification_rules import (
    _DELETE_CLAIM_PATTERNS,
    _DELETE_KEYWORDS,
    _PROCESS_STOP_CLAIM_PATTERNS,
    _PROCESS_STOP_KEYWORDS,
    _REMINDER_CLAIM_PATTERNS,
    _REMINDER_KEYWORDS,
    _SEND_CLAIM_PATTERNS,
    _SEND_KEYWORDS,
    _WRITE_CLAIM_PATTERNS,
    _WRITE_KEYWORDS,
)


@dataclass(frozen=True)
class VerificationEffectsDependencies:
    """Bound callbacks and live getters for this responsibility."""
    background_tasks: Callable[..., Any]
    claims_success: Callable[..., Any]
    extract_targets: Callable[..., Any]
    extract_user_messages: Callable[..., Any]
    missing_effect_reason: Callable[..., Any]
    normalize_target_values: Callable[..., Any]
    receipt: Callable[..., Any]
    successful_calls: Callable[..., Any]
    targets_match: Callable[..., Any]
    user_requested_action: Callable[..., Any]
    verify_requirement: Callable[..., Any]
    allowed_tool_names: Callable[..., Any]
    enabled: Callable[..., Any]
    restrict_tool_names: Callable[..., Any]
    tool_capabilities: Callable[..., Any]
    result_handlers: Callable[..., Any] | None = None


def successful_calls(
    state: AgentState,
    *,
    deps: VerificationEffectsDependencies,
) -> list[Mapping[str, Any]]:
    return [call for call in state.tool_calls if isinstance(call, Mapping)
            and str(call.get("status") or "").lower() == "ok"
            and (not deps.restrict_tool_names() or call.get("name") in deps.allowed_tool_names())]


def background_tasks(
    state: AgentState,
    *,
    deps: VerificationEffectsDependencies,
) -> dict[str, Mapping[str, Any]]:
    tasks: dict[str, Mapping[str, Any]] = {}
    for item in state.transcript:
        if item.get("role") != "tool":
            continue
        if deps.restrict_tool_names() and item.get("name") not in deps.allowed_tool_names():
            continue
        receipt = deps.receipt(item)
        task = receipt.get("background_task")
        session = receipt.get("execution_session", {})
        handler = deps.result_handlers().get(item.get("name")) if deps.result_handlers else None
        if (isinstance(task, Mapping) and isinstance(session, Mapping)
                and task.get("task_id") == session.get("session_id")
                and execution_event(handler, receipt, state.context) is not None):
            # Command attempts can be corrected. They are separate from durable
            # business jobs; original conditions still need actual evidence.
            continue
        if isinstance(task, Mapping) and task.get("task_id"):
            tasks[str(task["task_id"])] = task
    return tasks


def check_effects(
    state: AgentState,
    final_text: str,
    *,
    partial: bool=False,
    deps: VerificationEffectsDependencies,
) -> bool | str:
    if not deps.enabled():
        return True

    text = str(final_text or "").strip()
    if not text:
        return "汇报内容不能为空"

    # Failure disposition is resolved by the facade. Wording must never skip
    # the receipt checks for a structured completed result.
    text = assertion_text(text)
    background_tasks = deps.background_tasks(state)
    if any(not task.get("completed") for task in background_tasks.values()):
        delivery_text = re.sub(
            r"(?:已(?:经)?(?:创建|启动|提交|排队)(?:了)?(?:后台)?(?:下载)?任务|"
            r"(?:后台|下载)?任务(?:已(?:经)?)?(?:创建|启动|提交|排队))[^，。；;\n]*", "", text,
        )
        delivery_claim = any(
            (p.search(delivery_text) for p in (*_WRITE_CLAIM_PATTERNS, *_SEND_CLAIM_PATTERNS, *_REMINDER_CLAIM_PATTERNS, *_PROCESS_STOP_CLAIM_PATTERNS)),
        )
        nonnegated_delivery = re.sub(r"(?:尚未|还未|未能|未|还没|没有|没).{0,8}(?:完成|完了|成功|搞定)", "", delivery_text)
        async_success = re.search(r"(?:下载|文件|任务).{0,16}(?:完成|完了|成功|搞定)|(?:下载好了|下好了)", nonnegated_delivery)
        if delivery_claim or async_success:
            return "后台任务尚未完成或核验通过；只能报告任务号、进度或障碍，不能声称文件已经交付。"
        # Starting a durable job is an honest intermediate handoff, not a
        # missing write. The worker continues without another model call.
        return True

    # 汇总本次规划开始前的所有用户消息。恢复授权时，首个工具调用之后的
    # “好/同意”等用户消息不是新的任务目标，不能覆盖原始请求。
    user_messages = deps.extract_user_messages(state)
    successful_calls = deps.successful_calls(state)
    # 讨论宾语中的路径不要求被写入；原契约目标和成功声明仍走独立回执核验。
    requested_targets = deps.extract_targets("\n".join(execution_request_text(m) for m in user_messages))
    def matches_effect(effect: str, claimed: bool=False) -> bool:
        # 信息请求不要求实际修改，但成稿主动声称的具体目标仍须匹配回执；
        # 不能用写入另一文件的成功结果，替讨论对象伪造“已修改”。
        targets = list(dict.fromkeys([*requested_targets, *(deps.extract_targets(final_text) if claimed else [])]))
        if partial and targets:
            return any(
                (deps.verify_requirement(state, {'effect': effect, 'targets': [target]}, successful_calls) for target in targets),
            )
        return deps.verify_requirement(
            state,
            {'effect': effect, 'targets': targets},
            successful_calls,
        )

    # 1. 检查写文件/改配置/保存类操作
    user_wants_write = any(deps.user_requested_action(message, _WRITE_KEYWORDS) for message in user_messages)
    model_claims_write = any(p.search(text) for p in _WRITE_CLAIM_PATTERNS)
    if user_wants_write or model_claims_write:
        if not matches_effect("write", model_claims_write):
            # 用户有写入意图或模型声称写入，但未执行任何写入类工具
            return deps.missing_effect_reason("write", requested_targets)

    model_claims_delete = any(p.search(text) for p in _DELETE_CLAIM_PATTERNS)
    if any(deps.user_requested_action(message, _DELETE_KEYWORDS) for message in user_messages) or model_claims_delete:
        if not matches_effect("delete", model_claims_delete):
            return deps.missing_effect_reason("delete", requested_targets)

    # 2. 检查点播/发送音乐或文件类操作
    user_wants_send = any(deps.user_requested_action(message, _SEND_KEYWORDS) for message in user_messages)
    model_claims_send = any(p.search(text) for p in _SEND_CLAIM_PATTERNS)
    if user_wants_send or model_claims_send:
        if not matches_effect("send", model_claims_send):
            return deps.missing_effect_reason("send", requested_targets)

    # 3. 检查提醒设置类操作
    user_wants_reminder = any(
        (deps.user_requested_action(message, _REMINDER_KEYWORDS) for message in user_messages),
    )
    model_claims_reminder = any(p.search(text) for p in _REMINDER_CLAIM_PATTERNS)
    if user_wants_reminder or model_claims_reminder:
        if not matches_effect("reminder", model_claims_reminder):
            return deps.missing_effect_reason("reminder", requested_targets)

    # 4. 检查进程停止类操作
    user_wants_stop_proc = any(
        (deps.user_requested_action(message, _PROCESS_STOP_KEYWORDS) for message in user_messages),
    )
    model_claims_stop_proc = any(p.search(text) for p in _PROCESS_STOP_CLAIM_PATTERNS)
    if user_wants_stop_proc or model_claims_stop_proc:
        if not matches_effect("process_stop", model_claims_stop_proc):
            return deps.missing_effect_reason("process_stop", requested_targets)

    return True


def has_effect(
    calls: Sequence[Mapping[str, Any]],
    effect: str,
    requested_targets: Sequence[str],
    *,
    require_all: bool=True,
    deps: VerificationEffectsDependencies,
) -> bool:
    observed_targets: list[str] = []
    has_evidence = False
    for call in calls:
        name = str(call.get("name") or "").strip()
        capability = deps.tool_capabilities().get(name, {})
        if effect not in capability.get("effects", ()):
            continue
        data = call.get("data", {})
        updates = data.get("state_updates", {}) if isinstance(data, Mapping) else {}
        updates = updates if isinstance(updates, Mapping) else {}
        resource_read = effect == "read" or (effect.startswith("tool:") and "read" in capability.get("effects", ()))
        if resource_read and (call.get("status") != "ok" or call.get("executed") is False
                or effect not in updates.get("verified_effects", ())
                or not isinstance(updates.get("completion_targets"), (list, tuple))):
            continue
        if "verified_effects" in updates and effect not in updates["verified_effects"]:
            continue
        # Browser availability always requires an executor-issued receipt.
        if effect == "browser_access" and effect not in updates.get("verified_effects", ()):
            continue
        has_evidence = True
        target_arguments = capability.get("target_arguments", ())
        if not requested_targets:
            return True
        arguments = call.get("arguments")
        if "completion_targets" in updates:
            # Actual executor-issued targets take precedence over the model's
            # proposed arguments (a handler may resolve/redirect the target).
            raw_targets = updates["completion_targets"]
            if not isinstance(raw_targets, (list, tuple)):
                continue
            actual_targets = [normalized for value in raw_targets for normalized in deps.normalize_target_values(value)]
        elif isinstance(arguments, Mapping):
            actual_targets = [normalized for key in target_arguments for normalized in deps.normalize_target_values(arguments.get(key))]
        else:
            continue
        observed_targets.extend(actual_targets)
    matches = [any(deps.targets_match(expected, actual) for actual in observed_targets) for expected in requested_targets]
    return has_evidence and (all(matches) if require_all else any(matches))
