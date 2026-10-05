"""Agent 交付核验门禁 (Action Completion Verification Gate)。

用于防范大模型在多步执行任务中的“假性完成与行动脱节”幻觉：
即用户要求了修改、写入、发送或设置操作，模型读取信息后倾向于直接在文本中
声称“好的，已经帮你做好了”，但实际上根本没有调用对应的操作类工具。
"""

from __future__ import annotations

import ntpath
import re
from dataclasses import replace
from collections.abc import Mapping, Sequence
from typing import Any

from .loop import AgentState
from .completion import CompletionVerdict

__all__ = ["SmartCompletionVerifier", "CompletionVerdict"]

# 操作意图关键词匹配
_WRITE_KEYWORDS = (
    "修改", "改写", "改成", "改为", "写入", "写进", "保存", "创建", "新建", "覆盖", "更新", "保存到",
    "编辑", "追加", "替换", "换成", "设为", "设成",
)
_DELETE_KEYWORDS = ("删除", "移除", "清理", "清空")
_DELETE_CLAIM_PATTERNS = [re.compile(r"已(?:经|成功)?.*?(?:删除|删掉|删了|移除|清理|清空)")]

_WRITE_CLAIM_PATTERNS = [
    re.compile(r"已(经)?.*?(保存|修改|更新|写入|创建|替换|覆盖)"),
    re.compile(r"已(经)?.*?(将|把).*?(保存|写入|修改|更新|替换|覆盖|改成|改为)"),
    re.compile(r"已(成功)?保存(到|至)"),
    re.compile(r"已(成功)?(新建|创建)了?"),
    re.compile(r"(参数|配置|文件|内容).*?(已|已经|成功)(修改|变更为|更改为|更新为|设置为)"),
]

# Compatibility for legacy text-only final decisions. This does not select tools.
_ACTION_SUCCESS = re.compile(r"(?:^\s*(?:好了|完成了|搞定了)|(?:已(?:经)?|成功).{0,24}(?:完成|修好|打开|运行)|正常(?:打开|运行|访问)|修好了)")

_SEND_KEYWORDS = (
    "发送", "发到群", "发给我", "点播", "推送到", "发这首", "发这篇", "发文件", "发图", "发截图", "分享这首",
)

_SEND_CLAIM_PATTERNS = [
    re.compile(r"已(经)?.*?(发送|点播|推送|分享)"),
    re.compile(r"卡片已(发送|送达)"),
]

_REMINDER_KEYWORDS = ("提醒我", "定个闹钟", "设置提醒", "设个提醒", "添加提醒")
_REMINDER_CLAIM_PATTERNS = [
    re.compile(r"已(经)?.*?(设置|定好|添加)(了)?提醒"),
    re.compile(r"已(成功)?创建提醒"),
]

_PROCESS_STOP_KEYWORDS = ("停止进程", "杀掉进程", "关闭进程", "结束进程", "kill进程")
_PROCESS_STOP_CLAIM_PATTERNS = [
    re.compile(r"已(经)?.*?(停止|杀掉|关闭|结束)(了)?进程"),
]

# 如实汇报障碍或拒绝执行的声明，应直接放行，避免陷入死循环
_FAILURE_OR_REFUSAL_PATTERNS = [
    re.compile(r"(未能|无法|不能|没有办法|权限不足|找不到|不存在|失败|抱歉|对不起|没有找到)"),
    re.compile(r"(未执行|并没有|没有修改|未保存|无法完成|无法替你)"),
    re.compile(r"(不能替[你您]执行|这类工具操作只听主人的)"),
]

_INFORMATION_REQUEST = re.compile(
    r"^\s*(?:(?:请)?(?:告诉我|解释|说明|介绍|分析|评估|讲讲)|我(?:想知道|想了解|想问|想请教|想确认)|"
    r"为什么|怎么(?:做|才能|去)?|如何|能不能|是否|是不是|什么是|请问|你觉得|"
    r"给我(?:一个)?(?:方案|建议)|列出|有哪些)"
)
_INFORMATIONAL_SHAPE = re.compile(
    r"(?:工作原理|怎么工作|如何工作|是什么|什么意思|有什么作用|有什么区别|有哪些风险|"
    r"怎么实现|如何实现|原理|教程|操作步骤|为什么|是否|能否|可不可以|行不行)"
)
_EXPLICIT_ACTION_REQUEST = re.compile(
    r"(?:帮我|替我|帮忙).{0,40}(?:修改|改写|改成|改为|写入|写进|保存|创建|新建|覆盖|更新|编辑|追加|替换|删除|移除|清理|清空|设为|设成|发送|推送|设置|提醒|停止|杀掉|关闭|结束)"
    r"|(?:我想|我要|需要你).{0,40}(?:修改|改写|改成|改为|写入|写进|保存|创建|新建|覆盖|更新|编辑|追加|替换|删除|移除|清理|清空|设为|设成|发送|推送|设置|提醒|停止|杀掉|关闭|结束)"
    r"|^\s*(?:请)?\s*(?:直接)?\s*(?:修改|改写|改成|改为|写入|写进|保存|创建|新建|覆盖|更新|编辑|追加|替换|删除|移除|清理|清空|设为|设成|发送|推送|设置|提醒|停止|杀掉|关闭|结束)"
)
_NEGATED_ACTION_REQUEST = re.compile(
    r"(?:不要|别(?!人)|不必|无需|不需要|不用|禁止|不希望|不想|不让你).{0,20}"
    r"(?:修改|改写|改成|改为|写入|写进|保存|创建|新建|覆盖|更新|编辑|追加|替换|删除|移除|清理|清空|设为|设成|发送|推送|设置|提醒|停止|杀掉|关闭|结束)"
)

# Deliberately narrow compatibility guard: informational translation questions
# and plain in-chat translations remain unchanged. This NEVER grants a tool.
_PDF_TRANSLATION = re.compile(r"(?:翻译|汉化|译成|英译中).{0,200}(?:PDF|pdf)|(?:PDF|pdf).{0,200}(?:翻译|汉化|译成|英译中)", re.S)
_FULL_DOCUMENT = re.compile(r"全文|完整|全部|所有|整篇|全篇|(?:第\s*)?1\s*[-—到至]\s*\d+\s*页|(?<![第\d])\d+\s*页")

# 明确的工具能力登记：核验不再通过工具名中是否“包含 write/send”等片段猜测。
# 自定义 handler 可声明相同属性；未登记的工具不会被当成副作用证据。
_DEFAULT_TOOL_CAPABILITIES: dict[str, dict[str, tuple[str, ...]]] = {
    "private_file_read": {"effects": ("read", "tool:private_file_read"), "target_arguments": ("path",)},
    "file_write": {"effects": ("write",), "target_arguments": ("path",)},
    "write_file": {"effects": ("write",), "target_arguments": ("path",)},
    "private_file_write": {"effects": ("write",), "target_arguments": ("path",)},
    "file_save": {"effects": ("write",), "target_arguments": ("path",)},
    "private_file_delete": {"effects": ("delete",), "target_arguments": ("path",)},
    "file_delete": {"effects": ("delete",), "target_arguments": ("path",)},
    "private_browser_open": {"effects": ("browser_access",), "target_arguments": ("url",)},
    "private_qq_send_file": {"effects": ("send",), "target_arguments": ("path",)},
    "private_qq_send_netease_music": {"effects": ("send",), "target_arguments": ()},
    "private_reminder_set": {"effects": ("reminder",), "target_arguments": ("message",)},
    "private_process_stop": {"effects": ("process_stop",), "target_arguments": ("name", "pid")},
}

_FILE_TARGET = re.compile(
    r"(?<![\w])(?:[A-Za-z]:[\\/])?(?:[^\s\"'<>|，。；：！？、]+[\\/])*"
    r"[^\s\"'<>|，。；：！？、\\/]+\.[A-Za-z0-9]{1,10}(?![\w])"
)
_QUOTED_TARGET = re.compile(r"[\"“「『《]([^\"”」』》]{2,4096})[\"”」』》]")
_UNCHANGED_INPUTS = re.compile(
    r"(?:原文件|源文件|输入文件|原文).{0,12}(?:保持不变|不得修改|不能修改|不要修改|不修改)"
    r"|(?:不要|禁止|不得).{0,12}(?:修改|覆盖|更新).{0,6}(?:原文件|源文件|输入文件|原文)"
)


class SmartCompletionVerifier:
    """Evidence-based gate with a legacy text compatibility layer.

    Trusted hosts may supply ``agent_completion_contract.requirements`` in the
    execution context. Natural-language compatibility is not a semantic verifier.
    ``evaluate`` distinguishes an honest terminal report from goal completion.
    """

    supports_plain_text_completion_verification = True

    def __init__(
        self,
        *,
        allowed_tool_names: Sequence[str] | None = None,
        tool_capabilities: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
        enabled: bool = True,
        require_execution: bool = False,
        result_handlers: Mapping[str, Any] | None = None,
    ) -> None:
        self.allowed_tool_names = frozenset(str(n).strip() for n in (allowed_tool_names or ()) if str(n).strip())
        self.restrict_tool_names = allowed_tool_names is not None
        capabilities = {name: dict(value) for name, value in _DEFAULT_TOOL_CAPABILITIES.items()}
        if isinstance(tool_capabilities, Mapping):
            for raw_name, raw_value in tool_capabilities.items():
                name = str(raw_name or "").strip()
                if not name or not isinstance(raw_value, Mapping):
                    continue
                effects = tuple(str(item).strip().casefold() for item in raw_value.get("effects", ()) if str(item).strip())
                target_arguments = tuple(
                    str(item).strip() for item in raw_value.get("target_arguments", ()) if str(item).strip()
                )
                capabilities[name] = {"effects": effects, "target_arguments": target_arguments}
        self.tool_capabilities = capabilities
        self.enabled = bool(enabled)
        self.require_execution = bool(require_execution)
        self.result_handlers = dict(result_handlers or {})

    def __call__(self, state: AgentState, final_text: str) -> bool | str:
        """Keep bool/string callers compatible; AgentLoop uses the typed evaluator."""
        verdict = self.evaluate(state, final_text)
        return True if verdict.accepted else verdict.reason

    def evaluate(self, state: AgentState, final_text: str, *, outcome: str = "") -> CompletionVerdict:
        verdict = self._evaluate(state, final_text, outcome=outcome)
        evidence = []
        for call in state.tool_calls[-128:]:
            receipt = self._receipt(call)
            evidence.append({"tool": call.get("name"), "tool_execution": call.get("status") == "ok",
                             "effects": list(receipt.get("verified_effects", ())),
                             "targets": list(receipt.get("completion_targets", ())),
                             "external_service": receipt.get("site_access_ok"),
                             "http_status": receipt.get("http_status"),
                             "delivery_receipt": receipt.get("delivery_receipt", "")})
        evidence.append({"business_goal": verdict.status == "completed" and verdict.accepted,
                         "satisfied": list(verdict.satisfied), "missing": list(verdict.missing)})
        return replace(verdict, evidence=tuple(evidence))

    def _evaluate(self, state: AgentState, final_text: str, *, outcome: str = "") -> CompletionVerdict:
        from .request_scope import RequestScope

        context = state.context if isinstance(state.context, Mapping) else {}
        if "agent_request_scope" in context and not isinstance(context["agent_request_scope"], RequestScope):
            return CompletionVerdict(False, reason="宿主请求范围无效，不能确认完成")
        text = str(final_text or "").strip()
        if outcome and outcome not in {"completed", "partial", "failed", "blocked", "waiting_background"}:
            return CompletionVerdict(False, reason="任务结果状态无效")
        if not self.enabled:
            return CompletionVerdict(True, status=outcome or "completed")
        if not text:
            return CompletionVerdict(False, reason="汇报内容不能为空")
        if state.research.enabled:
            research_verdict = state.research.evaluate_answer(text, outcome)
            if not research_verdict.accepted:
                return research_verdict
        if state.local_answer.enabled:
            state.local_answer.refresh(self.result_handlers, state.context)
            local_verdict = state.local_answer.evaluate_answer(text, outcome)
            if not local_verdict.accepted:
                return local_verdict
        if state.history_answer.enabled:
            history_verdict = state.history_answer.evaluate_answer(text, outcome)
            if not history_verdict.accepted:
                return history_verdict
        calls = self._successful_calls(state)
        context = state.context if isinstance(state.context, Mapping) else {}
        contract = context.get("agent_completion_contract")
        if contract is not None and not isinstance(contract, Mapping):
            return CompletionVerdict(False, reason="验收契约格式无效")
        requirements = contract.get("requirements", ()) if isinstance(contract, Mapping) else ()
        satisfied: list[str] = []
        missing: list[str] = []
        if not isinstance(requirements, (list, tuple)):
            return CompletionVerdict(False, reason="验收契约 requirements 格式无效")
        for index, requirement in enumerate(requirements):
            if not isinstance(requirement, Mapping) or not str(requirement.get("effect") or "").strip():
                return CompletionVerdict(False, reason="验收契约条件格式无效")
            key = str(requirement.get("id") or f"condition_{index + 1}")
            targets = requirement.get("targets", ())
            if not isinstance(targets, (list, tuple)) or any(not isinstance(value, str) or not value.strip() for value in targets):
                return CompletionVerdict(False, reason=f"验收契约 {key} 的目标格式无效")
            effect = str(requirement["effect"]).strip()
            matched = self._verify_requirement(state, requirement, calls)
            (satisfied if matched else missing).append(key)

        user_messages = self._extract_user_messages(state)
        document_request = "\n".join(user_messages)
        if requirements and _UNCHANGED_INPUTS.search(document_request):
            # Newly discovered dependencies are inputs too. Re-reading after
            # modifying one must not reset the original observation baseline.
            roles = self._explicit_target_roles(user_messages)
            outputs = [target for row in requirements if row["effect"] in {"write", "document_translation"}
                       for target in row.get("targets", ())]
            read_targets = {target for call in calls
                if "read" in self.tool_capabilities.get(call.get("name"), {}).get("effects", ())
                for value in self._receipt(call).get("completion_targets", ())
                for target in self._normalize_target_values(value)}
            for target in sorted(read_targets):
                if "read" not in roles.get(target, ()) and any(
                        self._targets_match(target, n) for value in outputs for n in self._normalize_target_values(value)):
                    continue
                if not self._verify_read_unchanged(state, target):
                    missing.append(f"input_unchanged:{target}")
        document_inquiry = bool(user_messages) and all(
            (_INFORMATION_REQUEST.search(message) or _INFORMATIONAL_SHAPE.search(message)
             or re.search(r"怎么(?:做|翻译|汉化)|如何(?:翻译|汉化)", message))
            and not _EXPLICIT_ACTION_REQUEST.search(message) for message in user_messages)
        if (not document_inquiry and _PDF_TRANSLATION.search(document_request)
                and _FULL_DOCUMENT.search(document_request)
                and not re.search(r"(?:不要|别|无需|不用).{0,20}(?:翻译|汉化|英译中)", document_request)):
            document_requirement = {"id": "full_pdf_translation", "effect": "document_translation", "targets": []}
            requested_pages = re.search(r"(?:1\s*[-—到至]\s*)?(\d+)\s*页", document_request)
            if requested_pages and 1 <= int(requested_pages.group(1)) <= 100:
                document_requirement["checks"] = {"source_pages": int(requested_pages.group(1))}
            # Do not let a model-authored write-only contract erase full coverage.
            if not self._verify_requirement(state, document_requirement, calls):
                missing.append("full_pdf_translation")
        if requirements:
            # Structured model metadata cannot silently drop explicit file targets
            # in the user's request or the declared deliverables.
            verified_inputs = []
            # Explicitly labelled source inputs may be covered by a refreshed
            # translation receipt. They do NOT become write/send targets.
            if (not re.search(r"(?:修改|覆盖|替换|删除|更新)原文", document_request)
                    and re.search(r"原文|源文件|输入文件", document_request)):
                for call in calls:
                    if self._verify_requirement(state, {"effect": "document_translation", "targets": []}, [call]):
                        handler = self.result_handlers.get(call.get("name"))
                        try:
                            fresh = handler.agent_verify_result(receipt=self._receipt(call), context=state.context)
                        except Exception:
                            continue
                        source_path = fresh.get("source_path") if isinstance(fresh, Mapping) else None
                        if fresh.get("result_valid") is True and isinstance(source_path, str) and source_path:
                            verified_inputs.append(source_path)
            missing.extend(f"uncovered_target:{gap['target']}" for gap in self.target_coverage_gaps(
                contract, user_messages, additional_inputs=verified_inputs))
        inquiry = bool(user_messages) and all(
            (_INFORMATION_REQUEST.search(message) or _INFORMATIONAL_SHAPE.search(message))
            and not _EXPLICIT_ACTION_REQUEST.search(message)
            for message in user_messages
        )
        success_claim = self._claims_success(text)
        honest_failure = (not inquiry and outcome not in {"partial", "waiting_background"}
                          and bool(any(pattern.search(text) for pattern in _FAILURE_OR_REFUSAL_PATTERNS)) and not success_claim)
        background = self._background_tasks(state)
        pending_background = any(not task.get("completed") and task.get("status") in {"queued", "pending", "running", "retrying"} for task in background.values())

        # Honest failure must be deliverable, but must not become goal success.
        if honest_failure or outcome in {"failed", "blocked"}:
            if success_claim and (missing or not calls):
                return CompletionVerdict(False, reason="失败汇报同时宣称了未经核验的成功结果", missing=tuple(missing))
            return CompletionVerdict(
                True, status=(outcome if outcome in {"failed", "blocked"} else "failed"),
                reason="reported_failure", satisfied=tuple(satisfied), missing=tuple(missing),
            )

        if background and not pending_background and any(not task.get("completed") for task in background.values()) and outcome != "partial":
            return CompletionVerdict(False, reason="后台任务已停止或缺少运行状态，不能报告仍在运行或已经完成", missing=tuple(missing))

        # Trusted structured conditions supersede the coarse text-only target
        # extraction (different operations may intentionally target different files).
        legacy_check = (
            True if requirements and not pending_background
            else self._check_effects(state, text, partial=outcome == "partial")
        )
        if isinstance(legacy_check, str):
            return CompletionVerdict(False, reason=legacy_check, satisfied=tuple(satisfied), missing=tuple(missing))
        if outcome == "partial":
            research_partial = (state.research.enabled
                and any(str(requirement.get("effect", "")) == "research_answer" for requirement in requirements)
                and state.research.satisfies({"min_sources": 1}))
            history_partial = (state.history_answer.enabled and requirements
                and all(str(row.get("effect", "")) == "history_answer" for row in requirements)
                and state.history_answer.answer_check.get("valid") is True
                and state.history_answer.answer_check.get("disposition") in {"not_found", "ambiguous"})
            if not calls or (requirements and not satisfied and not research_partial and not history_partial):
                return CompletionVerdict(False, reason="部分完成也需要实际成功的工具证据", missing=tuple(missing))
            return CompletionVerdict(True, status="partial", reason="partial_completion", satisfied=tuple(satisfied), missing=tuple(missing))
        if pending_background:
            return CompletionVerdict(True, status="waiting_background", reason="background_pending", satisfied=tuple(satisfied), missing=tuple(missing))
        if outcome == "waiting_background":
            return CompletionVerdict(False, reason="没有仍在运行的后台任务证据")
        if missing:
            return CompletionVerdict(False, reason=f"完成验收未通过：缺少条件 {', '.join(missing)} 的实际结果证据", satisfied=tuple(satisfied), missing=tuple(missing))

        if not calls and (success_claim or bool(state.tool_calls) or (self.require_execution and not inquiry)):
            return CompletionVerdict(False, reason="完成验收未通过：没有成功执行的工具证据，不能将行动任务报告为成功")

        # A browser receipt describes actual site access, not driver installation.
        latest_browser = next((call for call in reversed(state.tool_calls) if isinstance(call, Mapping) and call.get("name") == "private_browser_open"), None)
        if latest_browser is not None and not requirements:
            receipt = self._receipt(latest_browser)
            if latest_browser.get("status") != "ok" or "browser_access" not in receipt.get("verified_effects", ()):
                return CompletionVerdict(False, reason="浏览器目标访问尚未验证成功；安装成功、导航返回或 HTTP 错误页不能代替目标站可用证据")
        if requirements and all(str(r.get("effect", "")).startswith("tool:") for r in requirements) and outcome != "completed":
            # Observation receipts verify collection, not the adequacy of an
            # answer. Do not silently upgrade an unspecified business outcome.
            return CompletionVerdict(True, status="partial", reason="answer_completion_not_declared", satisfied=tuple(satisfied))
        return CompletionVerdict(True, satisfied=tuple(satisfied))

    def target_coverage_gaps(self, contract, user_messages, *, additional_inputs=()):
        """Shared preflight/final coverage. Declaration is NEVER execution proof.

        Explicit operation/source labels are conservative role constraints, not
        a general semantic parser, a tool selector, or an authorization grant.
        A source binding supplied by a transforming handler cannot cover output.
        """
        requirements = contract.get("requirements", ())
        explicit = self._extract_targets("\n".join(user_messages))
        explicit.extend(t for d in contract.get("deliverables", ()) for t in self._extract_targets(str(d)))
        roles = self._explicit_target_roles(user_messages)

        def has_role(requirement, role):
            effect = str(requirement.get("effect", ""))
            if not role:
                return True
            if role == "write":
                return effect in {"write", "document_translation"}
            if role == "read" and effect.startswith("tool:"):
                capability = self.tool_capabilities.get(effect[5:], {})
                return (effect in capability.get("effects", ()) and bool(capability.get("target_arguments")))
            return effect == role

        def matches(requirement, target):
            return (any(self._targets_match(target, n) for d in requirement.get("targets", ())
                        for n in self._normalize_target_values(d))
                    or ("\\" not in target and ":" not in target
                        and self._selector_matches(requirement.get("target_selector"), target)))

        gaps = []
        for target in dict.fromkeys(explicit):
            required = sorted(roles.get(target, ()))
            missing_roles = []
            for role in required or [""]:
                input_binding = role in {"", "read"} and any(
                    self._targets_match(target, n) for value in additional_inputs
                    for n in self._normalize_target_values(value))
                if not input_binding and not any(matches(r, target) and has_role(r, role) for r in requirements):
                    missing_roles.append(role)
            if missing_roles:
                gaps.append({"target": target, "effects": [r for r in missing_roles if r]})
        return gaps

    @classmethod
    def _explicit_target_roles(cls, user_messages):
        roles = {}
        prefixes = {
            "read": r"(?:原文(?:文件|路径)?|源文件|输入(?:文件|路径)?|从(?:本地)?文件|读取(?:本地)?(?:文件)?|读入(?:文件)?|(?:source|input)(?:\s+(?:file|path))?)\s*[:：=]?\s*$",
            "write": r"(?:输出(?:文件|路径)?|写入(?:唯一)?(?:输出|文件|路径)?|保存(?:到|至)?|创建(?:文件)?|新建(?:文件)?|修改(?:文件)?|更新(?:文件)?|覆盖(?:文件)?|(?:output|write|save|create)(?:\s+(?:file|to|path))?)\s*[:：=]?\s*$",
            "delete": r"(?:删除(?:文件)?|移除(?:文件)?|delete(?:\s+file)?)\s*[:：=]?\s*$",
            "send": r"(?:发送(?:文件)?|推送(?:文件)?|send(?:\s+file)?)\s*[:：=]?\s*$",
        }
        for message in user_messages:
            source = str(message)
            mentions = [(m.start(), m.end(), m.group(1), True) for m in _QUOTED_TARGET.finditer(source)]
            unquoted = _QUOTED_TARGET.sub(lambda m: " " * len(m.group(0)), source)
            mentions.extend((m.start(), m.end(), m.group(0), False) for m in _FILE_TARGET.finditer(unquoted))
            for start, end, value, quoted in mentions:
                if not cls._is_file_mention(source, start, end, value, quoted=quoted):
                    continue
                prefix = re.split(r"[，。；;！？!?\r\n]", source[max(0, start - 80):start])[-1]
                if _NEGATED_ACTION_REQUEST.search(prefix):
                    continue
                for effect, pattern in prefixes.items():
                    if re.search(pattern, prefix, re.I):
                        for target in cls._normalize_target_values(value):
                            roles.setdefault(target, set()).add(effect)
        return roles

    def _verify_read_unchanged(self, state, target):
        observations = []
        for call in state.tool_calls:
            capability = self.tool_capabilities.get(call.get("name"), {})
            if "read" not in capability.get("effects", ()) or not self._has_effect([call], "read", [target]):
                continue
            receipt = self._receipt(call)
            refresh = getattr(self.result_handlers.get(call.get("name")), "agent_verify_result", None)
            if not receipt.get("sha256") or not callable(refresh):
                return False
            try:
                if refresh(receipt=receipt, context=state.context).get("result_valid") is not True:
                    return False
            except Exception:
                return False
            observations.append(receipt["sha256"])
        if not observations or len(set(observations)) != 1:
            return False
        # Known task-local source mutations cannot be erased by restoring bytes
        # or taking a new read receipt. This does not attest arbitrary external
        # filesystem history between two observations.
        for call in state.tool_calls:
            receipt = self._receipt(call)
            if call.get("status") != "ok" or not {"write", "delete"}.intersection(receipt.get("verified_effects", ())):
                continue
            if any(self._targets_match(target, n) for value in receipt.get("completion_targets", ())
                   for n in self._normalize_target_values(value)):
                return False
        return True

    def _verify_inplace_read(self, state, candidate, target, calls):
        """A requested in-place write may consume an observed pre-write input.

        Require the actual writer's pre-image hash AND its declared, currently
        valid output condition. Merely re-reading a changed input is not proof.
        """
        messages = self._extract_user_messages(state)
        if _UNCHANGED_INPUTS.search("\n".join(messages)):
            return False
        roles = self._explicit_target_roles(messages)
        if "write" not in roles.get(target, ()):
            return False
        requirements = (state.context or {}).get("agent_completion_contract", {}).get("requirements", ())
        output_conditions = [row for row in requirements if row.get("effect") == "write" and any(
            self._targets_match(target, n) for value in row.get("targets", ()) for n in self._normalize_target_values(value))]
        if not output_conditions or not any(self._verify_requirement(state, row, calls) for row in output_conditions):
            return False
        seen = False
        for call in state.tool_calls:
            if call is candidate:
                seen = True
                continue
            if not seen or not self._has_effect([call], "write", [target]):
                continue
            if call.get("status") == "ok" and self._receipt(call).get("previous_sha256") == self._receipt(candidate).get("sha256"):
                return True
            return False
        return False

    def _verify_requirement(self, state, requirement, calls):
        effect = str(requirement["effect"])
        selector = requirement.get("target_selector")
        if selector is not None:
            if effect not in {"delete", "write", "send", "document_translation"}:
                return False
            # An unresolved name is a predicate, not a placeholder path. Bind it
            # ONLY to executor-issued, effect-specific absolute file receipts.
            # Arguments/model notes never establish this binding. More than one
            # matching path is ambiguous and cannot complete a singular target.
            from .task_state import TaskContractError, validate_contract
            try:
                validate_contract({"goal": "verify", "deliverables": ["verify"], "requirements": [
                    {"id": "verify", "description": "verify", **requirement}]})
            except TaskContractError:
                return False
            paths = set()
            for call in calls:
                receipt = self._receipt(call)
                if effect not in self.tool_capabilities.get(call.get("name"), {}).get("effects", ()):
                    continue
                if effect not in receipt.get("verified_effects", ()):
                    continue
                actual = receipt.get("completion_targets", ())
                if not isinstance(actual, (list, tuple)):
                    continue
                for value in actual:
                    if (isinstance(value, str) and "://" not in value and ntpath.isabs(value)
                            and self._selector_matches(selector, value)):
                        paths.add(self._normalize_target_values(value)[0])
            if len(paths) != 1:
                return False
            resolved = {k: v for k, v in requirement.items() if k != "target_selector"}
            resolved["targets"] = list(paths)
            return self._verify_requirement(state, resolved, calls)
        if effect == "research_answer":
            checks = requirement.get("checks", {})
            return (isinstance(checks, Mapping) and not set(checks) - {"required_fields", "min_sources", "require_dates", "date_kinds"}
                    and state.research.satisfies(checks))
        if effect == "local_answer":
            return not requirement.get("targets") and state.local_answer.satisfies(requirement.get("checks", {}))
        if effect == "history_answer":
            return not requirement.get("targets") and state.history_answer.satisfies(requirement.get("checks", {}))
        targets = [self._normalize_target_values(t)[0] for t in requirement.get("targets", ())]
        if effect == "read" and not targets:
            return False
        if effect.startswith("tool:"):
            name = effect[5:]
            capability = self.tool_capabilities.get(name, {})
            if effect not in capability.get("effects", ()):
                return False
            # Free-form observation descriptions are not resource identifiers.
            if not capability.get("target_arguments"):
                targets = []
            matching = [c for c in state.tool_calls if c.get("name") == name and c.get("executed") is not False]
            if not matching or (not targets and matching[-1].get("status") != "ok"):
                return False
            receipt = self._receipt(matching[-1])
            required_fields = getattr(self.result_handlers.get(name), "agent_observation_required_fields", ())
            if any(not receipt.get(field) for field in required_fields):
                return False
        checks = requirement.get("checks", {})
        if not isinstance(checks, Mapping):
            return False
        # A narrowly recognizable mandatory content condition must not be
        # treated as proof merely because the model omitted result checks.
        mandatory = re.search(r"(?:文件)?内容必须为\s*([^。\n]+)", str(requirement.get("description", "")))
        if mandatory and checks.get("contains") != mandatory.group(1).strip():
            return False
        allowed = {"contains", "sha256", "min_bytes", "http_status", "page_contains", "receipt_required", "source_pages", "unchanged"}
        if set(checks) - allowed:
            return False
        if "unchanged" in checks and (effect != "read" or type(checks["unchanged"]) is not bool):
            return False
        for target in targets or [None]:
            if effect == "read" and checks.get("unchanged") is True and not self._verify_read_unchanged(state, target):
                return False
            candidates = [c for c in calls if self._has_effect([c], effect, [target] if target else [])]
            if not candidates:
                return False
            candidate = candidates[-1]
            # A newer executor receipt for the same target supersedes historical
            # success, including deletion, overwrite and failed site access.
            if target:
                for c in reversed(state.tool_calls):
                    related = self.tool_capabilities.get(c.get("name"), {}).get("effects", ())
                    relevant = {"write", "delete"} if effect in {"write", "delete"} else {effect}
                    if not relevant.intersection(related):
                        continue
                    actual = list(self._receipt(c).get("completion_targets", []))
                    arguments = c.get("arguments", {})
                    if isinstance(arguments, Mapping):
                        actual.extend(arguments[key] for key in self.tool_capabilities.get(c.get("name"), {}).get("target_arguments", ()) if key in arguments)
                    if any(self._targets_match(target, n) for a in actual for n in self._normalize_target_values(a)):
                        if not self._has_effect([c], effect, [target]) or c.get("status") != "ok":
                            return False
                        candidate = c
                        break
            receipt = dict(self._receipt(candidate))
            if effect == "browser_access" and (receipt.get("site_access_ok") is False or
                    isinstance(receipt.get("http_status"), int) and receipt["http_status"] >= 400):
                return False
            handler = self.result_handlers.get(candidate.get("name"))
            refresh = getattr(handler, "agent_verify_result", None)
            resource_read = "read" in self.tool_capabilities.get(candidate.get("name"), {}).get("effects", ())
            if (effect == "read" or effect.startswith("tool:") and resource_read) and (
                    receipt.get("read_complete") is not True or not callable(refresh) or not receipt.get("sha256")):
                return False
            if callable(refresh):
                try:
                    refreshed = refresh(receipt=receipt, context=state.context)
                except Exception:
                    return False
                if (effect == "read" and checks.get("unchanged") is not True
                        and refreshed.get("result_valid") is False
                        and self._verify_inplace_read(state, candidate, target, calls)):
                    # Keep the historical input's bytes/hash, not the output's
                    # refreshed contents. Unsupported input text checks fail.
                    receipt["result_valid"] = True
                else:
                    receipt.update(refreshed)
            if receipt.get("result_valid") is False:
                return False
            if (effect == "read" or effect.startswith("tool:") and resource_read) and receipt.get("result_valid") is not True:
                return False
            if effect == "document_translation":
                if (receipt.get("translation_verified") is not True or receipt.get("result_valid") is not True
                        or type(receipt.get("done_chunks")) is not int or receipt["done_chunks"] < 1
                        or receipt["done_chunks"] != receipt.get("chunk_count")
                        or not str(receipt.get("path", "")).lower().endswith(".pdf")
                        or not callable(refresh)):
                    return False
            if "source_pages" in checks and (effect != "document_translation" or type(checks["source_pages"]) is not int
                    or receipt.get("source_pages") != checks["source_pages"]):
                return False
            if "contains" in checks and checks["contains"] not in str(receipt.get("file_text", "")):
                return False
            if "sha256" in checks and str(checks["sha256"]).lower() != receipt.get("sha256"):
                return False
            if "min_bytes" in checks and (type(checks["min_bytes"]) is not int or receipt.get("bytes", -1) < checks["min_bytes"]):
                return False
            if "http_status" in checks and receipt.get("http_status") != checks["http_status"]:
                return False
            if "page_contains" in checks and checks["page_contains"] not in str(receipt.get("page_text", "")):
                return False
            if checks.get("receipt_required") and not receipt.get("delivery_receipt"):
                return False
        return True

    @staticmethod
    def _selector_matches(selector, path) -> bool:
        if not isinstance(selector, Mapping) or not isinstance(path, str) or "://" in path:
            return False
        name = ntpath.basename(path.replace("/", "\\")).casefold()
        value = selector.get("value")
        if not isinstance(value, str) or not value or any(c in value for c in '/\\:*?"<>|'):
            return False
        matched_suffixes = []
        if "suffixes" in selector:
            suffixes = selector["suffixes"]
            if not isinstance(suffixes, list) or not suffixes or any(not isinstance(s, str) for s in suffixes):
                return False
            matched_suffixes = [s.casefold() for s in suffixes if name.endswith(s.casefold())]
            if not matched_suffixes:
                return False
        if selector.get("kind") == "file_name":
            return name == value.casefold()
        if selector.get("kind") == "file_stem":
            if matched_suffixes:
                return any(name[:-len(s)] == value.casefold() for s in matched_suffixes)
            return ntpath.splitext(name)[0] == value.casefold()
        return False

    @staticmethod
    def _claims_success(text: str) -> bool:
        return bool(_ACTION_SUCCESS.search(text)) or any(
            pattern.search(text)
            for pattern in (*_WRITE_CLAIM_PATTERNS, *_DELETE_CLAIM_PATTERNS, *_SEND_CLAIM_PATTERNS, *_REMINDER_CLAIM_PATTERNS, *_PROCESS_STOP_CLAIM_PATTERNS)
        )

    def _successful_calls(self, state: AgentState) -> list[Mapping[str, Any]]:
        return [call for call in state.tool_calls if isinstance(call, Mapping)
                and str(call.get("status") or "").lower() == "ok"
                and (not self.restrict_tool_names or call.get("name") in self.allowed_tool_names)]

    @staticmethod
    def _receipt(call: Mapping[str, Any]) -> Mapping[str, Any]:
        data = call.get("data", {})
        updates = data.get("state_updates", {}) if isinstance(data, Mapping) else {}
        return updates if isinstance(updates, Mapping) else {}

    def _background_tasks(self, state: AgentState) -> dict[str, Mapping[str, Any]]:
        tasks: dict[str, Mapping[str, Any]] = {}
        for item in state.transcript:
            if item.get("role") != "tool" or item.get("status") != "ok":
                continue
            if self.restrict_tool_names and item.get("name") not in self.allowed_tool_names:
                continue
            task = self._receipt(item).get("background_task")
            if isinstance(task, Mapping) and task.get("task_id"):
                tasks[str(task["task_id"])] = task
        return tasks

    def _check_effects(self, state: AgentState, final_text: str, *, partial: bool = False) -> bool | str:
        if not self.enabled:
            return True

        text = str(final_text or "").strip()
        if not text:
            return "汇报内容不能为空"

        # 只有纯粹的失败/拒绝汇报才放行；不能让“失败了，不过已经完成”
        # 之类同时包含成功声明的文本绕过核验。
        has_success_claim = self._claims_success(text)
        background_tasks = self._background_tasks(state)
        if any(not task.get("completed") for task in background_tasks.values()):
            delivery_text = re.sub(
                r"(?:已(?:经)?(?:创建|启动|提交|排队)(?:了)?(?:后台)?(?:下载)?任务|"
                r"(?:后台|下载)?任务(?:已(?:经)?)?(?:创建|启动|提交|排队))[^，。；;\n]*", "", text,
            )
            delivery_claim = any(p.search(delivery_text) for p in (*_WRITE_CLAIM_PATTERNS, *_SEND_CLAIM_PATTERNS, *_REMINDER_CLAIM_PATTERNS, *_PROCESS_STOP_CLAIM_PATTERNS))
            nonnegated_delivery = re.sub(r"(?:尚未|还未|未能|未|还没|没有|没).{0,8}(?:完成|完了|成功|搞定)", "", delivery_text)
            async_success = re.search(r"(?:下载|文件|任务).{0,16}(?:完成|完了|成功|搞定)|(?:下载好了|下好了)", nonnegated_delivery)
            if delivery_claim or async_success:
                return "后台任务尚未完成或核验通过；只能报告任务号、进度或障碍，不能声称文件已经交付。"
            # Starting a durable job is an honest intermediate handoff, not a
            # missing write. The worker continues without another model call.
            return True
        if not has_success_claim and any(p.search(text) for p in _FAILURE_OR_REFUSAL_PATTERNS):
            return True

        # 汇总本次规划开始前的所有用户消息。恢复授权时，首个工具调用之后的
        # “好/同意”等用户消息不是新的任务目标，不能覆盖原始请求。
        user_messages = self._extract_user_messages(state)
        user_goal = "\n".join(user_messages)
        successful_calls = self._successful_calls(state)
        requested_targets = self._extract_targets(user_goal)
        def matches_effect(effect: str) -> bool:
            if partial and requested_targets:
                return any(self._verify_requirement(state, {"effect": effect, "targets": [target]}, successful_calls) for target in requested_targets)
            return self._verify_requirement(state, {"effect": effect, "targets": requested_targets}, successful_calls)

        # 1. 检查写文件/改配置/保存类操作
        user_wants_write = any(self._user_requested_action(message, _WRITE_KEYWORDS) for message in user_messages)
        model_claims_write = any(p.search(text) for p in _WRITE_CLAIM_PATTERNS)
        if user_wants_write or model_claims_write:
            if not matches_effect("write"):
                # 用户有写入意图或模型声称写入，但未执行任何写入类工具
                return self._missing_effect_reason("write", requested_targets)

        if any(self._user_requested_action(message, _DELETE_KEYWORDS) for message in user_messages) or any(p.search(text) for p in _DELETE_CLAIM_PATTERNS):
            if not matches_effect("delete"):
                return self._missing_effect_reason("delete", requested_targets)

        # 2. 检查点播/发送音乐或文件类操作
        user_wants_send = any(self._user_requested_action(message, _SEND_KEYWORDS) for message in user_messages)
        model_claims_send = any(p.search(text) for p in _SEND_CLAIM_PATTERNS)
        if user_wants_send or model_claims_send:
            if not matches_effect("send"):
                return self._missing_effect_reason("send", requested_targets)

        # 3. 检查提醒设置类操作
        user_wants_reminder = any(self._user_requested_action(message, _REMINDER_KEYWORDS) for message in user_messages)
        model_claims_reminder = any(p.search(text) for p in _REMINDER_CLAIM_PATTERNS)
        if user_wants_reminder or model_claims_reminder:
            if not matches_effect("reminder"):
                return self._missing_effect_reason("reminder", requested_targets)

        # 4. 检查进程停止类操作
        user_wants_stop_proc = any(self._user_requested_action(message, _PROCESS_STOP_KEYWORDS) for message in user_messages)
        model_claims_stop_proc = any(p.search(text) for p in _PROCESS_STOP_CLAIM_PATTERNS)
        if user_wants_stop_proc or model_claims_stop_proc:
            if not matches_effect("process_stop"):
                return self._missing_effect_reason("process_stop", requested_targets)

        return True

    @staticmethod
    def _user_requested_action(user_goal: str, keywords: Sequence[str]) -> bool:
        goal = str(user_goal or "").strip()
        if not goal or not any(keyword in goal for keyword in keywords):
            return False
        if _NEGATED_ACTION_REQUEST.search(goal):
            return False
        if (_INFORMATION_REQUEST.search(goal) or _INFORMATIONAL_SHAPE.search(goal)) and not _EXPLICIT_ACTION_REQUEST.search(goal):
            return False
        return True

    @staticmethod
    def _extract_user_messages(state: AgentState) -> list[str]:
        from .request_scope import RequestScope

        context = state.context if isinstance(state.context, Mapping) else {}
        if "agent_request_scope" in context:
            scope = context["agent_request_scope"]
            # No model-authored dict or restored arbitrary JSON is an intent
            # grant. Hosts rebuild this value from validated message/task state.
            return scope.user_messages() if isinstance(scope, RequestScope) else []
        messages: list[str] = []
        for item in state.transcript:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role == "assistant" and (item.get("tool_calls") or str(item.get("kind") or "").lower() == "tool"):
                break
            if role == "user":
                content = str(item.get("content") or "").strip()
                if content:
                    messages.append(content)
        return messages

    def _has_effect(self, calls: Sequence[Mapping[str, Any]], effect: str, requested_targets: Sequence[str], *, require_all: bool = True) -> bool:
        observed_targets: list[str] = []
        has_evidence = False
        for call in calls:
            name = str(call.get("name") or "").strip()
            capability = self.tool_capabilities.get(name, {})
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
                actual_targets = [normalized for value in raw_targets for normalized in self._normalize_target_values(value)]
            elif isinstance(arguments, Mapping):
                actual_targets = [normalized for key in target_arguments for normalized in self._normalize_target_values(arguments.get(key))]
            else:
                continue
            observed_targets.extend(actual_targets)
        matches = [any(self._targets_match(expected, actual) for actual in observed_targets) for expected in requested_targets]
        return has_evidence and (all(matches) if require_all else any(matches))

    @staticmethod
    def _is_file_mention(source: str, start: int, end: int, value: str, *, quoted: bool = False) -> bool:
        if not (_FILE_TARGET.fullmatch(value) or ntpath.isabs(value)):
            return False
        prefix, following = source[max(0, start - 80):start], source[end:]
        file_label = re.search(r"(?:文件|路径|file|path|read|write|读取|读入|保存|输出|输入|删除|发送)\s*[:：=]?\s*$", prefix, re.I)
        if any(c in value for c in "\\/") or ntpath.isabs(value):
            return True
        if file_label:
            return True
        if not quoted and ("=" in value or re.fullmatch(r"\d+(?:\.\d+)+", value)):
            return False
        if not file_label and (re.match(r"\s*[:=]", following)
                or re.search(r"(?:字段|属性|键|field|property|attribute|key)\s*$", prefix, re.I)):
            return False
        return True

    @staticmethod
    def _extract_targets(text: str) -> list[str]:
        source = str(text or "")
        quoted = [m.group(1) for m in _QUOTED_TARGET.finditer(source)
                  if SmartCompletionVerifier._is_file_mention(source, m.start(), m.end(), m.group(1), quoted=True)]
        unquoted_source = _QUOTED_TARGET.sub(lambda m: " " * len(m.group(0)), source)
        def unwrap_file_mention(match):
            value = (match.group(1) or match.group(2)).strip()
            prefix = re.split(r"[\s，。；：！？、]", unquoted_source[:match.start()])[-1]
            following = unquoted_source[match.end():]
            # Natural labels such as 文件（a.pdf） must not become a filename
            # containing the description. Literal path components and suffixes
            # such as C:\dir(a.pdf)\b.pdf or 报告（正式）.pdf stay intact;
            # quoted literal filenames bypass this prose normalization entirely.
            if (not any(c in prefix for c in "\\/:")
                    and not re.match(r"[\w\\/]|\.[A-Za-z0-9]", following)
                    and _FILE_TARGET.fullmatch(value)):
                return " " + value + " "
            return match.group(0)
        unquoted_source = re.sub(r"\(([^()（）\r\n]+)\)|（([^()（）\r\n]+)）",
                                 unwrap_file_mention, unquoted_source)
        # Quoted property values/prose are not extra file deliverables. Avoid
        # extracting a second, truncated filename from inside a quoted path.
        # Bare versions/decimal quantities are not file paths. Explicit quoted
        # numeric filenames and numeric filenames inside paths remain targets.
        unquoted = [m.group(0) for m in _FILE_TARGET.finditer(unquoted_source)
                    if SmartCompletionVerifier._is_file_mention(unquoted_source, m.start(), m.end(), m.group(0))]
        raw_targets = [*unquoted,
                       *(value for value in quoted if _FILE_TARGET.fullmatch(value) or ntpath.isabs(value))]
        targets: list[str] = []
        seen: set[str] = set()
        for raw in raw_targets:
            for normalized in SmartCompletionVerifier._normalize_target_values(raw):
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    targets.append(normalized)
        return targets

    @staticmethod
    def _normalize_target_values(value: Any) -> list[str]:
        if not isinstance(value, (str, int, float)):
            return []
        target = str(value).strip().strip(" \t\r\n\"'“”「」『』《》,，。；;：:！？!?)）]}>")
        if not target:
            return []
        if "://" in target:
            return [target.rstrip("/")]
        normalized = ntpath.normpath(target.replace("/", "\\")).casefold()
        return [normalized]

    @staticmethod
    def _targets_match(expected: str, actual: str) -> bool:
        if not expected or not actual:
            return False
        if "://" in expected or "://" in actual:
            return expected == actual
        # A basename request can match an absolute path; an explicit path cannot.
        return expected == actual or ("\\" not in expected and ":" not in expected and expected == ntpath.basename(actual))

    @staticmethod
    def _missing_effect_reason(effect: str, requested_targets: Sequence[str]) -> str:
        target_note = f"，且成功调用的工具没有作用于请求目标 {', '.join(requested_targets)}" if requested_targets else ""
        messages = {
            "write": "本轮未实际执行任何写入/保存类工具，或成功工具结果没有作用于请求目标",
            "delete": "本轮没有与请求目标匹配的成功删除工具结果；写入不能充当删除证据",
            "send": "本轮没有与请求目标匹配的成功发送工具结果",
            "reminder": "本轮没有与请求内容匹配的成功提醒设置结果",
            "process_stop": "本轮没有与请求目标匹配的成功进程停止结果",
        }
        return (
            f"完成核验未通过：{messages.get(effect, '没有对应的成功工具结果')}{target_note}。"
            "请先调用并核验对应工具；若无法完成，就如实说明阻塞原因。"
        )
