"""智能完成度核验器（SmartCompletionVerifier 门面）。

职责与上下游：
- 职责：作为 Agent 任务交付前的独立事实闸门。基于工具执行证据（Tool Execution Receipts）、
  目标文件/服务匹配（Target Verification）和契约要求（Contract Requirements），
  对模型声称的“已完成”进行事实性核验，杜绝口头幻觉。
- 上游：由 `complete_agent_decision`、`deliver_direct_chat` 等交付节点在输出给用户前调用。
- 下游：调用 `verification_contracts.py`、`verification_effects.py`、`verification_targets.py`
  分别核查契约满足度、工具副作用证据及目标路径。

设计原因与关键边界：
1. 诚实终态与目标完成的区分：
   - 允许模型如实报告失败（outcome="failed"）或阻断（outcome="blocked"）；
   - 普通问答可以在没有工具调用时通过核验，`outcome="completed"` 本身不要求工具回执；
   - 行动成功声明和契约要求须满足相应的执行证据、目标及结果检查，不能仅凭 completed 标签放行；
2. 动作声明与回执必须对应：
   - 包含“已发送”、“已修改”、“已删除”等语义断言时，必须有对应工具成功返回证据，
     禁止仅凭自然语言回复宣布任务完成。

阅读入口：
- 核心核验入口：`SmartCompletionVerifier.evaluate`
- 自然语言动作语义预检：`request_requires_action`
- 底层核验流转：`_evaluate`
"""

from __future__ import annotations

from shinku.phase_timing import timed

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .completion import CompletionVerdict
from .loop import AgentState
from .verification_rules import (
    _DEFAULT_TOOL_CAPABILITIES,
    _DELETE_KEYWORDS,
    _EXPLICIT_ACTION_REQUEST,
    _FULL_DOCUMENT,
    _INFORMATION_REQUEST,
    _INFORMATIONAL_SHAPE,
    _PDF_TRANSLATION,
    _PROCESS_STOP_KEYWORDS,
    _REMINDER_KEYWORDS,
    _SEND_KEYWORDS,
    _UNCHANGED_INPUTS,
    _WRITE_KEYWORDS,
)

from . import (
    verification_contracts,
    verification_effects,
    verification_targets,
    verification_text,
)
from .analysis_acceptance import analysis_delivery_verdict, ordinary_analysis
from .verification_contracts import VerificationContractsDependencies
from .verification_effects import VerificationEffectsDependencies
from .verification_targets import VerificationTargetsDependencies

# 公开核验入口在门面定义，规则模块不再声明它不拥有的导出。
__all__ = ["SmartCompletionVerifier", "CompletionVerdict"]


@dataclass
class _CompletionAssessment:
    """Condition results share the original lists throughout one evaluation."""

    contract: Mapping[str, Any] | None
    requirements: Sequence[Mapping[str, Any]]
    satisfied: list[str]
    missing: list[str]


class SmartCompletionVerifier:
    """Evidence-based gate with a legacy text compatibility layer.

    Trusted hosts may supply ``agent_completion_contract.requirements`` in the
    execution context. Natural-language compatibility is not a semantic verifier.
    ``evaluate`` distinguishes an honest terminal report from goal completion.
    """

    supports_plain_text_completion_verification = True

    @classmethod
    def request_requires_action(cls, request: str) -> bool:
        """The observation preflight uses the same action semantics as final QA."""
        return any(cls._user_requested_action(request, keywords) for keywords in (
            _WRITE_KEYWORDS, _DELETE_KEYWORDS, _SEND_KEYWORDS,
            _REMINDER_KEYWORDS, _PROCESS_STOP_KEYWORDS,
        )) or any(effects & {"write", "delete", "send"}
                  for effects in cls._explicit_target_roles([request]).values())

    def __init__(
        self,
        *,
        allowed_tool_names: Sequence[str] | None = None,
        tool_capabilities: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
        enabled: bool = True,
        require_execution: bool = False,
        result_handlers: Mapping[str, Any] | None = None,
    ) -> None:
        self.allowed_tool_names = frozenset(
            (str(n).strip() for n in allowed_tool_names or () if str(n).strip()),
        )
        self.restrict_tool_names = allowed_tool_names is not None
        capabilities = {name: dict(value) for name, value in _DEFAULT_TOOL_CAPABILITIES.items()}
        if isinstance(tool_capabilities, Mapping):
            for raw_name, raw_value in tool_capabilities.items():
                name = str(raw_name or "").strip()
                if not name or not isinstance(raw_value, Mapping):
                    continue
                effects = tuple(
                    (str(item).strip().casefold() for item in raw_value.get('effects', ()) if str(item).strip()),
                )
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

    @timed('verification')
    def evaluate(
        self,
        state: AgentState,
        final_text: str,
        *,
        outcome: str = "",
    ) -> CompletionVerdict:
        """评估汇报内容与状态是否满足任务完成标准。

        设计原因与证据提取：
        1. 抽取真实工具调用证据（Evidence）：核对最近 128 次工具调用的 status、verified_effects、
           completion_targets 与 delivery_receipt，形成可追溯的独立事实列表；
        2. 目标达成判定（business_goal）：只有当最终裁定 status=="completed" 且 accepted==True 时，
           才认可业务目标完成；
        3. 状态归属：返回不可变的 `CompletionVerdict`，携带 satisfied 和 missing 契约列表。
        """
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

    def _evaluate(
        self,
        state: AgentState,
        final_text: str,
        *,
        outcome: str = "",
    ) -> CompletionVerdict:
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
        from shinku.agent_runtime.capability_answer import capability_observer
        if capability_observer(state) is not None:
            # 纯自查契约虽只要求状态回执，也不能因此跳过正文额外声称的
            # 发送/修改/执行。复用普通分析的本轮行动识别，不放宽原效果核验。
            from .analysis_acceptance import _execution_claim_error
            claim_error = _execution_claim_error(self, state, text)
            if claim_error:
                return CompletionVerdict(False, reason=claim_error)
        answer_verdict = self._evaluate_answer_modes(state, text, outcome)
        if answer_verdict is not None:
            return answer_verdict
        calls = self._successful_calls(state)
        assessment = self._assess_completion_contract(state, calls)
        if isinstance(assessment, CompletionVerdict):
            return assessment
        user_messages = self._extract_user_messages(state)
        document_request = "\n".join(user_messages)
        self._check_unchanged_inputs(
            state, calls, assessment.requirements, user_messages, document_request, assessment.missing,
        )
        self._check_requested_translation(state, calls, user_messages, document_request, assessment.missing)
        self._check_requested_target_coverage(state, calls, assessment, user_messages, document_request)
        if ordinary_analysis(state) and outcome not in {"failed", "blocked", "waiting_background"}:
            return analysis_delivery_verdict(self, state, text, outcome, assessment)
        return self._finalize_completion_status(state, text, outcome, calls, assessment, user_messages)

    def _evaluate_answer_modes(self, state, text, outcome) -> CompletionVerdict | None:
        """Apply research, local and history answer gates in the original order."""
        if state.research.enabled:
            research_verdict = state.research.evaluate_answer(text, outcome)
            if not research_verdict.accepted:
                return research_verdict
        if state.local_answer.enabled:
            state.local_answer.refresh(self.result_handlers, state.context)
            local_verdict = state.local_answer.evaluate_answer(text, outcome)
            if not local_verdict.accepted and not ordinary_analysis(state):
                return local_verdict
        if state.history_answer.enabled:
            history_verdict = state.history_answer.evaluate_answer(text, outcome)
            if not history_verdict.accepted:
                return history_verdict
        return None

    def _assess_completion_contract(self, state, calls) -> _CompletionAssessment | CompletionVerdict:
        """Validate declared conditions and collect their evidence results."""
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
            matched = self._verify_requirement(state, requirement, calls)
            (satisfied if matched else missing).append(key)
        return _CompletionAssessment(contract, requirements, satisfied, missing)

    def _check_unchanged_inputs(
        self, state, calls, requirements, user_messages, document_request, missing,
    ):
        """Keep original input observations from being superseded by later mutations."""
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

    def _check_requested_translation(
        self, state, calls, user_messages, document_request, missing,
    ):
        """Retain full-document coverage demanded by the user's request."""
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

    def _check_requested_target_coverage(
        self, state, calls, assessment, user_messages, document_request,
    ):
        """Check explicit targets independently of model-authored conditions."""
        requirements = assessment.requirements
        missing = assessment.missing
        contract = assessment.contract
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
                            fresh = handler.agent_verify_result(
                                receipt=self._receipt(call),
                                context=state.context,
                            )
                        except Exception:
                            continue
                        source_path = fresh.get("source_path") if isinstance(fresh, Mapping) else None
                        if fresh.get("result_valid") is True and isinstance(source_path, str) and source_path:
                            verified_inputs.append(source_path)
            missing.extend(f"uncovered_target:{gap['target']}" for gap in self.target_coverage_gaps(
                contract, user_messages, additional_inputs=verified_inputs))

    def _partial_completion_verdict(self, state, calls, assessment, text) -> CompletionVerdict:
        """Require actual evidence even for a partial completion report."""
        requirements = assessment.requirements
        satisfied = assessment.satisfied
        missing = assessment.missing
        research_partial = (state.research.enabled
            and any(str(requirement.get("effect", "")) == "research_answer" for requirement in requirements)
            and state.research.satisfies({"min_sources": 1}))
        history_partial = (state.history_answer.enabled and requirements
            and all(str(row.get("effect", "")) == "history_answer" for row in requirements)
            and state.history_answer.answer_check.get("valid") is True
            and (state.history_answer.answer_check.get("disposition") in {"not_found", "ambiguous"}
                 or state.history_answer.validated_partial(text)))
        if not calls or (requirements and not satisfied and not research_partial and not history_partial):
            return CompletionVerdict(False, reason="部分完成也需要实际成功的工具证据", missing=tuple(missing))
        return CompletionVerdict(
            True,
            status='partial',
            reason='partial_completion',
            satisfied=tuple(satisfied),
            missing=tuple(missing),
        )

    def _finalize_completion_status(
        self, state, text, outcome, calls, assessment, user_messages,
    ) -> CompletionVerdict:
        """按结构化结果和执行证据收尾；正文措辞不能决定本轮成败。"""
        requirements = assessment.requirements
        satisfied = assessment.satisfied
        missing = assessment.missing
        inquiry = bool(user_messages) and all(
            (_INFORMATION_REQUEST.search(message) or _INFORMATIONAL_SHAPE.search(message))
            and not _EXPLICIT_ACTION_REQUEST.search(message)
            for message in user_messages
        )
        success_claim = self._claims_success(text)
        background = self._background_tasks(state)
        pending_background = any(
            (not task.get('completed') and task.get('status') in {'queued', 'pending', 'running', 'retrying'} for task in background.values()),
        )

        # 历史错误、否定句和礼貌措辞都可能出现在正常回答里，不能据此推断
        # 当前任务失败。失败/阻塞由结构化结果显式报告；未声明失败的行动任务
        # 仍须通过下面的契约及执行回执检查，completed 标签不能替代事实。
        if outcome in {"failed", "blocked"}:
            if success_claim and (missing or not calls):
                return CompletionVerdict(False, reason="失败汇报同时宣称了未经核验的成功结果", missing=tuple(missing))
            return CompletionVerdict(
                True, status=outcome,
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
            return CompletionVerdict(
                False,
                reason=legacy_check,
                satisfied=tuple(satisfied),
                missing=tuple(missing),
            )
        if outcome == "partial":
            return self._partial_completion_verdict(state, calls, assessment, text)
        if pending_background:
            return CompletionVerdict(
                True,
                status='waiting_background',
                reason='background_pending',
                satisfied=tuple(satisfied),
                missing=tuple(missing),
            )
        if outcome == "waiting_background":
            return CompletionVerdict(False, reason="没有仍在运行的后台任务证据")
        return self._evaluate_success_evidence(state, outcome, calls, assessment, success_claim, inquiry)

    def _evaluate_success_evidence(
        self, state, outcome, calls, assessment, success_claim, inquiry,
    ) -> CompletionVerdict:
        """Accept success only after missing conditions and executor receipts pass."""
        requirements = assessment.requirements
        satisfied = assessment.satisfied
        missing = assessment.missing
        if missing:
            return CompletionVerdict(
                False,
                reason=f"完成验收未通过：缺少条件 {', '.join(missing)} 的实际结果证据",
                satisfied=tuple(satisfied),
                missing=tuple(missing),
            )

        if not calls and (success_claim or bool(state.tool_calls) or (self.require_execution and not inquiry)):
            return CompletionVerdict(False, reason="完成验收未通过：没有成功执行的工具证据，不能将行动任务报告为成功")

        # A browser receipt describes actual site access, not driver installation.
        latest_browser = next(
            (call for call in reversed(state.tool_calls) if isinstance(call, Mapping) and call.get('name') == 'private_browser_open'),
            None,
        )
        if latest_browser is not None and not requirements:
            receipt = self._receipt(latest_browser)
            if latest_browser.get("status") != "ok" or "browser_access" not in receipt.get("verified_effects", ()):
                return CompletionVerdict(False, reason="浏览器目标访问尚未验证成功；安装成功、导航返回或 HTTP 错误页不能代替目标站可用证据")
        if requirements and all(str(r.get("effect", "")).startswith("tool:") for r in requirements) and outcome != "completed":
            # Observation receipts verify collection, not the adequacy of an
            # answer. Do not silently upgrade an unspecified business outcome.
            return CompletionVerdict(
                True,
                status='partial',
                reason='answer_completion_not_declared',
                satisfied=tuple(satisfied),
            )
        return CompletionVerdict(True, satisfied=tuple(satisfied))

    def target_coverage_gaps(self, contract, user_messages, *, additional_inputs=()):
        """Shared preflight/final coverage. Declaration is NEVER execution proof.

        Explicit operation/source labels are conservative role constraints, not
        a general semantic parser, a tool selector, or an authorization grant.
        A source binding supplied by a transforming handler cannot cover output.
        """
        requirements = contract.get("requirements", ())
        candidates = [value for row in requirements for value in row.get('targets', ()) if isinstance(value, str)]
        projected_messages = [verification_targets.bind_literal_paths(message, candidates) for message in user_messages]
        explicit = self._extract_targets("\n".join(projected_messages))
        from shinku.paths import absolute_file_path
        for deliverable in contract.get("deliverables", ()):
            # 契约中的完整绝对路径可含空格；不能把它再次当自然语言切成伪目标。
            value = str(deliverable).strip().strip('"')
            explicit.extend(self._normalize_target_values(value) if absolute_file_path(value)
                            else self._extract_targets(str(deliverable)))
        roles = self._explicit_target_roles(projected_messages)

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
            from .target_identity import resolve_target
            effect = requirement.get('effect', '')
            target = resolve_target(target, effect, self.result_handlers)
            return (any(self._targets_match(target, resolve_target(n, effect, self.result_handlers)) for d in requirement.get("targets", ())
                        for n in self._normalize_target_values(d))
                    or ("\\" not in target and ":" not in target
                        and self._selector_matches(requirement.get("target_selector"), target)))

        gaps = []
        if (verification_targets.file_delivery_requested(user_messages)
                and not any(r.get('effect') == 'send' for r in requirements)):
            # 缺少发送条件时拒绝契约；占位提示不是可执行目标或发送授权。
            gaps.append({'target': '', 'effects': ['send'], 'reason': 'requested_file_delivery_missing'})
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
        return verification_targets.explicit_target_roles(
            [message for message in user_messages if not verification_text.is_reply_composition(message)],
            deps=cls._verification_targets_dependencies(),
        )

    def _verify_read_unchanged(self, state, target):
        return verification_contracts.verify_read_unchanged(
            state,
            target,
            deps=self._verification_contracts_dependencies(),
        )

    def _verify_inplace_read(self, state, candidate, target, calls):
        return verification_contracts.verify_inplace_read(
            state,
            candidate,
            target,
            calls,
            deps=self._verification_contracts_dependencies(),
        )

    def _verify_requirement(self, state, requirement, calls):
        from .target_identity import resolve_target
        # 仅核验视图使用宿主路径身份；原任务、原始参数和工具回执都不改写。
        requirement = {**requirement, 'targets': [resolve_target(t, requirement['effect'],
            self.result_handlers, state.context) for t in requirement.get('targets', ())]}
        return verification_contracts.verify_requirement(
            state,
            requirement,
            calls,
            deps=self._verification_contracts_dependencies(),
        )

    @staticmethod
    def _selector_matches(selector, path) -> bool:
        return verification_text.selector_matches(
            selector,
            path,
        )

    @staticmethod
    def _claims_success(text: str) -> bool:
        return verification_text.claims_success(
            text,
        )

    def _successful_calls(self, state: AgentState) -> list[Mapping[str, Any]]:
        return verification_effects.successful_calls(
            state,
            deps=self._verification_effects_dependencies(),
        )

    @staticmethod
    def _receipt(call: Mapping[str, Any]) -> Mapping[str, Any]:
        return verification_text.receipt(
            call,
        )

    def _background_tasks(self, state: AgentState) -> dict[str, Mapping[str, Any]]:
        return verification_effects.background_tasks(
            state,
            deps=self._verification_effects_dependencies(),
        )

    def _check_effects(
        self,
        state: AgentState,
        final_text: str,
        *,
        partial: bool = False,
    ) -> bool | str:
        return verification_effects.check_effects(
            state,
            final_text,
            partial=partial,
            deps=self._verification_effects_dependencies(),
        )

    @staticmethod
    def _user_requested_action(user_goal: str, keywords: Sequence[str]) -> bool:
        return verification_text.user_requested_action(
            user_goal,
            keywords,
        )

    @staticmethod
    def _extract_user_messages(state: AgentState) -> list[str]:
        return verification_text.extract_user_messages(
            state,
        )

    def _has_effect(
        self,
        calls: Sequence[Mapping[str, Any]],
        effect: str,
        requested_targets: Sequence[str],
        *,
        require_all: bool = True,
    ) -> bool:
        return verification_effects.has_effect(
            calls,
            effect,
            requested_targets,
            require_all=require_all,
            deps=self._verification_effects_dependencies(),
        )

    @staticmethod
    def _is_file_mention(
        source: str,
        start: int,
        end: int,
        value: str,
        *,
        quoted: bool = False,
    ) -> bool:
        return verification_text.is_file_mention(
            source,
            start,
            end,
            value,
            quoted=quoted,
        )

    @staticmethod
    def _extract_targets(text: str) -> list[str]:
        return verification_text.extract_targets(
            text,
            file_mention=SmartCompletionVerifier._is_file_mention,
            http_target=SmartCompletionVerifier._valid_http_target,
            normalize=SmartCompletionVerifier._normalize_target_values,
        )

    @staticmethod
    def _valid_http_target(value: str) -> bool:
        return verification_text.valid_http_target(
            value,
        )

    @staticmethod
    def _normalize_target_values(value: Any) -> list[str]:
        return verification_text.normalize_target_values(
            value,
        )

    @staticmethod
    def _targets_match(expected: str, actual: str) -> bool:
        return verification_text.targets_match(
            expected,
            actual,
        )

    @staticmethod
    def _missing_effect_reason(effect: str, requested_targets: Sequence[str]) -> str:
        return verification_text.missing_effect_reason(
            effect,
            requested_targets,
        )

    def _verification_contracts_dependencies(self) -> VerificationContractsDependencies:
        return VerificationContractsDependencies(
            explicit_target_roles=self._explicit_target_roles,
            extract_user_messages=self._extract_user_messages,
            has_effect=self._has_effect,
            normalize_target_values=self._normalize_target_values,
            receipt=self._receipt,
            selector_matches=self._selector_matches,
            targets_match=self._targets_match,
            verify_inplace_read=self._verify_inplace_read,
            verify_read_unchanged=self._verify_read_unchanged,
            verify_requirement=self._verify_requirement,
            result_handlers=lambda: self.result_handlers,
            tool_capabilities=lambda: self.tool_capabilities,
        )

    def _verification_effects_dependencies(self) -> VerificationEffectsDependencies:
        return VerificationEffectsDependencies(
            background_tasks=self._background_tasks,
            result_handlers=lambda: self.result_handlers,
            claims_success=self._claims_success,
            extract_targets=self._extract_targets,
            extract_user_messages=self._extract_user_messages,
            missing_effect_reason=self._missing_effect_reason,
            normalize_target_values=self._normalize_target_values,
            receipt=self._receipt,
            successful_calls=self._successful_calls,
            targets_match=self._targets_match,
            user_requested_action=self._user_requested_action,
            verify_requirement=self._verify_requirement,
            allowed_tool_names=lambda: self.allowed_tool_names,
            enabled=lambda: self.enabled,
            restrict_tool_names=lambda: self.restrict_tool_names,
            tool_capabilities=lambda: self.tool_capabilities,
        )

    @classmethod
    def _verification_targets_dependencies(cls) -> VerificationTargetsDependencies:
        return VerificationTargetsDependencies(
            is_file_mention=cls._is_file_mention,
            normalize_target_values=cls._normalize_target_values,
        )
