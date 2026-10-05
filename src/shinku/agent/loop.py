"""有界的 Agent 规划-工具-结果循环。

循环内核不绑定 LLM SDK。调用方提供一个 planner，planner 读取 ``AgentState``
并返回最终文本、等待用户、或一次 ``ToolInvocation``。这样模型供应商只是规划
器适配器，工具宿主只是 handler 映射，循环本身可以离线测试。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from shinku.tools.execution import ExecutionPolicy, ToolHandler, execute_invocation, validate_invocation
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope

from .approval import ToolApprovalRequest
from .completion import CompletionVerdict
from .progress import ProgressLedger
from .research import ResearchLedger
from .local_answer import LocalAnswerLedger
from .history_answer import HistoryAnswerLedger

AGENT_MAX_ROUNDS_HARD_LIMIT = 64

__all__ = [
    "AGENT_MAX_ROUNDS_HARD_LIMIT",
    "AgentDecision",
    "AgentLoop",
    "AgentRunResult",
    "AgentRunCheckpoint",
    "AgentState",
    "ApprovalGate",
    "CompletionGate",
    "Planner",
]


@dataclass
class AgentState:
    """规划器可见的最小状态；列表刻意保持 JSON 形状，方便接 LLM。"""

    task_id: str = ""
    rounds: int = 0
    transcript: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    last_envelope: ToolResultEnvelope | None = None
    context: Any = None
    observation_calls: int = 0
    observation_budget_rejections: int = 0
    consecutive_observation_calls: int = 0
    consecutive_observation_budget_rejections: int = 0
    completion: dict[str, Any] = field(default_factory=dict)
    inflight_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    progress: dict[str, Any] = field(default_factory=dict)
    finalization_only: bool = False
    source_discovery_calls: int = 0
    suppressed_tool_names: set[str] = field(default_factory=set)
    final_reply_repair_attempts: int = 0
    plain_text_completion_verifiable: bool = False
    research: ResearchLedger = field(default_factory=ResearchLedger)
    local_answer: LocalAnswerLedger = field(default_factory=LocalAnswerLedger)
    history_answer: HistoryAnswerLedger = field(default_factory=HistoryAnswerLedger)
    last_final_text: str = ""


@dataclass(frozen=True)
class AgentDecision:
    """规划器的一次决定。``kind`` 只允许 final / wait / tool。"""

    kind: str
    text: str = ""
    invocation: ToolInvocation | None = None
    invocations: tuple[ToolInvocation, ...] = ()
    reason: str = ""
    outcome: str = ""
    task_contract: Any = None
    task_link: Any = None
    task_notes: Any = None
    reply_routing: Any = None

    def with_task_metadata(self, value: Mapping[str, Any]) -> "AgentDecision":
        return replace(self, **{
            key: value[key] for key in ("task_contract", "task_link", "task_notes", "reply_routing") if key in value
        })

    @classmethod
    def final(cls, text: str, *, outcome: str = "") -> "AgentDecision":
        return cls(kind="final", text=str(text or "").strip(), outcome=str(outcome or "").strip().lower())

    @classmethod
    def wait(cls, text: str, *, reason: str = "") -> "AgentDecision":
        return cls(kind="wait", text=str(text or "").strip(), reason=str(reason or "").strip())

    @classmethod
    def tool(cls, invocation: ToolInvocation) -> "AgentDecision":
        return cls(kind="tool", invocation=invocation, invocations=(invocation,))

    @classmethod
    def tools(cls, invocations: Sequence[ToolInvocation]) -> "AgentDecision":
        calls = tuple(item for item in invocations if isinstance(item, ToolInvocation))
        if not calls:
            return cls(kind="tool")
        return cls(kind="tool", invocation=calls[0], invocations=calls)

    @staticmethod
    def _invocation_from_mapping(raw_call: Mapping[str, Any]) -> ToolInvocation | None:
        malformed = raw_call.get("_malformed") is True or bool(raw_call.get("protocol_error"))
        function = raw_call.get("function")
        if isinstance(function, Mapping):
            merged = dict(function)
            merged.setdefault("id", raw_call.get("id", ""))
            merged.setdefault("source", raw_call.get("source", ""))
            raw_call = merged
        name = str(raw_call.get("name") or raw_call.get("type") or "").strip()
        raw_arguments = raw_call.get("arguments")
        protocol_error = "tool_arguments_malformed" if malformed or raw_call.get("_malformed") is True else ""
        if isinstance(raw_arguments, Mapping):
            arguments = dict(raw_arguments)
        elif isinstance(raw_arguments, str):
            try:
                decoded = json.loads(raw_arguments)
            except (TypeError, ValueError):
                decoded = None
            if not isinstance(decoded, Mapping):
                protocol_error = "tool_arguments_malformed"
            arguments = dict(decoded) if isinstance(decoded, Mapping) else {}
        else:
            if "arguments" in raw_call:
                protocol_error = "tool_arguments_malformed"
            arguments = {
                key: item for key, item in raw_call.items() if key not in {"type", "name", "arguments", "function"}
            }
        if not name:
            return None
        return ToolInvocation(
            name=name,
            arguments=dict(arguments),
            source=str(raw_call.get("source") or "legacy_json"),
            id=str(raw_call.get("id") or ""),
            protocol_error=protocol_error,
        )

    @staticmethod
    def _speech_segments_text(value: Mapping[str, Any]) -> str:
        """把既有回复宿主的分段格式折叠成 Agent 可交付的文本。

        QQ 回复层允许模型只返回 ``speech_segments``；Agent 规划层此前只认
        ``kind``/``reply``/``speech``，导致同一份有效回复被误判成
        ``invalid_decision``，还没走到发送层就结束。
        """

        raw_segments = value.get("speech_segments")
        if not isinstance(raw_segments, (list, tuple)):
            return ""
        segments: list[str] = []
        for raw_segment in raw_segments:
            if isinstance(raw_segment, Mapping):
                raw_segment = (
                    raw_segment.get("speech")
                    or raw_segment.get("text")
                    or raw_segment.get("content")
                    or raw_segment.get("message")
                )
            text = str(raw_segment or "").strip()
            if text:
                segments.append(text)
        return "\n".join(segments)

    @classmethod
    def from_value(cls, value: Any) -> "AgentDecision | None":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return None
        status = str(value.get("status") or "").strip().lower()
        kind = str(value.get("kind") or value.get("action") or "").strip().lower()
        if not kind and ("tool_call" in value or "tool_calls" in value or "invocation" in value):
            kind = "tool"
        text = str(
            value.get("text")
            or value.get("speech")
            or value.get("reply")
            or value.get("message")
            or value.get("content")
            or ""
        ).strip()
        if not text:
            text = cls._speech_segments_text(value)
        if not kind and status in {"reply", "respond", "response", "final", "answer", "done", "complete"}:
            kind = "final"
        if not kind and (text and ("speech" in value or "speech_segments" in value)):
            kind = "final"
        if not kind and status in {"wait", "waiting", "waiting_user", "ask"}:
            kind = "wait"
        if kind in {"final", "answer", "done", "complete"}:
            return cls.final(text, outcome=str(value.get("task_outcome") or value.get("outcome") or "")).with_task_metadata(value)
        if kind in {"wait", "waiting_user", "ask"}:
            return cls.wait(text, reason=str(value.get("reason") or "").strip()).with_task_metadata(value)
        raw_calls = value.get("tool_calls")
        if kind in {"tool", "call_tool", "tool_call"} and isinstance(raw_calls, list):
            calls = tuple(
                item
                for raw_item in raw_calls
                if isinstance(raw_item, Mapping)
                for item in (cls._invocation_from_mapping(raw_item),)
                if item is not None
            )
            if calls:
                return cls.tools(calls).with_task_metadata(value)
        raw_call = value.get("invocation") if isinstance(value.get("invocation"), Mapping) else value.get("tool_call")
        if kind in {"tool", "call_tool", "tool_call"} and isinstance(raw_call, Mapping):
            invocation = cls._invocation_from_mapping(raw_call)
            if invocation is not None:
                return cls.tool(invocation).with_task_metadata(value)
        return None


class Planner(Protocol):
    def __call__(self, *, state: AgentState) -> AgentDecision | Mapping[str, Any] | None: ...


CompletionGate = Callable[[AgentState, str], bool | str | tuple[bool, str] | CompletionVerdict]
ApprovalGate = Callable[..., ToolApprovalRequest | None]


@dataclass(frozen=True)
class AgentRunResult:
    status: str
    rounds: int
    final_text: str = ""
    reason: str = ""
    tool_envelopes: tuple[ToolResultEnvelope, ...] = ()
    transcript: tuple[dict[str, Any], ...] = ()
    pending_approval: ToolApprovalRequest | None = None
    observation_calls: int = 0
    observation_budget_rejections: int = 0
    completion: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    checkpoint: AgentRunCheckpoint | None = field(default=None, repr=False, compare=False)


@dataclass
class AgentRunCheckpoint:
    """Process-local executable state; never decode from model/JSON or auto-replay."""
    state: AgentState
    envelopes: list[ToolResultEnvelope] = field(default_factory=list)
    signature_attempts: dict[str, int] = field(default_factory=dict)
    successful_signatures: dict[str, tuple[str, int]] = field(default_factory=dict)
    duplicate_rejections: dict[str, int] = field(default_factory=dict)
    mutation_epoch: int = 0
    search_urls: set[str] = field(default_factory=set)
    search_stalled: int = 0
    progress_ledger: ProgressLedger = field(default_factory=ProgressLedger)
    slices: int = 0
    idle_plans: int = 0
    last_tool_count: int = 0
    rejected_finals: int = 0


def _call_signature(invocation: ToolInvocation) -> str:
    try:
        args = json.dumps(invocation.arguments or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        args = repr(invocation.arguments)
    return f"{str(invocation.name or '').strip()}:{args}"


class _RunInterrupted(Exception):
    def __init__(self, state, envelopes, reason):
        self.state, self.envelopes, self.reason = state, envelopes, reason


class AgentLoop:
    """执行一个有上限、可恢复的 Agent 任务循环。"""

    def __init__(
        self,
        *,
        planner: Planner,
        handlers: Mapping[str, ToolHandler] | None,
        max_rounds: int = 6,
        policy: ExecutionPolicy | None = None,
        completion_gate: CompletionGate | None = None,
        approval_gate: ApprovalGate | None = None,
        workspace: Any = None,
        max_consecutive_observations: int = 4,
        max_source_discovery_calls: int = 4,
        state_observer: Callable[[AgentState, str], None] | None = None,
        interruption_check: Callable[[AgentState], str] | None = None,
        sliced: bool = False,
    ) -> None:
        self.planner = planner
        self.handlers = handlers or {}
        self.max_rounds = max(1, min(AGENT_MAX_ROUNDS_HARD_LIMIT, int(max_rounds or 6)))
        self.policy = policy
        self.completion_gate = completion_gate
        self.approval_gate = approval_gate
        self.workspace = workspace
        self.state_observer = state_observer
        self.interruption_check = interruption_check
        self.sliced = bool(sliced)
        self._checkpoint: AgentRunCheckpoint | None = None
        self.max_consecutive_observations = max(1, min(AGENT_MAX_ROUNDS_HARD_LIMIT, int(max_consecutive_observations or 4)))
        self.max_source_discovery_calls = max(1, min(128, int(max_source_discovery_calls)))

    def run(
        self,
        *,
        task_id: str = "",
        context: Any = None,
        initial_transcript: list[dict[str, Any]] | None = None,
        first_decision: AgentDecision | Mapping[str, Any] | None = None,
        approved_invocation: ToolInvocation | None = None,
        approved_invocations: Sequence[ToolInvocation] | None = None,
        checkpoint: AgentRunCheckpoint | None = None,
    ) -> AgentRunResult:
        self._checkpoint = checkpoint if self.sliced else None
        try:
            return self._run(
                task_id=task_id, context=context, initial_transcript=initial_transcript,
                first_decision=first_decision, approved_invocation=approved_invocation,
                approved_invocations=approved_invocations,
            )
        except _RunInterrupted as exc:
            return self._result(
                exc.state, exc.envelopes,
                status="canceled" if exc.reason == "task_canceled" else "blocked",
                reason=exc.reason,
            )

    def _check_interruption(self, state: AgentState, envelopes: list[ToolResultEnvelope]) -> None:
        if self.interruption_check is None:
            return
        try:
            reason = self.interruption_check(state)
        except Exception:
            reason = "task_interruption_check_failed"
        if reason:
            allowed = {"task_canceled", "agent_disabled", "agent_permission_changed",
                       "responses_disabled", "task_time_budget_exhausted", "agent_authority_revoked",
                       "task_model_budget_exhausted", "task_token_budget_exhausted", "task_tool_budget_exhausted",
                       "task_budget_snapshot_invalid"}
            raise _RunInterrupted(state, envelopes, reason if reason in allowed else "task_interruption_check_failed")

    def _run(
        self,
        *,
        task_id: str = "",
        context: Any = None,
        initial_transcript: list[dict[str, Any]] | None = None,
        first_decision: AgentDecision | Mapping[str, Any] | None = None,
        approved_invocation: ToolInvocation | None = None,
        approved_invocations: Sequence[ToolInvocation] | None = None,
    ) -> AgentRunResult:
        resumed = self._checkpoint is not None
        state = self._checkpoint.state if resumed else AgentState(task_id=str(task_id or "").strip(), context=context)
        if resumed:
            context = context if context is not None else state.context
            state.context = context
        state.plain_text_completion_verifiable = bool(
            getattr(self.completion_gate, "supports_plain_text_completion_verification", False)
            and getattr(self.completion_gate, "enabled", False)
        )
        if not resumed:
            state.research.configure(self.handlers, context)
            state.local_answer.configure(context)
            state.history_answer.configure(context)
            state.transcript.extend(dict(item) for item in (initial_transcript or []) if isinstance(item, Mapping))
        if not resumed and isinstance(context, Mapping):
            restored = context.get("agent_restored_tool_calls", ())
            state.tool_calls.extend(dict(item) for item in restored if isinstance(item, Mapping))
        if self.sliced and not resumed:
            self._checkpoint = AgentRunCheckpoint(state)
        saved = self._checkpoint
        if saved:
            saved.slices += 1
        envelopes: list[ToolResultEnvelope] = saved.envelopes if saved else []
        self._check_interruption(state, envelopes)
        signature_attempts: dict[str, int] = saved.signature_attempts if saved else {}
        successful_signatures: dict[str, tuple[str, int]] = saved.successful_signatures if saved else {}
        duplicate_rejections: dict[str, int] = saved.duplicate_rejections if saved else {}
        mutation_epoch = saved.mutation_epoch if saved else 0
        search_urls: set[str] = saved.search_urls if saved else set()
        search_stalled = saved.search_stalled if saved else 0
        prior = context.get("agent_task_state", {}).get("progress", {}).get("fingerprints", []) if isinstance(context, Mapping) else []
        progress_ledger = saved.progress_ledger if resumed else ProgressLedger(prior)
        if saved:
            saved.progress_ledger = progress_ledger
        round_end = state.rounds + self.max_rounds if self.sliced else self.max_rounds

        def note_execution(invocation: ToolInvocation, envelope: ToolResultEnvelope) -> None:
            nonlocal mutation_epoch, search_stalled
            handler = self.handlers.get(invocation.name)
            prior_candidates = set(state.research.candidates)
            state.research.observe(invocation, envelope, handler)
            state.local_answer.observe(invocation, envelope, handler, state.context)
            state.history_answer.observe(envelope, handler)
            evidence_field = getattr(self.handlers.get(invocation.name), "agent_evidence_collection_field", "")
            if evidence_field and self._progress_class(invocation) == "observe":
                state.source_discovery_calls += 1
                if state.source_discovery_calls >= self.max_source_discovery_calls:
                    state.suppressed_tool_names.update(name for name in self.handlers if self._is_source_discovery(name))
                    state.transcript.append({"role": "system", "content":
                        "本轮资料发现额度已用完，但正文读取和其他核验工具仍可用。请评估已有候选来源；有相关来源就读取必要正文，没有相关来源则明确说明缺口，不再扩搜。"})
                rows = (envelope.data or {}).get("state_updates", {}).get(evidence_field, [])
                urls = {str(row.get("url")) for row in rows if isinstance(row, Mapping) and row.get("url")}
                search_stalled = 0 if urls - search_urls else search_stalled + 1
                search_urls.update(urls)
                if search_stalled >= 2 and search_urls:
                    if state.research.enabled:
                        state.suppressed_tool_names.update(name for name in self.handlers if self._is_source_discovery(name))
                        state.transcript.append({"role": "system", "content": "连续扩搜没有新增候选，暂停发现工具；已有关联候选可继续读正文，不能把发现停滞当成正文阅读也已结束。"})
                    else:
                        state.finalization_only = True
            event = progress_ledger.observe(invocation, envelope, progress_class=self._progress_class(invocation),
                candidate_only=bool(getattr(handler, "agent_research_candidate_field", "")))
            if event.get("stage") == "candidate_discovery":
                event["new_candidates"] = bool(set(state.research.candidates) - prior_candidates)
            state.progress = {**state.progress, "stalled": progress_ledger.stalled, "last_event": event,
                              "fingerprints": list(progress_ledger.seen)[-128:],
                              "source_discovery_calls": state.source_discovery_calls,
                              "source_discovery_limit": self.max_source_discovery_calls}
            state.consecutive_observation_calls = progress_ledger.stalled
            if event["obstacle"]:
                state.transcript.append({"role": "system", "content":
                    f"宿主诊断分类：{event['obstacle']}。依据实际错误选择替代方案；分类不是授权，修复环境不等于目标站可用。"})
            self._observe_state(state, "running")
            signature = _call_signature(invocation)
            duplicate_rejections.pop(signature, None)
            if envelope.status == "ok":
                # The three-attempt cap is for failures, not successful polling
                # with fresh results. Duplicate/no-progress guards still apply.
                signature_attempts[signature] = 0
                progress_class = self._progress_class(invocation)
                if event["verified_change"]:
                    mutation_epoch += 1
                successful_signatures[signature] = (progress_class, mutation_epoch)
            else:
                successful_signatures.pop(signature, None)
            if saved:
                saved.mutation_epoch, saved.search_stalled = mutation_epoch, search_stalled

        approved_calls = tuple(
            item
            for item in (
                approved_invocations
                if approved_invocations is not None
                else ((approved_invocation,) if approved_invocation is not None else ())
            )
            if isinstance(item, ToolInvocation)
        )

        # 用户已经在上一轮批准了确定的调用时，恢复这些调用本身，
        # 而不是让模型重新猜一次工具，避免重复调用或换错工具。
        if approved_calls:
            if len(approved_calls) + len(state.tool_calls) > 128:
                return self._result(state, envelopes, status="blocked", reason="tool_budget_exhausted")
            if state.rounds >= round_end:
                return self._result(state, envelopes, status="failed", reason="max_rounds")
            state.rounds += 1
            state.transcript.append(
                {
                    "role": "assistant",
                    "kind": "tool",
                    "content": "执行用户刚刚明确授权的工具调用。",
                    "tool_calls": [
                        {
                            "id": item.id,
                            "name": item.name,
                            "arguments": dict(item.arguments or {}),
                        }
                        for item in approved_calls
                    ],
                }
            )
            for approved in approved_calls:
                signature = _call_signature(approved)
                signature_attempts[signature] = signature_attempts.get(signature, 0) + 1
                note_execution(
                    approved,
                    self._execute_and_record(state, envelopes, approved, context=context, approved=True),
                )

        pending_first_decision = first_decision
        while state.rounds < round_end or (not self.sliced and state.rounds == round_end and search_urls):
            self._check_interruption(state, envelopes)
            if saved:
                saved.idle_plans = saved.idle_plans + 1 if state.rounds and len(state.tool_calls) == saved.last_tool_count else 0
                saved.last_tool_count = len(state.tool_calls)
                if saved.idle_plans >= 6:
                    return self._result(state, envelopes, status="blocked", reason="no_progress")
            if not self.sliced and state.rounds >= self.max_rounds:
                state.finalization_only = True
            if state.finalization_only:
                state.transcript.append({"role": "system", "content": "本轮发现预算已到或连续两次未新增来源。本轮只能 final，不允许工具；依据已有资料给部分结果，注明尚未核实的信息及遗漏，不能宣称完整核验。"})
            elif not self.sliced and state.rounds >= self.max_rounds - 1:
                state.transcript.append({"role": "system", "content": "接近规划上限，请优先核验已有来源并整理回复，不要继续换同义关键词扩搜。按原始任务交付所需信息及来源；缺少的证据明确标注，不把搜索摘要当作已核实事实。"})
            state.rounds += 1
            if pending_first_decision is not None:
                # A chat-first router may already have asked the model whether a
                # tool is needed. Reuse that exact decision so a second planner
                # call cannot silently change the selected tool or arguments.
                raw_decision = pending_first_decision
                pending_first_decision = None
            else:
                try:
                    raw_decision = self.planner(state=state)
                except Exception as exc:
                    self._check_interruption(state, envelopes)
                    if state.finalization_only:
                        return self._research_partial(state, envelopes)
                    return self._result(state, envelopes, status="failed", reason=f"planner_error:{type(exc).__name__}")
            self._check_interruption(state, envelopes)
            decision = AgentDecision.from_value(raw_decision)
            if decision is None:
                if state.finalization_only:
                    return self._research_partial(state, envelopes)
                return self._result(state, envelopes, status="failed", reason="invalid_decision")
            if state.finalization_only:
                if decision.kind != "final":
                    return self._research_partial(state, envelopes)
                # Research has its own evidence/coverage gate. Do not turn an
                # honest failed/no-answer closeout into an invalid empty partial.
                if not state.research.enabled:
                    decision = replace(decision, outcome="partial")
            if isinstance(context, dict) and isinstance(context.get("agent_task_state"), dict):
                from .task_state import TaskContractError, extend_contract, update_notes

                task = context["agent_task_state"]
                try:
                    if decision.task_contract is not None:
                        old_contract = task.get("contract")
                        if old_contract is None:
                            raise TaskContractError("contract_not_initialized_by_host")
                        task["contract"] = extend_contract(old_contract, decision.task_contract)
                        context["agent_completion_contract"] = task["contract"]
                    update_notes(task, decision.task_notes)
                except TaskContractError as exc:
                    state.transcript.append({"role": "system", "content": f"任务状态校验失败：{exc}；保留原始目标和条件，请修正内部字段。"})
                    continue
                self._observe_state(state, "planning")
            if decision.kind == "final":
                if not decision.text:
                    state.transcript.append({"role": "system", "content": "完成回复不能为空，请补充实际结果或说明阻塞原因。"})
                    continue
                state.last_final_text = str(decision.text or "").strip()
                research_verdict = state.research.evaluate_answer(decision.text, decision.outcome)
                if not research_verdict.accepted:
                    state.completion = research_verdict.as_dict()
                    if state.finalization_only:
                        return self._research_partial(state, envelopes)
                    if state.research.sources and (
                        state.research.answer_review_calls >= 2
                        or "invalid_semantic_review" in (research_verdict.reason or "")
                        or (not self.sliced and state.rounds >= self.max_rounds - 1)
                    ):
                        return self._result(
                            state,
                            envelopes,
                            status="partial" if decision.outcome != "failed" else "failed",
                            reason="research_evidence_partial_fallback",
                            final_text=decision.text,
                        )
                    state.transcript.append({"role": "system", "content": research_verdict.reason})
                    continue
                terminal_status = decision.outcome or "completed"
                terminal_reason = ""
                if terminal_status not in {"completed", "partial", "failed", "blocked", "waiting_background"}:
                    state.transcript.append({"role": "system", "content": "任务结果状态无效，请使用 completed/partial/failed/blocked/waiting_background。"})
                    continue
                if self.completion_gate is not None:
                    try:
                        evaluator = getattr(self.completion_gate, "evaluate", None)
                        gate_eval = (
                            evaluator(state, decision.text, outcome=decision.outcome)
                            if callable(evaluator)
                            else self.completion_gate(state, decision.text)
                        )
                    except Exception as exc:
                        return self._result(
                            state,
                            envelopes,
                            status="failed",
                            reason=f"completion_gate_error:{type(exc).__name__}",
                            final_text="完成核验暂时出错，我不能确认任务已经完成。",
                        )
                    passed = True
                    reject_reason = ""
                    if isinstance(gate_eval, CompletionVerdict):
                        passed = gate_eval.accepted
                        reject_reason = gate_eval.reason
                        state.completion = gate_eval.as_dict()
                        if passed:
                            terminal_status = gate_eval.status
                            terminal_reason = gate_eval.reason
                    elif isinstance(gate_eval, tuple):
                        passed = bool(gate_eval[0])
                        reject_reason = str(gate_eval[1] or "").strip() if len(gate_eval) > 1 else ""
                    elif isinstance(gate_eval, str):
                        if gate_eval.strip():
                            passed = False
                            reject_reason = gate_eval.strip()
                        else:
                            passed = True
                    elif gate_eval is False:
                        passed = False
                    elif gate_eval is not True:
                        passed = False
                        reject_reason = "完成验收器返回了未知格式，不能确认任务成功。"

                    if not passed:
                        if saved:
                            saved.rejected_finals += 1
                            if saved.rejected_finals >= 3:
                                return self._result(state, envelopes, status="failed", reason="completion_no_progress")
                        if state.finalization_only:
                            return self._research_partial(state, envelopes)
                        if reject_reason:
                            msg = reject_reason if "完成验收未通过" in reject_reason else f"完成验收未通过：{reject_reason}"
                        else:
                            msg = "完成验收未通过，请检查任务结果后再汇报。"
                        state.transcript.append({"role": "system", "content": msg})
                        if isinstance(gate_eval, CompletionVerdict):
                            state.transcript.append({"role": "system", "content":
                                f"保留已满足条件 {list(gate_eval.satisfied)}；仅补核验缺失条件 {list(gate_eval.missing)}，不要重做全部任务。"})
                        continue
                if terminal_status not in {"completed", "partial", "failed", "blocked", "waiting_background"}:
                    return self._result(state, envelopes, status="failed", reason="invalid_completion_status")
                return self._result(state, envelopes, status=terminal_status, reason=terminal_reason, final_text=decision.text)
            if decision.kind == "wait":
                if not decision.text:
                    return self._result(state, envelopes, status="failed", reason="empty_wait_message")
                return self._result(state, envelopes, status="waiting_user", final_text=decision.text, reason=decision.reason)
            invocations = decision.invocations or ((decision.invocation,) if decision.invocation is not None else ())
            if not invocations:
                return self._result(state, envelopes, status="failed", reason="missing_invocation")

            # 记录助理发起的工具调用，保持完整的时序上下文 (user -> assistant -> tool)
            assistant_content = str(decision.text or "").strip()
            if not assistant_content:
                call_descs = [
                    f"调用工具 `{inv.name}` 参数={json.dumps(inv.arguments or {}, ensure_ascii=False)}"
                    for inv in invocations
                ]
                assistant_content = "；".join(call_descs)
            state.transcript.append(
                {
                    "role": "assistant",
                    "kind": "tool",
                    "content": assistant_content,
                    "tool_calls": [
                        {
                            "id": inv.id,
                            "name": inv.name,
                            "arguments": dict(inv.arguments or {}),
                        }
                        for inv in invocations
                    ],
                }
            )

            ready_calls: list[ToolInvocation] = []
            round_signatures: set[str] = set()
            discovery_slots = max(0, self.max_source_discovery_calls - state.source_discovery_calls)
            for invocation in invocations:
                if self._is_source_discovery(invocation.name):
                    read_required = state.research.enabled and state.research.read_before_search()
                    suppressed = invocation.name in state.suppressed_tool_names
                    if discovery_slots == 0 or read_required or suppressed:
                        message = ("已有值得阅读的关联候选；本次扩搜未执行，先读必要正文或说明候选无法访问。" if read_required
                                   else "本轮发现已暂停或额度已用完；该调用未执行。仍可读取已有来源或如实总结资料缺口。")
                        code = "source_read_required" if read_required else "source_discovery_budget_exhausted"
                        envelope = ToolResultEnvelope(invocation.id, "error", message, {"code": code})
                        envelopes.append(envelope)
                        state.tool_calls.append({"name": invocation.name, "id": invocation.id, "arguments": dict(invocation.arguments), "status": "error", "executed": False, "data": envelope.data})
                        state.transcript.append({"role": "tool", "name": invocation.name, "invocation_id": invocation.id, "status": "error", "content": message,
                                                 "data": envelope.data})
                        continue
                    discovery_slots -= 1
                signature = _call_signature(invocation)
                previous_success = successful_signatures.get(signature)
                read_after_change = bool(
                    previous_success
                    and previous_success[0] == "observe"
                    and previous_success[1] < mutation_epoch
                )
                handler = self.handlers.get(invocation.name)
                repeatable_read = bool(
                    self._progress_class(invocation) == "observe"
                    and getattr(handler, "agent_repeatable_observation", False)
                )
                # Background observations may change independently of this
                # Agent's mutation epoch. Existing attempt/read budgets still apply.
                duplicate = signature in round_signatures or bool(previous_success and not read_after_change and not repeatable_read)
                if duplicate:
                    envelope = ToolResultEnvelope(
                        invocation_id=invocation.id,
                        status="error",
                        model_feedback="<tool_use_error>相同工具调用已经执行过，请使用上一次结果继续，或换一种方法。</tool_use_error>",
                        data={"code": "duplicate_tool_call", "tool": invocation.name},
                    )
                    envelopes.append(envelope)
                    state.last_envelope = envelope
                    state.tool_calls.append(
                        {"name": invocation.name, "id": invocation.id, "arguments": dict(invocation.arguments), "status": "error"}
                    )
                    state.transcript.append(
                        {"role": "tool", "name": invocation.name, "status": "error", "content": envelope.model_feedback}
                    )
                    duplicate_rejections[signature] = duplicate_rejections.get(signature, 0) + 1
                    if duplicate_rejections[signature] >= 2:
                        return self._result(state, envelopes, status="blocked", reason="repeated_tool_call")
                    continue
                if signature_attempts.get(signature, 0) >= 3:
                    return self._result(state, envelopes, status="blocked", reason="repeated_tool_call")
                signature_attempts[signature] = signature_attempts.get(signature, 0) + 1
                round_signatures.add(signature)
                validation = validate_invocation(
                    invocation,
                    self.handlers,
                    policy=self.policy,
                    context=context,
                )
                if validation.ok:
                    progress_class = self._progress_class(invocation)
                    if progress_class == "observe":
                        state.observation_calls += 1
                    ready_calls.append(invocation)
                else:
                    note_execution(invocation, self._execute_and_record(state, envelopes, invocation, context=context))

            # 先完成本批次的无副作用校验，再一次性请求授权，避免模型连续
            if len(ready_calls) + len(state.tool_calls) > 128:
                return self._result(state, envelopes, status="blocked", reason="tool_budget_exhausted",
                                    final_text="本轮已达到工具调用预算，尚未完成的部分先保留，不继续执行。")
            # 规划多个工具时出现“批准一个、漏掉其余”的半执行状态。默认策略
            # 对未知工具按 medium 处理；只有明确标为不需授权的低风险批次才自动执行。
            approval_calls = list(ready_calls)
            if self.approval_gate is None and self.policy is not None:
                approval_calls = [item for item in ready_calls if self.policy.requires_approval(item.name)]
                if approval_calls:
                    return self._result(
                        state,
                        envelopes,
                        status="failed",
                        reason="approval_gate_unavailable",
                        final_text="这项工具操作需要授权，但当前宿主没有可用的授权通道，因此我没有执行。",
                    )
            if ready_calls and self.approval_gate is not None and self.policy is not None:
                if not any(self.policy.requires_approval(item.name) for item in ready_calls):
                    approval_calls = []
            if approval_calls and self.approval_gate is not None:
                try:
                    try:
                        # 混合批次只要含有一个需要授权的调用，就整体等待授权，避免
                        # 用户批准前先执行同一批里的另一项副作用。
                        approval = self.approval_gate(invocations=tuple(ready_calls), state=state, context=context)
                    except TypeError:
                        if len(ready_calls) != 1:
                            raise
                        approval = self.approval_gate(invocation=ready_calls[0], state=state, context=context)
                except Exception as exc:
                    return self._result(state, envelopes, status="failed", reason=f"approval_gate_error:{type(exc).__name__}")
                if approval is not None:
                    state.transcript.append(
                        {
                            "role": "system",
                            "content": f"工具调用等待用户授权：{approval.question}",
                            "approval_id": approval.request_id,
                        }
                    )
                    return self._result(
                        state,
                        envelopes,
                        status="waiting_user",
                        final_text=approval.question,
                        reason="tool_approval_required",
                        pending_approval=approval,
                    )

            for invocation in ready_calls:
                if state.finalization_only:
                    break
                note_execution(invocation, self._execute_and_record(state, envelopes, invocation, context=context))
                if saved and state.progress.get("last_event", {}).get("progress"):
                    saved.rejected_finals = 0
                if progress_ledger.stalled == self.max_consecutive_observations:
                    state.transcript.append({"role": "system", "content": "连续调用没有增加证据或推进条件，请换策略或说明阻塞原因；不要为重置预算而做无关修改。"})
                    state.observation_budget_rejections += 1
                if progress_ledger.stalled >= self.max_consecutive_observations + 2:
                    return self._result(state, envelopes, status="blocked", reason="no_progress",
                                        final_text="连续尝试没有增加证据或推进任务，我先停在这里；这次还没有完成。")

        if search_urls:
            if self.sliced and not state.finalization_only:
                return self._result(state, envelopes, status="yielded", reason="slice_complete")
            return self._research_partial(state, envelopes)
        if self.sliced and not state.finalization_only:
            return self._result(state, envelopes, status="yielded", reason="slice_complete")
        if state.last_final_text and getattr(state.research, "sources", None):
            return self._result(
                state,
                envelopes,
                status="partial",
                reason="max_rounds_partial",
                final_text=state.last_final_text,
            )
        if state.observation_budget_rejections:
            final_text = "这轮已经到达规划上限，我仍没能完成后续操作或核验；所以不能算完成。"
        else:
            final_text = "这轮已经到达规划上限，我没能完成后续核验和整理；不会把未完成的结果说成成功。"
        return self._result(state, envelopes, status="failed", reason="max_rounds", final_text=final_text)

    def _research_partial(self, state, envelopes):
        if state.research.enabled:
            if state.research.sources and state.last_final_text:
                return self._result(
                    state,
                    envelopes,
                    status="partial",
                    reason="research_partial_findings",
                    final_text=state.last_final_text,
                )
            fields = state.research.required_fields(state.context)
            text = "本轮尚未交付经过正文核验的答案，原任务没有完成；搜索候选不等于已核实事实。"
            if fields:
                text += "\n尚未完成的回答字段：" + "、".join(fields) + "。"
            return self._result(state, envelopes, status="failed", reason="research_answer_unverified", final_text=text)
        sources = {}
        for call in state.tool_calls:
            evidence_field = getattr(self.handlers.get(call.get("name")), "agent_evidence_collection_field", "")
            if not evidence_field or call.get("status") != "ok":
                continue
            for row in call.get("data", {}).get("state_updates", {}).get(evidence_field, []):
                if isinstance(row, Mapping) and row.get("url"):
                    sources[str(row["url"])] = str(row.get("title") or "来源")
        text = "已取得部分候选来源，但尚未完成原问题的回答与核实；以下仅为待评估线索，不代表答案已确认：\n" + "\n".join(f"{title}：{url}" for url, title in list(sources.items())[:10])
        return self._result(state, envelopes, status="partial", reason="research_budget_closeout", final_text=text)

    def _is_source_discovery(self, name: str) -> bool:
        return bool(getattr(self.handlers.get(name), "agent_evidence_collection_field", "")) and self._progress_class(ToolInvocation(name, {})) == "observe"

    def _progress_class(self, invocation: ToolInvocation) -> str:
        handler = self.handlers.get(str(invocation.name or "").strip())
        classifier = getattr(handler, "agent_progress_class_for_call", None)
        if callable(classifier):
            try:
                value = str(classifier(dict(invocation.arguments or {})) or "").strip().lower()
            except Exception:
                value = "observe"
            if value in {"observe", "change"}:
                return value
        value = str(getattr(handler, "agent_progress_class", "") or "").strip().lower()
        # Unknown tools are conservatively bounded as observations. Handlers that
        # mutate state should declare ``change`` (or classify each call).
        return value if value in {"observe", "change"} else "observe"

    def _execute_and_record(
        self,
        state: AgentState,
        envelopes: list[ToolResultEnvelope],
        invocation: ToolInvocation,
        *,
        context: Any,
        approved: bool = False,
    ) -> ToolResultEnvelope:
        self._check_interruption(state, envelopes)
        from .budget import current_task_budget, TaskBudgetExceeded
        budget = current_task_budget()
        if budget is not None:
            try:
                budget.use_tool()
            except TaskBudgetExceeded:
                raise _RunInterrupted(state, envelopes, budget.reason)
        state.inflight_tool_calls = [{"name": invocation.name, "id": invocation.id,
                                      "arguments": dict(invocation.arguments or {}), "status": "unknown"}]
        # Checkpoint BEFORE a side effect; a process exit during execution must
        # leave an unknown receipt, not look like a safely unexecuted operation.
        self._observe_state(state, "running")
        result, envelope = execute_invocation(
            invocation,
            handlers=self.handlers,
            context=context,
            policy=self.policy,
        )
        del result  # planner 只消费规范 envelope，避免偷偷依赖 handler 的内部返回类型。
        call_record = {
            "name": invocation.name,
            "id": invocation.id,
            "arguments": dict(invocation.arguments or {}),
            "status": envelope.status,
        }
        handler = self.handlers.get(invocation.name)
        research_kind = ("candidate" if getattr(handler, "agent_research_candidate_field", "")
                         else "source_read" if getattr(handler, "agent_research_content_field", "") else "")
        if research_kind:
            call_record["research_kind"] = research_kind
        updates = (envelope.data or {}).get("state_updates", {})
        observation_fields = getattr(self.handlers.get(invocation.name), "agent_observation_receipt_fields", ())
        if observation_fields and isinstance(updates, Mapping):
            call_record["data"] = {"state_updates": {key: updates[key] for key in observation_fields if key in updates}}
        if isinstance(updates, Mapping) and "verified_effects" in updates:
            # Retain only compact delivery evidence, not a second copy of large
            # tool payloads in the model's action history.
            call_record["data"] = {"state_updates": {
                "verified_effects": updates["verified_effects"],
                "completion_targets": updates.get("completion_targets", []),
            }}
            for key in ("path", "bytes", "sha256", "http_status", "site_access_ok", "navigation_ok", "delivery_receipt", "result_valid", "background_task",
                        "content_selector", "content_total_chars", "content_offset", "content_next_offset", "content_truncated", "content_sha256",
                        "source_sha256", "source_path", "source_pages", "done_chunks", "chunk_count", "pdf_pages", "job_id", "translation_verified", "read_complete", "previous_sha256"):
                if key in updates:
                    call_record["data"]["state_updates"][key] = updates[key]
            if "page_text" in updates:
                call_record["data"]["state_updates"]["page_text"] = str(updates["page_text"])[:8000]
        if approved:
            call_record["approved"] = True
        state.tool_calls.append(call_record)
        envelopes.append(envelope)
        state.last_envelope = envelope
        tool_record = {
            "role": "tool",
            "name": invocation.name,
            "invocation_id": invocation.id,
            "status": envelope.status,
            "content": envelope.model_feedback,
            "data": dict(envelope.data or {}),
        }
        if research_kind:
            tool_record["research_kind"] = research_kind
        if approved:
            tool_record["approved"] = True
        state.transcript.append(tool_record)
        # Keep the newest output available for immediate analysis; older outputs
        # become explicit bounded excerpts, with compact executor receipts intact.
        for previous in state.transcript[:-1]:
            if previous.get("role") == "tool" and len(str(previous.get("content", ""))) > 1400:
                previous["content"] = str(previous["content"])[:1400] + "\n[宿主已压缩旧输出；需要完整内容时重新读取目标。]"
                old_data = previous.get("data", {})
                old_updates = old_data.get("state_updates", {}) if isinstance(old_data, Mapping) else {}
                previous["data"] = {"state_updates": {key: old_updates[key] for key in
                    ("verified_effects", "completion_targets", "background_task", "path", "bytes", "sha256", "http_status", "delivery_receipt",
                     "source_sha256", "source_path", "source_pages", "done_chunks", "chunk_count", "pdf_pages", "job_id", "translation_verified", "result_valid", "read_complete", "previous_sha256")
                    if key in old_updates}}
        state.inflight_tool_calls = []
        self._observe_state(state, "running")
        if self.workspace is not None and state.task_id:
            self._record_workspace_event(state, invocation, envelope)
        return envelope

    def _record_workspace_event(self, state: AgentState, invocation: ToolInvocation, envelope: ToolResultEnvelope) -> None:
        append = getattr(self.workspace, "append_event", None)
        if not callable(append):
            return
        try:
            append(
                task_id=state.task_id,
                event_type="tool_result",
                from_actor=f"tool:{invocation.name}",
                message=envelope.model_feedback,
                payload={"status": envelope.status, "data": dict(envelope.data or {}), "events": list(envelope.events)},
                status="handled" if envelope.status == "ok" else "pending",
            )
        except Exception:
            # 事件记录失败不能吞掉本轮工具结果；主循环仍能把结果交给 planner。
            return

    def _result(
        self,
        state: AgentState,
        envelopes: list[ToolResultEnvelope],
        *,
        status: str,
        final_text: str = "",
        reason: str = "",
        pending_approval: ToolApprovalRequest | None = None,
    ) -> AgentRunResult:
        if status not in {"canceled", "blocked"}:
            self._check_interruption(state, envelopes)
        self._observe_state(state, status)
        return AgentRunResult(
            status=status,
            rounds=state.rounds,
            final_text=str(final_text or ""),
            reason=str(reason or ""),
            tool_envelopes=tuple(envelopes),
            transcript=tuple(dict(item) for item in state.transcript),
            pending_approval=pending_approval,
            observation_calls=state.observation_calls,
            observation_budget_rejections=state.observation_budget_rejections,
            completion=dict(state.completion),
            progress=dict(state.progress),
            checkpoint=self._checkpoint if status in {"yielded", "waiting_user", "blocked"} else None,
        )

    def _observe_state(self, state: AgentState, status: str) -> None:
        if self.state_observer is not None:
            self.state_observer(state, status)
