"""有界的 Agent 规划-工具-结果循环。

循环内核不绑定 LLM SDK。调用方提供一个 planner，planner 读取 ``AgentState``
并返回最终文本、等待用户、或一次 ``ToolInvocation``。这样模型供应商只是规划
器适配器，工具宿主只是 handler 映射，循环本身可以离线测试。
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from shinku.conversation import AnswerEvidence
from shinku.tools.execution import (
    ExecutionPolicy,
    ToolHandler,
    execute_invocation,
)
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope

from .answer_protocol import ANSWER_EFFECT_ALIASES, normalize_answer_effect, local_failure_report
from .approval import ToolApprovalRequest
from .completion import CompletionVerdict
from .history_answer import HistoryAnswerLedger
from .local_answer import LocalAnswerLedger
from .loop_models import AgentLoopDependencies
from .loop_pipeline import run_agent_loop
from .progress import ProgressLedger
from .research import ResearchLedger

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
    # 仅宿主防循环收尾设置；与资料成稿分开，不能由模型元数据开启。
    tool_closeout_reason: str = ""
    source_discovery_calls: int = 0
    suppressed_tool_names: set[str] = field(default_factory=set)
    final_reply_repair_attempts: int = 0
    plain_text_completion_verifiable: bool = False
    research: ResearchLedger = field(default_factory=ResearchLedger)
    local_answer: LocalAnswerLedger = field(default_factory=LocalAnswerLedger)
    history_answer: HistoryAnswerLedger = field(default_factory=HistoryAnswerLedger)
    last_final_text: str = ""
    # Provider-only state. Omitted from host ledgers/results and preserved only
    # by the existing process-local slice checkpoint.
    native_tool_history: Any = field(default=None, repr=False, compare=False)


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
        if kind in ANSWER_EFFECT_ALIASES:
            normalized = normalize_answer_effect(value)
            if normalized is None:
                return None
            value, kind = normalized, "final"
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
    answer_evidence: AnswerEvidence = field(default_factory=AnswerEvidence)


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
                       "task_budget_snapshot_invalid", "task_budget_low"}
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
        return run_agent_loop(
            task_id=task_id, context=context, initial_transcript=initial_transcript,
            first_decision=first_decision, approved_invocation=approved_invocation,
            approved_invocations=approved_invocations, deps=self._loop_dependencies(),
        )

    def _loop_dependencies(self) -> AgentLoopDependencies:
        return AgentLoopDependencies(
            planner=lambda: self.planner,
            handlers=lambda: self.handlers,
            policy=lambda: self.policy,
            completion_gate=lambda: self.completion_gate,
            approval_gate=lambda: self.approval_gate,
            max_rounds=lambda: self.max_rounds,
            max_consecutive_observations=lambda: self.max_consecutive_observations,
            max_source_discovery_calls=lambda: self.max_source_discovery_calls,
            sliced=lambda: self.sliced,
            checkpoint=lambda: self._checkpoint,
            set_checkpoint=self._set_run_checkpoint,
            new_state=lambda **kwargs: AgentState(**kwargs),
            new_checkpoint=lambda state: AgentRunCheckpoint(state),
            parse_decision=lambda value: AgentDecision.from_value(value),
            call_signature=lambda invocation: _call_signature(invocation),
            result=lambda *args, **kwargs: self._result(*args, **kwargs),
            research_partial=lambda *args: self._research_partial(*args),
            check_interruption=lambda *args: self._check_interruption(*args),
            observe_state=lambda *args: self._observe_state(*args),
            progress_class=lambda invocation: self._progress_class(invocation),
            is_source_discovery=lambda name: self._is_source_discovery(name),
            execute_and_record=lambda *args, **kwargs: self._execute_and_record(*args, **kwargs),
        )

    def _set_run_checkpoint(self, checkpoint: AgentRunCheckpoint) -> None:
        self._checkpoint = checkpoint

    def _research_partial(self, state, envelopes, reason=""):
        # 本地分析不能借用研究文案或把协议错误称为预算耗尽；原完成门不放宽。
        if state.local_answer.enabled and not state.research.enabled:
            reason = reason or "local_analysis_incomplete"
            status, text = local_failure_report(state, self.handlers, reason)
            return self._result(state, envelopes, status=status, reason=reason, final_text=text)
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
        from .budget import TaskBudgetExceeded, current_task_budget
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
        material_session = context.get("agent_material_session") if isinstance(context, Mapping) else None
        material_reference = None
        if material_session is not None:
            try:
                material_reference = material_session.capture_tool_output(invocation, envelope)
            except Exception:
                # A capture failure cannot undo the real effect or authorize
                # another execution. The missing representation is explicit.
                material_reference = {"retrieval": "unavailable", "reason": "material_capture_failed"}
        call_record = {
            "name": invocation.name,
            "id": invocation.id,
            "arguments": dict(invocation.arguments or {}),
            "status": envelope.status,
            "call_fingerprint": hashlib.sha256(_call_signature(invocation).encode("utf-8")).hexdigest(),
        }
        if type(envelope.executed) is bool:
            call_record["executed"] = envelope.executed
        if envelope.executed is False and isinstance(envelope.data, Mapping):
            code = envelope.data.get("code")
            if code in {"tool_arguments_malformed", "missing_tool", "tool_blocked", "unknown_tool",
                        "invalid_handler", "bad_args", "validation_failed"}:
                call_record["data"] = {"code": code}
        task = context.get("agent_task_state", {}) if isinstance(context, Mapping) else {}
        if isinstance(task, Mapping) and type(task.get("execution_attempt")) is int:
            call_record["execution_binding"] = {
                "task_id": task.get("task_id", ""), "requester": task.get("requester", ""),
                "session": task.get("session", ""), "attempt": task["execution_attempt"],
                "work_id": str(context.get("agent_work_id") or ""),
                "work_attempt": context.get("agent_work_attempt", 0)}
        handler = self.handlers.get(invocation.name)
        if material_reference is not None:
            call_record["material_reference"] = material_reference
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
                        "source_sha256", "source_path", "source_pages", "done_chunks", "chunk_count", "pdf_pages", "job_id", "translation_verified", "read_complete", "previous_sha256", "execution_session", "artifact_receipts", "result_facts"):
                if key in updates:
                    call_record["data"]["state_updates"][key] = updates[key]
            if "page_text" in updates:
                call_record["data"]["state_updates"]["page_text"] = str(updates["page_text"])[:8000]
            # 具有效果回执的工具也可能需要宿主观察签名。只保留handler显式
            # 声明的字段，避免效果压缩丢掉凭据；不能从模型参数补造签名。
            for key in observation_fields:
                if key in updates:
                    call_record["data"]["state_updates"][key] = updates[key]
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
        if material_reference is not None:
            tool_record["material_reference"] = material_reference
        if research_kind:
            tool_record["research_kind"] = research_kind
        if approved:
            tool_record["approved"] = True
        state.transcript.append(tool_record)
        # Keep the newest output available for immediate analysis; older outputs
        # become explicit bounded excerpts, with compact executor receipts intact.
        for previous in state.transcript[:-1]:
            if previous.get("role") == "tool" and len(str(previous.get("content", ""))) > 1400:
                if material_session is not None:
                    full_chars = previous.get("feedback_information", {}).get("total_chars", len(str(previous["content"])))
                    previous["feedback_information"] = {"schema_version": 1, "total_chars": full_chars,
                        "retained_chars": 1400, "omitted_chars": max(0, full_chars - 1400),
                        "retrieval": "available" if previous.get("material_reference", {}).get("reference") else "unavailable"}
                    previous["content"] = str(previous["content"])[:1400] + "\n[旧输出已裁剪；仅通过绑定引用补读。]"
                else:
                    previous["content"] = str(previous["content"])[:1400] + "\n[宿主已压缩旧输出；需要完整内容时重新读取目标。]"
                old_data = previous.get("data", {})
                old_updates = old_data.get("state_updates", {}) if isinstance(old_data, Mapping) else {}
                previous["data"] = {"state_updates": {key: old_updates[key] for key in
                    ("verified_effects", "completion_targets", "background_task", "path", "bytes", "sha256", "http_status", "delivery_receipt",
                     "source_sha256", "source_path", "source_pages", "done_chunks", "chunk_count", "pdf_pages", "job_id", "translation_verified", "result_valid", "read_complete", "previous_sha256", "execution_session", "artifact_receipts")
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
        if status in {"completed", "partial", "failed", "blocked", "canceled"}:
            from .execution_sessions import stop_execution_sessions
            stop_execution_sessions(state, self.handlers, reason or status)
        self._observe_state(state, status)
        evidence = AnswerEvidence()
        if state.history_answer.enabled:
            evidence = AnswerEvidence.from_history(state.history_answer)
        elif state.local_answer.enabled:
            evidence = AnswerEvidence.from_source_ledger(state.local_answer, kind="local", query_count=len(state.tool_calls))
        elif state.research.enabled:
            evidence = AnswerEvidence.from_source_ledger(state.research, kind="public", query_count=len(state.tool_calls))
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
            answer_evidence=evidence.bound_to(str(final_text or "")),
        )

    def _observe_state(self, state: AgentState, status: str) -> None:
        if self.state_observer is not None:
            self.state_observer(state, status)
