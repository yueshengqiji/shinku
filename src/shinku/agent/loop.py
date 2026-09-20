"""有界的 Agent 规划-工具-结果循环。

循环内核不绑定 LLM SDK。调用方提供一个 planner，planner 读取 ``AgentState``
并返回最终文本、等待用户、或一次 ``ToolInvocation``。这样模型供应商只是规划
器适配器，工具宿主只是 handler 映射，循环本身可以离线测试。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from shinku.tools.execution import ExecutionPolicy, ToolHandler, execute_invocation, validate_invocation
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope

from .approval import ToolApprovalRequest

AGENT_MAX_ROUNDS_HARD_LIMIT = 64

__all__ = [
    "AGENT_MAX_ROUNDS_HARD_LIMIT",
    "AgentDecision",
    "AgentLoop",
    "AgentRunResult",
    "AgentState",
    "ApprovalGate",
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


@dataclass(frozen=True)
class AgentDecision:
    """规划器的一次决定。``kind`` 只允许 final / wait / tool。"""

    kind: str
    text: str = ""
    invocation: ToolInvocation | None = None
    invocations: tuple[ToolInvocation, ...] = ()
    reason: str = ""

    @classmethod
    def final(cls, text: str) -> "AgentDecision":
        return cls(kind="final", text=str(text or "").strip())

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
        function = raw_call.get("function")
        if isinstance(function, Mapping):
            merged = dict(function)
            merged.setdefault("id", raw_call.get("id", ""))
            merged.setdefault("source", raw_call.get("source", ""))
            raw_call = merged
        name = str(raw_call.get("name") or raw_call.get("type") or "").strip()
        raw_arguments = raw_call.get("arguments")
        if isinstance(raw_arguments, Mapping):
            arguments = dict(raw_arguments)
        elif isinstance(raw_arguments, str):
            try:
                decoded = json.loads(raw_arguments)
            except (TypeError, ValueError):
                decoded = {}
            arguments = dict(decoded) if isinstance(decoded, Mapping) else {}
        else:
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
        )

    @classmethod
    def from_value(cls, value: Any) -> "AgentDecision | None":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return None
        kind = str(value.get("kind") or value.get("action") or "").strip().lower()
        if not kind and ("tool_call" in value or "tool_calls" in value or "invocation" in value):
            kind = "tool"
        text = str(value.get("text") or value.get("speech") or value.get("message") or "").strip()
        if kind in {"final", "answer", "done", "complete"}:
            return cls.final(text)
        if kind in {"wait", "waiting_user", "ask"}:
            return cls.wait(text, reason=str(value.get("reason") or "").strip())
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
                return cls.tools(calls)
        raw_call = value.get("invocation") if isinstance(value.get("invocation"), Mapping) else value.get("tool_call")
        if kind in {"tool", "call_tool", "tool_call"} and isinstance(raw_call, Mapping):
            invocation = cls._invocation_from_mapping(raw_call)
            if invocation is not None:
                return cls.tool(invocation)
        return None


class Planner(Protocol):
    def __call__(self, *, state: AgentState) -> AgentDecision | Mapping[str, Any] | None: ...


CompletionGate = Callable[[AgentState, str], bool]
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


def _call_signature(invocation: ToolInvocation) -> str:
    try:
        args = json.dumps(invocation.arguments or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        args = repr(invocation.arguments)
    return f"{str(invocation.name or '').strip()}:{args}"


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
    ) -> None:
        self.planner = planner
        self.handlers = handlers or {}
        self.max_rounds = max(1, min(AGENT_MAX_ROUNDS_HARD_LIMIT, int(max_rounds or 6)))
        self.policy = policy
        self.completion_gate = completion_gate
        self.approval_gate = approval_gate
        self.workspace = workspace

    def run(
        self,
        *,
        task_id: str = "",
        context: Any = None,
        initial_transcript: list[dict[str, Any]] | None = None,
        approved_invocation: ToolInvocation | None = None,
        approved_invocations: Sequence[ToolInvocation] | None = None,
    ) -> AgentRunResult:
        state = AgentState(task_id=str(task_id or "").strip(), context=context)
        state.transcript.extend(dict(item) for item in (initial_transcript or []) if isinstance(item, Mapping))
        envelopes: list[ToolResultEnvelope] = []
        signatures: dict[str, int] = {}

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
            if state.rounds >= self.max_rounds:
                return self._result(state, envelopes, status="failed", reason="max_rounds")
            state.rounds += 1
            for approved in approved_calls:
                signature = _call_signature(approved)
                signatures[signature] = 1
                state.tool_calls.append(
                    {
                        "name": approved.name,
                        "id": approved.id,
                        "arguments": dict(approved.arguments),
                        "approved": True,
                    }
                )
                self._execute_and_record(state, envelopes, approved, context=context)

        while state.rounds < self.max_rounds:
            state.rounds += 1
            try:
                raw_decision = self.planner(state=state)
            except Exception as exc:
                return self._result(state, envelopes, status="failed", reason=f"planner_error:{type(exc).__name__}")
            decision = AgentDecision.from_value(raw_decision)
            if decision is None:
                return self._result(state, envelopes, status="failed", reason="invalid_decision")
            if decision.kind == "final":
                if not decision.text:
                    state.transcript.append({"role": "system", "content": "完成回复不能为空，请补充实际结果或说明阻塞原因。"})
                    continue
                if self.completion_gate is not None and not self.completion_gate(state, decision.text):
                    state.transcript.append({"role": "system", "content": "完成验收未通过，请检查任务结果后再汇报。"})
                    continue
                return self._result(state, envelopes, status="completed", final_text=decision.text)
            if decision.kind == "wait":
                if not decision.text:
                    return self._result(state, envelopes, status="failed", reason="empty_wait_message")
                return self._result(state, envelopes, status="waiting_user", final_text=decision.text, reason=decision.reason)
            invocations = decision.invocations or ((decision.invocation,) if decision.invocation is not None else ())
            if not invocations:
                return self._result(state, envelopes, status="failed", reason="missing_invocation")
            ready_calls: list[ToolInvocation] = []
            for invocation in invocations:
                signature = _call_signature(invocation)
                signatures[signature] = signatures.get(signature, 0) + 1
                if signatures[signature] > 1:
                    envelope = ToolResultEnvelope(
                        invocation_id=invocation.id,
                        status="error",
                        model_feedback="<tool_use_error>相同工具调用已经执行过，请使用上一次结果继续，或换一种方法。</tool_use_error>",
                        data={"code": "duplicate_tool_call", "tool": invocation.name},
                    )
                    envelopes.append(envelope)
                    state.last_envelope = envelope
                    state.transcript.append({"role": "tool", "name": invocation.name, "content": envelope.model_feedback})
                    if signatures[signature] >= 3:
                        return self._result(state, envelopes, status="blocked", reason="repeated_tool_call")
                    continue
                validation = validate_invocation(invocation, self.handlers, policy=self.policy)
                state.tool_calls.append({"name": invocation.name, "id": invocation.id, "arguments": dict(invocation.arguments)})
                if validation.ok:
                    ready_calls.append(invocation)
                else:
                    self._execute_and_record(state, envelopes, invocation, context=context)

            # 先完成本批次的无副作用校验，再一次性请求授权，避免模型连续
            # 规划多个工具时出现“批准一个、漏掉其余”的半执行状态。默认策略
            # 对未知工具按 medium 处理；只有明确标为不需授权的低风险批次才自动执行。
            approval_calls = list(ready_calls)
            if ready_calls and self.approval_gate is not None and self.policy is not None:
                if not any(self.policy.requires_approval(item.name) for item in ready_calls):
                    approval_calls = []
            if approval_calls and self.approval_gate is not None:
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
                self._execute_and_record(state, envelopes, invocation, context=context)

        return self._result(state, envelopes, status="failed", reason="max_rounds")

    def _execute_and_record(
        self,
        state: AgentState,
        envelopes: list[ToolResultEnvelope],
        invocation: ToolInvocation,
        *,
        context: Any,
    ) -> ToolResultEnvelope:
        result, envelope = execute_invocation(
            invocation,
            handlers=self.handlers,
            context=context,
            policy=self.policy,
        )
        del result  # planner 只消费规范 envelope，避免偷偷依赖 handler 的内部返回类型。
        envelopes.append(envelope)
        state.last_envelope = envelope
        state.transcript.append(
            {
                "role": "tool",
                "name": invocation.name,
                "invocation_id": invocation.id,
                "status": envelope.status,
                "content": envelope.model_feedback,
                "data": dict(envelope.data or {}),
            }
        )
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

    @staticmethod
    def _result(
        state: AgentState,
        envelopes: list[ToolResultEnvelope],
        *,
        status: str,
        final_text: str = "",
        reason: str = "",
        pending_approval: ToolApprovalRequest | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            status=status,
            rounds=state.rounds,
            final_text=str(final_text or ""),
            reason=str(reason or ""),
            tool_envelopes=tuple(envelopes),
            transcript=tuple(dict(item) for item in state.transcript),
            pending_approval=pending_approval,
        )
