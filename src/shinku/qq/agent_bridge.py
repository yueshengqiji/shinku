"""QQ 回合到 AgentLoop/LLMRuntime 的注入式桥接。"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from shinku.agent.approval import ConsentResolver, LLMConsentResolver, ToolApprovalRequest, make_tool_approval_request
from shinku.agent.loop import AgentState, AgentRunResult
from shinku.agent.loop import AGENT_MAX_ROUNDS_HARD_LIMIT
from shinku.agent.pending import InMemoryPendingApprovalStore, PendingApprovalRecord, PendingApprovalStore
from shinku.agent.planner import ChatRuntime, PromptBuilder
from shinku.hosts.tool_host import ToolHost
from shinku.memory import MemoryService
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.invocation import ToolInvocation
from shinku.tools.registry import ToolRegistry

from .delivery import DeliveryResult, OutgoingMessage, build_reply
from .turns import QQTurn

__all__ = ["QQAgentBridge", "QQAgentReply", "QQMessageSender"]


class QQMessageSender(Protocol):
    def __call__(self, message: OutgoingMessage) -> DeliveryResult: ...


@dataclass(frozen=True)
class QQAgentReply:
    """一轮 Agent 处理及可选出站投递的结果。"""

    agent_status: str
    final_text: str = ""
    sent: bool = False
    delivery_status: str = "not_sent"
    reason: str = ""
    delivery: DeliveryResult | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_status": self.agent_status,
            "final_text": self.final_text,
            "sent": self.sent,
            "delivery_status": self.delivery_status,
            "reason": self.reason,
            "delivery": self.delivery.as_dict() if self.delivery is not None else None,
        }


class QQAgentBridge:
    """把一个已经合并的 QQTurn 交给 AgentLoop，并可选投递最终文本。

    角色卡、记忆系统和工具注册表都从外部注入。桥接层不读取旧项目配置、不猜供应商，
    也不会因为模型失败而生成一条伪造的成功回复。
    """

    def __init__(
        self,
        *,
        runtime: ChatRuntime,
        system_prompt: str,
        sender: QQMessageSender | None = None,
        tool_host: ToolHost | None = None,
        allowed_tool_names: set[str] | None = None,
        build_user_prompt: PromptBuilder | None = None,
        max_rounds: int = 6,
        policy: ExecutionPolicy | None = None,
        completion_gate: Callable[[AgentState, str], bool] | None = None,
        prompt_cache_key: str = "",
        temperature: float = 0.7,
        history_turns: int = 12,
        recent_context_messages: int = 8,
        memory_service: MemoryService | None = None,
        consent_resolver: ConsentResolver | None = None,
        approval_ttl_seconds: float = 300.0,
        approval_store: PendingApprovalStore | None = None,
    ) -> None:
        if not callable(getattr(runtime, "call_chat_json", None)):
            raise TypeError("runtime must implement call_chat_json")
        self.runtime = runtime
        self.system_prompt = str(system_prompt or "").strip()
        self.sender = sender
        self.tool_host = tool_host or ToolHost(ToolRegistry())
        self.allowed_tool_names = set(allowed_tool_names or set())
        self.build_user_prompt = build_user_prompt
        self.max_rounds = max(1, min(AGENT_MAX_ROUNDS_HARD_LIMIT, int(max_rounds or 6)))
        self.policy = policy
        self.completion_gate = completion_gate
        self.prompt_cache_key = str(prompt_cache_key or "")
        self.temperature = float(temperature)
        self.history_turns = max(1, min(100, int(history_turns or 12)))
        self.recent_context_messages = max(1, min(50, int(recent_context_messages or 8)))
        self.memory_service = memory_service
        self.consent_resolver = consent_resolver or LLMConsentResolver(runtime)
        self.approval_ttl_seconds = max(30.0, float(approval_ttl_seconds or 300.0))
        self.approval_store = approval_store or InMemoryPendingApprovalStore()

    def __call__(self, turn: QQTurn) -> QQAgentReply:
        return self.handle_turn(turn)

    def handle_turn(self, turn: QQTurn) -> QQAgentReply:
        if not isinstance(turn, QQTurn):
            return QQAgentReply("failed", reason="invalid_qq_turn")
        images = self._images(turn)
        memory_context = None
        if self.memory_service is not None:
            try:
                memory_context = self.memory_service.prepare_turn(
                    conversation_id=turn.conversation_id,
                    conversation_type=turn.conversation_type,
                    user_message=turn.combined_text,
                    recent_context=self._recent_context(turn),
                )
                self.memory_service.record_turn(
                    conversation_id=turn.conversation_id,
                    conversation_type=turn.conversation_type,
                    message_id=turn.primary_message_id,
                    text=turn.combined_text,
                )
            except Exception:
                # Memory must never turn a valid QQ request into an Agent failure.
                memory_context = None
        context = {
            "qq_turn": turn,
            "user_images": images,
            "memory_context": memory_context.as_dict() if memory_context is not None else {},
        }

        # 上一轮已经选定了具体工具时，当前消息只负责语义授权；不重新走关键词
        # 路由，也不让模型重新猜一次工具，避免“先问是否执行、下一轮换了工具”。
        try:
            pending = self.approval_store.get(turn.conversation_id)
        except Exception as exc:
            return QQAgentReply("failed", reason=f"approval_store_error:{type(exc).__name__}")
        if pending is not None:
            if time.time() - pending.created_at > self.approval_ttl_seconds:
                try:
                    self.approval_store.delete(turn.conversation_id)
                except Exception as exc:
                    return QQAgentReply("failed", reason=f"approval_store_error:{type(exc).__name__}")
            else:
                decision = self.consent_resolver.resolve(
                    request=pending.request,
                    user_message=turn.combined_text,
                )
                if decision.verdict == "approve":
                    try:
                        self.approval_store.delete(turn.conversation_id)
                    except Exception as exc:
                        return QQAgentReply("failed", reason=f"approval_store_error:{type(exc).__name__}")
                    resumed_context = dict(pending.context)
                    resumed_context["qq_turn"] = turn
                    resumed_context["approval_response"] = turn.combined_text
                    resumed_transcript = list(pending.transcript)
                    resumed_transcript.append({"role": "user", "content": turn.combined_text})
                    resumed_transcript.append({"role": "system", "content": "用户已通过自然语言授权执行待处理工具。"})
                    try:
                        result = self._run_agent(
                            task_id=pending.task_id,
                            context=resumed_context,
                            initial_transcript=resumed_transcript,
                            approved_invocations=pending.request.invocations,
                        )
                    except Exception as exc:
                        return QQAgentReply("failed", reason=f"bridge_error:{type(exc).__name__}")
                    return self._finish_result(
                        turn,
                        result,
                        context=resumed_context,
                        initial_transcript=resumed_transcript,
                        task_id=pending.task_id,
                    )
                if decision.verdict == "deny":
                    try:
                        self.approval_store.delete(turn.conversation_id)
                    except Exception as exc:
                        return QQAgentReply("failed", reason=f"approval_store_error:{type(exc).__name__}")
                    return self._deliver_text(turn, "好，那就先不执行。", reason="tool_denied")
                return self._deliver_text(
                    turn,
                    "我还没判断出你是不是要执行这一步。要执行刚才那项工具吗？",
                    agent_status="waiting_user",
                    reason="tool_approval_unclear",
                )
        try:
            initial_transcript = self._transcript(turn)
            task_id = self._task_id(turn)
            result = self._run_agent(task_id=task_id, context=context, initial_transcript=initial_transcript)
        except Exception as exc:
            return QQAgentReply("failed", reason=f"bridge_error:{type(exc).__name__}")

        return self._finish_result(
            turn,
            result,
            context=context,
            initial_transcript=initial_transcript,
            task_id=task_id,
        )

    def _run_agent(
        self,
        *,
        task_id: str,
        context: dict[str, Any],
        initial_transcript: list[dict[str, Any]],
        approved_invocation: ToolInvocation | None = None,
        approved_invocations: Sequence[ToolInvocation] | None = None,
    ) -> AgentRunResult:
        return self.tool_host.run(
            runtime=self.runtime,
            system_prompt=self.system_prompt,
            task_id=task_id,
            context=context,
            initial_transcript=initial_transcript,
            allowed_tool_names=self.allowed_tool_names,
            build_user_prompt=self._build_prompt,
            build_user_images=self._build_user_images,
            max_rounds=self.max_rounds,
            policy=self.policy,
            completion_gate=self.completion_gate,
            approval_gate=self._approval_gate,
            approved_invocation=approved_invocation,
            approved_invocations=approved_invocations,
            prompt_cache_key=self.prompt_cache_key,
            temperature=self.temperature,
            history_limit=self.history_turns,
        )

    def _approval_gate(
        self,
        *,
        invocations: Sequence[ToolInvocation] | None = None,
        invocation: ToolInvocation | None = None,
        state: AgentState,
        context: Any,
    ) -> ToolApprovalRequest:
        del context
        calls = tuple(item for item in (invocations or ()) if isinstance(item, ToolInvocation))
        if not calls and isinstance(invocation, ToolInvocation):
            calls = (invocation,)
        risk_by_tool = {
            item.name: (self.policy.risk_of(item.name) if self.policy is not None else "medium")
            for item in calls
        }
        return make_tool_approval_request(
            invocation=invocation,
            invocations=invocations,
            task_id=state.task_id,
            risk_by_tool=risk_by_tool,
        )

    def _finish_result(
        self,
        turn: QQTurn,
        result: AgentRunResult,
        *,
        context: dict[str, Any],
        initial_transcript: list[dict[str, Any]],
        task_id: str,
    ) -> QQAgentReply:
        if result.pending_approval is not None:
            try:
                self.approval_store.put(
                    turn.conversation_id,
                    PendingApprovalRecord(
                        request=result.pending_approval,
                        task_id=task_id,
                        context=dict(context),
                        transcript=tuple(dict(item) for item in initial_transcript),
                        created_at=time.time(),
                    ),
                )
            except Exception as exc:
                return QQAgentReply("failed", reason=f"approval_store_error:{type(exc).__name__}")

        final_text = str(result.final_text or "").strip()
        if result.status not in {"completed", "waiting_user"} or not final_text:
            return QQAgentReply(result.status, final_text=final_text, reason=result.reason)

        return self._deliver_text(turn, final_text, agent_status=result.status, reason=result.reason)

    def _deliver_text(
        self,
        turn: QQTurn,
        final_text: str,
        *,
        agent_status: str = "completed",
        reason: str = "",
    ) -> QQAgentReply:
        final_text = str(final_text or "").strip()

        message = build_reply(
            turn.conversation_id,
            final_text,
            conversation_type=turn.conversation_type,
        )
        if self.sender is None:
            return QQAgentReply(
                agent_status,
                final_text=final_text,
                sent=False,
                delivery_status="dry_run",
                reason=reason or "sender_not_configured",
            )
        try:
            delivery = self.sender(message)
        except Exception as exc:
            return QQAgentReply(
                agent_status,
                final_text=final_text,
                sent=False,
                delivery_status="sender_error",
                reason=f"sender_error:{type(exc).__name__}",
            )
        if not isinstance(delivery, DeliveryResult):
            return QQAgentReply(
                agent_status,
                final_text=final_text,
                sent=False,
                delivery_status="sender_invalid",
                reason="sender_must_return_delivery_result",
            )
        return QQAgentReply(
            agent_status,
            final_text=final_text,
            sent=delivery.sent,
            delivery_status=delivery.status,
            reason=delivery.error or reason,
            delivery=delivery,
        )

    @staticmethod
    def _task_id(turn: QQTurn) -> str:
        conversation = str(turn.conversation_id or "unknown").strip() or "unknown"
        message = str(turn.primary_message_id or "turn").strip() or "turn"
        return f"qq:{conversation}:{message}"

    def _build_prompt(self, state: AgentState) -> str:
        """Keep memory as an audited side block, never as a persona rewrite."""

        if self.build_user_prompt is not None:
            base = str(self.build_user_prompt(state) or "")
        else:
            lines = [
                "根据当前任务继续处理。只做一件事：给出最终答复、等待用户，或选择一个工具。",
                "工具是否必要由任务意图和上下文决定，不要因为某个关键词出现就强行调用。",
            ]
            for item in state.transcript[-self.history_turns:]:
                if not isinstance(item, Mapping):
                    continue
                role = str(item.get("role") or "unknown").strip()
                content = str(item.get("content") or "").strip()
                if content:
                    lines.append(f"{role}: {content}")
            base = "\n".join(lines)
        context = state.context if isinstance(state.context, Mapping) else {}
        memory = context.get("memory_context")
        rendered = memory.get("rendered") if isinstance(memory, Mapping) else ""
        if rendered:
            return f"{base}\n\n{rendered}"
        return base

    def _recent_context(self, turn: QQTurn) -> str:
        lines: list[str] = []
        for message in turn.messages[-self.recent_context_messages:]:
            text = str(getattr(message, "text", "") or "").strip()
            if text:
                lines.append(text)
        return "\n".join(lines)

    @staticmethod
    def _transcript(turn: QQTurn) -> list[dict[str, str]]:
        transcript: list[dict[str, str]] = []
        for message in turn.messages:
            text = str(getattr(message, "text", "") or "").strip()
            if bool(getattr(message, "has_media", False)):
                text = f"{text}\n[图片附件已保留给视觉模型]".strip()
            if not text:
                text = "[未解析的消息附件]"
            transcript.append({"role": "user", "content": text})
        return transcript

    @staticmethod
    def _images(turn: QQTurn) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        for _, inputs in turn.visual_inputs:
            for item in inputs:
                model_item = item.as_model_item()
                if isinstance(model_item, Mapping):
                    images.append(dict(model_item))
        return images

    @staticmethod
    def _build_user_images(state: AgentState) -> list[dict[str, Any]]:
        context = state.context if isinstance(state.context, Mapping) else {}
        raw = context.get("user_images", [])
        return [dict(item) for item in raw if isinstance(item, Mapping)]
