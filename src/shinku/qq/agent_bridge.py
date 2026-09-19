"""QQ 回合到 AgentLoop/LLMRuntime 的注入式桥接。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from shinku.agent.loop import AgentRunResult, AgentState
from shinku.agent.planner import ChatRuntime, PromptBuilder
from shinku.hosts.tool_host import ToolHost
from shinku.tools.execution import ExecutionPolicy
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
    ) -> None:
        if not callable(getattr(runtime, "call_chat_json", None)):
            raise TypeError("runtime must implement call_chat_json")
        self.runtime = runtime
        self.system_prompt = str(system_prompt or "").strip()
        self.sender = sender
        self.tool_host = tool_host or ToolHost(ToolRegistry())
        self.allowed_tool_names = set(allowed_tool_names or set())
        self.build_user_prompt = build_user_prompt
        self.max_rounds = max(1, min(12, int(max_rounds or 6)))
        self.policy = policy
        self.completion_gate = completion_gate
        self.prompt_cache_key = str(prompt_cache_key or "")
        self.temperature = float(temperature)

    def __call__(self, turn: QQTurn) -> QQAgentReply:
        return self.handle_turn(turn)

    def handle_turn(self, turn: QQTurn) -> QQAgentReply:
        if not isinstance(turn, QQTurn):
            return QQAgentReply("failed", reason="invalid_qq_turn")
        images = self._images(turn)
        context = {"qq_turn": turn, "user_images": images}
        try:
            result = self.tool_host.run(
                runtime=self.runtime,
                system_prompt=self.system_prompt,
                task_id=self._task_id(turn),
                context=context,
                initial_transcript=self._transcript(turn),
                allowed_tool_names=self.allowed_tool_names,
                build_user_prompt=self.build_user_prompt,
                build_user_images=self._build_user_images,
                max_rounds=self.max_rounds,
                policy=self.policy,
                completion_gate=self.completion_gate,
                prompt_cache_key=self.prompt_cache_key,
                temperature=self.temperature,
            )
        except Exception as exc:
            return QQAgentReply("failed", reason=f"bridge_error:{type(exc).__name__}")

        final_text = str(result.final_text or "").strip()
        if result.status not in {"completed", "waiting_user"} or not final_text:
            return QQAgentReply(result.status, final_text=final_text, reason=result.reason)

        message = build_reply(
            turn.conversation_id,
            final_text,
            conversation_type=turn.conversation_type,
        )
        if self.sender is None:
            return QQAgentReply(
                result.status,
                final_text=final_text,
                sent=False,
                delivery_status="dry_run",
                reason="sender_not_configured",
            )
        try:
            delivery = self.sender(message)
        except Exception as exc:
            return QQAgentReply(
                result.status,
                final_text=final_text,
                sent=False,
                delivery_status="sender_error",
                reason=f"sender_error:{type(exc).__name__}",
            )
        if not isinstance(delivery, DeliveryResult):
            return QQAgentReply(
                result.status,
                final_text=final_text,
                sent=False,
                delivery_status="sender_invalid",
                reason="sender_must_return_delivery_result",
            )
        return QQAgentReply(
            result.status,
            final_text=final_text,
            sent=delivery.sent,
            delivery_status=delivery.status,
            reason=delivery.error,
            delivery=delivery,
        )

    @staticmethod
    def _task_id(turn: QQTurn) -> str:
        conversation = str(turn.conversation_id or "unknown").strip() or "unknown"
        message = str(turn.primary_message_id or "turn").strip() or "turn"
        return f"qq:{conversation}:{message}"

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
