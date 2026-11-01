"""Transport and turn-result contracts shared by the QQ bridge and delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .delivery import DeliveryResult, OutgoingMessage


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
    background_task_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_status": self.agent_status,
            "final_text": self.final_text,
            "sent": self.sent,
            "delivery_status": self.delivery_status,
            "reason": self.reason,
            "delivery": self.delivery.as_dict() if self.delivery is not None else None,
            "background_task_id": self.background_task_id,
        }
