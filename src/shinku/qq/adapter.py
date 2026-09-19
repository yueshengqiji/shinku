"""QQ 入站适配器的组合边界。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .attention import AttentionBatch, AttentionBatcher, AttentionDecision, AttentionPolicy
from .message import IncomingMessage
from .routing import ReplyRoutePolicy, RouteDecision

__all__ = ["QQIngressAdapter", "QQIngressResult"]


@dataclass(frozen=True)
class QQIngressResult:
    message: IncomingMessage
    route: RouteDecision
    attention: AttentionDecision
    closed_batch: AttentionBatch | None = None

    @property
    def scheduled(self) -> bool:
        return self.attention.should_schedule

    def as_dict(self) -> dict[str, Any]:
        return {
            "message": self.message.as_dict(),
            "route": self.route.as_dict(),
            "attention": self.attention.as_dict(),
            "closed_batch": self.closed_batch.as_dict() if self.closed_batch else None,
        }


class QQIngressAdapter:
    """将原始 QQ 事件组合成可交给 Agent 的批次。

    这里不导入 QQ SDK，也不发网络请求。真实账号适配器只需把收到的原始事件交给
    ``ingest``，再消费返回的 ``closed_batch`` 或显式调用 ``flush``。
    """

    def __init__(
        self,
        *,
        route_policy: ReplyRoutePolicy | None = None,
        attention_policy: AttentionPolicy | None = None,
        batcher: AttentionBatcher | None = None,
    ) -> None:
        self.route_policy = route_policy or ReplyRoutePolicy()
        self.attention_policy = attention_policy or AttentionPolicy()
        self.batcher = batcher or AttentionBatcher()

    def ingest(self, event: Mapping[str, Any], *, at: float) -> QQIngressResult:
        message = IncomingMessage.from_event(event)
        route = self.route_policy.decide(message)
        attention = self.attention_policy.evaluate(message, route)
        closed_batch = self.batcher.push(message, attention, at=at)
        return QQIngressResult(
            message=message,
            route=route,
            attention=attention,
            closed_batch=closed_batch,
        )

    def flush(self) -> AttentionBatch | None:
        return self.batcher.flush()
