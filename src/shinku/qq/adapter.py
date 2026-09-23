"""QQ 入站适配器的组合边界。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
from typing import Any, Mapping

from .attention import AttentionBatch, AttentionBatcher, AttentionDecision, AttentionPolicy
from .commands import QQCommandState
from .group_context import GroupContextTracker
from .ingress_guard import IngressAdmission, IngressPolicy
from .message import IncomingMessage
from .routing import ReplyRoutePolicy, RouteDecision

__all__ = ["QQIngressAdapter", "QQIngressResult"]


@dataclass(frozen=True)
class QQIngressResult:
    message: IncomingMessage
    route: RouteDecision
    attention: AttentionDecision
    closed_batch: AttentionBatch | None = None
    admitted: bool = True
    admission: IngressAdmission | None = None

    @property
    def scheduled(self) -> bool:
        return self.attention.should_schedule

    def as_dict(self) -> dict[str, Any]:
        return {
            "message": self.message.as_dict(),
            "route": self.route.as_dict(),
            "attention": self.attention.as_dict(),
            "closed_batch": self.closed_batch.as_dict() if self.closed_batch else None,
            "admitted": self.admitted,
            "admission": self.admission.as_dict() if self.admission is not None else None,
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
        ingress_policy: IngressPolicy | None = None,
        group_context: GroupContextTracker | None = None,
    ) -> None:
        self.route_policy = route_policy or ReplyRoutePolicy.from_environment()
        self.attention_policy = attention_policy or AttentionPolicy()
        self.batcher = batcher or AttentionBatcher()
        self.ingress_policy = ingress_policy or IngressPolicy.from_environment()
        self.group_context = group_context or GroupContextTracker()
        self._seen_event_fingerprints: dict[str, float] = {}
        self._ingress_lock = threading.RLock()
        self.command_state: QQCommandState | None = None

    def set_command_state(self, state: QQCommandState | None) -> None:
        """Attach the live command switches after the Agent is assembled."""

        self.command_state = state
        if state is None:
            self.attention_policy = replace(
                self.attention_policy,
                autonomous_enabled_getter=None,
                response_enabled_getter=None,
            )
            return
        self.attention_policy = replace(
            self.attention_policy,
            autonomous_enabled_getter=state.is_autonomous_reply_enabled,
            response_enabled_getter=state.responses_enabled,
        )

    def ingest(self, event: Mapping[str, Any], *, at: float) -> QQIngressResult:
        return self.ingest_message(IncomingMessage.from_event(event), at=at)

    def admit_message(self, message: IncomingMessage, *, at: float) -> IngressAdmission:
        if not isinstance(message, IncomingMessage):
            raise TypeError("message must be an IncomingMessage")
        return self.ingress_policy.admit(
            message,
            now=float(at),
            seen=self._seen_event_fingerprints,
            lock=self._ingress_lock,
        )

    def ingest_message(
        self,
        message: IncomingMessage,
        *,
        at: float,
        admission: IngressAdmission | None = None,
    ) -> QQIngressResult:
        if not isinstance(message, IncomingMessage):
            raise TypeError("message must be an IncomingMessage")
        admission = admission or self.admit_message(message, at=at)
        if not admission.accepted:
            route = RouteDecision(False, admission.reason)
            attention = AttentionDecision(False, False, 0, admission.reason)
            return QQIngressResult(
                message=message,
                route=route,
                attention=attention,
                closed_batch=None,
                admitted=False,
                admission=admission,
            )
        route = self.route_policy.decide(message)
        self.group_context.enrich(message, route, now=float(at))
        attention = self.attention_policy.evaluate(message, route)
        closed_batch = self.batcher.push(message, attention, at=at)
        return QQIngressResult(
            message=message,
            route=route,
            attention=attention,
            closed_batch=closed_batch,
            admitted=True,
            admission=admission,
        )

    def flush(self) -> AttentionBatch | None:
        return self.batcher.flush()
