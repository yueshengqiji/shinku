"""QQ 入站适配器的组合边界。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import threading
from typing import Any, Mapping

from .attention import AttentionBatch, AttentionBatcher, AttentionDecision, AttentionPolicy, AutonomousAttentionTracker
from .attention_trace import AttentionTraceStore
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
        attention_trace_store: AttentionTraceStore | None = None,
        autonomous_tracker: AutonomousAttentionTracker | None = None,
    ) -> None:
        self.route_policy = route_policy or ReplyRoutePolicy.from_environment()
        self.attention_policy = attention_policy or AttentionPolicy.from_environment()
        self.batcher = batcher or AttentionBatcher()
        self.ingress_policy = ingress_policy or IngressPolicy.from_environment()
        self.group_context = group_context or GroupContextTracker()
        self.attention_trace_store = attention_trace_store or AttentionTraceStore()
        self.autonomous_tracker = autonomous_tracker or AutonomousAttentionTracker()
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
                autonomous_enabled_for_message_getter=None,
                response_enabled_getter=None,
            )
            return
        self.attention_policy = replace(
            self.attention_policy,
            autonomous_enabled_getter=state.is_autonomous_reply_enabled,
            autonomous_enabled_for_message_getter=lambda message: state.is_autonomous_reply_enabled(
                message.conversation_id
                if str(message.conversation_type or "").casefold() in {"group", "guild"}
                else ""
            ),
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
            self._record_attention_trace(
                message,
                at=at,
                admission=admission,
                route=route,
                attention=attention,
                queued=False,
            )
            return QQIngressResult(
                message=message,
                route=route,
                attention=attention,
                closed_batch=None,
                admitted=False,
                admission=admission,
            )
        with self._ingress_lock:
            route = self.route_policy.decide(message)
            signals = self.autonomous_tracker.signals(message, now=float(at))
            recent = tuple(self.group_context.snapshot(message.conversation_id, now=float(at)))
            other_focus = self.group_context.focus_snapshot(message.conversation_id, now=float(at))
            attention = self.attention_policy.evaluate(
                message, route, signals=signals, recent=recent, other_focus=other_focus, now=float(at),
            )
            self.autonomous_tracker.observe(message, route, attention, now=float(at))
            self.group_context.enrich(
                message,
                route,
                now=float(at),
                include_context=attention.should_schedule,
            )
            group_snapshot = message.raw_event.get("_shinku_group_context", {})
            if isinstance(group_snapshot.get("current"), dict):
                group_snapshot["current"]["bot_exchange_confirmed"] = bool(
                    route.addressed or signals.reply_to_delivered_message
                    or signals.recent_directed_turn or signals.recent_bot_exchange)
            closed_batch = self.batcher.push(message, attention, at=at)
            self._record_attention_trace(
                message,
                at=at,
                admission=admission,
                route=route,
                attention=attention,
                queued=attention.should_schedule,
                closed_batch=closed_batch,
            )
        return QQIngressResult(
            message=message,
            route=route,
            attention=attention,
            closed_batch=closed_batch,
            admitted=True,
            admission=admission,
        )

    def note_group_reply(
        self, conversation_id: str, *, recipient_ids: tuple[str, ...],
        message_ids: tuple[str, ...], autonomous: bool, at: float,
    ) -> None:
        """Count a turn only after transport confirms at least one segment."""

        with self._ingress_lock:
            self.autonomous_tracker.note_delivered(
                conversation_id, recipient_ids=recipient_ids, message_ids=message_ids,
                autonomous=autonomous, now=float(at),
            )

    def note_group_continuation(self, group_id: str, recipient_id: str, at: float) -> None:
        """与入站共用锁；调用方必须先核验承接凭据及完整送达，不更改受话/任务归属。"""
        with self._ingress_lock:
            self.autonomous_tracker.note_continuation(group_id, recipient_id, now=float(at))

    def _record_attention_trace(
        self,
        message: IncomingMessage,
        *,
        at: float,
        admission: IngressAdmission,
        route: RouteDecision,
        attention: AttentionDecision,
        queued: bool,
        closed_batch: AttentionBatch | None = None,
    ) -> None:
        """Record diagnostics without letting observability affect ingress."""

        raw = message.raw_event if isinstance(message.raw_event, Mapping) else {}
        topic = raw.get("_shinku_group_context")
        topic = topic if isinstance(topic, Mapping) else {}
        topic_items = topic.get("items")
        topic_items = topic_items if isinstance(topic_items, (list, tuple)) else ()
        try:
            self.attention_trace_store.record(
                observed_at=float(at),
                conversation_id=message.conversation_id,
                conversation_type=message.conversation_type,
                message_id=message.message_id,
                sender_id=message.sender_id,
                text_chars=len(str(message.text or "")),
                has_media=message.has_media,
                is_poke=message.is_poke,
                admission={"accepted": admission.accepted, "reason": admission.reason},
                route=route.as_dict(),
                attention=attention.as_dict(),
                queued=queued,
                closed_batch_size=len(closed_batch.messages) if closed_batch is not None else 0,
                closed_batch_primary_message_id=(
                    closed_batch.primary_message.message_id if closed_batch is not None else ""
                ),
                topic_state={
                    "reply_message_count": len(message.reply_message_ids),
                    "group_context_count": len(topic_items),
                    "group_context_reason": str(topic.get("reason") or "")[:80],
                    "group_context_focus": bool(topic.get("focus")),
                    "current_at_target_count": len(message.at_targets),
                },
            )
        except Exception:
            # Diagnostics are best-effort and must never change reply behavior.
            return

    def flush(self) -> AttentionBatch | None:
        with self._ingress_lock:
            return self.batcher.flush()
