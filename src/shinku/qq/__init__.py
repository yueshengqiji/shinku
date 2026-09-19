"""QQ/消息适配边界。"""

from .message import IncomingMessage, MessageSegment
from .attention import AttentionBatch, AttentionBatcher, AttentionDecision, AttentionPolicy
from .routing import ReplyRoutePolicy, RouteDecision
from .delivery import DeliveryResult, DeliveryTransport, OutgoingMessage, OutgoingSegment, build_reply, deliver

__all__ = [
    "AttentionBatch",
    "AttentionBatcher",
    "AttentionDecision",
    "AttentionPolicy",
    "DeliveryResult",
    "DeliveryTransport",
    "IncomingMessage",
    "MessageSegment",
    "OutgoingMessage",
    "OutgoingSegment",
    "ReplyRoutePolicy",
    "RouteDecision",
    "build_reply",
    "deliver",
]
