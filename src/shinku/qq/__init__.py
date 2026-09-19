"""QQ/消息适配边界。"""

from .message import IncomingMessage, MessageSegment
from .attention import AttentionBatch, AttentionBatcher, AttentionDecision, AttentionPolicy
from .routing import ReplyRoutePolicy, RouteDecision
from .delivery import DeliveryResult, DeliveryTransport, OutgoingMessage, OutgoingSegment, build_reply, deliver
from .adapter import QQIngressAdapter, QQIngressResult
from .napcat import (
    NapCatActionTransport,
    NapCatEventAdapter,
    NapCatEventDecoder,
    NapCatHttpActionCaller,
    NapCatVisualBatch,
    NapCatVisualInput,
    NapCatVisualInputBridge,
    NapCatImageMaterializer,
)

__all__ = [
    "AttentionBatch",
    "AttentionBatcher",
    "AttentionDecision",
    "AttentionPolicy",
    "DeliveryResult",
    "DeliveryTransport",
    "IncomingMessage",
    "MessageSegment",
    "NapCatActionTransport",
    "NapCatEventAdapter",
    "NapCatEventDecoder",
    "NapCatHttpActionCaller",
    "NapCatVisualBatch",
    "NapCatVisualInput",
    "NapCatVisualInputBridge",
    "NapCatImageMaterializer",
    "QQIngressAdapter",
    "QQIngressResult",
    "OutgoingMessage",
    "OutgoingSegment",
    "ReplyRoutePolicy",
    "RouteDecision",
    "build_reply",
    "deliver",
]
