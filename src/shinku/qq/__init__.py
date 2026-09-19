"""QQ/消息适配边界。"""

from .message import IncomingMessage, MessageSegment
from .attention import AttentionBatch, AttentionBatcher, AttentionDecision, AttentionPolicy
from .routing import ReplyRoutePolicy, RouteDecision
from .delivery import DeliveryResult, DeliveryTransport, OutgoingMessage, OutgoingSegment, build_reply, deliver
from .adapter import QQIngressAdapter, QQIngressResult
from .host import NapCatHost, NapCatHostResult
from .config import NapCatConnectionConfig
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
    "NapCatHost",
    "NapCatHostResult",
    "NapCatConnectionConfig",
    "OutgoingMessage",
    "OutgoingSegment",
    "ReplyRoutePolicy",
    "RouteDecision",
    "build_reply",
    "deliver",
]
