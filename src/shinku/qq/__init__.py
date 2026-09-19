"""QQ/消息适配边界。"""

from .message import IncomingMessage, MessageSegment
from .attention import AttentionBatch, AttentionBatcher, AttentionDecision, AttentionPolicy
from .routing import ReplyRoutePolicy, RouteDecision

__all__ = [
    "AttentionBatch",
    "AttentionBatcher",
    "AttentionDecision",
    "AttentionPolicy",
    "IncomingMessage",
    "MessageSegment",
    "ReplyRoutePolicy",
    "RouteDecision",
]
