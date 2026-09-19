"""入站消息是否进入 Agent 的确定性路由。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .message import IncomingMessage, MessageSegment

__all__ = ["ReplyRoutePolicy", "RouteDecision"]


_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class RouteDecision:
    should_reply: bool
    reason: str
    addressed: bool = False
    image_only: bool = False
    priority: int = 0
    reply_to_message_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "should_reply": self.should_reply,
            "reason": self.reason,
            "addressed": self.addressed,
            "image_only": self.image_only,
            "priority": self.priority,
            "reply_to_message_id": self.reply_to_message_id,
        }


@dataclass(frozen=True)
class ReplyRoutePolicy:
    """默认保守的消息入口策略。

    ``bot_ids`` 是适配器已经解析出的机器人账号 ID；名称匹配只负责文本直呼，
    不负责猜测群里提到某个名字是不是在叫机器人。
    """

    bot_ids: frozenset[str] = frozenset()
    bot_names: tuple[str, ...] = ("真红", "二阶堂真红", "Nikaidou Shinku")
    allow_private: bool = True
    allow_group_name_address: bool = True
    allow_group_at: bool = True
    allow_reply_to_bot: bool = True
    allow_unaddressed_image: bool = False

    def decide(self, message: IncomingMessage) -> RouteDecision:
        image_only = message.has_media and not message.text.strip()
        if message.conversation_type == "private" and self.allow_private:
            return RouteDecision(True, "private_message", priority=100, image_only=image_only)

        at_bot = self._mentions_bot(message)
        if at_bot and self.allow_group_at:
            return RouteDecision(True, "mentioned_bot", addressed=True, priority=90, image_only=image_only)

        name_addressed = self._starts_with_bot_name(message.text)
        if name_addressed and self.allow_group_name_address:
            return RouteDecision(True, "direct_name", addressed=True, priority=80, image_only=image_only)

        reply_target = self._reply_target_id(message)
        reply_to_bot = self._reply_targets_bot(message)
        if reply_to_bot and self.allow_reply_to_bot:
            return RouteDecision(
                True,
                "reply_to_bot",
                addressed=True,
                priority=85,
                image_only=image_only,
                reply_to_message_id=reply_target,
            )

        if image_only and not self.allow_unaddressed_image:
            return RouteDecision(False, "unaddressed_image", image_only=True)
        return RouteDecision(False, "unaddressed_group_message", image_only=image_only)

    def _mentions_bot(self, message: IncomingMessage) -> bool:
        if bool(message.raw_event.get("mentioned_bot")) or bool(message.raw_event.get("at_bot")):
            return True
        configured = {str(item).strip() for item in self.bot_ids if str(item).strip()}
        return any(target in configured for segment in message.segments if segment.kind == "at" for target in _segment_targets(segment))

    def _reply_targets_bot(self, message: IncomingMessage) -> bool:
        if bool(message.raw_event.get("reply_to_bot")) or bool(message.raw_event.get("is_reply_to_bot")):
            return True
        configured = {str(item).strip() for item in self.bot_ids if str(item).strip()}
        if not configured:
            return False
        return any(
            target in configured
            for segment in message.segments
            if segment.kind == "reply"
            for target in _segment_targets(segment)
        )

    @staticmethod
    def _reply_target_id(message: IncomingMessage) -> str:
        for segment in message.segments:
            if segment.kind == "reply":
                data = segment.payload.get("data") if isinstance(segment.payload.get("data"), dict) else {}
                for source in (data, segment.payload):
                    for key in ("id", "message_id", "messageId"):
                        value = source.get(key)
                        if value is not None and str(value).strip():
                            return str(value).strip()
        return ""

    def _starts_with_bot_name(self, text: str) -> bool:
        normalized = _SPACE_RE.sub(" ", str(text or "").strip()).casefold()
        if not normalized:
            return False
        for name in sorted((str(item).strip().casefold() for item in self.bot_names), key=len, reverse=True):
            if not name:
                continue
            if normalized == name or normalized.startswith(name + " "):
                return True
            if normalized.startswith(name) and len(normalized) > len(name) and normalized[len(name)] in "，,：:、!?！？。":
                return True
        return False


def _segment_targets(segment: MessageSegment) -> tuple[str, ...]:
    values: list[str] = []
    raw = segment.payload
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    for source in (raw, data):
        for key in ("qq", "target", "user_id", "userId", "id", "message_id", "messageId"):
            value = source.get(key)
            if value is not None and str(value).strip():
                values.append(str(value).strip())
    return tuple(dict.fromkeys(values))
