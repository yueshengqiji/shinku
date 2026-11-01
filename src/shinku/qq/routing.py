"""入站消息是否进入 Agent 的确定性路由。"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .commands import command_text_for_conversation
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

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "ReplyRoutePolicy":
        """从独立配置生成路由策略；没有配置时保持默认行为。"""

        env = os.environ if environ is None else environ
        raw_ids = str(env.get("SHINKU_QQ_BOT_IDS", "") or "")
        raw_names = str(env.get("SHINKU_QQ_BOT_NAMES", "") or "")
        if env.get("SHINKU_CHARACTER_FILE"):
            from shinku.character import load_character
            character = load_character(env["SHINKU_CHARACTER_FILE"])
            raw_names = ",".join(character.aliases)
        bot_ids = frozenset(item.strip() for item in raw_ids.split(",") if item.strip())
        bot_names = tuple(item.strip() for item in raw_names.split(",") if item.strip())

        def flag(key: str, default: bool) -> bool:
            raw = str(env.get(key, "") or "").strip().lower()
            if not raw:
                return default
            return raw in {"1", "true", "yes", "on"}

        return cls(
            bot_ids=bot_ids,
            bot_names=bot_names or cls.bot_names,
            allow_private=flag("SHINKU_QQ_ALLOW_PRIVATE", cls.allow_private),
            allow_group_name_address=flag("SHINKU_QQ_ALLOW_GROUP_NAME_ADDRESS", cls.allow_group_name_address),
            allow_group_at=flag("SHINKU_QQ_ALLOW_GROUP_AT", cls.allow_group_at),
            allow_reply_to_bot=flag("SHINKU_QQ_ALLOW_REPLY_TO_BOT", cls.allow_reply_to_bot),
            allow_unaddressed_image=flag("SHINKU_QQ_ALLOW_UNADDRESSED_IMAGE", cls.allow_unaddressed_image),
        )

    def decide(self, message: IncomingMessage) -> RouteDecision:
        image_only = message.has_media and not message.text.strip()
        if message.is_poke:
            if not self._poke_targets_bot(message):
                return RouteDecision(False, "poke_not_for_bot")
            return RouteDecision(
                True,
                "qq_poke",
                addressed=True,
                priority=95,
            )
        # Private commands are available directly.  In a group the slash is
        # control-plane traffic only when the message also explicitly @s this
        # bot.  An unaddressed group ``/...`` therefore follows the ordinary
        # group route instead of changing state or entering the interpreter.
        at_bot = self._mentions_bot(message)
        if command_text_for_conversation(
            message.text,
            message.conversation_type,
            mentioned_bot=at_bot and self.allow_group_at,
        ):
            return RouteDecision(
                True,
                "slash_command",
                addressed=True,
                priority=110,
                image_only=image_only,
            )
        if message.conversation_type == "private" and self.allow_private:
            return RouteDecision(True, "private_message", priority=100, image_only=image_only)

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
        configured = self._effective_bot_ids(message)
        return any(target in configured for segment in message.segments if segment.kind == "at" for target in _segment_targets(segment))

    def _reply_targets_bot(self, message: IncomingMessage) -> bool:
        if bool(message.raw_event.get("reply_to_bot")) or bool(message.raw_event.get("is_reply_to_bot")):
            return True
        configured = self._effective_bot_ids(message)
        if not configured:
            return False
        return any(
            target in configured
            for segment in message.segments
            if segment.kind == "reply"
            for target in _segment_targets(segment)
        )

    def _effective_bot_ids(self, message: IncomingMessage) -> set[str]:
        """Use explicit IDs first, then the OneBot event's ``self_id``.

        NapCat's normal OneBot message payload identifies the logged-in bot as
        ``self_id``; it does not add the test-only ``at_bot`` flag.  Without
        this fallback a real ``[CQ:at,qq=...]`` segment is treated as an
        ordinary group message whenever no bot ID was duplicated into Shinku's
        environment.
        """

        configured = {str(item).strip() for item in self.bot_ids if str(item).strip()}
        self_id = message.raw_event.get("self_id")
        if self_id is not None and str(self_id).strip():
            configured.add(str(self_id).strip())
        return configured

    def _poke_targets_bot(self, message: IncomingMessage) -> bool:
        targets = {str(message.raw_event.get(key) or "").strip()
                   for key in ("target_id", "targetId")
                   if str(message.raw_event.get(key) or "").strip()}
        # Match the decoder's nonempty alias handling, but fail closed when
        # two aliases contradict each other instead of guessing ownership.
        if len(targets) != 1:
            return False
        return next(iter(targets)) in self._effective_bot_ids(message)

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
            # Chinese vocatives naturally omit punctuation. A second-person
            # continuation is directed speech; a possessive/name mention is not.
            if normalized.startswith(name) and re.match(
                r"(?:你|您|在吗|在不在|有空吗|怎么看|能不能|可以帮|帮我|帮忙|回答一下|说说)", normalized[len(name):],
            ):
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
