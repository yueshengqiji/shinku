"""QQ 入站事件的本地准入与去重边界。

这层只做低成本、确定性的判断：不调用模型、不访问网络，也不把被拒绝的
事件交给旁观记忆。它把 NapCat 的重复投递、旧事件和机器人自发消息挡在
消息路由之前，避免后面的 Agent 只能靠提示词猜测“这条是不是已经处理过”。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .message import IncomingMessage

__all__ = ["IngressAdmission", "IngressPolicy"]


def _flag(value: Any, *, default: bool) -> bool:
    raw = str(value or "").strip().casefold()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "开启", "开"}


def _ids(value: Any) -> frozenset[str]:
    raw = str(value or "")
    return frozenset(item.strip() for item in raw.replace("，", ",").split(",") if item.strip())


@dataclass(frozen=True)
class IngressAdmission:
    """一次入站事件是否允许继续进入观察和回复路由。"""

    accepted: bool
    reason: str = "accepted"
    fingerprint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class IngressPolicy:
    """入站事件过滤策略。

    ``poke`` 默认跳过过期过滤，因为部分 NapCat 版本会给 notice 事件填入
    过期的消息时间；它仍然受自消息、忽略对象和事件去重保护。
    """

    event_max_age_seconds: float = 300.0
    allow_stale_events: bool = False
    dedup_ttl_seconds: float = 600.0
    dedup_max_entries: int = 4096
    ignored_group_ids: frozenset[str] = field(default_factory=frozenset)
    ignored_user_ids: frozenset[str] = field(default_factory=frozenset)
    ignore_self_events: bool = True

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "IngressPolicy":
        env = os.environ if environ is None else environ

        def number(key: str, default: float, minimum: float) -> float:
            try:
                return max(minimum, float(str(env.get(key, default) or default).strip()))
            except (TypeError, ValueError):
                return default

        try:
            max_entries = max(128, min(100_000, int(str(env.get("SHINKU_QQ_EVENT_DEDUP_MAX_ENTRIES", 4096) or 4096))))
        except (TypeError, ValueError):
            max_entries = 4096
        return cls(
            event_max_age_seconds=number("SHINKU_QQ_EVENT_MAX_AGE_SECONDS", 300.0, 0.0),
            allow_stale_events=_flag(env.get("SHINKU_QQ_ALLOW_STALE_EVENTS"), default=False),
            dedup_ttl_seconds=number("SHINKU_QQ_EVENT_DEDUP_TTL_SECONDS", 600.0, 1.0),
            dedup_max_entries=max_entries,
            ignored_group_ids=_ids(env.get("SHINKU_QQ_IGNORED_GROUP_IDS")),
            ignored_user_ids=_ids(env.get("SHINKU_QQ_IGNORED_USER_IDS")),
            ignore_self_events=_flag(env.get("SHINKU_QQ_IGNORE_SELF_EVENTS", "true"), default=True),
        )

    def fingerprint(self, message: IncomingMessage) -> str:
        """生成稳定的事件指纹，不把完整原始 payload 当作数据库键。"""

        message_id = str(message.message_id or "").strip()
        if message_id:
            material = {
                "kind": message.event_kind,
                "conversation": message.conversation_id,
                "message_id": message_id,
            }
        else:
            material = {
                "kind": message.event_kind,
                "conversation": message.conversation_id,
                "sender": message.sender_id,
                "timestamp": message.timestamp,
                "segments": [segment.as_dict() for segment in message.segments],
            }
        encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def admit(
        self,
        message: IncomingMessage,
        *,
        now: float,
        seen: dict[str, float],
        lock: threading.RLock,
    ) -> IngressAdmission:
        if not isinstance(message, IncomingMessage):
            raise TypeError("message must be an IncomingMessage")
        current = float(now)
        fingerprint = self.fingerprint(message)
        # ``IncomingMessage(...)`` is also a useful in-process test/domain
        # object.  Without a raw transport event or timestamp it has no stable
        # delivery identity, so treating repeated calls as a NapCat duplicate
        # would break callers that deliberately re-evaluate the same object
        # after changing live command switches.
        transport_event = bool(message.raw_event) or message.timestamp is not None
        with lock:
            cutoff = current - self.dedup_ttl_seconds
            for key, timestamp in tuple(seen.items()):
                if timestamp < cutoff:
                    seen.pop(key, None)
            if transport_event and fingerprint in seen:
                return IngressAdmission(False, "duplicate_event", fingerprint)

        raw = message.raw_event if isinstance(message.raw_event, Mapping) else {}
        group_id = str(raw.get("group_id") or (message.conversation_id if message.conversation_type == "group" else "")).strip()
        sender_id = str(message.sender_id or "").strip()
        if group_id and group_id in self.ignored_group_ids:
            return IngressAdmission(False, "ignored_group", fingerprint)
        if sender_id and sender_id in self.ignored_user_ids:
            return IngressAdmission(False, "ignored_user", fingerprint)

        self_id = str(raw.get("self_id") or raw.get("selfId") or "").strip()
        sender = raw.get("sender") if isinstance(raw.get("sender"), Mapping) else {}
        raw_sender_id = str(
            raw.get("user_id") or raw.get("sender_id") or sender.get("user_id") or sender.get("userId") or ""
        ).strip()
        if self.ignore_self_events and self_id and sender_id and sender_id == self_id:
            return IngressAdmission(False, "self_event", fingerprint)
        if self.ignore_self_events and self_id and raw_sender_id and raw_sender_id == self_id:
            return IngressAdmission(False, "self_event", fingerprint)

        if (
            not message.is_poke
            and not self.allow_stale_events
            and message.timestamp is not None
            and self.event_max_age_seconds > 0
            and current - float(message.timestamp) > self.event_max_age_seconds
        ):
            return IngressAdmission(False, "stale_event", fingerprint)

        if transport_event:
            with lock:
                seen[fingerprint] = current
                if len(seen) > self.dedup_max_entries:
                    excess = len(seen) - self.dedup_max_entries
                    for key, _ in sorted(seen.items(), key=lambda item: item[1])[:excess]:
                        seen.pop(key, None)
        return IngressAdmission(True, "accepted", fingerprint)
