"""外部消息事件的不可逆信息保护层。

本模块只负责把不同适配器送来的事件归一化为消息和 segment，不负责 QQ 登录、
NapCat、发送消息或调用模型。尤其不把图片、文件和引用折叠成 ``[图片]`` 等
纯文本占位符；模型是否查看附件由后续能力路由决定。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["IncomingMessage", "MessageSegment"]


_MEDIA_KINDS = frozenset({"image", "file", "audio", "video"})
_KNOWN_KINDS = _MEDIA_KINDS | frozenset({"text", "at", "reply", "face", "unknown"})
_ID_KEYS = ("message_id", "messageId", "id")


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class MessageSegment:
    """一段可独立处理的消息内容。"""

    kind: str
    text: str = ""
    media_id: str = ""
    url: str = ""
    local_path: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> "MessageSegment":
        if isinstance(value, str):
            return cls(kind="text", text=value)
        if not isinstance(value, Mapping):
            return cls(kind="unknown", payload={"value": value})
        raw_kind = str(value.get("type") or value.get("kind") or "unknown").strip().lower()
        kind = raw_kind if raw_kind in _KNOWN_KINDS else "unknown"
        data = _mapping_copy(value.get("data"))
        merged = {**dict(value), **data}
        text = _first_text(merged, "text", "content", "name") if kind in {"text", "at", "face"} else ""
        return cls(
            kind=kind,
            text=text,
            media_id=_first_text(merged, "file", "file_id", "fileId", "id"),
            url=_first_text(merged, "url", "src", "image_url", "imageUrl"),
            local_path=_first_text(merged, "path", "file_path", "local_path"),
            payload=dict(value),
        )

    @property
    def is_media(self) -> bool:
        return self.kind in _MEDIA_KINDS

    @property
    def attachment_ref(self) -> dict[str, str] | None:
        if not self.is_media:
            return None
        return {
            "kind": self.kind,
            "media_id": self.media_id,
            "url": self.url,
            "local_path": self.local_path,
        }

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        if self.text:
            result["text"] = self.text
        if self.media_id:
            result["media_id"] = self.media_id
        if self.url:
            result["url"] = self.url
        if self.local_path:
            result["local_path"] = self.local_path
        if self.payload:
            result["payload"] = dict(self.payload)
        return result


@dataclass(frozen=True)
class IncomingMessage:
    """跨 QQ/HTTP 适配器的统一入站消息。"""

    message_id: str
    sender_id: str
    conversation_id: str
    conversation_type: str
    segments: tuple[MessageSegment, ...]
    timestamp: float | None = None
    raw_event: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> "IncomingMessage":
        if not isinstance(event, Mapping):
            raise TypeError("message event must be a mapping")
        sender = event.get("sender") if isinstance(event.get("sender"), Mapping) else {}
        segments_value = event.get("segments")
        if segments_value is None:
            segments_value = event.get("message")
        if isinstance(segments_value, Sequence) and not isinstance(segments_value, (str, bytes, bytearray)):
            segments = tuple(MessageSegment.from_value(item) for item in segments_value)
        elif segments_value is None:
            text = event.get("raw_message", event.get("text", ""))
            segments = (MessageSegment.from_value(str(text)),) if str(text or "") else ()
        else:
            segments = (MessageSegment.from_value(segments_value),)

        conversation_type = _first_text(event, "conversation_type", "message_type", "chat_type").lower()
        if conversation_type not in {"private", "group", "guild", "unknown"}:
            conversation_type = "group" if event.get("group_id") else "private"
        return cls(
            message_id=_first_text(event, *_ID_KEYS),
            sender_id=_first_text(event, "sender_id", "user_id", "userId") or _first_text(sender, *_ID_KEYS, "user_id"),
            conversation_id=(
                _first_text(event, "conversation_id", "conversationId", "group_id", "guild_id")
                or _first_text(sender, "conversation_id", "group_id")
            ),
            conversation_type=conversation_type,
            segments=segments,
            timestamp=_timestamp(event.get("time", event.get("timestamp"))),
            raw_event=dict(event),
        )

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments if segment.kind in {"text", "at", "face"})

    @property
    def attachments(self) -> tuple[dict[str, str], ...]:
        return tuple(ref for segment in self.segments if (ref := segment.attachment_ref) is not None)

    @property
    def has_media(self) -> bool:
        return bool(self.attachments)

    @property
    def has_reply(self) -> bool:
        return any(segment.kind == "reply" for segment in self.segments)

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "conversation_id": self.conversation_id,
            "conversation_type": self.conversation_type,
            "segments": [segment.as_dict() for segment in self.segments],
            "text": self.text,
            "attachments": [dict(item) for item in self.attachments],
            "timestamp": self.timestamp,
        }


def _timestamp(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
