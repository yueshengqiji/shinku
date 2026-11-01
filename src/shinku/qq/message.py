"""外部消息事件的不可逆信息保护层。

本模块只负责把不同适配器送来的事件归一化为消息和 segment，不负责 QQ 登录、
NapCat、发送消息或调用模型。尤其不把图片、文件和引用折叠成 ``[图片]`` 等
纯文本占位符；模型是否查看附件由后续能力路由决定。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import html
import json
import re
from typing import Any

from .cards import CARD_KINDS, card_context, cq_segments

__all__ = ["IncomingMessage", "MessageSegment", "is_group_increase_event", "is_poke_event"]


_MEDIA_KINDS = frozenset({"image", "file", "audio", "video"})
_KNOWN_KINDS = _MEDIA_KINDS | frozenset({"text", "at", "reply", "face", "poke", "unknown"}) | CARD_KINDS
_ID_KEYS = ("message_id", "messageId", "id")
_HTML_IMAGE_RE = re.compile(
    r"<img\b[^>]*?\bsrc\s*=\s*(?P<quote>['\"])(?P<src>.*?)(?P=quote)[^>]*?/?>",
    re.IGNORECASE | re.DOTALL,
)
_CQ_SEGMENT_RE = re.compile(r"\[CQ:\w+(?:,[^\]]*)?\]", re.IGNORECASE)


def _first_text(mapping: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _inline_message_segments(value: str) -> tuple["MessageSegment", ...]:
    """兼容字符串消息中的 CQ 段和 HTML 图片，保留卡片/引用/媒体的原有顺序。

    CQ 的属性由独立单层解码器处理，JSON 内编码逗号不能变成新的属性。
    这里只归一化，不访问网络；HTML 仅识别现有 img src 兼容入口。
    """

    text = str(value or "")
    if not text:
        return ()
    matches: list[tuple[int, int, MessageSegment]] = []
    for match in _HTML_IMAGE_RE.finditer(text):
        source = html.unescape(str(match.group("src") or "").strip())
        if not source:
            continue
        matches.append(
            (
                match.start(),
                match.end(),
                MessageSegment.from_value({"type": "image", "data": {"url": source}}),
            )
        )
    for match in _CQ_SEGMENT_RE.finditer(text):
        decoded = cq_segments(match.group(0))
        if decoded:
            matches.append((match.start(), match.end(), MessageSegment.from_value(decoded[0])))
    if not matches:
        return (MessageSegment(kind="text", text=text),)

    matches.sort(key=lambda item: (item[0], item[1]))
    segments: list[MessageSegment] = []
    cursor = 0
    for start, end, image in matches:
        if start < cursor:
            continue
        if start > cursor:
            segments.append(MessageSegment(kind="text", text=text[cursor:start]))
        segments.append(image)
        cursor = end
    if cursor < len(text):
        segments.append(MessageSegment(kind="text", text=text[cursor:]))
    return tuple(segment for segment in segments if segment.text or segment.kind != "text")


def is_poke_event(event: Mapping[str, Any]) -> bool:
    """Return whether a OneBot event is one of NapCat's poke variants.

    NapCat installations in the wild use both ``notice/poke`` and
    ``notify/notify/poke`` shapes.  Keeping the shape check here gives every
    adapter the same event boundary without accepting unrelated notices such
    as group uploads or recalls as conversational messages.
    """

    if not isinstance(event, Mapping):
        return False
    post_type = str(event.get("post_type") or "").strip().casefold()
    notice_type = str(event.get("notice_type") or "").strip().casefold()
    sub_type = str(event.get("sub_type") or event.get("notice_sub_type") or "").strip().casefold()
    if post_type not in {"notice", "notify"}:
        return False
    if notice_type in {"poke", "group_poke"}:
        return True
    return notice_type == "notify" and sub_type in {"poke", "group_poke"}


def is_group_increase_event(event: Mapping[str, Any]) -> bool:
    """Return whether a OneBot event announces a new group member.

    The event is kept separate from chat messages so a welcome policy can act
    on it without creating an LLM turn or a passive memory record.
    """

    if not isinstance(event, Mapping):
        return False
    post_type = str(event.get("post_type") or "").strip().casefold()
    notice_type = str(event.get("notice_type") or "").strip().casefold()
    return post_type in {"notice", "notify"} and notice_type == "group_increase"


def _poke_actor_id(event: Mapping[str, Any], *, target_id: str = "") -> str:
    sender = event.get("sender") if isinstance(event.get("sender"), Mapping) else {}
    candidates = (
        _first_text(event, "operator_id", "operatorId"),
        _first_text(event, "sender_id", "senderId", "user_id", "userId"),
        _first_text(sender, "user_id", "userId", "id"),
    )
    for candidate in candidates:
        if candidate and candidate != target_id:
            return candidate
    return next((candidate for candidate in candidates if candidate), "")


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
        result = {
            "kind": self.kind,
            "media_id": self.media_id,
            "url": self.url,
            "local_path": self.local_path,
        }
        if self.kind == "file":
            data = _mapping_copy(self.payload.get("data"))
            merged = {**self.payload, **data}
            # NapCat 的 file 是展示文件名，file_id 才是平台读取标识；不能互相覆盖。
            result.update(file_id=_first_text(merged, "file_id", "fileId", "id"),
                          name=_first_text(merged, "name", "file", "file_name"),
                          size=_first_text(merged, "file_size", "size"))
        return result

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
    event_kind: str = "message"
    quoted_message: dict[str, Any] = field(default_factory=dict)
    # 仅宿主解析器生成；不接受事件自报的已读取资料，正文仍独立供控制指令判断。
    card_context: str = ""

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> "IncomingMessage":
        if not isinstance(event, Mapping):
            raise TypeError("message event must be a mapping")
        sender = event.get("sender") if isinstance(event.get("sender"), Mapping) else {}
        event_kind = (
            "poke"
            if is_poke_event(event)
            else "group_increase"
            if is_group_increase_event(event)
            else "message"
        )
        if event_kind == "poke":
            target_id = _first_text(event, "target_id", "targetId")
            actor_id = _poke_actor_id(event, target_id=target_id)
            actor_label = _first_text(sender, "card", "nickname", "remark")
            marker = "[QQ戳一戳]"
            segments = (
                MessageSegment(
                    kind="poke",
                    text=marker,
                    payload={
                        "target_id": target_id,
                        "actor_id": actor_id,
                        "actor_label": actor_label,
                    },
                ),
            )
        elif event_kind == "group_increase":
            # This notice has no conversational content. Keep its identity so
            # a deterministic welcome policy can consume it, but never make
            # up prompt text for the Agent.
            segments = ()
        else:
            segments_value = event.get("segments")
            if segments_value is None:
                segments_value = event.get("message")
            if isinstance(segments_value, Sequence) and not isinstance(segments_value, (str, bytes, bytearray)):
                normalized: list[MessageSegment] = []
                for item in segments_value:
                    if isinstance(item, str):
                        normalized.extend(_inline_message_segments(item))
                    else:
                        normalized.append(MessageSegment.from_value(item))
                segments = tuple(normalized)
            elif segments_value is None:
                text = event.get("raw_message", event.get("text", ""))
                segments = _inline_message_segments(str(text)) if str(text or "") else ()
            elif isinstance(segments_value, str):
                segments = _inline_message_segments(segments_value)
            else:
                segments = (MessageSegment.from_value(segments_value),)

        conversation_type = _first_text(event, "conversation_type", "message_type", "chat_type").lower()
        if conversation_type not in {"private", "group", "guild", "unknown"}:
            conversation_type = "group" if event.get("group_id") else "private"
        if event_kind == "poke":
            target_id = _first_text(event, "target_id", "targetId")
            sender_id = _poke_actor_id(event, target_id=target_id)
        else:
            sender_id = _first_text(event, "sender_id", "user_id", "userId") or _first_text(
                sender, *_ID_KEYS, "user_id", "userId"
            )
        conversation_id = _first_text(event, "conversation_id", "conversationId", "group_id", "guild_id") or _first_text(
            sender, "conversation_id", "group_id"
        )
        # OneBot private-message events normally expose only ``user_id``.
        # Keep the normalized conversation identity usable for both memory
        # scope and send_private_msg instead of letting it become ``unknown``.
        if not conversation_id and conversation_type == "private":
            conversation_id = sender_id
        message_id = _first_text(event, *_ID_KEYS, "event_id", "eventId")
        if event_kind == "poke" and not message_id:
            target_id = _first_text(event, "target_id", "targetId") or "unknown"
            message_id = f"poke:{conversation_id or 'unknown'}:{sender_id or 'unknown'}:{target_id}:{event.get('time', '')}"
        if event_kind == "group_increase" and not message_id:
            sub_type = _first_text(event, "sub_type", "notice_sub_type") or "unknown"
            message_id = (
                f"group_increase:{conversation_id or 'unknown'}:{sender_id or 'unknown'}:"
                f"{sub_type}:{event.get('time', '')}"
            )
        quoted_message: dict[str, Any] = {}
        embedded_quote = _embedded_quote_event(segments)
        if embedded_quote:
            embedded_quote.setdefault("message_id", _reply_id_from_segments(segments))
            embedded_quote.setdefault("conversation_type", conversation_type)
            embedded_quote.setdefault("conversation_id", conversation_id)
            # An omitted original author is unknown, not the current quoter.
            try:
                quoted_message = cls.from_event(embedded_quote).as_dict()
            except (TypeError, ValueError):
                quoted_message = {}
        return cls(
            message_id=message_id,
            sender_id=sender_id,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            segments=segments,
            timestamp=_timestamp(event.get("time", event.get("timestamp"))),
            raw_event=dict(event),
            event_kind=event_kind,
            quoted_message=quoted_message,
            card_context=card_context(segments),
        )

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments if segment.kind in {"text", "at", "face", "poke"})

    @property
    def readable_text(self) -> str:
        """给模型/只读上下文的资料视图；text 保持当前用户正文，防止引用触发控制。"""
        if not self.card_context:
            return self.text
        origin = json.dumps({"message_id": self.message_id, "sender_id": self.sender_id,
                             "conversation_id": self.conversation_id,
                             "conversation_type": self.conversation_type}, ensure_ascii=False)
        return f"{self.text}\n[QQ资料来源] {origin}\n{self.card_context}".strip()

    @property
    def has_at_mention(self) -> bool:
        return any(segment.kind == "at" for segment in self.segments)

    @property
    def text_without_at_mentions(self) -> str:
        """Message body without the target labels carried by OneBot ``at`` segments."""

        return "".join(
            segment.text
            for segment in self.segments
            if segment.kind in {"text", "face", "poke"}
        )

    @property
    def at_target_labels(self) -> tuple[str, ...]:
        """Human-readable @ targets, falling back to the QQ ID when no name is present."""

        return tuple(label for _, label in self.at_targets)

    @property
    def at_target_ids(self) -> tuple[str, ...]:
        """Stable QQ identities for @ targets, kept in the same order as labels."""

        return tuple(identifier for identifier, _ in self.at_targets)

    @property
    def at_targets(self) -> tuple[tuple[str, str], ...]:
        """Pairs of target ID and display label for attribution-sensitive paths."""

        targets: list[tuple[str, str]] = []
        seen: set[str] = set()
        for segment in self.segments:
            if segment.kind != "at":
                continue
            payload = segment.payload if isinstance(segment.payload, Mapping) else {}
            data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
            target = ""
            for source in (data, payload):
                target = _first_text(source, "qq", "target", "user_id", "userId", "id")
                if target:
                    break
            label = str(segment.text or "").strip().lstrip("@").strip()
            display = label or (f"QQ {target}" if target else "未知成员")
            key = target or display.casefold()
            if key not in seen:
                seen.add(key)
                targets.append((target, display))
        return tuple(targets)

    @property
    def attachments(self) -> tuple[dict[str, str], ...]:
        return tuple(ref for segment in self.segments if (ref := segment.attachment_ref) is not None)

    @property
    def has_media(self) -> bool:
        return bool(self.attachments)

    @property
    def has_reply(self) -> bool:
        return any(segment.kind == "reply" for segment in self.segments)

    @property
    def reply_message_ids(self) -> tuple[str, ...]:
        values: list[str] = []
        for segment in self.segments:
            if segment.kind != "reply":
                continue
            data = segment.payload.get("data") if isinstance(segment.payload.get("data"), Mapping) else {}
            for source in (data, segment.payload):
                for key in ("id", "message_id", "messageId"):
                    value = source.get(key)
                    if value is not None and str(value).strip():
                        values.append(str(value).strip())
        return tuple(dict.fromkeys(values))

    @property
    def sender_label(self) -> str:
        raw = self.raw_event if isinstance(self.raw_event, Mapping) else {}
        sender = raw.get("sender") if isinstance(raw.get("sender"), Mapping) else {}
        for source in (sender, raw):
            for key in ("card", "nickname", "remark", "display_name", "displayName"):
                value = str(source.get(key) or "").strip()
                if value:
                    return value
        return ""

    @property
    def quoted_text(self) -> str:
        value = self.quoted_message if isinstance(self.quoted_message, Mapping) else {}
        return str(value.get("readable_text") or value.get("text") or value.get("raw_message") or "").strip()

    @property
    def quoted_attachments(self) -> tuple[dict[str, str], ...]:
        value = self.quoted_message if isinstance(self.quoted_message, Mapping) else {}
        attachments = value.get("attachments")
        if isinstance(attachments, list):
            return tuple(dict(item) for item in attachments if isinstance(item, Mapping))
        return ()

    def with_quoted_message(self, quoted: "IncomingMessage | Mapping[str, Any] | None") -> "IncomingMessage":
        """返回带引用快照的副本，不改动当前消息的正文和事件 ID。"""

        if isinstance(quoted, IncomingMessage):
            payload = quoted.as_dict()
        elif isinstance(quoted, Mapping):
            payload = dict(quoted)
        else:
            payload = {}
        return replace(self, quoted_message=payload)

    @property
    def is_poke(self) -> bool:
        return self.event_kind == "poke" or any(segment.kind == "poke" for segment in self.segments)

    @property
    def is_group_increase(self) -> bool:
        return self.event_kind == "group_increase"

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "conversation_id": self.conversation_id,
            "conversation_type": self.conversation_type,
            "event_kind": self.event_kind,
            "sender_label": self.sender_label,
            "segments": [segment.as_dict() for segment in self.segments],
            "text": self.text,
            "readable_text": self.readable_text,
            "card_context": self.card_context,
            "attachments": [dict(item) for item in self.attachments],
            "reply_message_ids": list(self.reply_message_ids),
            "quoted_message": dict(self.quoted_message) if self.quoted_message else None,
            "timestamp": self.timestamp,
        }


def _timestamp(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _reply_id_from_segments(segments: Sequence[MessageSegment]) -> str:
    for segment in segments:
        if segment.kind != "reply":
            continue
        data = segment.payload.get("data") if isinstance(segment.payload.get("data"), Mapping) else {}
        for source in (data, segment.payload):
            for key in ("id", "message_id", "messageId"):
                value = source.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
    return ""


def _embedded_quote_event(segments: Sequence[MessageSegment]) -> dict[str, Any]:
    """取 OneBot reply 段里可能随附的原始引用内容。"""

    for segment in segments:
        if segment.kind != "reply":
            continue
        data = segment.payload.get("data") if isinstance(segment.payload.get("data"), Mapping) else {}
        for source in (data, segment.payload):
            for key in ("message", "segments", "raw_message", "text"):
                value = source.get(key)
                if value in (None, "", []):
                    continue
                quoted = dict(value) if isinstance(value, Mapping) else (
                    {"message": value} if key != "raw_message" else {"raw_message": value}
                )
                # NapCat can put original-author metadata next to the quoted
                # segment list rather than inside it. Keep that provenance.
                for origin in (data, segment.payload):
                    for field_name in ("sender_id", "user_id", "userId", "sender", "time", "message_id", "message_type", "group_id"):
                        if field_name in origin:
                            quoted.setdefault(field_name, origin[field_name])
                return quoted
    return {}
