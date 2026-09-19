"""出站消息的统一投递边界。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["DeliveryResult", "DeliveryTransport", "OutgoingMessage", "OutgoingSegment", "build_reply"]


_ALLOWED_KINDS = frozenset({"text", "image", "file", "audio", "video", "reply"})


@dataclass(frozen=True)
class OutgoingSegment:
    kind: str
    text: str = ""
    url: str = ""
    local_path: str = ""
    media_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if str(self.kind or "").strip().lower() not in _ALLOWED_KINDS:
            raise ValueError("unsupported outgoing segment kind")
        if self.kind == "text" and not str(self.text or "").strip():
            raise ValueError("outgoing text segment cannot be empty")
        if self.kind in {"image", "file", "audio", "video"} and not any(
            str(value or "").strip() for value in (self.url, self.local_path, self.media_id)
        ):
            raise ValueError("outgoing media segment needs a reference")

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        for key, value in (
            ("text", self.text),
            ("url", self.url),
            ("local_path", self.local_path),
            ("media_id", self.media_id),
        ):
            if str(value or "").strip():
                result[key] = str(value)
        if self.payload:
            result["payload"] = dict(self.payload)
        return result


@dataclass(frozen=True)
class OutgoingMessage:
    conversation_id: str
    segments: tuple[OutgoingSegment, ...]
    conversation_type: str = "unknown"
    reply_to_message_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.conversation_id or "").strip():
            raise ValueError("conversation_id is required")
        if not self.segments:
            raise ValueError("outgoing message needs at least one segment")
        if str(self.conversation_type or "unknown").strip().lower() not in {"private", "group", "guild", "unknown"}:
            raise ValueError("unsupported conversation type")

    def as_dict(self) -> dict[str, Any]:
        result = {
            "conversation_id": self.conversation_id,
            "conversation_type": self.conversation_type,
            "segments": [segment.as_dict() for segment in self.segments],
        }
        if self.reply_to_message_id:
            result["reply_to_message_id"] = self.reply_to_message_id
        return result


class DeliveryTransport(Protocol):
    def send(self, message: OutgoingMessage) -> Any: ...


@dataclass(frozen=True)
class DeliveryResult:
    sent: bool
    status: str
    message_ids: tuple[str, ...] = ()
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "sent": self.sent,
            "status": self.status,
            "message_ids": list(self.message_ids),
            "error": self.error,
        }


def build_reply(
    conversation_id: str,
    text: str = "",
    *,
    attachments: Sequence[Mapping[str, Any]] = (),
    conversation_type: str = "unknown",
    reply_to_message_id: str = "",
) -> OutgoingMessage:
    """把一轮回复构造成一个不可拆分的出站消息。"""

    segments: list[OutgoingSegment] = []
    if str(text or "").strip():
        segments.append(OutgoingSegment(kind="text", text=str(text).strip()))
    for attachment in attachments:
        if not isinstance(attachment, Mapping):
            continue
        kind = str(attachment.get("kind") or "image").strip().lower()
        if kind not in {"image", "file", "audio", "video"}:
            continue
        segments.append(
            OutgoingSegment(
                kind=kind,
                url=str(attachment.get("url") or ""),
                local_path=str(attachment.get("local_path") or ""),
                media_id=str(attachment.get("media_id") or ""),
                payload=dict(attachment),
            )
        )
    return OutgoingMessage(
        conversation_id=str(conversation_id or "").strip(),
        segments=tuple(segments),
        conversation_type=str(conversation_type or "unknown").strip().lower(),
        reply_to_message_id=str(reply_to_message_id or "").strip(),
    )


def deliver(transport: DeliveryTransport, message: OutgoingMessage) -> DeliveryResult:
    """调用一次外部 transport；不生成替代文本，不拆分图片和文字。"""

    if not callable(getattr(transport, "send", None)):
        return DeliveryResult(False, "transport_invalid", error="delivery_transport_invalid")
    try:
        raw = transport.send(message)
    except Exception as exc:
        return DeliveryResult(False, "transport_error", error=type(exc).__name__)
    if not isinstance(raw, Mapping) or raw.get("ok") is False:
        return DeliveryResult(False, "rejected", error=str((raw or {}).get("error") or "delivery_rejected") if isinstance(raw, Mapping) else "delivery_rejected")
    raw_ids = raw.get("message_ids", raw.get("message_id", ()))
    if isinstance(raw_ids, str):
        ids = (raw_ids,) if raw_ids.strip() else ()
    elif isinstance(raw_ids, Sequence):
        ids = tuple(str(item).strip() for item in raw_ids if str(item).strip())
    else:
        ids = ()
    return DeliveryResult(True, "sent", message_ids=ids)


__all__.append("deliver")
