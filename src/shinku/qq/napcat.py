"""OneBot/NapCat 消息格式适配，不包含网络客户端。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .delivery import DeliveryTransport, OutgoingMessage
from .message import IncomingMessage

__all__ = ["NapCatActionTransport", "NapCatEventAdapter"]


class NapCatEventAdapter:
    """只负责验证并归一化 OneBot 事件。"""

    def normalize(self, event: Mapping[str, Any]) -> IncomingMessage:
        if not isinstance(event, Mapping):
            raise TypeError("napcat event must be a mapping")
        if str(event.get("post_type") or "message").strip().lower() != "message":
            raise ValueError("napcat event is not a message event")
        return IncomingMessage.from_event(event)


class NapCatActionTransport(DeliveryTransport):
    """将出站消息转换成 OneBot action，由调用方注入真正的 HTTP/WebSocket。"""

    def __init__(self, action_call: Callable[..., Any]) -> None:
        if not callable(action_call):
            raise TypeError("action_call must be callable")
        self.action_call = action_call

    def send(self, message: OutgoingMessage) -> dict[str, Any]:
        action, params = self._build_action(message)
        raw = self.action_call(action=action, params=params)
        return self._normalize_response(raw)

    @staticmethod
    def _build_action(message: OutgoingMessage) -> tuple[str, dict[str, Any]]:
        conversation_type = str(message.conversation_type or "unknown").strip().lower()
        if conversation_type == "private":
            action = "send_private_msg"
            target_key = "user_id"
        elif conversation_type == "group":
            action = "send_group_msg"
            target_key = "group_id"
        else:
            raise ValueError("napcat target conversation type is required")
        onebot_segments: list[dict[str, Any]] = []
        if message.reply_to_message_id:
            onebot_segments.append({"type": "reply", "data": {"id": message.reply_to_message_id}})
        for segment in message.segments:
            if segment.kind == "text":
                onebot_segments.append({"type": "text", "data": {"text": segment.text}})
            elif segment.kind in {"image", "file", "audio", "video"}:
                file_ref = segment.url or segment.local_path or segment.media_id
                onebot_segments.append({"type": segment.kind, "data": {"file": file_ref, **dict(segment.payload)}})
            elif segment.kind == "reply":
                onebot_segments.append({"type": "reply", "data": dict(segment.payload)})
        return action, {target_key: message.conversation_id, "message": onebot_segments}

    @staticmethod
    def _normalize_response(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            return {"ok": False, "error": "invalid_napcat_response"}
        status = str(raw.get("status") or "").strip().lower()
        retcode = raw.get("retcode", 0)
        try:
            failed = int(retcode) != 0
        except (TypeError, ValueError):
            failed = True
        if status and status not in {"ok", "async"}:
            failed = True
        if failed:
            return {"ok": False, "error": str(raw.get("message") or raw.get("wording") or "napcat_action_failed")}
        data = raw.get("data") if isinstance(raw.get("data"), Mapping) else {}
        message_id = data.get("message_id") or raw.get("message_id")
        return {"ok": True, "message_ids": [str(message_id)] if message_id is not None else []}
