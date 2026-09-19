"""OneBot/NapCat 消息格式适配与可注入 HTTP action caller。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .delivery import DeliveryTransport, OutgoingMessage
from .message import IncomingMessage

__all__ = ["NapCatActionTransport", "NapCatEventAdapter", "NapCatHttpActionCaller"]


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


class NapCatHttpActionCaller:
    """通过 OneBot HTTP API 执行 action。

    网络边界留在这个 caller 里，消息域仍只依赖 ``NapCatActionTransport`` 的
    callable。默认使用标准库传输；测试或上层宿主可以注入 ``opener``，因此
    构造对象不会自动发请求。
    """

    def __init__(
        self,
        base_url: str,
        *,
        access_token: str = "",
        timeout: float = 10.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        normalized = str(base_url or "").strip()
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("napcat base_url must use http or https")
        if timeout <= 0:
            raise ValueError("napcat timeout must be positive")
        self.base_url = normalized.rstrip("/") + "/"
        self.access_token = str(access_token or "").strip()
        self.timeout = float(timeout)
        self._opener = opener or urlopen

    def __call__(self, *, action: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        action_name = str(action or "").strip().strip("/")
        if not action_name or "/" in action_name or "\\" in action_name:
            raise ValueError("napcat action name must be a single path segment")
        if not isinstance(params, Mapping):
            raise TypeError("napcat action params must be a mapping")

        request = Request(
            urljoin(self.base_url, action_name),
            data=json.dumps(dict(params), ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            method="POST",
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        if self.access_token:
            request.add_header("Authorization", f"Bearer {self.access_token}")

        try:
            response = self._opener(request, timeout=self.timeout)
            payload = response.read()
        except HTTPError as exc:
            return {
                "status": "failed",
                "retcode": int(exc.code),
                "message": "napcat_http_error",
            }
        except URLError as exc:
            raise ConnectionError("napcat HTTP action request failed") from exc

        try:
            decoded = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
        except (TypeError, ValueError):
            return {"status": "failed", "retcode": -1, "message": "napcat_invalid_json"}
        if not isinstance(decoded, Mapping):
            return {"status": "failed", "retcode": -1, "message": "napcat_invalid_response"}
        return dict(decoded)
