"""OneBot/NapCat 消息格式适配与可注入 HTTP action caller。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import base64
from dataclasses import dataclass
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .delivery import DeliveryTransport, OutgoingMessage
from .message import IncomingMessage

__all__ = [
    "NapCatActionTransport",
    "NapCatEventAdapter",
    "NapCatEventDecoder",
    "NapCatHttpActionCaller",
    "NapCatVisualInput",
    "NapCatVisualInputBridge",
    "NapCatVisualBatch",
]


class NapCatEventAdapter:
    """只负责验证并归一化 OneBot 事件。"""

    def normalize(self, event: Mapping[str, Any]) -> IncomingMessage:
        if not isinstance(event, Mapping):
            raise TypeError("napcat event must be a mapping")
        if str(event.get("post_type") or "message").strip().lower() != "message":
            raise ValueError("napcat event is not a message event")
        return IncomingMessage.from_event(event)


class NapCatEventDecoder:
    """解码 NapCat webhook 的 JSON body，再交给事件适配器。"""

    def __init__(self, adapter: NapCatEventAdapter | None = None) -> None:
        self.adapter = adapter or NapCatEventAdapter()

    def decode(self, payload: bytes | bytearray | str | Mapping[str, Any]) -> IncomingMessage:
        if isinstance(payload, Mapping):
            event = dict(payload)
        else:
            if isinstance(payload, (bytes, bytearray)):
                try:
                    payload = bytes(payload).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError("napcat event body must be UTF-8 JSON") from exc
            try:
                event = json.loads(str(payload))
            except (TypeError, ValueError) as exc:
                raise ValueError("napcat event body must be valid JSON") from exc
        if not isinstance(event, Mapping):
            raise ValueError("napcat event JSON must be an object")
        return self.adapter.normalize(event)


@dataclass(frozen=True)
class NapCatVisualInput:
    """一张图片从 QQ 引用到模型输入之间的显式状态。"""

    source_kind: str
    reference: str
    data_url: str = ""

    @property
    def ready(self) -> bool:
        return self.data_url.lower().startswith("data:image/")

    def as_model_item(self) -> dict[str, Any] | None:
        if not self.ready:
            return None
        return {"data_url": self.data_url}


@dataclass(frozen=True)
class NapCatVisualBatch:
    """图片输入准备结果；未解析的图片不会被伪装成“没有图片”。"""

    inputs: tuple[NapCatVisualInput, ...]

    @property
    def ready(self) -> bool:
        return bool(self.inputs) and all(item.ready for item in self.inputs)

    @property
    def pending(self) -> bool:
        return any(not item.ready for item in self.inputs)

    def model_images(self) -> list[dict[str, Any]]:
        return [item for item in (entry.as_model_item() for entry in self.inputs) if item is not None]


class NapCatVisualInputBridge:
    """把 NapCat 图片引用转换成 runtime 可消费的 data URL。

    ``materialize`` 是唯一的下载/读文件边界。没有注入它时，远程 URL、本地路径和
    media id 会保持为 pending，不会被折叠成 ``[图片]``，也不会触发隐式网络或文件读取。
    """

    def __init__(self, materialize: Callable[[Mapping[str, Any]], Any] | None = None) -> None:
        self.materialize = materialize

    def build(self, message: IncomingMessage) -> NapCatVisualBatch:
        inputs: list[NapCatVisualInput] = []
        for attachment in message.attachments:
            if str(attachment.get("kind") or "").strip().lower() != "image":
                continue
            reference = next(
                (
                    str(attachment.get(key) or "").strip()
                    for key in ("url", "local_path", "media_id")
                    if str(attachment.get(key) or "").strip()
                ),
                "",
            )
            source_kind = next(
                (key for key in ("url", "local_path", "media_id") if str(attachment.get(key) or "").strip()),
                "unknown",
            )
            data_url = reference if reference.lower().startswith("data:image/") else ""
            if not data_url and self.materialize is not None:
                data_url = self._coerce_data_url(self.materialize(dict(attachment)), attachment)
            inputs.append(NapCatVisualInput(source_kind, reference, data_url))
        return NapCatVisualBatch(tuple(inputs))

    @staticmethod
    def _coerce_data_url(value: Any, attachment: Mapping[str, Any]) -> str:
        if isinstance(value, Mapping):
            value = value.get("data_url") or value.get("dataUrl") or value.get("content")
        if isinstance(value, str):
            candidate = value.strip()
            return candidate if candidate.lower().startswith("data:image/") else ""
        if not isinstance(value, (bytes, bytearray)):
            return ""
        mime = str(attachment.get("mime_type") or attachment.get("mime") or "image/png").strip()
        if not mime.lower().startswith("image/"):
            mime = "image/png"
        encoded = base64.b64encode(bytes(value)).decode("ascii")
        return f"data:{mime};base64,{encoded}"


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
