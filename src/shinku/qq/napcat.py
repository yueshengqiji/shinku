"""OneBot/NapCat 消息格式适配与可注入 HTTP action caller。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import base64
from dataclasses import dataclass
import json
import mimetypes
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
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
    "NapCatImageMaterializer",
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


class NapCatImageMaterializer:
    """按显式引用把图片准备成 data URL。

    这层可以实际读取本地图片或请求远程图片，但只有上层主动调用
    ``NapCatVisualInputBridge(materialize=...)`` 时才会发生。``media_id`` 必须由
    宿主注入 loader，避免把 NapCat action 调用偷偷塞进视觉路径。
    """

    def __init__(
        self,
        *,
        url_opener: Callable[..., Any] | None = None,
        file_reader: Callable[[str], bytes] | None = None,
        media_loader: Callable[[str], Any] | None = None,
        allowed_roots: tuple[str, ...] = (),
        timeout: float = 10.0,
        max_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        if timeout <= 0:
            raise ValueError("image materializer timeout must be positive")
        if max_bytes <= 0:
            raise ValueError("image materializer max_bytes must be positive")
        self._url_opener = url_opener or urlopen
        self._file_reader = file_reader or (lambda path: Path(path).read_bytes())
        self._media_loader = media_loader
        self._allowed_roots = tuple(Path(root).resolve() for root in allowed_roots if str(root).strip())
        self.timeout = float(timeout)
        self.max_bytes = int(max_bytes)

    def __call__(self, attachment: Mapping[str, Any]) -> str | None:
        if not isinstance(attachment, Mapping):
            return None
        url = str(attachment.get("url") or "").strip()
        if url.lower().startswith("data:image/"):
            return url
        if url.lower().startswith(("http://", "https://")):
            return self._from_url(url, attachment)

        local_path = str(attachment.get("local_path") or "").strip()
        if local_path:
            return self._from_local(local_path, attachment)

        media_id = str(attachment.get("media_id") or "").strip()
        if media_id and self._media_loader is not None:
            try:
                value = self._media_loader(media_id)
            except Exception:
                return None
            return self._value_to_data_url(value, attachment, "")
        return None

    def _from_url(self, url: str, attachment: Mapping[str, Any]) -> str | None:
        request = Request(url, headers={"Accept": "image/*"}, method="GET")
        try:
            response = self._url_opener(request, timeout=self.timeout)
            payload = self._read_limited(response)
        except (HTTPError, URLError, OSError, ValueError):
            return None
        headers = getattr(response, "headers", None)
        response_mime = ""
        if headers is not None:
            try:
                response_mime = str(headers.get_content_type() or "")
            except (AttributeError, TypeError):
                response_mime = ""
        mime = response_mime if response_mime.lower().startswith("image/") else ""
        if not mime:
            mime = str(attachment.get("mime_type") or attachment.get("mime") or "").strip()
        if not mime:
            mime = mimetypes.guess_type(url)[0] or ""
        return self._value_to_data_url(payload, attachment, mime)

    def _from_local(self, raw_path: str, attachment: Mapping[str, Any]) -> str | None:
        path = self._path_from_reference(raw_path)
        if path is None or not self._path_allowed(path):
            return None
        try:
            payload = self._read_limited(self._file_reader(str(path)))
        except (OSError, ValueError, TypeError):
            return None
        mime = str(attachment.get("mime_type") or attachment.get("mime") or "").strip()
        mime = mime or mimetypes.guess_type(str(path))[0] or ""
        return self._value_to_data_url(payload, attachment, mime)

    def _read_limited(self, source: Any) -> bytes:
        if isinstance(source, (bytes, bytearray)):
            payload = bytes(source)
        else:
            reader = getattr(source, "read", None)
            if not callable(reader):
                raise ValueError("image source is not readable")
            payload = reader(self.max_bytes + 1)
            if not isinstance(payload, (bytes, bytearray)):
                raise ValueError("image source did not return bytes")
            payload = bytes(payload)
        if not payload or len(payload) > self.max_bytes:
            raise ValueError("image payload exceeds materializer limit")
        return payload

    def _path_allowed(self, path: Path) -> bool:
        if not self._allowed_roots:
            return True
        resolved = path.resolve()
        return any(resolved == root or root in resolved.parents for root in self._allowed_roots)

    @staticmethod
    def _path_from_reference(raw_path: str) -> Path | None:
        value = str(raw_path or "").strip()
        if not value:
            return None
        if value.lower().startswith("file://"):
            parsed = urlparse(value)
            path = unquote(parsed.path or "")
            if parsed.netloc and parsed.netloc.lower() != "localhost":
                path = f"{parsed.netloc}:{path}" if len(parsed.netloc) == 1 else f"//{parsed.netloc}{path}"
            if len(path) >= 3 and path[0] == "/" and path[2] == ":":
                path = path[1:]
            return Path(path)
        return Path(value)

    @staticmethod
    def _value_to_data_url(value: Any, attachment: Mapping[str, Any], mime: str) -> str | None:
        if isinstance(value, Mapping):
            value = value.get("data_url") or value.get("dataUrl") or value.get("content")
        if isinstance(value, str):
            return value.strip() if value.lower().startswith("data:image/") else None
        if not isinstance(value, (bytes, bytearray)):
            return None
        normalized_mime = str(mime or attachment.get("mime_type") or attachment.get("mime") or "image/png").strip()
        if not normalized_mime.lower().startswith("image/"):
            normalized_mime = "image/png"
        encoded = base64.b64encode(bytes(value)).decode("ascii")
        return f"data:{normalized_mime};base64,{encoded}"


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
