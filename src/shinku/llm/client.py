"""LLM 传输客户端：把两种上游协议收进同一个 OpenAI 形状的调用面。

本项目的其余部分只认一种调用形状——``client.chat.completions.create(**options)``——
并按 OpenAI 的字段读回结果（``choices`` / ``message.content`` / ``usage``）。
OpenAI 兼容端点直接复用官方 SDK；Anthropic 的 Messages 端点形状差得远，
这里在本地做一层适配：请求体改写、HTTP 直发、SSE 逐事件翻译、响应回装成 OpenAI 形状。

``build_llm_client`` 只负责「按协议挑一个客户端」：不缓存、不重试、不读配置文件，
收到什么参数就用什么参数。协议名优先取显式传参，没传就按 ``base_url`` 猜。

关于一个可选的项目级兼容标记：上游把这个客户端设计成「项目侧可以再挂一个私有的
协议别名，供旧运行时代码读」。本项目的运行时是同批重写的，统一读规范属性
``protocol``，因此**这里不提供任何别名**（契约文档 §5 决定 1 记了理由与回退路径）。
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from types import SimpleNamespace

import requests
from openai import OpenAI

from ..providers.config import normalize_api_protocol, normalize_base_url

__all__ = [
    "AnthropicCompatClient",
    "build_llm_client",
]

#: 未配置密钥时给 SDK 的占位串。SDK 在构造期就校验密钥，占位串让它先立起来，
#: 真正的鉴权失败留给第一次请求去报（错误信息更准确）。
_KEY_PLACEHOLDERS = {"ollama": "ollama"}
_KEY_FALLBACK = "not-configured"

#: Anthropic 的 ``max_tokens`` 是必填项，缺省时用它兜底。
_DEFAULT_MAX_TOKENS = 1024

#: 系统提示词块允许带 ``cache_control`` 的块数上限。超出部分照常发送，只是不再缓存。
_SYSTEM_CACHE_SLOTS = 4

#: Anthropic Messages API 的版本头取值（线路协议，不随本项目改名而改）。
_ANTHROPIC_API_VERSION = "2023-06-01"

#: ``data:image/...;base64,...`` 形式的内联图片。
_INLINE_IMAGE_RE = re.compile(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.+)$", re.IGNORECASE)

#: Anthropic 的停止原因 → OpenAI 的 ``finish_reason``。
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def build_llm_client(
    *,
    api_key: str,
    base_url: str,
    timeout: float = 60.0,
    max_retries: int = 0,
    protocol: str = "auto",
    default_max_tokens: int = _DEFAULT_MAX_TOKENS,
    system_cache_slots: int = _SYSTEM_CACHE_SLOTS,
):
    """按协议造客户端：显式协议优先，``auto`` 时按 ``base_url`` 猜，兜底 openai。

    Anthropic 走本地适配层，其余协议交给官方 SDK。无论走哪条路，返回对象上都有
    可读的 ``protocol`` 与 OpenAI 形状的 ``chat.completions``。
    """

    resolved = normalize_api_protocol(protocol=protocol, base_url=base_url)
    if resolved == "anthropic":
        return AnthropicCompatClient(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_max_tokens=default_max_tokens,
            system_cache_slots=system_cache_slots,
        )

    client = OpenAI(
        api_key=str(api_key or "").strip() or _placeholder_key(resolved),
        base_url=normalize_base_url(protocol=resolved, base_url=base_url),
        timeout=timeout,
        max_retries=max_retries,
    )
    # 挂协议名用 setattr：SDK 的客户端对象是否允许加属性由它自己决定，
    # 不允许时也不该让构造失败——协议名只是给调用层看的。
    try:
        setattr(client, "protocol", resolved)
    except Exception:
        pass
    return client


class AnthropicCompatClient:
    """Anthropic Messages 端点的 OpenAI 形状外壳。

    只实现调用层真正用到的那一路（``chat.completions.create``）；
    其余字段挂在实例上供外部读取，构造时就地归一化。
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
        max_retries: int = 0,
        default_max_tokens: int = _DEFAULT_MAX_TOKENS,
        system_cache_slots: int = _SYSTEM_CACHE_SLOTS,
    ):
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "").strip()
        self.timeout = float(timeout or 60.0)
        self.max_retries = int(max_retries or 0)
        self.default_max_tokens = max(1, int(default_max_tokens or _DEFAULT_MAX_TOKENS))
        self.system_cache_slots = max(0, int(system_cache_slots or 0))
        self.protocol = "anthropic"
        self.chat = _ChatSurface(self)


class _ChatSurface:
    """``client.chat``：只往下一层挂 ``completions``。"""

    def __init__(self, owner: AnthropicCompatClient) -> None:
        self.completions = _CompletionSurface(owner)


class _CompletionSurface:
    """``client.chat.completions``：把一次调用翻译成一次 Messages 请求。"""

    def __init__(self, owner: AnthropicCompatClient) -> None:
        self._owner = owner

    def create(self, **options):
        planned = _prepare_request(
            options,
            fallback_timeout=self._owner.timeout,
            default_max_tokens=self._owner.default_max_tokens,
            system_cache_slots=self._owner.system_cache_slots,
        )
        reply = requests.post(
            _messages_endpoint(self._owner.base_url),
            headers=_request_headers(self._owner.api_key),
            json=planned.payload,
            timeout=planned.timeout,
            stream=planned.stream,
        )
        reply.encoding = "utf-8"
        _reject_bad_status(reply)
        label = planned.payload.get("model", "")
        if planned.stream:
            return _MessageEventStream(reply, label)
        return _as_chat_completion(reply.json(), label)


@dataclass(frozen=True)
class _PreparedRequest:
    """一次出站请求的三件套：请求体、超时、是否流式。"""

    payload: dict
    timeout: float
    stream: bool


def _placeholder_key(protocol: str) -> str:
    return _KEY_PLACEHOLDERS.get(protocol, _KEY_FALLBACK)


def _prepare_request(
    options: dict,
    *,
    fallback_timeout: float,
    default_max_tokens: int = _DEFAULT_MAX_TOKENS,
    system_cache_slots: int = _SYSTEM_CACHE_SLOTS,
) -> _PreparedRequest:
    """把 OpenAI 形状的调用参数改写成 Messages 请求体。"""

    system_text, messages = _split_roles(options.get("messages") or [])
    extras = _clean_strings(options.get("system_extra_blocks"))
    payload: dict = {
        "model": options.get("model", ""),
        "messages": messages,
        "max_tokens": int(
            options.get("max_tokens")
            or options.get("max_completion_tokens")
            or max(1, int(default_max_tokens or _DEFAULT_MAX_TOKENS))
        ),
    }
    blocks = _system_block_list([system_text, *extras], cache_slots=system_cache_slots)
    if blocks:
        payload["system"] = blocks

    if options.get("temperature") is not None:
        payload["temperature"] = min(1.0, max(0.0, float(options["temperature"])))
    if options.get("top_p") is not None:
        payload["top_p"] = float(options["top_p"])

    stop = options.get("stop")
    if isinstance(stop, str) and stop.strip():
        payload["stop_sequences"] = [stop.strip()]
    elif isinstance(stop, list):
        # 注意：这里**不走** `_clean_strings`。停止序列列表按字面 str() 取值
        # （`None` 会变成 "None" 并留下），与系统提示词块的取值语义不同。
        stops = [str(item).strip() for item in stop if str(item).strip()]
        if stops:
            payload["stop_sequences"] = stops

    if options.get("stream") is not None:
        payload["stream"] = bool(options.get("stream"))

    extra_body = options.get("extra_body")
    if isinstance(extra_body, dict):
        for name, value in extra_body.items():
            # 已由上面的显式字段定过的一律不覆盖。
            if name not in payload:
                payload[name] = value

    chosen = options.get("timeout", fallback_timeout)
    return _PreparedRequest(
        payload=payload,
        timeout=float(chosen or fallback_timeout),
        stream=bool(options.get("stream")),
    )


def _clean_strings(values) -> list[str]:
    """留下去空白后非空的那些；非字符串一律先 ``str()``。"""
    return [text for item in (values or []) if (text := str(item or "").strip())]


def _system_block_list(parts: list, *, cache_slots: int = _SYSTEM_CACHE_SLOTS) -> list[dict]:
    """把系统提示词切成块；前 ``cache_slots`` 块带缓存标记。"""

    blocks: list[dict] = []
    slots = max(0, int(cache_slots or 0))
    for part in parts:
        text = str(part or "").strip()
        if not text:
            continue
        block: dict = {"type": "text", "text": text}
        if slots:
            block["cache_control"] = {"type": "ephemeral"}
            slots -= 1
        blocks.append(block)
    return blocks


def _split_roles(messages: list) -> tuple[str, list]:
    """把消息表拆成「系统提示词文本」与「user/assistant 轮次」。"""

    system_parts: list[str] = []
    converted: list[dict] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user") or "user").strip().lower()
        content = item.get("content", "")
        if role in {"system", "developer"}:
            text = _as_plain_text(content)
            if text:
                system_parts.append(text)
            continue
        # Messages 端点只认 user/assistant，其余一律当 user 发。
        if role not in {"user", "assistant"}:
            role = "user"
        converted.append({"role": role, "content": _as_content_blocks(content)})
    return "\n".join(system_parts).strip(), converted


def _as_plain_text(content) -> str:
    """把任意内容压成纯文本：字符串原样，块列表取其中的 text 块。"""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = [
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict) and str(item.get("type", "")).strip() == "text"
        ]
        return "\n".join(piece for piece in pieces if piece).strip()
    return str(content or "").strip()


def _as_content_blocks(content):
    """把消息内容转成 Messages 端点的块列表；转不出块就退回纯文本。"""

    if isinstance(content, str):
        return content
    collected: list[dict] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type", "")).strip()
            if kind == "text":
                collected.append({"type": "text", "text": str(item.get("text", ""))})
            elif kind == "image_url":
                source = _as_image_source(item.get("image_url"))
                if source is not None:
                    collected.append(source)
        if collected:
            return collected
    return _as_plain_text(content)


def _as_image_source(value):
    """``image_url`` 块 → Messages 端点的 image 源；空 URL 返回 ``None``。"""

    if isinstance(value, dict):
        url = str(value.get("url", "")).strip()
    else:
        url = str(value or "").strip()
    if not url:
        return None
    inline = _INLINE_IMAGE_RE.match(url)
    if inline:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": inline.group(1),
                "data": inline.group(2),
            },
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _messages_endpoint(base_url: str) -> str:
    """由 base_url 拼出 Messages 端点，三种已归一的写法都不重复拼 ``/v1``。"""

    base = str(base_url or "").strip().rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return base + "/messages"
    return base + "/v1/messages"


def _request_headers(api_key: str) -> dict:
    return {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_API_VERSION,
    }


def _reject_bad_status(reply: requests.Response) -> None:
    """非 2xx 一律抛 ``RuntimeError``，尽量带上服务端给的错误说明。"""

    if reply.ok:
        return
    detail = ""
    try:
        body = reply.json()
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message", "")).strip()
            elif error:
                detail = str(error).strip()
    except Exception:
        detail = reply.text.strip()
    raise RuntimeError(detail or f"Anthropic request failed: HTTP {reply.status_code}")


def _joined_text(body: dict) -> str:
    return "".join(
        str(item.get("text", ""))
        for item in body.get("content") or []
        if isinstance(item, dict) and str(item.get("type", "")).strip() == "text"
    ).strip()


def _as_chat_completion(body: dict, requested_model: str):
    """把 Messages 响应回装成调用层认得的样子。"""

    usage = body.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    return SimpleNamespace(
        id=body.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
        object="chat.completion",
        created=0,
        model=body.get("model") or requested_model,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason=_canonical_finish_reason(body.get("stop_reason")),
                message=SimpleNamespace(
                    role="assistant",
                    content=_joined_text(body),
                    tool_calls=[],
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
            cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
        ),
    )


class _MessageEventStream:
    """把 Messages 端点的 SSE 事件流转成一条条 OpenAI 形状的增量块。

    事件之间用空行分隔；``event:`` 给事件名，``data:`` 给载荷（可能多行）。
    流尾若还挂着没收尾的 ``data:``，也要照常翻一个块出来。
    无论正常读完还是中途异常，退出时都关掉底层连接。
    """

    def __init__(self, reply: requests.Response, model: str) -> None:
        self._reply = reply
        self._label = model
        #: 由 ``message_start`` 事件里带的 usage 填上，没有就是 ``None``。
        self.usage = None

    def __iter__(self):
        event = ""
        data: list[str] = []
        try:
            for raw in self._reply.iter_lines(decode_unicode=False):
                if raw is None:
                    continue
                if isinstance(raw, bytes):
                    line = raw.decode("utf-8", errors="replace").strip()
                else:
                    line = str(raw).strip()
                if not line:
                    if event == "message_start" and data:
                        self._absorb_usage(data)
                    chunk = _event_to_chunk(event, data, self._label)
                    if chunk is not None:
                        yield chunk
                    event, data = "", []
                    continue
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data.append(line.split(":", 1)[1].strip())
            if data:
                chunk = _event_to_chunk(event, data, self._label)
                if chunk is not None:
                    yield chunk
        finally:
            self._reply.close()

    def _absorb_usage(self, data: list[str]) -> None:
        """``message_start`` 里带缓存读写计数，取不到就算了（不影响正文）。"""

        try:
            payload = json.loads("\n".join(data).strip())
            usage = (payload.get("message") or {}).get("usage") or {}
            self.usage = SimpleNamespace(
                cache_read_input_tokens=int(usage.get("cache_read_input_tokens", 0) or 0),
                cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens", 0) or 0),
            )
        except Exception:
            pass


def _event_to_chunk(event: str, data: list[str], model: str):
    """一个 SSE 事件 → 一个增量块；不产生正文与结束信号的事件返回 ``None``。"""

    if not data:
        return None
    raw = "\n".join(data).strip()
    if not raw or raw == "[DONE]":
        return None
    payload = json.loads(raw)
    if event == "content_block_start":
        text = str((payload.get("content_block") or {}).get("text", "") or "")
        return _delta_chunk(text, model) if text else None
    if event == "content_block_delta":
        text = str((payload.get("delta") or {}).get("text", "") or "")
        return _delta_chunk(text, model) if text else None
    if event == "message_delta":
        reason = (payload.get("delta") or {}).get("stop_reason")
        return _delta_chunk("", model, _canonical_finish_reason(reason))
    return None


def _delta_chunk(text: str, model: str, finish_reason: str | None = None):
    return SimpleNamespace(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion.chunk",
        created=0,
        model=model,
        choices=[
            SimpleNamespace(
                index=0,
                delta=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
    )


def _canonical_finish_reason(reason) -> str | None:
    return _STOP_REASON_MAP.get(str(reason or "").strip())
