"""LLM runtime facade; channels, payload, transport and response parsing are separate responsibilities: runtime_response."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from shinku.tools.invocation import NATIVE_TOOL_CALL_FIELD

from .runtime_reply_format import normalize_reply_medium
from .runtime_stream_parser import StreamTap
from .runtime_types import JSON_RE


@dataclass(frozen=True)
class RuntimeResponseDependencies:
    """Bound callbacks and live getters for this responsibility."""
    first_json_object: Callable[..., Any]
    flatten_content: Callable[..., Any]
    heal_json: Callable[..., Any]
    parse_line: Callable[..., Any]
    read_json_value: Callable[..., Any]
    scan_stream_tool_call: Callable[..., Any]
    skip_ws: Callable[..., Any]


def chat_reply_needs_rescue(parsed: Any) -> bool:
    """判断流式结果还能不能当一轮回复用。

        供应商可能干净地关掉流却一个字都没给，也可能吐半截 / 非 JSON 正文。
        这两种过去会直接滑进 persona 兜底。注意「有意沉默」是合法结果，
        不能再额外打一次模型。
        """

    if not isinstance(parsed, dict):
        return True
    if parsed.get("tool_call") or parsed.get(NATIVE_TOOL_CALL_FIELD):
        return False
    status = str(parsed.get("status") or "final").strip().lower()
    if status in {"skip", "silent", "decline"}:
        return False
    if str(parsed.get("speech") or "").strip():
        return False
    if str(parsed.get("text") or parsed.get("reply") or "").strip() and str(parsed.get("kind") or "") in {"final", "wait"}:
        return False
    segments = parsed.get("speech_segments")
    return not isinstance(segments, list) or not any(str(item or "").strip() for item in segments)


def scan_stream_tool_call(
    text: str,
    *,
    deps: RuntimeResponseDependencies,
) -> tuple[str, dict[str, Any] | None]:
    raw = str(text or "")
    length = len(raw)
    idx = deps.skip_ws(raw, 0)
    if idx >= length:
        return "pending", None
    if raw[idx] != "{":
        return "none", None
    decoder = json.JSONDecoder()
    idx += 1

    while True:
        idx = deps.skip_ws(raw, idx)
        if idx >= length:
            return "pending", None
        if raw[idx] == "}":
            return "none", None
        if raw[idx] != '"':
            return "pending", None
        try:
            key, key_end = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            return "pending", None
        if not isinstance(key, str):
            return "none", None
        idx = deps.skip_ws(raw, key_end)
        if idx >= length:
            return "pending", None
        if raw[idx] != ":":
            return "pending", None
        idx = deps.skip_ws(raw, idx + 1)
        if idx >= length:
            return "pending", None
        try:
            value, value_end = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            return "pending", None
        if key == "tool_call":
            if value is None:
                return "null", None
            return ("object", value) if isinstance(value, dict) else ("none", None)
        idx = deps.skip_ws(raw, value_end)
        if idx >= length:
            return "pending", None
        if raw[idx] == ",":
            idx += 1
            continue
        if raw[idx] == "}":
            return "none", None
        return "pending", None


def scan_leading_tool_call(
    text: str,
    *,
    deps: RuntimeResponseDependencies,
) -> tuple[str, dict[str, Any] | None]:
    return deps.scan_stream_tool_call(text)


def skip_ws(text: str, start: int) -> int:
    idx = int(start)
    while idx < len(text) and text[idx] in " \t\r\n":
        idx += 1
    return idx


def read_content_text(response: Any, *, deps: RuntimeResponseDependencies) -> str:
    try:
        return deps.flatten_content(response.choices[0].message.content).strip()
    except Exception:
        return ""


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is None:
                    text = item.get("content")
                if text is None and isinstance(item.get("input_text"), str):
                    text = item.get("input_text")
            else:
                text = getattr(item, "text", None)
                if text is None:
                    text = getattr(item, "content", None)
            if text is not None:
                parts.append(str(text))
        return "".join(parts)
    return str(content or "")


def read_stream_text(chunk: Any) -> str:
    try:
        choice = chunk.choices[0]
    except Exception:
        return ""

    delta = getattr(choice, "delta", None)
    if delta is None and isinstance(choice, dict):
        delta = choice.get("delta")

    if isinstance(delta, dict):
        content = delta.get("content")
    else:
        content = getattr(delta, "content", None)

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                text = str(getattr(item, "text", "") or getattr(item, "content", "") or "").strip()
            if text:
                parts.append(text)
        return "".join(parts)
    return str(content or "")


def drain_buffer(
    buffer: str,
    *,
    deps: RuntimeResponseDependencies,
) -> tuple[str, list[dict[str, Any]]]:
    remaining = str(buffer or "")
    events: list[dict[str, Any]] = []
    while True:
        newline_index = remaining.find("\n")
        if newline_index < 0:
            break
        line = remaining[:newline_index]
        remaining = remaining[newline_index + 1:]
        parsed = deps.parse_line(line)
        if parsed is not None:
            events.append(parsed)
    return remaining, events


def parse_line(line: Any) -> dict[str, Any] | None:
    raw = str(line or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def close_response(response: Any) -> None:
    if response is None:
        return
    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
            return
        except Exception:
            pass
    raw_response = getattr(response, "_response", None)
    close = getattr(raw_response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def parse_json(text: str, *, deps: RuntimeResponseDependencies) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    parsed = deps.first_json_object(raw)
    if isinstance(parsed, dict):
        return parsed
    match = JSON_RE.search(raw)
    if not match:
        repaired = deps.heal_json(raw)
        return repaired
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except Exception:
        repaired = deps.heal_json(match.group(0))
        return repaired


def heal_json(text: str) -> dict[str, Any] | None:
    try:
        import json_repair

        repaired = json_repair.repair_json(text, return_objects=True)
        return repaired if isinstance(repaired, dict) else None
    except Exception:
        return None


def first_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "")
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw):
        start = match.start()
        prefix = raw[:start].rstrip()
        if prefix and prefix[-1] in {":", "[", ","}:
            continue
        try:
            payload, _stop = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def recover_partial(
    text: str,
    *,
    fallback: dict[str, Any],
    deps: RuntimeResponseDependencies,
) -> dict[str, Any] | None:
    tap = StreamTap()
    tap.feed(text)
    speech = str(tap.latest_speech or "").strip()
    emotion = str(tap.latest_emotion or "").strip()
    reply_medium = normalize_reply_medium(tap.latest_reply_medium)
    raw_segments = deps.read_json_value(text, "speech_segments")
    segments: list[str] = []
    if isinstance(raw_segments, list):
        for item in raw_segments:
            value = (item.get("speech") or item.get("text") or "") if isinstance(item, dict) else item
            segment = " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip(
            )
            if segment:
                segments.append(segment[:500])
            if len(segments) >= 3:
                break
    if not speech and not emotion and not segments and not reply_medium:
        return None
    recovered = dict(fallback)
    if emotion:
        recovered["emotion"] = emotion
    if reply_medium:
        recovered["reply_medium"] = reply_medium
    if segments:
        recovered["speech"] = "\n".join(segments)
        recovered["speech_segments"] = segments
    elif speech:
        recovered["speech"] = speech
        recovered["speech_segments"] = []
    if (speech or segments) and recovered.pop("_agent_transport_fallback", False) is True:
        # Real recovered speech takes precedence over the local failure
        # template. Emotion/media metadata alone must not authorize it.
        recovered.pop("text", None)
        if recovered.pop("_agent_plain_text_verify", False) is True and recovered.get("task_outcome") == "failed":
            explicit_outcome = deps.read_json_value(text, "task_outcome") or deps.read_json_value(text, "outcome")
            if isinstance(explicit_outcome, str) and explicit_outcome.strip():
                recovered["task_outcome"] = explicit_outcome.strip().lower()
            else:
                recovered.pop("task_outcome", None)
    return recovered


def read_json_value(text: str, key: str, *, deps: RuntimeResponseDependencies) -> Any:
    raw = str(text or "")
    key_text = str(key or "").strip()
    if not key_text:
        return None
    pattern = re.compile(r'"' + re.escape(key_text) + r'"\s*:\s*')
    for match in pattern.finditer(raw):
        start = deps.skip_ws(raw, match.end())
        if start >= len(raw):
            continue
        try:
            # A complete scalar (e.g. task_outcome) is recoverable too;
            # the compound-value scanner only recognizes arrays/objects.
            return json.JSONDecoder().raw_decode(raw[start:])[0]
        except Exception:
            continue
    return None
