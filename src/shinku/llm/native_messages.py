"""Bounded OpenAI assistant/tool messages for explicitly opted-in execution.

Provider reasoning stays in process-local protocol state by default. It is never
added to the parsed decision, tool evidence, task ledger, or prompt audit.
Angelos 可按用户显式要求通过宿主接收器保存独立脱敏展示副本，不能回灌为证据。
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import json
from typing import Any

MAX_NATIVE_MESSAGES = 256
MAX_NATIVE_BYTES = 2 * 1024 * 1024
MAX_NATIVE_HISTORY_BYTES = 16 * 1024 * 1024
MAX_NATIVE_HISTORY_MESSAGES = 4096


class NativeProtocolError(ValueError):
    """A static error code; never include provider text or credentials."""


def _value(item: Any, key: str, default=None):
    return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)


def assistant_message(response: Any) -> dict[str, Any]:
    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, TypeError):
        raise NativeProtocolError("native_assistant_message_missing") from None
    content = _value(message, "content")
    reasoning = _value(message, "reasoning_content")
    if (content is not None and not isinstance(content, str)) or (
            reasoning is not None and not isinstance(reasoning, str)):
        raise NativeProtocolError("native_assistant_content_invalid")
    # Empty content is accepted by gateways requiring a non-null assistant body.
    result = {"role": "assistant", "content": content or ""}
    if reasoning is not None:
        result["reasoning_content"] = reasoning
    calls = _value(message, "tool_calls")
    if calls:
        if not isinstance(calls, (list, tuple)) or len(calls) > 32:
            raise NativeProtocolError("native_tool_calls_invalid")
        normalized, ids = [], set()
        for call in calls:
            ident, kind = _value(call, "id"), _value(call, "type")
            function = _value(call, "function")
            name, arguments = _value(function, "name"), _value(function, "arguments")
            if (not isinstance(ident, str) or not ident or len(ident) > 512 or ident in ids
                    or kind != "function" or not isinstance(name, str) or not name or len(name) > 128
                    or not isinstance(arguments, str)):
                raise NativeProtocolError("native_tool_call_invalid")
            ids.add(ident)
            normalized.append({"id": ident, "type": "function", "function": {
                "name": name, "arguments": arguments}})
        result["tool_calls"] = normalized
    _check_size([result])
    return result


def _check_size(messages, *, history=False):
    # 单条供应商消息仍有资源保护；完整旧链可在模型窗口处摘要，不先撞2MiB历史限制。
    max_messages = MAX_NATIVE_HISTORY_MESSAGES if history else MAX_NATIVE_MESSAGES
    max_bytes = MAX_NATIVE_HISTORY_BYTES if history else MAX_NATIVE_BYTES
    if (len(messages) > max_messages or len(json.dumps(messages, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")) > max_bytes):
        raise NativeProtocolError("native_history_resource_limit")


def validate_native_messages(value: Any) -> list[dict[str, Any]]:
    """Reject incomplete/forged pairs before SDK serialization; no silent trimming."""
    if not isinstance(value, (list, tuple)):
        raise NativeProtocolError("native_history_invalid")
    _check_size(value, history=True)
    result, pending, seen = [], set(), set()
    for row in value:
        if not isinstance(row, Mapping):
            raise NativeProtocolError("native_history_invalid")
        role = row.get("role")
        if role == "assistant":
            if pending or set(row) - {"role", "content", "reasoning_content", "tool_calls"}:
                raise NativeProtocolError("native_history_unpaired")
            class Response:
                choices = [type("Choice", (), {"message": row})()]
            item = assistant_message(Response())
            for call in item.get("tool_calls", []):
                ident = call["id"]
                if ident in seen:
                    raise NativeProtocolError("native_tool_id_reused")
                pending.add(ident)
                seen.add(ident)
        elif role == "tool":
            if (set(row) != {"role", "content", "tool_call_id"}
                    or row.get("tool_call_id") not in pending or not isinstance(row.get("content"), str)):
                raise NativeProtocolError("native_tool_receipt_unpaired")
            pending.remove(row["tool_call_id"])
            item = dict(row)
        elif role == "user":
            if pending or set(row) != {"role", "content"} or not isinstance(row.get("content"), str):
                raise NativeProtocolError("native_user_turn_invalid")
            item = dict(row)
        else:
            raise NativeProtocolError("native_history_role_invalid")
        result.append(item)
    if pending:
        raise NativeProtocolError("native_tool_receipt_missing")
    return copy.deepcopy(result)


def without_reasoning(messages):
    return [{k: copy.deepcopy(v) for k, v in row.items() if k != "reasoning_content"}
            for row in messages or []]
