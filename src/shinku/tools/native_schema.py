"""把 Shinku 的 handler 描述成 provider 原生（OpenAI functions 形态）的工具 schema。

供应商的原生工具调用只接受保守的 schema：名字限字符集与长度、描述限长度、参数必须是 object。
这个模块从 handler 上已有的元数据（``tool_metadata().input_schema`` 优先，否则
``build_prompt_instruction()``）生出那种 schema，并把仍在讲「老式调用信封」的整句从描述里剔掉。
"""

from __future__ import annotations

import re
from typing import Any, Mapping

#: 原生工具名的合法形态。
NATIVE_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,80}$")

#: 原生工具描述的长度上限。
NATIVE_TOOL_DESCRIPTION_MAX_CHARS = 900

#: 描述里出现这些字样，说明那句话还在讲老式调用信封（JSON 块/格式说明），整句剔除。
_ENVELOPE_MARKERS = ("格式为", "调用格式", "tool_call", '{"type"')

#: 句末标点，用于把描述切成句子。
_CLAUSE_ENDING = "。"


def build_openai_native_tool_specs(
    handlers: Mapping[str, Any] | None,
    *,
    allowed_tool_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """从 handler 映射生成原生工具 schema 列表；输入不是映射就返回空列表。"""

    if not isinstance(handlers, Mapping):
        return []
    allowed = {str(name or "").strip() for name in allowed_tool_names or set()}
    specs: list[dict[str, Any]] = []
    taken: set[str] = set()
    for raw_key, handler in sorted(handlers.items(), key=lambda pair: str(pair[0] or "")):
        name = str(getattr(handler, "tool_type", "") or raw_key or "").strip()
        if not _admits(name, taken, allowed):
            continue
        schema = _input_schema(handler)
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": _describe(handler, name, schema),
                    "parameters": _parameters(schema),
                },
            }
        )
        taken.add(name)
    return specs


def _admits(name: str, taken: set[str], allowed: set[str]) -> bool:
    """名字能不能进：非空、不重复、在候选集内（若有限定）、形态合法。"""

    if not name or name in taken:
        return False
    if allowed and name not in allowed:
        return False
    return NATIVE_TOOL_NAME_RE.fullmatch(name) is not None


def _input_schema(handler: Any) -> Mapping[str, Any] | None:
    """读 handler 的元数据 schema；读不到或不是映射都返回 None。"""

    getter = getattr(handler, "tool_metadata", None)
    if not callable(getter):
        return None
    try:
        metadata = getter()
    except Exception:
        return None
    schema = getattr(metadata, "input_schema", None)
    return schema if isinstance(schema, Mapping) else None


def _describe(handler: Any, tool_name: str, schema: Mapping[str, Any] | None) -> str:
    """定出描述：优先元数据里的，其次 handler 的提示语，最后给一句兜底。"""

    if schema:
        from_schema = str(schema.get("description") or schema.get("x_description") or "").strip()
        if from_schema:
            return " ".join(from_schema.split())[:NATIVE_TOOL_DESCRIPTION_MAX_CHARS]
    builder = getattr(handler, "build_prompt_instruction", None)
    try:
        text = str(builder() or "").strip() if callable(builder) else ""
    except Exception:
        text = ""
    text = text or f"Call Shinku tool {tool_name}."
    return _drop_envelope_clauses(" ".join(text.split()))[:NATIVE_TOOL_DESCRIPTION_MAX_CHARS]


def _parameters(schema: Mapping[str, Any] | None) -> dict[str, Any]:
    """参数用元数据 schema（去掉描述键、保证 object）；没有就给宽松的 object。"""

    if not schema:
        return {"type": "object", "additionalProperties": True}
    parameters = {
        str(key): value
        for key, value in dict(schema).items()
        if str(key) not in {"description", "x_description"}
    }
    if str(parameters.get("type") or "").strip() != "object":
        parameters["type"] = "object"
    return parameters


def _drop_envelope_clauses(text: str) -> str:
    """剔掉仍带老式信封字样的整句；剔空了就退回原文。"""

    if not text:
        return text
    kept = [clause for clause in _clauses(text) if not _mentions_envelope(clause)]
    return "".join(kept).strip() or text


def _clauses(text: str) -> list[str]:
    """按句号切句，句号留在前一句末尾。"""

    clauses: list[str] = []
    pending: list[str] = []
    for char in text:
        pending.append(char)
        if char == _CLAUSE_ENDING:
            clauses.append("".join(pending))
            pending = []
    if pending:
        clauses.append("".join(pending))
    return clauses


def _mentions_envelope(clause: str) -> bool:
    return any(marker in clause for marker in _ENVELOPE_MARKERS)


__all__ = [
    "NATIVE_TOOL_DESCRIPTION_MAX_CHARS",
    "NATIVE_TOOL_NAME_RE",
    "build_openai_native_tool_specs",
]
