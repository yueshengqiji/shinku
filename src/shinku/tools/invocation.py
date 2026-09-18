"""provider 无关的工具调用协议。

上游供应商各有各的工具调用语法：老式的 JSON 块、OpenAI 的 native tool_calls、
Anthropic 的 tool_use。它们表达的是同一件事——"要调哪个工具、参数是什么"。
这个模块定义那件事的**中间形态**，以及它与老式 JSON 形态之间的双向转换。

三个下划线开头的字段名（``_tool_source`` / ``_tool_invocation_id`` / ``_native_tool_call``）
是跨模块按字符串匹配的**线路约定**，不随命名风格改动。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

#: 工具调用来源标记。
LEGACY_JSON = "legacy_json"
NATIVE_OPENAI = "native_openai"
NATIVE_ANTHROPIC = "native_anthropic"

#: 嵌在工具调用载荷里的元数据键。属线路协议。
TOOL_SOURCE_FIELD = "_tool_source"
TOOL_INVOCATION_ID_FIELD = "_tool_invocation_id"

#: 供应商原始工具调用在载荷里的存放键。属线路协议。
NATIVE_TOOL_CALL_FIELD = "_native_tool_call"

#: 元数据键的公共前缀。转换时带此前缀的键一律不进 ``arguments``。
_METADATA_PREFIX = "_tool_"

#: 自动生成的调用 ID 的前缀与随机段长度。
_ID_PREFIX = "call_"
_ID_SUFFIX_CHARS = 16


@dataclass
class ToolInvocation:
    """一次归一化后的工具请求，与供应商语法无关。"""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    source: str = LEGACY_JSON
    id: str = ""

    def __post_init__(self) -> None:
        # 调用方没给 ID 时补一个；给了就一律尊重原值。
        if not str(self.id or "").strip():
            self.id = _ID_PREFIX + uuid.uuid4().hex[:_ID_SUFFIX_CHARS]


@dataclass
class ValidationResult:
    """工具调用校验的结果。"""

    ok: bool
    message: str = ""
    code: str = ""

    @classmethod
    def success(cls) -> "ValidationResult":
        return cls(ok=True)

    @classmethod
    def fail(cls, code: str, message: str) -> "ValidationResult":
        return cls(ok=False, code=str(code or ""), message=str(message or ""))


@dataclass
class ToolResultEnvelope:
    """回传给模型的一次工具执行结果。"""

    invocation_id: str
    status: str
    model_feedback: str
    data: dict[str, Any] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


def _arguments_from(tool_call: dict[str, Any]) -> dict[str, Any]:
    """取出载荷里的工具参数：去掉 ``type`` 本身，以及所有 ``_tool_`` 元数据键。"""

    return {
        key: value
        for key, value in tool_call.items()
        if key != "type" and not str(key).startswith(_METADATA_PREFIX)
    }


def legacy_tool_call_to_invocation(
    tool_call: Any,
    *,
    source: str = LEGACY_JSON,
    invocation_id: str = "",
) -> ToolInvocation | None:
    """把一条老式 JSON 工具调用转成归一化形态。

    载荷里若已经嵌了来源与调用 ID，以**载荷里**的为准——它比调用参数更接近事实。
    没有工具名或输入不是字典时返回 ``None``。
    """

    if not isinstance(tool_call, dict):
        return None

    name = str(tool_call.get("type") or "").strip()
    if not name:
        return None

    embedded_source = str(tool_call.get(TOOL_SOURCE_FIELD) or "").strip()
    embedded_id = str(tool_call.get(TOOL_INVOCATION_ID_FIELD) or "").strip()

    return ToolInvocation(
        name=name,
        arguments=_arguments_from(tool_call),
        source=embedded_source or source,
        id=embedded_id or invocation_id,
    )


def invocation_to_legacy_tool_call(
    invocation: ToolInvocation,
    *,
    include_metadata: bool = False,
) -> dict[str, Any]:
    """把归一化形态还原成老式 JSON 工具调用。

    ``include_metadata`` 只在来源不是 ``legacy_json`` 时才有意义——
    回写老式形态却带上 native 来源标记，等于把来源信息丢在了半路。
    """

    payload: dict[str, Any] = {"type": invocation.name, **dict(invocation.arguments or {})}
    if include_metadata and str(invocation.source or LEGACY_JSON) != LEGACY_JSON:
        payload[TOOL_SOURCE_FIELD] = str(invocation.source or "")
        payload[TOOL_INVOCATION_ID_FIELD] = str(invocation.id or "")
    return payload


def round_trip_legacy_tool_call(tool_call: Any) -> dict[str, Any] | None:
    """老式 JSON → 归一化 → 老式 JSON。

    结果是**规范化后**的形态：``_tool_`` 元数据会被剥掉，非字符串键会被保留原样。
    输入不成形时返回 ``None``。
    """

    invocation = legacy_tool_call_to_invocation(tool_call)
    if invocation is None:
        return None
    return invocation_to_legacy_tool_call(invocation)


__all__ = [
    "LEGACY_JSON",
    "NATIVE_ANTHROPIC",
    "NATIVE_OPENAI",
    "NATIVE_TOOL_CALL_FIELD",
    "TOOL_INVOCATION_ID_FIELD",
    "TOOL_SOURCE_FIELD",
    "ToolInvocation",
    "ToolResultEnvelope",
    "ValidationResult",
    "invocation_to_legacy_tool_call",
    "legacy_tool_call_to_invocation",
    "round_trip_legacy_tool_call",
]
