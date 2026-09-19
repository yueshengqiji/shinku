"""外部工具宿主的无传输适配边界。

QQ、浏览器、MCP 或其他独立进程只需要提供工具描述和一个注入式调用通道；
本模块负责把描述包装成 Shinku 的 ``ToolHandler``，不启动进程、不读密钥、
也不直接发 HTTP 请求。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol

from shinku.tools.native_schema import NATIVE_TOOL_NAME_RE
from shinku.tools.registry import ToolRegistry

__all__ = ["ExternalToolBridge", "ExternalToolDescriptor", "ExternalToolTransport"]


class ExternalToolTransport(Protocol):
    """外部宿主必须提供的最小调用面；具体 socket/HTTP/stdio 留在宿主侧。"""

    def call(self, *, tool_name: str, arguments: dict[str, Any], context: Any = None) -> Any: ...


@dataclass(frozen=True)
class ExternalToolDescriptor:
    name: str
    description: str = ""
    input_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if NATIVE_TOOL_NAME_RE.fullmatch(name) is None:
            raise ValueError("external tool name is invalid")
        schema = self.input_schema
        if schema is not None and not isinstance(schema, Mapping):
            raise TypeError("external tool input_schema must be a mapping")
        if isinstance(schema, Mapping) and str(schema.get("type") or "object") != "object":
            raise ValueError("external tool input_schema must describe an object")

    def schema(self) -> dict[str, Any]:
        raw = dict(self.input_schema or {})
        raw["type"] = "object"
        return raw


class _ExternalToolHandler:
    def __init__(self, descriptor: ExternalToolDescriptor, transport: ExternalToolTransport) -> None:
        self.descriptor = descriptor
        self.transport = transport
        self.tool_type = descriptor.name

    def tool_metadata(self) -> SimpleNamespace:
        return SimpleNamespace(
            description=self.descriptor.description,
            input_schema=self.descriptor.schema(),
        )

    def build_prompt_instruction(self) -> str:
        return self.descriptor.description or f"调用外部工具 {self.tool_type}。"

    def normalize_call(self, call: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(call, Mapping):
            return None
        arguments = {
            str(key): value
            for key, value in call.items()
            if str(key) not in {"type", "_tool_source", "_tool_invocation_id"}
        }
        schema = self.descriptor.schema()
        required = schema.get("required")
        if isinstance(required, list) and any(str(key) not in arguments for key in required):
            return None
        properties = schema.get("properties")
        if schema.get("additionalProperties") is False and isinstance(properties, Mapping):
            if any(key not in properties for key in arguments):
                return None
        return arguments

    def execute(self, *, call: dict[str, Any], context: Any) -> Any:
        arguments = self.normalize_call(call)
        if arguments is None:
            raise ValueError("external_tool_arguments_invalid")
        return self.transport.call(tool_name=self.tool_type, arguments=arguments, context=context)


class ExternalToolBridge:
    """把一个外部宿主的能力描述包装进本地 ``ToolRegistry``。"""

    def __init__(
        self,
        *,
        host_id: str,
        transport: ExternalToolTransport,
        descriptors: Sequence[ExternalToolDescriptor],
    ) -> None:
        clean_id = str(host_id or "").strip()
        if not clean_id:
            raise ValueError("external host id is required")
        if not callable(getattr(transport, "call", None)):
            raise TypeError("external transport must implement call")
        handlers: dict[str, _ExternalToolHandler] = {}
        for descriptor in descriptors:
            if not isinstance(descriptor, ExternalToolDescriptor):
                raise TypeError("external descriptors must be ExternalToolDescriptor values")
            if descriptor.name in handlers:
                raise ValueError(f"external tool already registered: {descriptor.name}")
            handlers[descriptor.name] = _ExternalToolHandler(descriptor, transport)
        self.host_id = clean_id
        self.transport = transport
        self._registry = ToolRegistry(handlers)

    def registry(self) -> ToolRegistry:
        return self._registry

    def health(self) -> dict[str, Any]:
        names = self._registry.names()
        return {
            "host_id": self.host_id,
            "ready": bool(names),
            "tool_names": list(names),
            "reason": "" if names else "no_external_tools",
        }
