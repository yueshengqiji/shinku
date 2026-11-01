"""工具宿主注册边界。

宿主负责把具体能力注册为 handler；Agent 只拿到这个注册表的只读视图和原生
schema。注册表不执行工具，也不负责网络、线程或 QQ 生命周期。
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .execution import ToolHandler
from .native_schema import build_openai_native_tool_specs

__all__ = ["ToolRegistry"]


class ToolRegistry:
    """对 handler 做唯一命名和只读暴露。"""

    def __init__(self, handlers: Mapping[str, ToolHandler] | None = None) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        for name, handler in (handlers or {}).items():
            self.register(handler, name=name)

    def register(self, handler: ToolHandler, *, name: str = "", replace: bool = False) -> str:
        tool_name = str(name or getattr(handler, "tool_type", "") or "").strip()
        if not tool_name:
            raise ValueError("tool handler must have a name")
        if not callable(getattr(handler, "normalize_call", None)) or not callable(getattr(handler, "execute", None)):
            raise TypeError(f"tool handler {tool_name!r} does not implement the handler boundary")
        if tool_name in self._handlers and not replace:
            raise ValueError(f"tool handler already registered: {tool_name}")
        self._handlers[tool_name] = handler
        return tool_name

    def get(self, name: str) -> ToolHandler | None:
        return self._handlers.get(str(name or "").strip())

    def unregister(self, name: str, *, expected: ToolHandler) -> bool:
        """只撤回调用方原来注册的对象，保留其他宿主后来替换的 handler。

        插件停用在装配/退出边界调用；在途 AgentLoop 持有的旧 handler 由插件
        生命周期包装器另行封住，不把删除目录误当作停止已发生的执行。
        """
        key = str(name or "").strip()
        if self._handlers.get(key) is not expected:
            return False
        del self._handlers[key]
        return True

    def view(self) -> Mapping[str, ToolHandler]:
        return MappingProxyType(self._handlers)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    def native_specs(self, *, allowed_tool_names: set[str] | None = None) -> list[dict[str, Any]]:
        return build_openai_native_tool_specs(self._handlers, allowed_tool_names=allowed_tool_names)
