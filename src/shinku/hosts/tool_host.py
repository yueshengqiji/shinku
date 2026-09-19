"""本地工具宿主与 Agent 接线。

这是一个宿主适配层，不是具体能力实现：它接收已注册的 handler，向 Agent 暴露
只读工具视图和 native schema，并负责组装 planner 与 loop。未来 QQ、浏览器或
独立进程宿主都可以实现同样的外部边界，不需要修改 Agent 核心。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shinku.agent.loop import AgentLoop, AgentRunResult
from shinku.agent.planner import ChatRuntime, LLMPlanner, PromptBuilder
from shinku.tools.execution import ExecutionPolicy, ToolHandler
from shinku.tools.registry import ToolRegistry

__all__ = ["ToolHost", "ToolHostHealth"]


@dataclass(frozen=True)
class ToolHostHealth:
    host_id: str
    ready: bool
    tool_names: tuple[str, ...]
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "host_id": self.host_id,
            "ready": self.ready,
            "tool_names": list(self.tool_names),
            "reason": self.reason,
        }


class ToolHost:
    """把一个工具注册表接到 AgentLoop 的本地宿主。"""

    def __init__(self, registry: ToolRegistry, *, host_id: str = "local-tools") -> None:
        self.registry = registry
        self.host_id = str(host_id or "local-tools").strip() or "local-tools"

    def health(self) -> ToolHostHealth:
        names = self.registry.names()
        return ToolHostHealth(
            host_id=self.host_id,
            ready=bool(names),
            tool_names=names,
            reason="" if names else "no_tools_registered",
        )

    def build_loop(
        self,
        *,
        runtime: ChatRuntime,
        system_prompt: str,
        allowed_tool_names: set[str] | None = None,
        build_user_prompt: PromptBuilder | None = None,
        max_rounds: int = 6,
        policy: ExecutionPolicy | None = None,
        completion_gate: Callable[[Any, str], bool] | None = None,
        prompt_cache_key: str = "",
        temperature: float = 0.2,
    ) -> AgentLoop:
        specs = self.registry.native_specs(allowed_tool_names=allowed_tool_names)
        registered = self.registry.view()
        handlers: dict[str, ToolHandler] = {}
        allowed = {str(name or "").strip() for name in allowed_tool_names or set()}
        for raw_name, handler in registered.items():
            tool_name = str(getattr(handler, "tool_type", "") or raw_name or "").strip()
            if not tool_name or (allowed and tool_name not in allowed) or tool_name in handlers:
                continue
            handlers[tool_name] = handler
        planner = LLMPlanner(
            runtime=runtime,
            system_prompt=system_prompt,
            tool_specs=specs,
            build_user_prompt=build_user_prompt,
            prompt_cache_key=prompt_cache_key,
            temperature=temperature,
        )
        return AgentLoop(
            planner=planner,
            handlers=handlers,
            max_rounds=max_rounds,
            policy=policy,
            completion_gate=completion_gate,
        )

    def run(
        self,
        *,
        runtime: ChatRuntime,
        system_prompt: str,
        task_id: str = "",
        context: Any = None,
        initial_transcript: list[dict[str, Any]] | None = None,
        allowed_tool_names: set[str] | None = None,
        build_user_prompt: PromptBuilder | None = None,
        max_rounds: int = 6,
        policy: ExecutionPolicy | None = None,
        completion_gate: Callable[[Any, str], bool] | None = None,
        prompt_cache_key: str = "",
        temperature: float = 0.2,
    ) -> AgentRunResult:
        loop = self.build_loop(
            runtime=runtime,
            system_prompt=system_prompt,
            allowed_tool_names=allowed_tool_names,
            build_user_prompt=build_user_prompt,
            max_rounds=max_rounds,
            policy=policy,
            completion_gate=completion_gate,
            prompt_cache_key=prompt_cache_key,
            temperature=temperature,
        )
        return loop.run(task_id=task_id, context=context, initial_transcript=initial_transcript)
