"""本地工具宿主与 Agent 接线。

这是一个宿主适配层，不是具体能力实现：它接收已注册的 handler，向 Agent 暴露
只读工具视图和 native schema，并负责组装 planner 与 loop。未来 QQ、浏览器或
独立进程宿主都可以实现同样的外部边界，不需要修改 Agent 核心。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from shinku.agent.loop import AgentLoop, AgentRunResult, ApprovalGate
from shinku.agent.planner import ChatRuntime, LLMPlanner, PromptBuilder, UserImagesBuilder
from shinku.tools.execution import ExecutionPolicy, ToolHandler
from shinku.tools.invocation import ToolInvocation
from shinku.tools.registry import ToolRegistry

__all__ = ["ToolHost", "ToolHostHealth", "ToolHostRuntimeHealth"]


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


@dataclass(frozen=True)
class ToolHostRuntimeHealth:
    """工具宿主与当前模型 runtime 的能力交集。"""

    host_id: str
    ready: bool
    native_tools_supported: bool | None
    tool_names: tuple[str, ...]
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "host_id": self.host_id,
            "ready": self.ready,
            "native_tools_supported": self.native_tools_supported,
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

    def runtime_health(self, runtime: ChatRuntime) -> ToolHostRuntimeHealth:
        """检查宿主是否能和当前 runtime 共同提供 native tools。

        runtime 没有能力探针时保留 ``None``，兼容离线测试桩和未来宿主；只有
        明确返回 ``False`` 时才判定不可用，避免把未知 runtime 误报成支持或不支持。
        """

        base = self.health()
        supported = self._native_tools_supported(runtime)
        if not base.ready:
            reason = base.reason
            ready = False
        elif supported is False:
            reason = "runtime_native_tools_unsupported"
            ready = False
        else:
            reason = "" if supported is True else "runtime_capability_unknown"
            ready = True
        return ToolHostRuntimeHealth(
            host_id=base.host_id,
            ready=ready,
            native_tools_supported=supported,
            tool_names=base.tool_names,
            reason=reason,
        )

    def build_loop(
        self,
        *,
        runtime: ChatRuntime,
        system_prompt: str,
        allowed_tool_names: set[str] | None = None,
        build_user_prompt: PromptBuilder | None = None,
        build_user_images: UserImagesBuilder | None = None,
        max_rounds: int = 6,
        policy: ExecutionPolicy | None = None,
        completion_gate: Callable[[Any, str], bool] | None = None,
        approval_gate: ApprovalGate | None = None,
        prompt_cache_key: str = "",
        temperature: float = 0.2,
        history_limit: int = 12,
    ) -> AgentLoop:
        native_supported = self._native_tools_supported(runtime)
        tools_enabled = native_supported is not False
        specs = (
            self.registry.native_specs(allowed_tool_names=allowed_tool_names)
            if tools_enabled
            else []
        )
        registered = self.registry.view() if tools_enabled else {}
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
            build_user_images=build_user_images,
            prompt_cache_key=prompt_cache_key,
            temperature=temperature,
            history_limit=history_limit,
        )
        return AgentLoop(
            planner=planner,
            handlers=handlers,
            max_rounds=max_rounds,
            policy=policy,
            completion_gate=completion_gate,
            approval_gate=approval_gate,
        )

    @staticmethod
    def _native_tools_supported(runtime: ChatRuntime) -> bool | None:
        probe = getattr(runtime, "chat_supports_native_tools", None)
        if not callable(probe):
            return None
        try:
            return bool(probe())
        except Exception:
            return None

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
        build_user_images: UserImagesBuilder | None = None,
        max_rounds: int = 6,
        policy: ExecutionPolicy | None = None,
        completion_gate: Callable[[Any, str], bool] | None = None,
        approval_gate: ApprovalGate | None = None,
        approved_invocation: ToolInvocation | None = None,
        approved_invocations: Sequence[ToolInvocation] | None = None,
        prompt_cache_key: str = "",
        temperature: float = 0.2,
        history_limit: int = 12,
    ) -> AgentRunResult:
        loop = self.build_loop(
            runtime=runtime,
            system_prompt=system_prompt,
            allowed_tool_names=allowed_tool_names,
            build_user_prompt=build_user_prompt,
            build_user_images=build_user_images,
            max_rounds=max_rounds,
            policy=policy,
            completion_gate=completion_gate,
            approval_gate=approval_gate,
            prompt_cache_key=prompt_cache_key,
            temperature=temperature,
            history_limit=history_limit,
        )
        return loop.run(
            task_id=task_id,
            context=context,
            initial_transcript=initial_transcript,
            approved_invocation=approved_invocation,
            approved_invocations=approved_invocations,
        )
