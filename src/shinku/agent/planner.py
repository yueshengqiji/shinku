"""把 Shinku 的 LLMRuntime 适配为 C3 Agent planner。

这个适配器只做两件事：拼接当前循环状态、解析模型返回的最终文本或工具决定。
工具 schema 通过 runtime 的结构化参数传入，不把 handler 的实现说明复制进普通
对话文本。模型供应商仍然由调用方注入，因而本模块可以用假 runtime 离线测试。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from .loop import AgentDecision, AgentState
from shinku.tools.invocation import NATIVE_TOOL_CALL_FIELD, legacy_tool_call_to_invocation

__all__ = ["ChatRuntime", "LLMPlanner", "PromptBuilder", "UserImagesBuilder"]


class ChatRuntime(Protocol):
    def call_chat_json(self, **kwargs: Any) -> dict[str, Any]: ...


PromptBuilder = Callable[[AgentState], str]
UserImagesBuilder = Callable[[AgentState], list[dict[str, Any]]]


def _default_user_prompt(state: AgentState) -> str:
    lines = ["根据当前任务继续处理。只做一件事：给出最终答复、等待用户，或选择一个工具。"]
    for item in state.transcript[-12:]:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "unknown").strip()
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


class LLMPlanner:
    """将一次 `call_chat_json` 结果降为 `AgentDecision`。"""

    def __init__(
        self,
        *,
        runtime: ChatRuntime,
        system_prompt: str,
        tool_specs: Sequence[dict[str, Any]] | None = None,
        build_user_prompt: PromptBuilder | None = None,
        build_user_images: UserImagesBuilder | None = None,
        temperature: float = 0.2,
        prompt_cache_key: str = "",
        native_tool_choice: Any = "auto",
        fallback_text: str = "暂时无法完成这一步。",
    ) -> None:
        self.runtime = runtime
        self.system_prompt = str(system_prompt or "").strip()
        self.tool_specs = [dict(item) for item in (tool_specs or []) if isinstance(item, Mapping)]
        self.build_user_prompt = build_user_prompt or _default_user_prompt
        self.build_user_images = build_user_images
        self.temperature = temperature
        self.prompt_cache_key = str(prompt_cache_key or "")
        self.native_tool_choice = native_tool_choice
        self.fallback_text = str(fallback_text or "").strip()

    def __call__(self, *, state: AgentState) -> AgentDecision | None:
        user_images = self.build_user_images(state) if self.build_user_images is not None else None
        if not isinstance(user_images, list):
            user_images = []
        response = self.runtime.call_chat_json(
            system_prompt=self.system_prompt,
            user_prompt=str(self.build_user_prompt(state) or ""),
            fallback={"kind": "final", "text": self.fallback_text},
            temperature=self.temperature,
            prompt_cache_key=self.prompt_cache_key,
            user_images=user_images or None,
            native_tools=self.tool_specs or None,
            native_tool_choice=self.native_tool_choice if self.tool_specs else "",
            history_turns=self._history(state),
        )
        return self._parse(response)

    @staticmethod
    def _history(state: AgentState) -> list[dict[str, str]]:
        return [
            {"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")}
            for item in state.transcript[-12:]
            if isinstance(item, Mapping) and str(item.get("content") or "").strip()
        ]

    @classmethod
    def _parse(cls, response: Any) -> AgentDecision | None:
        if not isinstance(response, Mapping):
            return None
        direct = AgentDecision.from_value(response)
        if direct is not None:
            return direct
        native_payload = response.get(NATIVE_TOOL_CALL_FIELD)
        if isinstance(native_payload, Mapping):
            invocation = legacy_tool_call_to_invocation(native_payload)
            if invocation is not None:
                return AgentDecision.tool(invocation)
        native = response.get("tool_calls")
        if isinstance(native, list) and native:
            call = native[0]
            if isinstance(call, Mapping):
                function = call.get("function") if isinstance(call.get("function"), Mapping) else call
                if isinstance(function, Mapping):
                    payload = {
                        "kind": "tool",
                        "tool_call": {
                            "name": function.get("name"),
                            "arguments": function.get("arguments", {}),
                            "id": call.get("id", ""),
                        },
                    }
                    direct = AgentDecision.from_value(payload)
                    if direct is not None:
                        return direct
        if response.get("reply") or response.get("speech"):
            return AgentDecision.final(str(response.get("reply") or response.get("speech") or ""))
        return None
