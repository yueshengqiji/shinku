"""给 Agent 的只读自身状态工具。

它只返回宿主主动提供的安全快照，不读取系统提示词原文、密钥或文件树。
这样模型在被问到“你现在用的是什么/工具是否可用”时可以查事实，避免凭
模型记忆猜测项目配置。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Callable

__all__ = ["SelfStateTool"]


class SelfStateTool:
    """Expose a bounded, read-only runtime/agent/tool snapshot."""

    tool_type = "read_self_state"
    agent_progress_class = "observe"
    agent_permission_class = "read"
    agent_fast_perception = True
    _SCOPES = frozenset({"all", "runtime", "agent", "tools", "persona", "config", "changes"})

    def __init__(
        self,
        provider: Callable[..., Mapping[str, Any]] | None = None,
        *,
        fresh_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.provider = provider
        self.fresh_provider = fresh_provider

    def tool_metadata(self) -> SimpleNamespace:
        return SimpleNamespace(
            description=(
                "只读查看当前 Shinku 实例的运行模型、Agent 配置、可用工具、主人设状态，"
                "以及最近加载的代码/MCP 变更。用户询问自身构成、当前模型、工具能力或最近改动时使用；"
                "不会返回密钥、源码正文或系统提示词原文。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["all", "runtime", "agent", "tools", "persona", "config", "changes"],
                        "description": "要查看的状态范围，默认 all。",
                    }
                },
                "additionalProperties": False,
            },
        )

    def build_prompt_instruction(self) -> str:
        return (
            "当用户明确询问你当前的模型、供应方、工具、Agent 配置、主人设是否生效，"
            "最近新增了哪些 MCP、代码修改是否生效，或询问自身运行状态时调用 read_self_state。"
            "普通闲聊不要调用；只读，不修改配置。"
        )

    def normalize_call(self, call: dict[str, Any]) -> dict[str, str] | None:
        if not isinstance(call, Mapping):
            return None
        scope = str(call.get("scope") or "all").strip().casefold() or "all"
        return {"scope": scope} if scope in self._SCOPES else None

    def execute(self, *, call: dict[str, Any], context: Any) -> dict[str, Any]:
        arguments = self.normalize_call(call)
        if arguments is None:
            raise ValueError("read_self_state_arguments_invalid")
        # 显式自查允许刷新代码/MCP 快照；普通回合使用的 context 快照不会因此
        # 每轮重新扫描整个项目。
        snapshot = self._snapshot(context, fresh=True)
        scope = arguments["scope"]
        selected = snapshot if scope == "all" else {scope: snapshot.get(scope, {})}
        rendered = json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True)
        # Keep an unexpectedly verbose provider from destroying the prompt budget.
        rendered = rendered[:8000]
        return {
            "tool_type": self.tool_type,
            "followup_context": "当前实例的只读状态如下：\n" + rendered,
            "state_updates": {"self_state": selected},
        }

    def _snapshot(self, context: Any, *, fresh: bool = False) -> dict[str, Any]:
        if fresh and self.fresh_provider is not None:
            try:
                provided = self.fresh_provider()
            except Exception:
                provided = {}
            if isinstance(provided, Mapping):
                return _safe_copy(provided)
        raw = context.get("self_state") if isinstance(context, Mapping) else None
        if isinstance(raw, Mapping):
            return _safe_copy(raw)
        if self.provider is not None:
            try:
                provided = self.provider()
            except Exception:
                provided = {}
            if isinstance(provided, Mapping):
                return _safe_copy(provided)
        return {"status": "unavailable", "reason": "self_state_provider_unconfigured"}


def _safe_copy(value: Any, *, depth: int = 0) -> Any:
    """Copy JSON-like values while dropping obvious secret-shaped keys."""

    if depth > 5:
        return "<truncated>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            lowered = name.casefold()
            if any(marker in lowered for marker in ("key", "token", "secret", "password", "authorization")):
                result[name] = "<redacted>"
            else:
                result[name] = _safe_copy(item, depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_safe_copy(item, depth=depth + 1) for item in value[:100]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
