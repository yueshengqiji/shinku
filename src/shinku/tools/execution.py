"""工具调用的纯执行边界。

模型只提交 provider 无关的 :class:`ToolInvocation`；本模块负责找到对应 handler、
校验参数、执行一次调用并把结果转成可回传给模型的 envelope。它不决定本轮是否
应该调用工具，也不负责多轮循环——那些属于更上层的 Agent 编排器。
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from .invocation import (
    ToolInvocation,
    ToolResultEnvelope,
    ValidationResult,
    invocation_to_legacy_tool_call,
)

__all__ = [
    "ExecutionPolicy",
    "ToolHandler",
    "execute_invocation",
    "result_to_envelope",
    "validation_to_envelope",
    "validate_invocation",
]


class ToolHandler(Protocol):
    """执行器需要的最小 handler 面。"""

    tool_type: str

    def normalize_call(self, call: dict[str, Any]) -> dict[str, Any] | None: ...

    def execute(self, *, call: dict[str, Any], context: Any) -> Any: ...


@dataclass(frozen=True)
class ExecutionPolicy:
    """一次执行允许/禁止的工具集合。

    空的 ``allowed`` 代表不额外限制；``blocked`` 永远优先于允许集合。

    ``risk_by_tool`` 只影响是否需要宿主授权，不会绕过 ``allowed`` / ``blocked``
    校验。未知工具默认按 ``default_risk`` 处理，默认是 ``medium``，因此新增工具
    不会因为忘记登记风险级别而悄悄变成自动执行。
    """

    allowed: frozenset[str] = field(default_factory=frozenset)
    blocked: frozenset[str] = field(default_factory=frozenset)
    risk_by_tool: Mapping[str, str] = field(default_factory=dict)
    approval_required_by_risk: frozenset[str] = field(default_factory=lambda: frozenset({"medium", "high"}))
    default_risk: str = "medium"

    def __post_init__(self) -> None:
        allowed = frozenset(str(item or "").strip() for item in self.allowed if str(item or "").strip())
        blocked = frozenset(str(item or "").strip() for item in self.blocked if str(item or "").strip())
        raw_risks = self.risk_by_tool if isinstance(self.risk_by_tool, Mapping) else {}
        risks = {
            str(name or "").strip(): self._normalize_risk(value, fallback="medium")
            for name, value in raw_risks.items()
            if str(name or "").strip()
        }
        required = frozenset(
            self._normalize_risk(item, fallback="medium")
            for item in self.approval_required_by_risk
            if str(item or "").strip()
        )
        object.__setattr__(self, "allowed", allowed)
        object.__setattr__(self, "blocked", blocked)
        object.__setattr__(self, "risk_by_tool", MappingProxyType(risks))
        object.__setattr__(self, "approval_required_by_risk", required)
        object.__setattr__(self, "default_risk", self._normalize_risk(self.default_risk, fallback="medium"))

    @staticmethod
    def _normalize_risk(value: object, *, fallback: str = "medium") -> str:
        risk = str(value or "").strip().lower()
        return risk if risk in {"low", "medium", "high"} else fallback

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "ExecutionPolicy":
        """从独立项目环境读取风险策略；无效配置安全回退到默认值。"""

        env = os.environ if environ is None else environ
        default_risk = cls._normalize_risk(env.get("SHINKU_TOOL_DEFAULT_RISK", "medium"), fallback="medium")
        raw_required = str(env.get("SHINKU_TOOL_APPROVAL_RISKS", "medium,high") or "medium,high")
        required = frozenset(
            cls._normalize_risk(item, fallback="")
            for item in raw_required.split(",")
            if cls._normalize_risk(item, fallback="")
        ) or frozenset({"medium", "high"})
        risks: dict[str, str] = {}
        raw_overrides = str(env.get("SHINKU_TOOL_RISK_OVERRIDES_JSON", "") or "").strip()
        if raw_overrides:
            try:
                decoded = json.loads(raw_overrides)
            except (TypeError, ValueError):
                decoded = {}
            if isinstance(decoded, Mapping):
                for name, value in decoded.items():
                    tool_name = str(name or "").strip()
                    if tool_name:
                        risks[tool_name] = cls._normalize_risk(value, fallback=default_risk)
        return cls(
            risk_by_tool=risks,
            approval_required_by_risk=required,
            default_risk=default_risk,
        )

    def accepts(self, tool_name: str) -> bool:
        name = str(tool_name or "").strip()
        return bool(name) and name not in self.blocked and (not self.allowed or name in self.allowed)

    def risk_of(self, tool_name: str) -> str:
        name = str(tool_name or "").strip()
        return self.risk_by_tool.get(name, self.default_risk)

    def requires_approval(self, tool_name: str) -> bool:
        return self.risk_of(tool_name) in self.approval_required_by_risk


def _handler_name(handler: Any, fallback: Any = "") -> str:
    return str(getattr(handler, "tool_type", "") or fallback or "").strip()


def validate_invocation(
    invocation: ToolInvocation | None,
    handlers: Mapping[str, ToolHandler] | None,
    *,
    raw_call: dict[str, Any] | None = None,
    policy: ExecutionPolicy | None = None,
) -> ValidationResult:
    """校验工具名称、调用策略和 handler 参数，不执行任何副作用。"""

    if invocation is None:
        return ValidationResult.success()
    tool_name = str(invocation.name or "").strip()
    if not tool_name:
        return ValidationResult.fail("missing_tool", "工具调用缺少名称。")
    if policy is not None and not policy.accepts(tool_name):
        return ValidationResult.fail("tool_blocked", f"工具「{tool_name}」不在本轮允许的执行范围内。")
    if not isinstance(handlers, Mapping):
        return ValidationResult.fail("unknown_tool", f"工具「{tool_name}」在本轮不可用。")
    handler = handlers.get(tool_name)
    if handler is None:
        available = "、".join(sorted(_handler_name(item, key) for key, item in handlers.items() if _handler_name(item, key)))
        suffix = f"可用工具：{available}。" if available else "本轮没有可用工具。"
        return ValidationResult.fail("unknown_tool", f"工具「{tool_name}」在本轮不可用。{suffix}")
    normalizer = getattr(handler, "normalize_call", None)
    if not callable(normalizer):
        return ValidationResult.fail("invalid_handler", f"工具「{tool_name}」没有参数校验器。")
    candidate = raw_call if isinstance(raw_call, dict) else invocation_to_legacy_tool_call(invocation)
    try:
        normalized = normalizer(candidate)
    except Exception:
        normalized = None
    if normalized is None:
        return ValidationResult.fail("bad_args", f"工具「{tool_name}」的参数格式不完整，无法执行。")
    return ValidationResult.success()


def validation_to_envelope(*, invocation: ToolInvocation, validation: ValidationResult) -> ToolResultEnvelope:
    message = str(validation.message or validation.code or "tool_validation_failed").strip()
    return ToolResultEnvelope(
        invocation_id=invocation.id,
        status="error",
        model_feedback=f"<tool_use_error>{message}</tool_use_error>",
        data={"code": str(validation.code or "validation_failed"), "tool": str(invocation.name or "")},
    )


def _result_value(result: Any, name: str, default: Any) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def result_to_envelope(*, invocation: ToolInvocation, result: Any) -> ToolResultEnvelope:
    if result is None:
        return ToolResultEnvelope(
            invocation_id=invocation.id,
            status="error",
            model_feedback="<tool_use_error>工具执行没有返回结果。</tool_use_error>",
            data={"code": "empty_result", "tool": str(invocation.name or "")},
        )
    feedback = str(_result_value(result, "followup_context", "") or "").strip()
    updates = _result_value(result, "state_updates", {})
    events = _result_value(result, "stream_events", [])
    return ToolResultEnvelope(
        invocation_id=invocation.id,
        status="ok",
        model_feedback=feedback,
        data={
            "tool_type": str(_result_value(result, "tool_type", "") or invocation.name),
            "state_updates": dict(updates or {}) if isinstance(updates, Mapping) else {},
        },
        events=[dict(item) for item in list(events or []) if isinstance(item, Mapping)],
    )


def execute_invocation(
    invocation: ToolInvocation,
    *,
    handlers: Mapping[str, ToolHandler] | None,
    context: Any = None,
    raw_call: dict[str, Any] | None = None,
    policy: ExecutionPolicy | None = None,
) -> tuple[Any | None, ToolResultEnvelope]:
    """执行一次工具调用；验证失败和 handler 异常都转成 envelope。"""

    validation = validate_invocation(invocation, handlers, raw_call=raw_call, policy=policy)
    if not validation.ok:
        return None, validation_to_envelope(invocation=invocation, validation=validation)
    assert handlers is not None
    tool_name = str(invocation.name or "").strip()
    handler = handlers[tool_name]
    call = raw_call if isinstance(raw_call, dict) else invocation_to_legacy_tool_call(invocation)
    try:
        result = handler.execute(call=call, context=context)
    except Exception as exc:
        return None, ToolResultEnvelope(
            invocation_id=invocation.id,
            status="error",
            model_feedback="<tool_use_error>工具执行失败，可以根据错误重新规划或直接说明无法完成。</tool_use_error>",
            data={
                "code": "handler_error",
                "tool": tool_name,
                "error_type": type(exc).__name__,
            },
        )
    return result, result_to_envelope(invocation=invocation, result=result)
