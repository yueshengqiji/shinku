"""Agent 工具调用的语义授权边界。

工具是否值得调用由模型通过 native tool calling 自己判断；本模块只处理另一件
事：在真正产生副作用前，把具体调用交给用户确认。确认不是一组写死的关键词，
而是一个没有工具权限的轻量语义判断请求。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

from shinku.tools.invocation import ToolInvocation

__all__ = [
    "ConsentDecision",
    "ConsentResolver",
    "LLMConsentResolver",
    "ToolApprovalRequest",
    "make_tool_approval_request",
]


ConsentVerdict = Literal["approve", "deny", "unclear"]


@dataclass(frozen=True)
class ToolApprovalRequest:
    """一次尚未执行的工具调用授权请求。"""

    request_id: str
    task_id: str
    invocation: ToolInvocation
    question: str
    created_at: float
    invocations: tuple[ToolInvocation, ...] = ()
    risk_by_tool: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        calls = tuple(item for item in self.invocations if isinstance(item, ToolInvocation))
        if not calls:
            calls = (self.invocation,)
        object.__setattr__(self, "invocations", calls)
        object.__setattr__(self, "invocation", calls[0])
        object.__setattr__(
            self,
            "risk_by_tool",
            MappingProxyType(
                {
                    str(name or "").strip(): str(value or "medium").strip().lower()
                    for name, value in (self.risk_by_tool.items() if isinstance(self.risk_by_tool, Mapping) else ())
                    if str(name or "").strip()
                }
            ),
        )

    def risk_of(self, tool_name: str) -> str:
        return str(self.risk_by_tool.get(str(tool_name or "").strip(), "medium") or "medium")

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task_id": self.task_id,
            "tool": self.invocation.name,
            "arguments": _safe_arguments(self.invocation.arguments),
            "tools": [
                {
                    "tool": item.name,
                    "risk": self.risk_of(item.name),
                    "arguments": _safe_arguments(item.arguments),
                }
                for item in self.invocations
            ],
            "risk": self.risk_of(self.invocation.name),
            "question": self.question,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ConsentDecision:
    """语义授权器的保守结果。``unclear`` 永远不执行工具。"""

    verdict: ConsentVerdict
    confidence: float = 0.0
    reason: str = ""

    @property
    def approved(self) -> bool:
        return self.verdict == "approve"


class ConsentResolver(Protocol):
    def resolve(self, *, request: ToolApprovalRequest, user_message: str) -> ConsentDecision: ...


def _safe_arguments(arguments: dict[str, Any] | None) -> dict[str, Any]:
    """授权提示中隐藏明显的凭据字段，避免把密钥回显给用户或模型。"""

    hidden_markers = ("password", "passwd", "secret", "token", "api_key", "cookie", "authorization")
    safe: dict[str, Any] = {}
    for raw_key, value in (arguments or {}).items():
        key = str(raw_key)
        lowered = key.lower()
        safe[key] = "<已隐藏>" if any(marker in lowered for marker in hidden_markers) else value
    return safe


def _compact_arguments(arguments: dict[str, Any] | None) -> str:
    try:
        text = json.dumps(_safe_arguments(arguments), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        text = repr(_safe_arguments(arguments))
    return text if len(text) <= 420 else text[:417] + "..."


def make_tool_approval_request(
    *,
    invocation: ToolInvocation | None = None,
    invocations: Sequence[ToolInvocation] | None = None,
    task_id: str = "",
    risk_by_tool: Mapping[str, str] | None = None,
) -> ToolApprovalRequest:
    """为一个或多个已校验的调用生成用户可读授权请求。"""

    calls = tuple(item for item in (invocations or ()) if isinstance(item, ToolInvocation))
    if not calls and isinstance(invocation, ToolInvocation):
        calls = (invocation,)
    if not calls:
        raise ValueError("at least one tool invocation is required")
    if len(calls) == 1:
        item = calls[0]
        tool_name = str(item.name or "").strip() or "未命名工具"
        arguments = _compact_arguments(item.arguments)
        risk = str((risk_by_tool or {}).get(item.name, "medium") or "medium").strip().lower()
        question = f"我准备调用「{tool_name}」（风险级别：{risk}），参数是 {arguments}。这一步可能读取或操作外部资源，要现在执行吗？"
    else:
        details = "；".join(
            f"「{str(item.name or '').strip() or '未命名工具'}」（风险级别：{str((risk_by_tool or {}).get(item.name, 'medium') or 'medium').strip().lower()}）"
            f"参数 {_compact_arguments(item.arguments)}"
            for item in calls
        )
        question = f"我准备连续调用这些工具：{details}。这一步可能读取或操作外部资源，要现在执行吗？"
    return ToolApprovalRequest(
        request_id="approval_" + uuid.uuid4().hex[:16],
        task_id=str(task_id or "").strip(),
        invocation=calls[0],
        question=question,
        created_at=time.time(),
        invocations=calls,
        risk_by_tool=risk_by_tool or {},
    )


class LLMConsentResolver:
    """用无工具的轻量模型判断用户是否在语义上批准当前请求。"""

    def __init__(self, runtime: Any, *, prompt_cache_key: str = "shinku:tool-consent:v1") -> None:
        self.runtime = runtime
        self.prompt_cache_key = str(prompt_cache_key or "shinku:tool-consent:v1")

    def resolve(self, *, request: ToolApprovalRequest, user_message: str) -> ConsentDecision:
        payload = {
            "pending_request": {
                "tool": request.invocation.name,
                "risk": request.risk_of(request.invocation.name),
                "arguments": _safe_arguments(request.invocation.arguments),
                "question": request.question,
            },
            "requests": [
                {
                    "tool": item.name,
                    "risk": request.risk_of(item.name),
                    "arguments": _safe_arguments(item.arguments),
                }
                for item in request.invocations
            ],
            "user_reply": str(user_message or "").strip(),
        }
        system_prompt = (
            "你是一个只负责判断工具授权的内部分类器。"
            "请结合待执行请求和用户的这条回复，判断用户是否明确同意执行这一次具体调用。"
            "不要执行工具，不要回答用户问题。只返回 JSON："
            '{"decision":"approve|deny|unclear","confidence":0到1之间的数字,"reason":"简短原因"}。'
            "如果回复只是闲聊、答非所问、对象不明确或无法确定，就返回 unclear；"
            "不要把模糊的礼貌回应强行当成授权。"
        )
        kwargs = {
            "system_prompt": system_prompt,
            "user_prompt": json.dumps(payload, ensure_ascii=False),
            "fallback": {"decision": "unclear", "confidence": 0.0, "reason": "consent_judge_fallback"},
            "temperature": 0.0,
            "prompt_cache_key": self.prompt_cache_key,
        }
        try:
            aux = getattr(self.runtime, "call_aux_json", None)
            result = aux(**kwargs) if callable(aux) else self.runtime.call_chat_json(**kwargs)
        except Exception as exc:
            return ConsentDecision("unclear", 0.0, f"judge_error:{type(exc).__name__}")
        return self._parse(result)

    @staticmethod
    def _parse(value: Any) -> ConsentDecision:
        if not isinstance(value, dict):
            return ConsentDecision("unclear", 0.0, "invalid_judge_result")
        raw = value.get("decision", value.get("verdict", value.get("result", "")))
        decision = str(raw or "").strip().lower()
        if decision not in {"approve", "deny", "unclear"}:
            return ConsentDecision("unclear", 0.0, "unknown_judge_decision")
        try:
            confidence = max(0.0, min(1.0, float(value.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        return ConsentDecision(decision, confidence, str(value.get("reason") or "").strip())
