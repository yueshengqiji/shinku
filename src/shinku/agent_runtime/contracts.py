"""Caller data has no permission, runtime, budget or success fields."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol

from shinku.agent.task_state import validate_contract


class AgentBoundaryError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def bounded_text(value: Any, limit: int, code: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise AgentBoundaryError(code)
    return value


@dataclass(frozen=True)
class AgentScope:
    """Established by a trusted host, never decoded from a model request."""
    requester: str
    session: str

    def __post_init__(self):
        bounded_text(self.requester, 256, "invalid_scope")
        bounded_text(self.session, 256, "invalid_scope")


@dataclass(frozen=True)
class AgentRequest:
    request_key: str
    original_request: str
    contract: dict[str, Any]
    constraints: tuple[str, ...] = ()
    sources: tuple[dict[str, Any], ...] = ()
    # 会话背景单独留存；不属于必须全文读取的任务附件，更不能恢复授权。
    history_context: str = ""

    def payload(self) -> dict[str, Any]:
        return _request_payload(self, self.contract)


@dataclass(frozen=True)
class AgentTurnRequest:
    """原始会话输入；只有显式启用会话循环的宿主可接纳，不接受模型契约或权限。"""
    request_key: str
    original_request: str
    constraints: tuple[str, ...] = ()
    sources: tuple[dict[str, Any], ...] = ()
    history_context: str = ""

    def payload(self) -> dict[str, Any]:
        return _request_payload(self, None, conversation=True)


def _request_payload(request, contract, *, conversation=False):
    """共用原输入上限；契约为空只属于宿主会话入口，不放宽普通AgentRequest。"""
    bounded_text(request.request_key, 256, "invalid_request_key")
    bounded_text(request.original_request, 12000, "invalid_original_request")
    if len(request.constraints) > 16 or len(request.sources) > 16:
        raise AgentBoundaryError("execution_input_oversized")
    constraints = [bounded_text(v, 2000, "invalid_constraint") for v in request.constraints]
    sources = []
    for row in request.sources:
        if not isinstance(row, dict) or set(row) - {"id", "content", "reference"}:
            raise AgentBoundaryError("invalid_source_data")
        reference = row.get("reference") or ""
        if not isinstance(reference, str) or len(reference) > 2000:
            raise AgentBoundaryError("invalid_source_data")
        sources.append({"id": bounded_text(row.get("id"), 256, "invalid_source_data"),
            "content": bounded_text(row.get("content"), 12000, "invalid_source_data"),
            "reference": reference, "source": "caller_data"})
    payload = {"request_key": request.request_key, "original_request": request.original_request,
               "contract": None if conversation else validate_contract(contract), "constraints": constraints, "sources": sources}
    if conversation:
        payload['input_mode'] = 'conversation'
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > 65536:
        raise AgentBoundaryError("execution_input_oversized")
    if not isinstance(request.history_context, str) or len(request.history_context.encode("utf-8")) > 16 * 1024 * 1024:
        raise AgentBoundaryError("conversation_history_resource_limit")
    if request.history_context:
        payload["history_context"] = request.history_context
    return copy.deepcopy(payload)


@dataclass(frozen=True)
class TaskHandle:
    task_id: str
    attempt: int
    version: int


@dataclass(frozen=True)
class AgentSupplement:
    """Caller clarification, without paths, credentials or host source authority."""
    request_key: str
    clarification: str = ""
    constraints: tuple[str, ...] = ()

    def payload(self):
        bounded_text(self.request_key, 256, "invalid_request_key")
        if not isinstance(self.clarification, str) or len(self.clarification) > 12000 or len(self.constraints) > 16:
            raise AgentBoundaryError("execution_input_oversized")
        return {"request_key": self.request_key, "clarification": self.clarification,
                "constraints": [bounded_text(v, 2000, "invalid_constraint") for v in self.constraints]}


class AgentPort(Protocol):
    """A client is bound to host-verified scope, independent of transport."""
    def submit(self, request: AgentRequest | AgentTurnRequest) -> TaskHandle: ...
    def status(self, task_id: str) -> dict[str, Any]: ...
    def cancel(self, handle: TaskHandle) -> dict[str, Any]: ...
    def resume(self, handle: TaskHandle, *, request_key: str) -> TaskHandle: ...
    def recovery(self, handle: TaskHandle) -> dict[str, Any]: ...
    def result(self, task_id: str) -> dict[str, Any]: ...
    def supplement(self, handle: TaskHandle, request: AgentSupplement) -> TaskHandle: ...
    # 问答是可验证的补充+恢复组合；答案不替代宿主授权。
    def question(self, task_id: str) -> dict[str, Any] | None: ...
    def answer_question(self, handle: TaskHandle, *, question_id: str, text: str) -> TaskHandle: ...
