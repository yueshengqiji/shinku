"""Explicit data and effects passed between QQ turn stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from shinku.agent.loop import AgentDecision
from shinku.tasks.handoff import TaskHandoff

from .chat_flow import ChatFlowDecision
from .reply_models import QQAgentReply
from .turns import QQTurn


@dataclass(frozen=True)
class QQTurnEffects:
    trace: Callable[..., None]
    conversation_step: Callable[..., None]
    deliver: Callable[..., QQAgentReply]
    terminal_failure_text: Callable[..., str]
    direct_chat_fallback: str
    non_master_tool_refusal: str


@dataclass(frozen=True)
class QQPreparedTurn:
    turn: QQTurn
    trace_id: str
    is_poke: bool
    chat_flow: ChatFlowDecision | None
    audience_context: dict[str, Any]
    context: dict[str, Any]
    requester: str
    session: str


@dataclass(frozen=True)
class QQTurnDecision:
    turn: QQTurn
    trace_id: str
    is_poke: bool
    context: dict[str, Any]
    task_id: str
    initial_transcript: list[dict[str, Any]]
    decision: AgentDecision
    available_tool_names: set[str]

    @property
    def requires_agent(self) -> bool:
        return self.decision.kind == "tool" or (
            isinstance(self.decision.task_link, Mapping) and self.decision.task_link.get("mode") == "continue"
        )


@dataclass(frozen=True)
class _PreparedAgentTask:
    """Validated business-task state, not yet committed or executable."""

    handoff: TaskHandoff
    task: dict[str, Any]
    decision: AgentDecision | None
