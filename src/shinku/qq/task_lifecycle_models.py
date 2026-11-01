"""Narrow live dependencies for QQ task lifecycle responsibilities."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shinku.agent.loop import AgentDecision
    from shinku.agent.pending import PendingApprovalStore
    from shinku.agent.planner import ChatRuntime
    from shinku.agent.task_state import AgentTaskStore
    from shinku.memory import MemoryService
    from shinku.tasks.coordinator import AgentTaskCoordinator
    from .commands import QQCommandState
    from .reply_models import QQAgentReply
    from .turns import QQTurn


@dataclass(frozen=True)
class QQTaskContractDependencies:
    build_agent_task_state: Callable[..., tuple[dict[str, Any], str]]
    primary_task_request: Callable[..., str]
    registry_handlers: Callable[..., Mapping[str, Any]]
    repair_agent_task_metadata: Callable[..., AgentDecision]
    request_scope: Callable[..., Any]
    system_prompt_for: Callable[..., str]
    task_effect_catalog: Callable[..., dict[str, list[str]]]
    task_guidance: Callable[..., str]
    task_metadata_repair_anchor: Callable[..., dict[str, Any] | None]
    task_scope: Callable[..., tuple[str, str]]
    task_target_coverage_gaps: Callable[..., Any]
    trace: Callable[..., None]
    validate_task_effects: Callable[..., None]
    prompt_cache_key: Callable[[], str]
    runtime: Callable[[], ChatRuntime]
    task_contract_enabled: Callable[[], bool]
    task_coordinator: Callable[[], AgentTaskCoordinator | None]
    task_store: Callable[[], AgentTaskStore]
    repairable_errors: Callable[[], frozenset[str]]
    configure_budget: Callable[..., None] | None = None
    recovery_check: Callable[..., dict] | None = None
    execution_decides: Callable[[], bool] = lambda: False
    metadata_context: Callable[..., dict[str, Any]] | None = None


@dataclass(frozen=True)
class QQTaskCheckpointDependencies:
    conversation_queued_lease: Callable[..., float]
    conversation_step: Callable[..., None]
    task_coordinator: Callable[[], AgentTaskCoordinator | None]
    task_store: Callable[[], AgentTaskStore]


@dataclass(frozen=True)
class QQTaskMemoryDependencies:
    effective_turn_sender_id: Callable[..., str]
    memory_message_meta: Callable[..., dict[str, Any]]
    primary_task_request: Callable[..., str]
    task_scope: Callable[..., tuple[str, str]]
    trace: Callable[..., None]
    memory_service: Callable[[], MemoryService | None]
    task_store: Callable[[], AgentTaskStore]


@dataclass(frozen=True)
class QQTaskResultDependencies:
    approval_session: Callable[..., str]
    conversation_step: Callable[..., None]
    deliver_text: Callable[..., QQAgentReply]
    is_poke_turn: Callable[..., bool]
    record_agent_task_memory: Callable[..., None]
    remember_task_snapshot: Callable[..., None]
    response_turn: Callable[..., QQTurn]
    terminal_failure_text: Callable[..., str]
    trace: Callable[..., None]
    approval_store: Callable[[], PendingApprovalStore]
    command_state: Callable[[], QQCommandState]
    task_store: Callable[[], AgentTaskStore]
    express_result: Callable[..., QQAgentReply] | None = None
