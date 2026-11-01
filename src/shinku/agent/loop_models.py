"""Explicit phase dependencies and process-local state for one AgentLoop run."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from shinku.tools.execution import ExecutionPolicy, ToolHandler
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope

from .progress import ProgressLedger

if TYPE_CHECKING:
    from .loop import (
        AgentDecision, AgentRunCheckpoint, AgentRunResult, AgentState,
        ApprovalGate, CompletionGate, Planner,
    )


@dataclass(frozen=True)
class AgentLoopDependencies:
    planner: Callable[[], Planner]
    handlers: Callable[[], Mapping[str, ToolHandler]]
    policy: Callable[[], ExecutionPolicy | None]
    completion_gate: Callable[[], CompletionGate | None]
    approval_gate: Callable[[], ApprovalGate | None]
    max_rounds: Callable[[], int]
    max_consecutive_observations: Callable[[], int]
    max_source_discovery_calls: Callable[[], int]
    sliced: Callable[[], bool]
    checkpoint: Callable[[], AgentRunCheckpoint | None]
    set_checkpoint: Callable[[AgentRunCheckpoint], None]
    new_state: Callable[..., AgentState]
    new_checkpoint: Callable[[AgentState], AgentRunCheckpoint]
    parse_decision: Callable[[Any], AgentDecision | None]
    call_signature: Callable[[ToolInvocation], str]
    result: Callable[..., AgentRunResult]
    research_partial: Callable[..., AgentRunResult]
    check_interruption: Callable[..., None]
    observe_state: Callable[..., None]
    progress_class: Callable[[ToolInvocation], str]
    is_source_discovery: Callable[[str], bool]
    execute_and_record: Callable[..., ToolResultEnvelope]


@dataclass
class AgentLoopRun:
    state: AgentState
    context: Any
    saved: AgentRunCheckpoint | None
    rejected_finals: int
    envelopes: list[ToolResultEnvelope]
    signature_attempts: dict[str, int]
    successful_signatures: dict[str, tuple[str, int]]
    duplicate_rejections: dict[str, int]
    mutation_epoch: int
    search_urls: set[str]
    search_stalled: int
    progress_ledger: ProgressLedger
    round_end: int
    pending_first_decision: AgentDecision | Mapping[str, Any] | None


@dataclass(frozen=True)
class LoopPlanOutcome:
    decision: AgentDecision | None = None
    result: AgentRunResult | None = None


@dataclass(frozen=True)
class LoopGateReview:
    accepted: bool
    status: str = ""
    reason: str = ""
    result: AgentRunResult | None = None


@dataclass
class LoopToolBatch:
    invocations: Sequence[ToolInvocation]
    ready_calls: list[ToolInvocation]
    round_signatures: set[str]
    discovery_slots: int
