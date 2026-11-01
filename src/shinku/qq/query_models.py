"""Per-turn model dependencies and isolated public-query state."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from shinku.agent.loop import AgentDecision
from shinku.agent.planner import AgentPromptParts, ChatRuntime
from shinku.tools.execution import ExecutionPolicy

from .commands import QQCommandState
from .public_query_evidence import PublicQueryEvidence
from .public_query_feedback import PublicQueryFeedback
from .turn_models import QQTurnEffects
from .turns import QQTurn

PUBLIC_QUERY_MAX_TOOLS = 3


@dataclass(frozen=True)
class QQModelCallDependencies:
    runtime: ChatRuntime
    build_prompt: Callable[..., AgentPromptParts]
    system_prompt_for: Callable[[Any], str]
    user_images: Callable[[Any], list[dict[str, Any]]]
    temperature: float
    prompt_cache_key: str
    history_turns: int
    effects: QQTurnEffects
    stream_enabled: Callable[[], bool] = lambda: False

    source_projection: Callable[..., Any] | None = None

    def call_json(self, *, trace_id: str = "", allow_stream: bool = True,
                  relevance_guard: Callable[[str], str] | None = None, source_context=None, **options):
        from .model_stream import call_reply_model
        from .dialogue_grounding import capture_sources
        before = capture_sources(self, source_context, phase="before_generation", trace_id=trace_id) if source_context is not None else None
        if source_context is not None and before is None:
            source_context["qq_dialogue_sources_changed"] = True
            return dict(options["fallback"])
        response = call_reply_model(self, trace_id=trace_id, allow_stream=allow_stream,
                                    relevance_guard=relevance_guard, **options)
        if source_context is not None and (before is None or before != capture_sources(
                self, source_context, phase="after_generation", trace_id=trace_id)):
            # 只撤销本次候选；不追加模型重写，不恢复已被遗忘的资料。
            source_context["qq_dialogue_sources_changed"] = True
        return response


@dataclass(frozen=True)
class QQReadOnlyRouteDependencies:
    command_state: QQCommandState
    tool_names: Callable[[Mapping[str, Any]], set[str]]
    registry_handlers: Callable[[], Mapping[str, Any]]
    policy_for_permission_mode: Callable[[str], ExecutionPolicy]
    is_master_turn: Callable[[QQTurn], bool]
    primary_request: Callable[[QQTurn], str]
    contract_enabled: Callable[[], bool]
    async_enabled: Callable[[], bool]
    public_handlers: Callable[[Mapping[str, Any]], dict[str, Any]]
    tool_decision_mode: str = "legacy"
    read_capabilities: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class QQQueryDependencies:
    model: QQModelCallDependencies
    routes: QQReadOnlyRouteDependencies
    public_handlers: Callable[[Mapping[str, Any]], dict[str, Any]]
    is_public_query: Callable[..., bool]
    native_specs: Callable[..., list[dict[str, Any]]]
    permission_guidance: Callable[[str], str]
    compact_spec: Callable[..., tuple[dict[str, Any], str]]
    resolve_dispatch: Callable[..., tuple[AgentDecision, bool]]
    response_turn: Callable[[QQTurn, Any], QQTurn]
    is_poke: Callable[[QQTurn], bool]
    express_query: Callable[..., Any] | None = None
    foreground_policy: Any = None
    continue_query: Callable[..., Any] | None = None
    existing_query: Callable[..., Any] | None = None


@dataclass
class QQPublicQueryRun:
    turn: QQTurn
    decision: AgentDecision
    context: dict[str, Any]
    trace_id: str
    handlers: dict[str, Any]
    authority: int
    parts: AgentPromptParts
    dialogue: list[dict[str, str]]
    current_message: str
    goal: str
    evidence: PublicQueryEvidence
    diagnostics: PublicQueryFeedback
    safe_specs: list[dict[str, Any]]
    feedback: list[str] = field(default_factory=list)
    envelopes: list[Any] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    source_read: bool = False
    schema_repaired: bool = False
    model_outcome: str = ""
    goal_status: str = "unverified"
    search_calls: int = 0
    rejection: dict[str, Any] | None = None
    model_error: str = ""
    text: str = "这次没有取得足够的来源资料，还不能确认答案。"
    outcome: str = "failed"
    reason: str = "lightweight_public_query_fast_track"
    calls: list[Any] = field(default_factory=list)
    started: float = 0.0
