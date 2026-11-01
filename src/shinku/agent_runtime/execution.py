"""Reuse ToolHost/AgentLoop with an explicit execution runtime and host grant."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from shinku.agent.budget import TaskBudget, bind_task_budget, current_task_budget
from shinku.agent.loop import AgentRunResult
from shinku.hosts.tool_host import ToolHost
from shinku.tools.execution import ExecutionPolicy

from .context import EXECUTION_SYSTEM_PROMPT, build_execution_prompt
from .contracts import AgentBoundaryError


@dataclass(frozen=True)
class ExecutionProfile:
    runtime: Any
    profile_id: str = "explicit-executor"
    max_rounds: int = 6
    temperature: float = 0.2
    allow_unknown_capability: bool = False
    # Existing LLMRuntime accounts usage at provider calls. Alternate injected
    # runtimes are metered conservatively at this boundary by default.
    runtime_meters_budget: bool = False
    # A host/provider probe must establish this separately from tool support.
    supports_images: bool | None = None
    # Trusted host opt-in. No caller/model request can enable P&E.
    stage_policy: Any = None
    # Provider protocol state is opt-in and separate from task evidence.
    native_tool_history: bool = False
    # Explicit host configuration for the no-tool research answer review.
    # None preserves the original shared transport. Tool planning, admission
    # and foreground progress review always use the execution runtime above.
    research_review_runtime: Any = None
    research_review_profile_id: str = ""
    research_review_runtime_meters_budget: bool = False

    def __post_init__(self):
        if (type(self.research_review_runtime_meters_budget) is not bool
                or not isinstance(self.research_review_profile_id, str)
                or self.research_review_runtime is None and (
                    self.research_review_profile_id or self.research_review_runtime_meters_budget)
                or self.research_review_runtime is not None and (
                    not callable(getattr(self.research_review_runtime, "call_chat_json", None))
                    or not self.research_review_profile_id.strip()
                    or len(self.research_review_profile_id) > 256)):
            raise AgentBoundaryError("invalid_research_review_profile")


@dataclass(frozen=True)
class ExecutionGrant:
    """Returned by trusted host policy, never accepted in AgentRequest."""
    authorization_ref: str
    allowed_tools: frozenset[str]
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    permission_mode: str = "ask"
    budget_limits: Mapping[str, Any] = field(default_factory=dict)
    tool_context: Mapping[str, Any] = field(default_factory=dict)
    still_authorized: Callable[[], bool] = field(default=lambda: True, repr=False, compare=False)
    resume_safe: bool = False
    material_sources: tuple[Any, ...] = ()
    material_gaps: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self):
        from .materials import MaterialSource
        if (not isinstance(self.material_sources, tuple) or len(self.material_sources) > 64
                or any(not isinstance(row, MaterialSource) for row in self.material_sources)
                or not isinstance(self.material_gaps, tuple) or len(self.material_gaps) > 128
                or any(not isinstance(row, Mapping) for row in self.material_gaps)
                or not callable(self.still_authorized)):
            raise AgentBoundaryError("invalid_host_grant")
        if self.permission_mode not in {"read", "ask", "traditional", "confirm", "readonly", "full", "ultra"}:
            raise AgentBoundaryError("invalid_host_grant")
        object.__setattr__(self, "allowed_tools", frozenset(self.allowed_tools))
        object.__setattr__(self, "budget_limits", MappingProxyType(dict(self.budget_limits)))
        object.__setattr__(self, "tool_context", MappingProxyType(dict(self.tool_context)))

    def budget(self):
        keys = {"max_requests", "max_tokens", "max_tools", "max_active_seconds"}
        if not self.authorization_ref or set(self.budget_limits) - keys:
            raise AgentBoundaryError("invalid_host_grant")
        return TaskBudget(**dict(self.budget_limits), managed=True, profile="independent-agent")


class _MeteredRuntime:
    def __init__(self, runtime):
        self.runtime = runtime

    def chat_supports_native_tools(self):
        return self.runtime.chat_supports_native_tools()

    def native_message_scope(self):
        return self.runtime.native_message_scope()

    @property
    def context_budget_advisory(self):
        # 可选宿主能力穿过计量包装；底层没有时仍表现为AttributeError，不能伪造支持。
        return self.runtime.context_budget_advisory

    @property
    def context_awareness_enabled(self):
        return getattr(self.runtime, 'context_awareness_enabled', False)

    @property
    def register_work_context(self):
        return self.runtime.register_work_context

    def call_chat_json(self, **kwargs):
        budget = current_task_budget()
        reservation = budget.reserve_request({"messages": [
            {"role": "system", "content": kwargs.get("system_prompt", "")},
            {"role": "system", "content": kwargs.get("system_extra_blocks", [])},
            {"role": "user", "content": kwargs.get("user_prompt", "")},
            {"role": "user", "content": kwargs.get("user_images") or []},
            *(kwargs.get("history_turns") or []), *(kwargs.get("native_history") or [])],
            "tools": kwargs.get("native_tools") or []}) if budget else None
        # This wrapper owns this runtime's accounting; prevent provider hooks
        # from charging the same request twice. Nested tool calls remain bound.
        with bind_task_budget(None):
            response = self.runtime.call_chat_json(**kwargs)
        if budget and reservation is not None:
            budget.reconcile(reservation, response)
        return response


class AgentExecutor:
    def __init__(self, tool_host: ToolHost, profile: ExecutionProfile):
        self.tool_host, self.profile = tool_host, profile

    def run(self, *, task_id, context, grant, observer=None, checkpoint=None,
            first_decision=None, approved_invocations=None, completion_gate=None, approval_gate=None):
        runtime = self.profile.runtime
        if runtime is None or not callable(getattr(runtime, "call_chat_json", None)):
            return AgentRunResult("blocked", 0, reason="execution_runtime_unavailable")
        health = self.tool_host.runtime_health(runtime)
        if health.native_tools_supported is False:
            return AgentRunResult("blocked", 0, reason="execution_native_tools_unsupported")
        if health.native_tools_supported is None and not self.profile.allow_unknown_capability:
            return AgentRunResult("blocked", 0, reason="execution_capability_unknown")
        # 只接收宿主运行时的模型容量，调用者不能借请求字段扩大输入或权限。
        context.pop('agent_context_window_tokens', None)
        context['agent_context_compaction_supported'] = getattr(runtime, 'compaction_supported', False) is True
        context_window = getattr(runtime, 'context_window_tokens', None)
        if callable(context_window):
            window = context_window()
            if type(window) is int and 0 < window <= 2000000:
                context['agent_context_window_tokens'] = window
        allowed = set(grant.allowed_tools) & set(self.tool_host.registry.names())
        policy = ExecutionPolicy(allowed=frozenset(allowed),
            blocked=grant.policy.blocked | (set(self.tool_host.registry.names()) - allowed),
            risk_by_tool=grant.policy.risk_by_tool,
            approval_required_by_risk=grant.policy.approval_required_by_risk,
            default_risk=grant.policy.default_risk)

        def interrupt(state):
            if not grant.still_authorized():
                return "agent_permission_changed"
            budget = current_task_budget()
            if budget is not None and budget.stop_reason():
                return budget.stop_reason()
            check = context.get("agent_task_interrupt")
            return str(check() or "") if callable(check) else ""

        loop = None
        stages = None
        material_session = context.get("agent_material_session")
        try:
            host = self.tool_host
            query = context.get("agent_query_checkpoint")
            if context.get("agent_task_state", {}).get("foreground_query") and query is None:
                raise AgentBoundaryError("query_live_checkpoint_unavailable")
            if query is not None:
                from .query_handoff import ForegroundQueryCheckpoint
                if not isinstance(query, ForegroundQueryCheckpoint):
                    raise AgentBoundaryError("host_query_checkpoint_required")
                query.validate(context.get("agent_task_state", {}), grant, context)
                host = query.tool_host(host)
                allowed &= set(host.registry.names())
            from .conversation import conversation_turn, CONVERSATION_GUIDANCE
            turn = conversation_turn(context)
            direct = turn is not None and turn.reply_delivery
            from .capability_answer import capability_task, capability_host, capability_first_decision
            if turn is None and capability_task(context):
                host, allowed, policy = capability_host(host, allowed, policy, context, grant)
                if checkpoint is None and first_decision is None:
                    first_decision = capability_first_decision()
            if turn is not None:
                host, allowed, policy = turn.tool_host(host, allowed, policy)
            if material_session is not None:
                from .material_session import TaskMaterialSession
                if not isinstance(material_session, TaskMaterialSession):
                    raise AgentBoundaryError("host_material_session_required")
                context["agent_execution_images"] = material_session.media(supported=self.profile.supports_images)
                unavailable = material_session.unavailable_reason()
                if unavailable:
                    return AgentRunResult("waiting_user", 0, reason=unavailable,
                        final_text="必要任务资料尚未取得完整来源，请补充原文、图片或附件后继续。",
                        completion={"accepted": False, "reason": unavailable, "missing": [unavailable]})
                host, allowed, policy = material_session.tool_host(host, allowed, policy)
                # Reject an oversized required context before the admitted
                # first invocation can create an effect.
                from types import SimpleNamespace
                build_execution_prompt(SimpleNamespace(context=context, transcript=[]),
                                       native_rules=self.profile.native_tool_history)
            def observe(state, status):
                if context.get("agent_query_source_refreshes"):
                    state.progress["query_source_refreshes"] = list(context["agent_query_source_refreshes"])
                if stages is not None:
                    stages.refresh(state)
                if material_session is not None:
                    material_session.checkpoint()
                if observer is not None:
                    observer(state, status)
                # 先把真实回执和材料引用提交到任务库，再构建首次模型视图。
                # 可补读检查只承认持久任务里的引用，不能把尚未提交的临时引用
                # 当成已保存原件；此顺序也保证视图失败不妨碍执行事实落盘。
                if self.profile.native_tool_history and state.native_tool_history is not None:
                    state.native_tool_history.observe(state)
            metered = runtime if self.profile.runtime_meters_budget else _MeteredRuntime(runtime)
            if self.profile.research_review_runtime is not None and not direct:
                from .research_review import ResearchAnswerReviewRuntime
                review = self.profile.research_review_runtime
                review = review if self.profile.research_review_runtime_meters_budget else _MeteredRuntime(review)
                metered = ResearchAnswerReviewRuntime(metered, review,
                    execution_profile_id=self.profile.profile_id,
                    review_profile_id=self.profile.research_review_profile_id)
            if self.profile.stage_policy is not None and not direct:
                from .stages import StageController, StagePolicy
                if not isinstance(self.profile.stage_policy, StagePolicy):
                    raise AgentBoundaryError('invalid_stage_policy')
                context['agent_stage_read_tools'] = {name for name, handler in host.registry.view().items()
                    if name in allowed and getattr(handler, 'agent_permission_class', '') == 'read'}
                stages = StageController(context, self.profile.stage_policy, metered,
                    profile_id=self.profile.profile_id, temperature=self.profile.temperature,
                    interrupt=interrupt, save=observe)
            def prompt(state):
                if turn is not None:
                    turn.attach(state, host.registry.view())
                parts = build_execution_prompt(state, native_rules=self.profile.native_tool_history)
                if turn is not None:
                    from shinku.qq.task_contracts import task_effect_catalog
                    from types import SimpleNamespace
                    import json
                    if not direct:
                        catalog = task_effect_catalog(deps=SimpleNamespace(registry_handlers=host.registry.view))
                        parts = replace(parts, user_prompt=parts.user_prompt + '\n[Registered effect catalog]\n' + json.dumps(catalog))
                return stages.prepare(state, parts) if stages is not None else parts
            system_prompt = EXECUTION_SYSTEM_PROMPT
            if turn is not None:
                from shinku.qq.task_contracts import task_guidance
                from shinku.angelos.input_routing import INITIATIVE_GUIDANCE
                from .work_planning import PLAN_GUIDANCE
                if direct:
                    from .reply_delivery import GUIDANCE
                    system_prompt = system_prompt.replace('host permissions and completion contract.', 'host permissions.')
                    system_prompt += '\n' + GUIDANCE + '\n' + INITIATIVE_GUIDANCE + '\n' + turn.work_plan.build_prompt_instruction()
                else:
                    system_prompt += '\n' + CONVERSATION_GUIDANCE + '\n' + INITIATIVE_GUIDANCE + '\n' + task_guidance() + '\n' + PLAN_GUIDANCE
            loop = host.build_loop(runtime=metered,
                system_prompt=system_prompt,
                allowed_tool_names=allowed or {"__independent_agent_no_tools__"},
                build_user_prompt=prompt,
                build_user_images=lambda state: list(state.context.get("agent_execution_images", [])),
                max_rounds=self.profile.max_rounds,
                policy=policy, completion_gate=completion_gate, approval_gate=approval_gate,
                temperature=self.profile.temperature,
                prompt_cache_key="agent-execution:" + self.profile.profile_id, history_limit=12,
                state_observer=observe, interruption_check=interrupt,
                native_tool_history=self.profile.native_tool_history, sliced=True)
            if turn is not None:
                loop.completion_gate = turn.gate(loop.completion_gate)
                if checkpoint is not None:
                    checkpoint.state.context = context
                    turn.attach(checkpoint.state, host.registry.view())
            if stages is not None:
                stages.gate = loop.completion_gate
            if query is not None and checkpoint is None:
                checkpoint = query.seed(context, host.registry.view())
            # 仅可信运行时可绑定本轮本地展示通道；模型输入不能伪造绑定ID。
            bind = getattr(runtime, 'bind_task', None)
            from shinku.llm.context_compaction import continuation_guard
            with (bind(task_id) if callable(bind) else nullcontext()), continuation_guard(lambda: not interrupt(None)):
                result = loop.run(task_id=task_id, context=context, initial_transcript=[],
                    first_decision=first_decision, approved_invocations=approved_invocations, checkpoint=checkpoint)
            # Preserve only process-local executable state for an explicitly
            # authorized continuation, including duplicate-effect counters.
            return replace(result, checkpoint=loop._checkpoint)
        except Exception as exc:
            from shinku.agent.budget import TaskBudgetExceeded
            from .stages import StageStrategyStopped
            reason = (str(exc) if isinstance(exc, TaskBudgetExceeded) else exc.code if isinstance(exc, AgentBoundaryError)
                      else str(exc) if isinstance(exc, StageStrategyStopped) else "execution_runtime_error:" + type(exc).__name__)
            saved = getattr(loop, "_checkpoint", None) or checkpoint
            if saved is not None:
                from shinku.agent.execution_sessions import stop_execution_sessions
                stop_execution_sessions(saved.state, self.tool_host.registry.view(), reason)
            return AgentRunResult("canceled" if reason == 'task_canceled' else "blocked" if isinstance(exc, StageStrategyStopped)
                else "waiting_user" if isinstance(exc, AgentBoundaryError) and material_session is not None
                else "blocked" if isinstance(exc, TaskBudgetExceeded) else "failed", 0, reason=reason,
                checkpoint=saved, completion={"accepted": False, "reason": reason, "missing": [reason]}
                    if isinstance(exc, AgentBoundaryError) and material_session is not None
                    else dict(saved.state.completion) if saved else {},
                progress=dict(saved.state.progress) if saved else {})
