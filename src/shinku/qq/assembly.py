"""独立 QQ Agent 的配置与装配边界。

本模块只负责把独立 persona、LLMRuntime、ToolHost 和 QQAgentBridge 组起来。
它不读取旧项目配置，也不主动发送网络请求。真实出站必须由
``SHINKU_QQ_AGENT_SEND_ENABLED=true`` 显式打开。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

from shinku.agent.planner import ChatRuntime, PromptBuilder
from shinku.agent.pending import JsonPendingApprovalStore
from shinku.agent.task_state import AgentTaskStore
from shinku.agent.loop import AGENT_MAX_ROUNDS_HARD_LIMIT
from shinku.agent.self_state import SelfStateTool
from shinku.agent.change_awareness import RuntimeChangeTracker
from shinku.hosts.tool_host import ToolHost
from shinku.llm.runtime import LLMRuntime
from shinku.memory import LLMMemoryJudge, MemoryPolicy, MemoryRouter, MemorySearchTool, MemoryService, MemoryStore
from shinku.memory.embedding_factory import build_embedding_provider
from shinku.memory.lifecycle_review import LLMLifecycleReviewer
from shinku.persona import PersonaDocument, load_persona
from shinku.extensions import RuntimeExtension, RuntimeExtensionError, load_runtime_extension
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry
from shinku.tasks.coordinator import AgentTaskCoordinator

from .agent_bridge import QQAgentBridge, QQMessageSender
from .chat_flow import ChatFlowController
from .commands import QQCommandState
from .reply_host import ReplyHost
from .trace import TurnTraceStore

__all__ = [
    "QQAgentConfig",
    "QQAgentConfigError",
    "QQAgentAssembly",
    "load_qq_agent_config",
    "assemble_qq_agent",
]


def _flag(value: object, *, default: bool = False) -> bool:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _csv_values(value: object) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in str(value or "").replace("，", ",").split(",")
        if item.strip()
    )


class QQAgentConfigError(ValueError):
    """QQ Agent 被显式启用但配置不完整时抛出。"""


def _validate_private_owner_gate(
    extension: RuntimeExtension | None,
    owner_ids: Sequence[str],
) -> None:
    """A private extension must never start without a configured owner."""

    if extension is not None and not any(str(item or "").strip() for item in owner_ids):
        raise QQAgentConfigError("private_master_user_ids_missing")


@dataclass(frozen=True)
class QQAgentConfig:
    enabled: bool = False
    send_enabled: bool = False
    persona_file: str = ""
    chat_api_key: str = ""
    chat_base_url: str = ""
    chat_api_protocol: str = "auto"
    chat_model_name: str = ""
    metrics_path: str = ""
    max_rounds: int = 6
    history_turns: int = 12
    recent_context_messages: int = 8
    approval_ttl_seconds: float = 300.0
    approval_store_path: str = ""
    temperature: float = 0.7
    prompt_cache_key: str = "shinku:qq:v1"
    memory_enabled: bool = True
    memory_db_path: str = ""
    memory_judge_enabled: bool = True
    reply_host_enabled: bool = True
    reply_long_trigger_chars: int = 360
    reply_max_segment_chars: int = 600
    reply_segment_delay_seconds: float = 0.0
    reply_strip_final_periods: bool = True
    command_state_path: str = ""
    agent_task_store_path: str = ""
    agent_state_store_path: str = ""
    trace_store_path: str = ""
    runtime_extension_module: str = ""
    async_agent_enabled: bool = False
    async_queue_limit: int = 8
    task_max_model_requests: int = 120
    task_max_tokens: int = 600000
    task_max_tools: int = 128
    task_max_active_seconds: float = 1800.0

    def diagnostics(self) -> dict[str, object]:
        """返回安全配置快照，绝不返回 key 内容。"""

        return {
            "enabled": self.enabled,
            "send_enabled": self.send_enabled,
            "persona_file": self.persona_file or "(unset)",
            "chat_api_key": "set" if self.chat_api_key else "unset",
            "chat_base_url": self.chat_base_url or "(unset)",
            "chat_api_protocol": self.chat_api_protocol,
            "chat_model_name": self.chat_model_name or "(unset)",
            "metrics_path": self.metrics_path or "(memory)",
            "max_rounds": self.max_rounds,
            "history_turns": self.history_turns,
            "recent_context_messages": self.recent_context_messages,
            "approval_ttl_seconds": self.approval_ttl_seconds,
            "approval_store_path": self.approval_store_path or "(memory)",
            "temperature": self.temperature,
            "prompt_cache_key": self.prompt_cache_key or "(unset)",
            "memory_enabled": self.memory_enabled,
            "memory_db_path": self.memory_db_path or "(default data root)",
            "memory_judge_enabled": self.memory_judge_enabled,
            "reply_host_enabled": self.reply_host_enabled,
            "reply_long_trigger_chars": self.reply_long_trigger_chars,
            "reply_max_segment_chars": self.reply_max_segment_chars,
            "reply_segment_delay_seconds": self.reply_segment_delay_seconds,
            "reply_strip_final_periods": self.reply_strip_final_periods,
            "command_state_path": self.command_state_path or "(memory)",
            "agent_task_store_path": self.agent_task_store_path or "(memory)",
            "agent_state_store_path": self.agent_state_store_path or self.agent_task_store_path or "data/agent_conversation_tasks.sqlite3",
            "trace_store_path": self.trace_store_path or "(beside task state; bounded metadata only)",
            "runtime_extension_module": self.runtime_extension_module or "(none)",
            "async_agent_enabled": self.async_agent_enabled,
            "async_queue_limit": self.async_queue_limit,
            "task_budget": {"max_requests": self.task_max_model_requests, "max_tokens": self.task_max_tokens,
                            "max_tools": self.task_max_tools, "max_active_seconds": self.task_max_active_seconds},
        }

    def validation_errors(self) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        errors: list[str] = []
        if not self.persona_file.strip():
            errors.append("persona_file_missing")
        if not self.chat_base_url.strip():
            errors.append("chat_base_url_missing")
        if not self.chat_model_name.strip():
            errors.append("chat_model_name_missing")
        protocol = self.chat_api_protocol.strip().lower()
        if protocol not in {"", "auto", "openai", "anthropic", "ollama"}:
            errors.append("chat_api_protocol_invalid")
        if protocol not in {"ollama"} and not self.chat_api_key.strip():
            errors.append("chat_api_key_missing")
        if not 1 <= self.max_rounds <= AGENT_MAX_ROUNDS_HARD_LIMIT:
            errors.append("max_rounds_invalid")
        if not 1 <= self.history_turns <= 100:
            errors.append("history_turns_invalid")
        if not 1 <= self.recent_context_messages <= 50:
            errors.append("recent_context_messages_invalid")
        if not 30.0 <= self.approval_ttl_seconds <= 86400.0:
            errors.append("approval_ttl_seconds_invalid")
        if not 0.0 <= self.temperature <= 1.0:
            errors.append("temperature_invalid")
        if not 1 <= self.async_queue_limit <= 64:
            errors.append("async_queue_limit_invalid")
        if not (1 <= self.task_max_model_requests <= 1000 and 1000 <= self.task_max_tokens <= 10000000
                and 1 <= self.task_max_tools <= 128 and 1 <= self.task_max_active_seconds <= 86400):
            errors.append("task_budget_invalid")
        return tuple(errors)


def load_qq_agent_config(environ: Mapping[str, str] | None = None) -> QQAgentConfig:
    """读取独立 ``SHINKU_*`` 配置，不兼容旧项目键名。"""

    env = os.environ if environ is None else environ
    def budget_number(key, default, cast=int):
        try:
            return cast(env.get(key, default))
        except (ValueError, TypeError):
            return default
    raw_rounds = str(env.get("SHINKU_QQ_AGENT_MAX_ROUNDS", "6") or "6").strip()
    try:
        max_rounds = int(raw_rounds)
    except ValueError:
        max_rounds = 6
    raw_temperature = str(env.get("SHINKU_QQ_AGENT_TEMPERATURE", "0.7") or "0.7").strip()
    try:
        temperature = float(raw_temperature)
    except ValueError:
        temperature = 0.7
    raw_history = str(env.get("SHINKU_QQ_AGENT_HISTORY_TURNS", "12") or "12").strip()
    try:
        history_turns = int(raw_history)
    except ValueError:
        history_turns = 12
    raw_recent = str(env.get("SHINKU_QQ_AGENT_RECENT_CONTEXT_MESSAGES", "8") or "8").strip()
    try:
        recent_context_messages = int(raw_recent)
    except ValueError:
        recent_context_messages = 8
    raw_ttl = str(env.get("SHINKU_QQ_AGENT_APPROVAL_TTL_SECONDS", "300") or "300").strip()
    try:
        approval_ttl_seconds = float(raw_ttl)
    except ValueError:
        approval_ttl_seconds = 300.0
    raw_reply_trigger = str(env.get("SHINKU_QQ_REPLY_LONG_TRIGGER_CHARS", "360") or "360").strip()
    try:
        reply_long_trigger_chars = int(raw_reply_trigger)
    except ValueError:
        reply_long_trigger_chars = 360
    raw_reply_max_chars = str(env.get("SHINKU_QQ_REPLY_MAX_SEGMENT_CHARS", "600") or "600").strip()
    try:
        reply_max_segment_chars = int(raw_reply_max_chars)
    except ValueError:
        reply_max_segment_chars = 600
    raw_reply_delay = str(env.get("SHINKU_QQ_REPLY_SEGMENT_DELAY_SECONDS", "0") or "0").strip()
    try:
        reply_segment_delay_seconds = float(raw_reply_delay)
    except ValueError:
        reply_segment_delay_seconds = 0.0
    reply_strip_final_periods = _flag(
        env.get("SHINKU_QQ_REPLY_STRIP_FINAL_PERIODS", "true"),
        default=True,
    )
    try:
        async_queue_limit = int(str(env.get("SHINKU_QQ_AGENT_ASYNC_QUEUE_LIMIT", "8") or "8"))
    except ValueError:
        async_queue_limit = 8
    return QQAgentConfig(
        enabled=_flag(env.get("SHINKU_QQ_AGENT_ENABLED")),
        send_enabled=_flag(env.get("SHINKU_QQ_AGENT_SEND_ENABLED")),
        persona_file=str(env.get("SHINKU_PERSONA_FILE", "") or "").strip(),
        chat_api_key=str(env.get("SHINKU_CHAT_API_KEY", "") or "").strip(),
        chat_base_url=str(env.get("SHINKU_CHAT_BASE_URL", "") or "").strip(),
        chat_api_protocol=str(env.get("SHINKU_CHAT_API_PROTOCOL", "auto") or "auto").strip().lower(),
        chat_model_name=str(env.get("SHINKU_CHAT_MODEL_NAME", "") or "").strip(),
        metrics_path=str(env.get("SHINKU_CHAT_METRICS_PATH", "") or "").strip(),
        max_rounds=max_rounds,
        history_turns=history_turns,
        recent_context_messages=recent_context_messages,
        approval_ttl_seconds=approval_ttl_seconds,
        approval_store_path=str(env.get("SHINKU_QQ_AGENT_APPROVAL_STORE_PATH", "") or "").strip(),
        temperature=temperature,
        prompt_cache_key=str(env.get("SHINKU_QQ_AGENT_PROMPT_CACHE_KEY", "shinku:qq:v1") or "shinku:qq:v1").strip(),
        memory_enabled=_flag(env.get("SHINKU_MEMORY_ENABLED", "true"), default=True),
        memory_db_path=str(env.get("SHINKU_MEMORY_DB_PATH", "") or "").strip(),
        memory_judge_enabled=_flag(env.get("SHINKU_MEMORY_JUDGE_ENABLED", "true"), default=True),
        reply_host_enabled=_flag(env.get("SHINKU_QQ_REPLY_HOST_ENABLED", "true"), default=True),
        reply_long_trigger_chars=reply_long_trigger_chars,
        reply_max_segment_chars=reply_max_segment_chars,
        reply_segment_delay_seconds=max(0.0, min(10.0, reply_segment_delay_seconds)),
        reply_strip_final_periods=reply_strip_final_periods,
        command_state_path=str(env.get("SHINKU_QQ_COMMAND_STATE_PATH", "") or "").strip(),
        agent_task_store_path=str(env.get("SHINKU_QQ_AGENT_TASKS_DB_PATH", "") or "").strip(),
        agent_state_store_path=str(env.get("SHINKU_QQ_AGENT_STATE_DB_PATH", "") or "").strip(),
        trace_store_path=str(env.get("SHINKU_QQ_TRACE_STORE_PATH", "") or "").strip(),
        runtime_extension_module=str(env.get("SHINKU_RUNTIME_EXTENSION_MODULE", "") or "").strip(),
        async_agent_enabled=_flag(env.get("SHINKU_QQ_AGENT_ASYNC_ENABLED")),
        async_queue_limit=async_queue_limit,
        task_max_model_requests=budget_number("SHINKU_QQ_AGENT_TASK_MAX_MODEL_REQUESTS", 120),
        task_max_tokens=budget_number("SHINKU_QQ_AGENT_TASK_MAX_TOKENS", 600000),
        task_max_tools=budget_number("SHINKU_QQ_AGENT_TASK_MAX_TOOLS", 128),
        task_max_active_seconds=budget_number("SHINKU_QQ_AGENT_TASK_MAX_ACTIVE_SECONDS", 1800.0, float),
    )


@dataclass(frozen=True)
class QQAgentAssembly:
    config: QQAgentConfig
    persona: PersonaDocument
    bridge: QQAgentBridge
    task_coordinator: AgentTaskCoordinator
    extension: RuntimeExtension | None = None
    change_tracker: RuntimeChangeTracker | None = None

    def diagnostics(self) -> dict[str, object]:
        memory_service = getattr(self.bridge, "memory_service", None)
        memory_store = getattr(memory_service, "store", None)
        registered_tools = ()
        tool_host = getattr(self.bridge, "tool_host", None)
        registry = getattr(tool_host, "registry", None)
        names_getter = getattr(registry, "names", None)
        if callable(names_getter):
            try:
                registered_tools = names_getter()
            except Exception:
                registered_tools = ()
        return {
            "config": self.config.diagnostics(),
            "persona": self.persona.diagnostics(),
            "sender_mode": "enabled" if self.bridge.sender is not None else "dry_run",
            "voice_renderer": "enabled" if self.bridge.voice_renderer is not None else "disabled",
            "chat_flow": self.bridge.chat_flow.diagnostics(),
            "task_coordinator": {
                "enabled": True,
                "tasks": len(self.task_coordinator.list()),
                **self.task_coordinator.storage_status(),
            },
            "runtime_extension": self.extension.diagnostics() if self.extension is not None else {"configured": False},
            "memory_semantic": memory_store.semantic_status() if memory_store is not None else {"enabled": False},
            "memory_owner": memory_service.owner_status() if memory_service is not None else {"enabled": False},
            "changes": (
                self.change_tracker.diagnostics(registered_tools=registered_tools)
                if self.change_tracker is not None
                else {"enabled": False, "status": "not_attached"}
            ),
            "owner_tool_gate": {
                "enabled": bool(getattr(self.bridge, "master_user_ids", ())),
                "configured_master_count": len(getattr(self.bridge, "master_user_ids", ())),
            },
            "command_auth": {
                "owner_required": bool(
                    getattr(getattr(self.bridge, "command_interpreter", None), "require_owner", False)
                ),
                "configured_owner_count": len(getattr(getattr(self.bridge, "command_state", None), "owner_ids", ())),
            },
        }


def assemble_qq_agent(
    config: QQAgentConfig,
    *,
    sender: QQMessageSender | None = None,
    group_delivery_observer: Callable[[str, tuple[str, ...], tuple[str, ...], bool, float], None] | None = None,
    runtime: ChatRuntime | None = None,
    tool_host: ToolHost | None = None,
    allowed_tool_names: set[str] | None = None,
    build_user_prompt: PromptBuilder | None = None,
    policy: ExecutionPolicy | None = None,
) -> QQAgentAssembly:
    """建立 QQ Agent；未显式开启时拒绝装配，避免半配置运行。"""

    errors = config.validation_errors()
    if errors:
        raise QQAgentConfigError(",".join(errors))
    if not config.enabled:
        raise QQAgentConfigError("qq_agent_disabled")
    effective_policy = policy if policy is not None else ExecutionPolicy.from_environment()
    approval_store = JsonPendingApprovalStore(config.approval_store_path) if config.approval_store_path else None
    persona = load_persona(config.persona_file)
    if runtime is None:
        metrics = Path(config.metrics_path) if config.metrics_path else None
        runtime = LLMRuntime(metrics_path=metrics)
    memory_policy = MemoryPolicy.from_environment()
    try:
        extension = load_runtime_extension(config.runtime_extension_module)
    except RuntimeExtensionError as exc:
        raise QQAgentConfigError(str(exc)) from exc
    _validate_private_owner_gate(extension, memory_policy.master_user_ids)

    memory_service = None
    if config.memory_enabled:
        db_path = Path(config.memory_db_path) if config.memory_db_path else Path("data") / "memory" / "shinku_memory.sqlite3"
        embedding_build = build_embedding_provider(
            memory_policy,
            default_cache_dir=db_path.parent / "models" / "bge-m3",
        )
        judge = (
            LLMMemoryJudge(runtime, recent_context_chars=memory_policy.recent_context_chars)
            if config.memory_judge_enabled
            else None
        )
        memory_service = MemoryService(
            store=MemoryStore(
                db_path,
                embedding_provider=embedding_build.provider,
                semantic_enabled=memory_policy.semantic_enabled,
                semantic_backfill_batch_size=memory_policy.semantic_backfill_batch_size,
                semantic_min_score=memory_policy.semantic_min_score,
                embedding_status=embedding_build.status,
                embedding_error=embedding_build.error,
            ),
            router=MemoryRouter(
                judge=judge,
                deterministic_guards=memory_policy.deterministic_guards,
                max_query_chars=memory_policy.max_query_chars,
            ),
            policy=memory_policy,
            lifecycle_reviewer=(
                LLMLifecycleReviewer(runtime)
                if memory_policy.lifecycle.semantic_review_enabled else None
            ),
        )
        if embedding_build.provider is not None and memory_service.store.semantic_enabled:
            memory_service.store.start_embedding_worker(
                interval_seconds=memory_policy.semantic_index_interval_seconds,
            )
    effective_tool_host = tool_host or ToolHost(ToolRegistry())
    change_tracker = RuntimeChangeTracker.from_environment()
    if extension is not None:
        try:
            extension.register_tools(registry=effective_tool_host.registry, config=config)
        except RuntimeExtensionError as exc:
            raise QQAgentConfigError(str(exc)) from exc
    if memory_service is not None and effective_tool_host.registry.get("memory_search") is None:
        effective_tool_host.registry.register(MemorySearchTool(memory_service), name="memory_search")
    task_coordinator = AgentTaskCoordinator(
        state_path=config.agent_task_store_path or (config.agent_state_store_path or "data/agent_background_tasks.sqlite3" if config.async_agent_enabled else None),
        max_workers=1 if config.async_agent_enabled else 2,
        max_pending_tasks=config.async_queue_limit,
    )

    def self_state_provider(*, refresh: bool = False) -> Mapping[str, object]:
        chat = getattr(runtime, "chat", None)
        return {
            "runtime": {
                "model": str(getattr(chat, "model", "") or getattr(runtime, "chat_model", "") or ""),
                "provider": str(getattr(chat, "provider", "") or getattr(runtime, "provider", "") or ""),
            },
            "config": config.diagnostics(),
            "persona": persona.diagnostics(),
            "agent": {
                "max_rounds": config.max_rounds,
                "async_agent_enabled": config.async_agent_enabled,
                "async_queue_limit": config.async_queue_limit,
                "history_turns": config.history_turns,
                "recent_context_messages": config.recent_context_messages,
                "sender_configured": bool(config.send_enabled and sender is not None),
                "owner_tool_gate_enabled": bool(memory_policy.master_user_ids),
                "configured_master_count": len(memory_policy.master_user_ids),
            },
            "tools": {"registered": list(effective_tool_host.registry.names())},
            "changes": change_tracker.diagnostics(
                registered_tools=effective_tool_host.registry.names(),
                refresh=refresh,
            ),
        }

    if effective_tool_host.registry.get("read_self_state") is None:
        effective_tool_host.registry.register(
            SelfStateTool(
                self_state_provider,
                fresh_provider=lambda: self_state_provider(refresh=True),
            ),
            name="read_self_state",
        )
    change_tracker.start(registered_tools=effective_tool_host.registry.names())
    risk_by_tool = dict(effective_policy.risk_by_tool)
    risk_by_tool.setdefault("memory_search", "low")
    risk_by_tool.setdefault("read_self_state", "low")
    if extension is not None:
        risk_by_tool.setdefault("private_browser_status", "low")
        risk_by_tool.setdefault("private_browser_setup", "high")
    for name, handler in effective_tool_host.registry.view().items():
        if (
            str(getattr(handler, "agent_permission_class", "") or "").strip().casefold() == "read"
            and str(getattr(handler, "agent_progress_class", "") or "").strip().casefold() == "observe"
            and not getattr(handler, "agent_research_candidate_field", "")
            and not getattr(handler, "agent_research_content_field", "")
            and not getattr(handler, "agent_evidence_collection_field", "")
        ):
            risk_by_tool.setdefault(name, "low")
    effective_policy = replace(effective_policy, risk_by_tool=risk_by_tool)
    reply_host = ReplyHost(
        enabled=config.reply_host_enabled,
        long_trigger_chars=config.reply_long_trigger_chars,
        max_segment_chars=config.reply_max_segment_chars,
        strip_final_periods=config.reply_strip_final_periods,
    )
    command_state = QQCommandState.from_environment(state_path=config.command_state_path)
    voice_renderer = None
    if extension is not None and _flag(os.environ.get("SHINKU_VOICE_ENABLED")):
        from shinku_private.voice import PrivateVoiceRenderer

        voice_renderer = PrivateVoiceRenderer.from_environment()
    bot_ids = _csv_values(os.environ.get("SHINKU_QQ_BOT_IDS", ""))
    chat_flow = ChatFlowController.from_environment()
    system_prompt = persona.text
    command_interpreter = None
    if extension is not None:
        try:
            extension.bind_runtime(
                registry=effective_tool_host.registry, runtime=runtime,
                permission_provider=command_state.permission_mode,
                enabled_provider=command_state.is_agent_enabled,
            )
            system_prompt = extension.augment_system_prompt(prompt=system_prompt, config=config)
            command_interpreter = extension.build_command_interpreter(
                state=command_state,
                runtime=runtime,
                memory_inspector=memory_service.inspect if memory_service is not None else None,
                memory_reset=memory_service.reset_all if memory_service is not None else None,
                runtime_snapshot=self_state_provider,
                task_controller=task_coordinator.control,
                config=config,
            )
        except RuntimeExtensionError as exc:
            raise QQAgentConfigError(str(exc)) from exc
    bridge = QQAgentBridge(
        runtime=runtime,
        system_prompt=system_prompt,
        sender=sender if config.send_enabled else None,
        group_delivery_observer=group_delivery_observer,
        voice_renderer=voice_renderer,
        tool_host=effective_tool_host,
        allowed_tool_names=allowed_tool_names,
        build_user_prompt=build_user_prompt,
        max_rounds=config.max_rounds,
        policy=effective_policy,
        prompt_cache_key=config.prompt_cache_key,
        temperature=config.temperature,
        history_turns=config.history_turns,
        recent_context_messages=config.recent_context_messages,
        bot_ids=bot_ids,
        master_user_ids=memory_policy.master_user_ids,
        owner_gate_required=extension is not None,
        approval_ttl_seconds=config.approval_ttl_seconds,
        approval_store=approval_store,
        memory_service=memory_service,
        reply_host=reply_host,
        reply_segment_delay_seconds=config.reply_segment_delay_seconds,
        command_state=command_state,
        command_interpreter=command_interpreter,
        chat_flow=chat_flow,
        self_state_provider=self_state_provider,
        task_controller=task_coordinator.control,
        task_store=AgentTaskStore(
            config.agent_state_store_path or config.agent_task_store_path or "data/agent_conversation_tasks.sqlite3",
        ),
        task_contract_enabled=True,
        trace_store=TurnTraceStore(path=config.trace_store_path or str(Path(
            config.agent_state_store_path or config.agent_task_store_path or "data/agent_conversation_tasks.sqlite3",
        ).with_name("qq_turn_traces.json"))),
        task_coordinator=task_coordinator,
        async_agent_enabled=config.async_agent_enabled,
        task_budget_limits={"max_requests": config.task_max_model_requests, "max_tokens": config.task_max_tokens,
                            "max_tools": config.task_max_tools, "max_active_seconds": config.task_max_active_seconds},
    )
    return QQAgentAssembly(
        config=config,
        persona=persona,
        bridge=bridge,
        task_coordinator=task_coordinator,
        extension=extension,
        change_tracker=change_tracker,
    )
