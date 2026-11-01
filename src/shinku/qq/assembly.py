"""独立 QQ Agent 的配置与装配边界。

本模块只负责把独立 persona、LLMRuntime、ToolHost 和 QQAgentBridge 组起来。
它不读取旧项目配置，也不主动发送网络请求。真实出站必须由
``SHINKU_QQ_AGENT_SEND_ENABLED=true`` 显式打开。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from shinku.character import SHINKU_CHARACTER, load_character
from shinku.agent.change_awareness import RuntimeChangeTracker
from shinku.agent.pending import JsonPendingApprovalStore
from shinku.agent.planner import ChatRuntime, PromptBuilder
from shinku.agent.self_state import SelfStateTool
from shinku.agent.task_state import AgentTaskStore
from shinku.agent_runtime.execution import ExecutionProfile
from shinku.agent_runtime.materials import MaterialPolicy, TaskMaterialStore
from shinku.conversation import ConversationRuntime
from shinku.extensions import (
    RuntimeExtension,
    RuntimeExtensionError,
    load_runtime_extension,
)
from shinku.hosts.tool_host import ToolHost
from shinku.llm.runtime import LLMRuntime
from shinku.memory import (
    LLMMemoryJudge,
    MemoryPolicy,
    MemoryRouter,
    MemorySearchTool,
    MemoryService,
    MemoryStore,
)
from shinku.memory.embedding_factory import build_embedding_provider
from shinku.memory.lifecycle_review import LLMLifecycleReviewer
from shinku.persona import PersonaDocument, PersonaLoadError, load_persona
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry

from .agent_bridge import QQAgentBridge, QQMessageSender
from .chat_flow import ChatFlowController
from .commands import QQCommandState
from .reply_host import ReplyHost
from .trace import TurnTraceStore

from . import assembly_memory, assembly_tools, assembly_extension
from .assembly_config import _flag, _csv_values, _persona_scene_files, QQAgentConfigError, _validate_private_owner_gate, QQAgentConfig, load_qq_agent_config

logger = logging.getLogger(__name__)

__all__ = [
    "QQAgentConfig",
    "QQAgentConfigError",
    "QQAgentAssembly",
    "load_qq_agent_config",
    "assemble_qq_agent",
]


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
            "supervision": {
                "accepts_handoffs": bool(getattr(self.bridge, "supervisor_incoming", False)),
                "target": str(getattr(self.bridge, "supervisor_target", "")),
            },
            "sender_mode": "enabled" if self.bridge.sender is not None else "dry_run",
            "voice_renderer": "enabled" if self.bridge.voice_renderer is not None else "disabled",
            "chat_flow": self.bridge.chat_flow.diagnostics(),
            "conversation_runtime": self.bridge.conversation_runtime.storage_status(),
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
    group_continuation_observer: Callable[[str, str, float], None] | None = None,
    runtime: ChatRuntime | None = None,
    tool_host: ToolHost | None = None,
    allowed_tool_names: set[str] | None = None,
    build_user_prompt: PromptBuilder | None = None,
    policy: ExecutionPolicy | None = None,
    execution_profile: ExecutionProfile | None = None,
    delegation: Any = None,
    expression_profile: Any = None,
    material_policy: MaterialPolicy | None = None,
    material_roots: Sequence[str | Path] = (),
    experience_enabled: bool = False,
    recovery_enabled: bool = False,
    tool_decision_mode: str = "legacy",
    chat_read_capabilities: Mapping[str, Any] | None = None,
    foreground_query_policy: Any = None,
    runtime_extension: RuntimeExtension | None = None,
    file_reader: Any = None,
) -> QQAgentAssembly:
    """建立 QQ Agent；未显式开启时拒绝装配，避免半配置运行。"""

    errors = config.validation_errors()
    if errors:
        raise QQAgentConfigError(",".join(errors))
    if not config.enabled:
        raise QQAgentConfigError("qq_agent_disabled")
    # 在创建状态库、工具或后台组件之前验证角色输入。
    character = load_character(config.character_file) if config.character_file else SHINKU_CHARACTER
    effective_policy = policy if policy is not None else ExecutionPolicy.from_environment()
    approval_store = JsonPendingApprovalStore(config.approval_store_path) if config.approval_store_path else None
    persona = load_persona(config.persona_file)
    if runtime is None:
        metrics = Path(config.metrics_path) if config.metrics_path else None
        runtime = LLMRuntime(metrics_path=metrics)
    memory_policy = MemoryPolicy.from_environment()
    try:
        # 显式宿主对象在启动上下文内只创建一次；旧环境入口继续使用原加载语义。
        extension = runtime_extension if runtime_extension is not None else load_runtime_extension(config.runtime_extension_module)
    except RuntimeExtensionError as exc:
        raise QQAgentConfigError(str(exc)) from exc
    _validate_private_owner_gate(extension, memory_policy.master_user_ids)

    memory_service = assembly_memory.build_memory_service(config, runtime, memory_policy, character=character, deps=assembly_memory.MemoryAssemblyDependencies(
        MemoryService=MemoryService,
        MemoryStore=MemoryStore,
        MemoryRouter=MemoryRouter,
        LLMMemoryJudge=LLMMemoryJudge,
        LLMLifecycleReviewer=LLMLifecycleReviewer,
        build_embedding_provider=build_embedding_provider,
    ))
    effective_tool_host = tool_host or ToolHost(ToolRegistry())
    change_tracker = RuntimeChangeTracker.from_environment()
    if extension is not None:
        try:
            extension.register_tools(registry=effective_tool_host.registry, config=config)
        except RuntimeExtensionError as exc:
            raise QQAgentConfigError(str(exc)) from exc
    if runtime_extension is not None:
        # 在发布工具之后获取目录，让冲突插件先完整撤回；宿主明确传入的只读项优先。
        try:
            chat_read_capabilities = {
                **extension.build_chat_read_capabilities(config=config),
                **dict(chat_read_capabilities or {}),
            }
        except RuntimeExtensionError as exc:
            raise QQAgentConfigError(str(exc)) from exc
    if memory_service is not None and effective_tool_host.registry.get("memory_search") is None:
        effective_tool_host.registry.register(MemorySearchTool(memory_service), name="memory_search")
    chat_read_capabilities = assembly_tools.memory_read_capabilities(
        chat_read_capabilities, memory_service, effective_tool_host)
    task_coordinator = AgentTaskCoordinator(
        state_path=config.agent_task_store_path or (config.agent_state_store_path or "data/agent_background_tasks.sqlite3" if config.async_agent_enabled else None),
        max_workers=1 if config.async_agent_enabled else 2,
        max_pending_tasks=config.async_queue_limit,
    )

    self_state_provider = assembly_tools.make_self_state_provider(
        config,
        runtime,
        persona,
        memory_policy,
        effective_tool_host,
        change_tracker,
        sender,
        chat_read_catalog=lambda: chat_read_capabilities,
        delegation_configured=delegation is not None,
    )

    effective_policy = assembly_tools.register_state_and_policy(
        effective_tool_host,
        self_state_provider,
        change_tracker,
        effective_policy,
        extension,
        self_state_tool=SelfStateTool,
    )
    # 先绑定本实例 provider，再发布只读目录；仅注册工具不会让 task 聊天看到它。
    chat_read_capabilities = assembly_tools.self_state_read_capabilities(
        chat_read_capabilities, self_state_provider, effective_tool_host)
    reply_host = ReplyHost(
        enabled=config.reply_host_enabled,
        long_trigger_chars=config.reply_long_trigger_chars,
        max_segment_chars=config.reply_max_segment_chars,
        strip_final_periods=config.reply_strip_final_periods,
    )
    command_state = QQCommandState.from_environment(state_path=config.command_state_path)
    voice_renderer = None
    if extension is not None and extension.owner_gate_required and _flag(os.environ.get("SHINKU_VOICE_ENABLED")):
        from shinku_private.voice import PrivateVoiceRenderer

        voice_renderer = PrivateVoiceRenderer.from_environment()
    bot_ids = _csv_values(os.environ.get("SHINKU_QQ_BOT_IDS", ""))
    chat_flow = ChatFlowController.from_environment(character=character)
    system_prompt, command_interpreter = assembly_extension.bind_extension(
        config,
        runtime,
        persona,
        extension,
        command_state,
        memory_service,
        self_state_provider,
        task_coordinator,
        effective_tool_host,
        extension_error=RuntimeExtensionError,
        config_error=QQAgentConfigError,
    )
    # 按会话来源覆盖场景层：每个显式路由的会话类型各装配一份完整 prompt，
    # 每份都各自跑一遍 augment_system_prompt，保证私用运行层不缺席。
    # 单份加载失败只跳过该类型并告警，退化成默认人设，绝不让装配起不来。
    scene_prompts = assembly_extension.build_scene_prompts(
        config,
        extension,
        scene_files=_persona_scene_files,
        environ=os.environ,
        persona_loader=load_persona,
        persona_error=PersonaLoadError,
        extension_error=RuntimeExtensionError,
        logger=logger,
    )
    task_store = AgentTaskStore(
        config.agent_state_store_path or config.agent_task_store_path or "data/agent_conversation_tasks.sqlite3")
    materials = TaskMaterialStore(task_store, policy=material_policy, approved_roots=material_roots) if material_policy is not None else None
    bridge = QQAgentBridge(
        runtime=runtime, character=character,
        peer_ids=config.peer_ids,
        peer_exchange_path=str(config.conversation_runtime_path.with_name("peer_exchange.sqlite3")),
        peer_private_unlimited_ids=config.peer_private_unlimited_ids,
        system_prompt=system_prompt,
        system_prompts=scene_prompts,
        sender=sender if config.send_enabled else None,
        group_delivery_observer=group_delivery_observer,
        group_continuation_observer=group_continuation_observer,
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
        owner_gate_required=extension.owner_gate_required if extension is not None else False,
        approval_ttl_seconds=config.approval_ttl_seconds,
        approval_store=approval_store,
        memory_service=memory_service,
        reply_host=reply_host,
        reply_segment_delay_seconds=config.reply_segment_delay_seconds,
        command_state=command_state,
        command_interpreter=command_interpreter,
        chat_flow=chat_flow,
        conversation_runtime=ConversationRuntime(config.conversation_runtime_path,
            lease_seconds=config.task_max_active_seconds + 120),
        self_state_provider=self_state_provider,
        task_controller=task_coordinator.control,
        task_store=task_store,
        material_store=materials,
        task_contract_enabled=True,
        trace_store=TurnTraceStore(path=config.trace_store_path or str(Path(
            config.agent_state_store_path or config.agent_task_store_path or "data/agent_conversation_tasks.sqlite3",
        ).with_name("qq_turn_traces.json"))),
        task_coordinator=task_coordinator,
        async_agent_enabled=config.async_agent_enabled,
        adaptive_budget_enabled=config.adaptive_budget_enabled,
        execution_profile=execution_profile,
        delegation=delegation,
        expression_profile=expression_profile,
        experience_enabled=experience_enabled,
        recovery_enabled=recovery_enabled,
        tool_decision_mode=tool_decision_mode,
        chat_read_capabilities=chat_read_capabilities,
        file_reader=file_reader,
        foreground_query_policy=foreground_query_policy,
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
