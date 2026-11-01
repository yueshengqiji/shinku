"""Self-state tools, change tracking and tool risk policy assembly."""
from __future__ import annotations
from dataclasses import replace
from typing import Mapping


def memory_read_capabilities(capabilities, memory_service, tool_host):
    """Host opt-in for the built-in scoped reader; custom handlers get no grant."""
    from shinku.memory import MemorySearchTool
    result = dict(capabilities or {})
    handler = tool_host.registry.get("memory_search")
    if (type(handler) is MemorySearchTool and handler.service is memory_service
            and handler.model_retrieval):
        result.setdefault("memory_search", handler.chat_read_capability())
    return result


def self_state_read_capabilities(capabilities, self_state_provider, tool_host):
    """只给本次装配绑定的内置自查器补充聊天目录，不把注册表当作授权目录。"""
    from shinku.agent.self_state import SelfStateTool
    from .interaction_tools import ChatReadCapability

    result = dict(capabilities or {})
    handler = tool_host.registry.get("read_self_state")
    # 同名插件、子类及其他执行宿主的 provider 不能自动取得这份聊天准入。
    # fresh_provider 和原有状态回执负责新鲜度、范围及回合绑定；这里仅发布能力，
    # 每次实际调用仍由聊天只读适配器和当前权限策略共同核验。
    if (type(handler) is SelfStateTool and handler.provider is self_state_provider
            and callable(handler.fresh_provider)):
        result.setdefault("read_self_state", ChatReadCapability(
            "read_self_state",
            "只读核对当前聊天实例 Shinku 自己的状态，不是远端 Angelos 执行实例。"
            "询问当前模型、工具、配置、人设或加载变更时，可直接读取相应 scope，无需新建任务。"
            "按问题选择必要范围，整体自查才选 all。模型字段表示运行客户端当前配置，"
            "不证明供应商底层模型权重；状态快照不证明实际部署、业务健康或消息送达。"
            "不读取密钥、源码正文或系统提示词原文。",
            {"type": "object", "properties": {"scope": {
                "type": "string", "enum": ["runtime", "agent", "tools", "persona", "config", "changes", "all"],
                "maxLength": 16}}, "required": ["scope"], "additionalProperties": False},
            max_calls=1, max_result_bytes=64000, max_seconds=10,
        ))
    return result


def make_self_state_provider(
    config,
    runtime,
    persona,
    memory_policy,
    effective_tool_host,
    change_tracker,
    sender,
    *,
    chat_read_catalog=None,
    delegation_configured=False,
):
    def self_state_provider(*, refresh: bool = False) -> Mapping[str, object]:
        from .interaction_tools import audited_handlers
        chat = getattr(runtime, "chat", None)
        bound_reads = audited_handlers(effective_tool_host.registry.view(),
            chat_read_catalog() or {} if chat_read_catalog else {})
        return {
            "runtime": {
                # 状态属于这个聊天 runtime；委托侧有独立 provider，不能互作模型证据。
                "instance": "shinku_chat",
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
            "tools": {
                "registered": list(effective_tool_host.registry.names()),
                # 注册表可能包含远端元数据占位；聊天目录由装配持有，不能把名字当执行许可。
                # getter 在读取时取得装配最终目录，包含后注册的本实例自查工具。
                "chat_read_catalog": sorted(bound_reads),
                "delegation_configured": bool(delegation_configured),
                "registration_grants_execution": False,
            },
            "changes": change_tracker.diagnostics(
                registered_tools=effective_tool_host.registry.names(),
                refresh=refresh,
            ),
        }
    return self_state_provider


def register_state_and_policy(
    effective_tool_host,
    self_state_provider,
    change_tracker,
    effective_policy,
    extension,
    *,
    self_state_tool,
):
    if effective_tool_host.registry.get("read_self_state") is None:
        effective_tool_host.registry.register(
            self_state_tool(
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
    return effective_policy
