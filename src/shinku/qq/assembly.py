"""独立 QQ Agent 的配置与装配边界。

本模块只负责把独立 persona、LLMRuntime、ToolHost 和 QQAgentBridge 组起来。
它不读取旧项目配置，也不主动发送网络请求。真实出站必须由
``SHINKU_QQ_AGENT_SEND_ENABLED=true`` 显式打开。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from shinku.agent.planner import ChatRuntime, PromptBuilder
from shinku.agent.pending import JsonPendingApprovalStore
from shinku.agent.loop import AGENT_MAX_ROUNDS_HARD_LIMIT
from shinku.hosts.tool_host import ToolHost
from shinku.llm.runtime import LLMRuntime
from shinku.memory import LLMMemoryJudge, MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.persona import PersonaDocument, load_persona
from shinku.tools.execution import ExecutionPolicy

from .agent_bridge import QQAgentBridge, QQMessageSender

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


class QQAgentConfigError(ValueError):
    """QQ Agent 被显式启用但配置不完整时抛出。"""


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
        return tuple(errors)


def load_qq_agent_config(environ: Mapping[str, str] | None = None) -> QQAgentConfig:
    """读取独立 ``SHINKU_*`` 配置，不兼容旧项目键名。"""

    env = os.environ if environ is None else environ
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
    )


@dataclass(frozen=True)
class QQAgentAssembly:
    config: QQAgentConfig
    persona: PersonaDocument
    bridge: QQAgentBridge

    def diagnostics(self) -> dict[str, object]:
        return {
            "config": self.config.diagnostics(),
            "persona": self.persona.diagnostics(),
            "sender_mode": "enabled" if self.bridge.sender is not None else "dry_run",
        }


def assemble_qq_agent(
    config: QQAgentConfig,
    *,
    sender: QQMessageSender | None = None,
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
    memory_service = None
    if config.memory_enabled:
        memory_policy = MemoryPolicy.from_environment()
        db_path = Path(config.memory_db_path) if config.memory_db_path else Path("data") / "memory" / "shinku_memory.sqlite3"
        judge = (
            LLMMemoryJudge(runtime, recent_context_chars=memory_policy.recent_context_chars)
            if config.memory_judge_enabled
            else None
        )
        memory_service = MemoryService(
            store=MemoryStore(db_path),
            router=MemoryRouter(
                judge=judge,
                deterministic_guards=memory_policy.deterministic_guards,
                max_query_chars=memory_policy.max_query_chars,
            ),
            policy=memory_policy,
        )
    bridge = QQAgentBridge(
        runtime=runtime,
        system_prompt=persona.text,
        sender=sender if config.send_enabled else None,
        tool_host=tool_host,
        allowed_tool_names=allowed_tool_names,
        build_user_prompt=build_user_prompt,
        max_rounds=config.max_rounds,
        policy=effective_policy,
        prompt_cache_key=config.prompt_cache_key,
        temperature=config.temperature,
        history_turns=config.history_turns,
        recent_context_messages=config.recent_context_messages,
        approval_ttl_seconds=config.approval_ttl_seconds,
        approval_store=approval_store,
        memory_service=memory_service,
    )
    return QQAgentAssembly(config=config, persona=persona, bridge=bridge)
