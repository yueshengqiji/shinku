"""QQ Agent configuration, validation and environment parsing."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from shinku.config import load_settings
from shinku.agent.loop import AGENT_MAX_ROUNDS_HARD_LIMIT
from shinku.extensions import RuntimeExtension

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


def _persona_scene_files(env: Mapping[str, str]) -> dict[str, str]:
    """读取 SHINKU_PERSONA_SCENE_FILES，解析成「会话类型 -> 人设文件」。

    条目以 ``os.pathsep`` 分隔，每条是 ``会话类型=路径``（路径按首个 ``=`` 之后
    整体取值，故盘符与目录中的 ``=`` 不影响）。这是默认人设之外的**按会话来源
    覆盖**：只列需要偏离的会话类型（例如群聊用另一份场景层）。空值表示不覆盖。
    """

    raw = str(env.get("SHINKU_PERSONA_SCENE_FILES", "") or "").strip()
    mapping: dict[str, str] = {}
    for entry in raw.split(os.pathsep):
        item = entry.strip()
        if not item or "=" not in item:
            continue
        conversation_type, path = item.split("=", 1)
        conversation_type = conversation_type.strip().casefold()
        path = path.strip()
        if conversation_type and path:
            mapping[conversation_type] = path
    return mapping


class QQAgentConfigError(ValueError):
    """QQ Agent 被显式启用但配置不完整时抛出。"""


def _validate_private_owner_gate(
    extension: RuntimeExtension | None,
    owner_ids: Sequence[str],
) -> None:
    """A private extension must never start without a configured owner."""

    if extension is not None and extension.owner_gate_required and not any(str(item or "").strip() for item in owner_ids):
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
    conversation_state_store_path: str = ""
    trace_store_path: str = ""
    runtime_extension_module: str = ""
    async_agent_enabled: bool = False
    async_queue_limit: int = 8
    adaptive_budget_enabled: bool = True
    task_max_model_requests: int = 800
    task_max_tokens: int = 8000000
    task_max_tools: int = 512
    task_max_active_seconds: float = 14400.0
    character_file: str = ""
    peer_ids: tuple[str, ...] = ()
    peer_private_unlimited_ids: tuple[str, ...] = ()

    @property
    def conversation_runtime_path(self) -> Path:
        if self.conversation_state_store_path:
            return Path(self.conversation_state_store_path).expanduser().resolve(strict=False)
        task_path = self.agent_state_store_path or self.agent_task_store_path
        return (Path(task_path).with_name("conversation_state.sqlite3") if task_path
                else load_settings().data_root / "runtime" / "conversation_state.sqlite3").resolve(strict=False)

    def diagnostics(self) -> dict[str, object]:
        """返回安全配置快照，绝不返回 key 内容。"""

        return {
            "enabled": self.enabled,
            "send_enabled": self.send_enabled,
            "persona_file": self.persona_file or "(unset)",
            "character_file": self.character_file or "(legacy default)",
            "peer_exchange_enabled": bool(self.peer_ids),
            "peer_exchange_limit": 2 if self.peer_ids else 0,
            # 只公开豁免数量，避免在诊断接口暴露账号名单。
            "peer_private_unlimited_count": len(set(self.peer_private_unlimited_ids)),
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
            "conversation_state_store_path": str(self.conversation_runtime_path),
            "runtime_extension_module": self.runtime_extension_module or "(none)",
            "async_agent_enabled": self.async_agent_enabled,
            "adaptive_budget_enabled": self.adaptive_budget_enabled,
            "async_queue_limit": self.async_queue_limit,
            "task_budget": {"max_requests": self.task_max_model_requests, "max_tokens": self.task_max_tokens,
                            "max_tools": self.task_max_tools, "max_active_seconds": self.task_max_active_seconds},
        }

    def validation_errors(self) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        errors: list[str] = []
        if not set(self.peer_private_unlimited_ids) <= set(self.peer_ids):
            errors.append("peer_private_unlimited_not_peer")
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
                and 1 <= self.task_max_tools <= 1024 and 1 <= self.task_max_active_seconds <= 86400):
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
        character_file=str(env.get("SHINKU_CHARACTER_FILE", "") or "").strip(),
        peer_ids=tuple(_csv_values(env.get("SHINKU_QQ_AGENT_PEER_IDS", ""))),
        peer_private_unlimited_ids=tuple(_csv_values(env.get("SHINKU_QQ_AGENT_PEER_PRIVATE_UNLIMITED_IDS", ""))),
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
        conversation_state_store_path=str(env.get("SHINKU_QQ_CONVERSATION_STATE_DB_PATH", "") or "").strip()
            or ("" if env.get("SHINKU_QQ_AGENT_STATE_DB_PATH") or env.get("SHINKU_QQ_AGENT_TASKS_DB_PATH")
                else str(load_settings(env).data_root / "runtime" / "conversation_state.sqlite3")),
        trace_store_path=str(env.get("SHINKU_QQ_TRACE_STORE_PATH", "") or "").strip(),
        runtime_extension_module=str(env.get("SHINKU_RUNTIME_EXTENSION_MODULE", "") or "").strip(),
        async_agent_enabled=_flag(env.get("SHINKU_QQ_AGENT_ASYNC_ENABLED")),
        async_queue_limit=async_queue_limit,
        adaptive_budget_enabled=_flag(env.get("SHINKU_QQ_AGENT_ADAPTIVE_BUDGET_ENABLED", "true")),
        task_max_model_requests=budget_number("SHINKU_QQ_AGENT_TASK_MAX_MODEL_REQUESTS", 800),
        task_max_tokens=budget_number("SHINKU_QQ_AGENT_TASK_MAX_TOKENS", 8000000),
        task_max_tools=budget_number("SHINKU_QQ_AGENT_TASK_MAX_TOOLS", 512),
        task_max_active_seconds=budget_number("SHINKU_QQ_AGENT_TASK_MAX_ACTIVE_SECONDS", 14400.0, float),
    )
