"""QQ slash-command parsing and small, deterministic control state.

The parser itself remains narrow: only a message whose first character is the
ASCII ``/`` is a slash command.  The conversation boundary is decided before
parsing: private messages may enter directly, while group messages must also
be explicitly addressed to Shinku.  Ordinary group prose that merely starts
with ``/`` is left to the normal QQ/Agent route.

This module does not know NapCat, LLMs, or the Agent loop.  It parses and
applies the controls that belong to the QQ host, which keeps command traffic
out of persona prompts and memory history.
"""

from __future__ import annotations

import json
import inspect
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = [
    "CommandOutcome",
    "QQCommandInterpreter",
    "QQCommandState",
    "SlashCommand",
    "command_text_for_conversation",
    "is_slash_command",
    "parse_slash_command",
]


_COMMAND_NAME_RE = re.compile(r"^[A-Za-z0-9_\-\u4e00-\u9fff]+$")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+@-]{0,63}$")
_PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,47}$")
_STATE_VERSION = 1

AGENT_PERMISSION_MODES = frozenset({"traditional", "readonly", "confirm", "full", "ultra"})
AGENT_PERMISSION_LABELS = {
    "traditional": "传统（沿用风险分级）",
    "readonly": "只读（仅放行明确只读工具）",
    "confirm": "调用/修改确认（每次工具调用均确认）",
    "full": "完全访问（全电脑范围自动读写）",
    "ultra": "Ultra（解除 Shinku 工具层限制）",
}


def is_slash_command(value: Any) -> bool:
    """Return true only when the literal first character is ``/``.

    Deliberately do not call ``strip`` here.  The boundary is the user's
    requested one: ``/`` must be at the beginning, not somewhere in prose or
    after leading whitespace.
    """

    return isinstance(value, str) and value.startswith("/")


def command_text_for_conversation(
    value: Any,
    conversation_type: Any,
    *,
    mentioned_bot: bool = False,
) -> str:
    """Return the command text only when this conversation may use commands.

    Private/direct conversations keep the convenient ``/command`` entry point.
    A group command additionally requires an explicit @ mention of Shinku.
    The returned value is unchanged so ``parse_slash_command`` remains the
    single parser and cannot accidentally accept a slash embedded in prose.
    """

    if not is_slash_command(value):
        return ""
    kind = str(conversation_type or "").strip().casefold()
    if kind in {"private", "friend", "direct"}:
        return str(value)
    if kind == "group" and bool(mentioned_bot):
        return str(value)
    return ""


@dataclass(frozen=True)
class SlashCommand:
    """A parsed command and its unmodified argument tail."""

    name: str
    args: tuple[str, ...]
    argument_text: str
    raw_text: str

    def arg(self, index: int = 0, default: str = "") -> str:
        try:
            return self.args[index]
        except IndexError:
            return default


def parse_slash_command(value: Any) -> SlashCommand | None:
    """Parse a slash command without prefix or substring matching.

    ``/at reply on`` is normalized to ``at_reply``.  All other command names
    are kept lower-case.  Unknown names still parse successfully so the
    interpreter can return help instead of accidentally invoking the LLM.
    """

    if not is_slash_command(value):
        return None
    raw = str(value)
    body = raw[1:]
    stripped = body.strip()
    if not stripped:
        return SlashCommand("help", (), "", raw)
    pieces = re.split(r"\s+", stripped)
    name = pieces[0].casefold()
    if not _COMMAND_NAME_RE.fullmatch(name):
        return SlashCommand("unknown", tuple(pieces), stripped, raw)
    args = pieces[1:]
    # Keep the old, user-facing spelling while giving handlers one stable key.
    if name == "at" and args and args[0].casefold() == "reply":
        name, args = "at_reply", args[1:]
    elif name == "reply" and args and args[0].casefold() == "mode":
        name, args = "reply_mode", args[1:]
    elif name == "provider" and args and args[0].casefold() in {"list", "current", "status"}:
        args = [args[0].casefold(), *args[1:]]
    elif name == "model" and args and args[0].casefold() in {"list", "current", "status", "default"}:
        args = [args[0].casefold(), *args[1:]]
    elif name == "memory" and args and args[0].casefold() in {"recent", "retrieve", "summary", "search"}:
        args = [args[0].casefold(), *args[1:]]
    elif name == "timeline":
        name = "memory"
        args = ["summary", *args]
    argument_text = stripped[len(pieces[0]):].strip()
    if name == "at_reply" and argument_text.casefold().startswith("reply"):
        argument_text = argument_text[5:].strip()
    return SlashCommand(name, tuple(args), argument_text, raw)


@dataclass(frozen=True)
class CommandOutcome:
    """Observable result of interpreting one slash command."""

    handled: bool = True
    reply: str = ""
    status: str = "handled"
    reason: str = "slash_command"
    quiet: bool = False


def _flag(value: Any, *, default: bool) -> bool:
    raw = str(value or "").strip().casefold()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "开启", "开"}


class QQCommandState:
    """Small thread-safe state shared by routing, observation, and commands."""

    def __init__(
        self,
        *,
        stream_enabled: bool = False,
        autonomous_reply_enabled: bool = False,
        agent_enabled: bool = True,
        permission_mode: str = "traditional",
        stopped: bool = False,
        reply_mode: str = "auto",
        state_path: str | Path = "",
        owner_ids: tuple[str, ...] = (),
    ) -> None:
        self._lock = threading.RLock()
        # Process-local revocation epoch: toggling off/on must not revive an
        # already running background task. Never persist a grant across boots.
        self._authority_revision = 0
        self._stream_enabled = bool(stream_enabled)
        self._autonomous_reply_enabled = bool(autonomous_reply_enabled)
        self._agent_enabled = bool(agent_enabled)
        normalized_permission_mode = str(permission_mode or "traditional").strip().casefold()
        self._permission_mode = normalized_permission_mode if normalized_permission_mode in AGENT_PERMISSION_MODES else "traditional"
        self._stopped = bool(stopped)
        self._reply_mode = str(reply_mode or "auto").strip().casefold() or "auto"
        self._autonomous_reply_group_overrides: dict[str, bool] = {}
        self._onlooker_disabled: set[str] = set()
        self.state_path = Path(state_path) if str(state_path or "").strip() else None
        self.owner_ids = frozenset(str(item).strip() for item in owner_ids if str(item).strip())
        self._load()

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        state_path: str | Path = "",
    ) -> "QQCommandState":
        env = os.environ if environ is None else environ
        # The private runtime has one canonical owner list.  Keep the older
        # command-specific variable as a public/legacy fallback, but never let
        # it override the configured private master identities.
        raw_owners = str(
            env.get("SHINKU_MEMORY_MASTER_USER_IDS")
            or env.get("SHINKU_QQ_COMMAND_USER_IDS", "")
            or ""
        )
        owners = tuple(item.strip() for item in raw_owners.replace("，", ",").split(",") if item.strip())
        return cls(
            stream_enabled=_flag(env.get("SHINKU_QQ_STREAM_ENABLED"), default=False),
            autonomous_reply_enabled=_flag(
                env.get("SHINKU_QQ_AUTONOMOUS_REPLY_ENABLED"), default=False
            ),
            state_path=state_path or env.get("SHINKU_QQ_COMMAND_STATE_PATH", ""),
            owner_ids=owners,
        )

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(raw, Mapping) or raw.get("version") not in {None, _STATE_VERSION}:
            return
        with self._lock:
            self._stream_enabled = bool(raw.get("stream_enabled", self._stream_enabled))
            self._autonomous_reply_enabled = bool(
                raw.get("autonomous_reply_enabled", self._autonomous_reply_enabled)
            )
            group_overrides = raw.get("autonomous_reply_groups", {})
            if isinstance(group_overrides, Mapping):
                self._autonomous_reply_group_overrides = {
                    str(group_id).strip(): (
                        value if isinstance(value, bool) else _flag(value, default=False)
                    )
                    for group_id, value in group_overrides.items()
                    if str(group_id or "").strip()
                }
            self._agent_enabled = bool(raw.get("agent_enabled", self._agent_enabled))
            permission_mode = str(raw.get("permission_mode", self._permission_mode) or "traditional").casefold()
            self._permission_mode = permission_mode if permission_mode in AGENT_PERMISSION_MODES else "traditional"
            self._stopped = bool(raw.get("stopped", self._stopped))
            mode = str(raw.get("reply_mode", self._reply_mode) or self._reply_mode).casefold()
            self._reply_mode = mode if mode in {"text", "voice", "both", "auto"} else "auto"
            disabled = raw.get("onlooker_disabled", [])
            if isinstance(disabled, list):
                self._onlooker_disabled = {str(item).strip() for item in disabled if str(item).strip()}

    def _save(self) -> bool:
        if self.state_path is None:
            return True
        payload = self.snapshot()
        payload["version"] = _STATE_VERSION
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.state_path)
        except OSError:
            return False
        return True

    def _set(self, attr: str, value: Any) -> bool:
        with self._lock:
            changed = getattr(self, attr) != value
            setattr(self, attr, value)
            if changed and attr in {"_agent_enabled", "_stopped"}:
                self._authority_revision += 1
            return self._save()

    def set_stream_enabled(self, enabled: bool) -> bool:
        return self._set("_stream_enabled", bool(enabled))

    def is_stream_enabled(self) -> bool:
        with self._lock:
            return self._stream_enabled

    def set_autonomous_reply_enabled(self, enabled: bool, conversation_id: str = "") -> bool:
        key = str(conversation_id or "").strip()
        with self._lock:
            if key:
                self._autonomous_reply_group_overrides[key] = bool(enabled)
            else:
                self._autonomous_reply_enabled = bool(enabled)
            return self._save()

    def clear_autonomous_reply_override(self, conversation_id: str) -> bool:
        key = str(conversation_id or "").strip()
        if not key:
            return False
        with self._lock:
            self._autonomous_reply_group_overrides.pop(key, None)
            return self._save()

    def autonomous_reply_override(self, conversation_id: str) -> bool | None:
        key = str(conversation_id or "").strip()
        if not key:
            return None
        with self._lock:
            return self._autonomous_reply_group_overrides.get(key)

    def is_autonomous_reply_enabled(self, conversation_id: str = "") -> bool:
        key = str(conversation_id or "").strip()
        with self._lock:
            if key in self._autonomous_reply_group_overrides:
                return self._autonomous_reply_group_overrides[key]
            return self._autonomous_reply_enabled

    def set_agent_enabled(self, enabled: bool) -> bool:
        return self._set("_agent_enabled", bool(enabled))

    def is_agent_enabled(self) -> bool:
        with self._lock:
            return self._agent_enabled

    def set_permission_mode(self, mode: str) -> bool:
        normalized = str(mode or "").strip().casefold()
        if normalized not in AGENT_PERMISSION_MODES:
            return False
        with self._lock:
            previous = self._permission_mode
            self._permission_mode = normalized
            if self._save():
                if previous != normalized:
                    self._authority_revision += 1
                return True
            # A failed persistence must not leave a high-risk profile active
            # only in memory while reporting that the switch failed.
            self._permission_mode = previous
            return False

    def permission_mode(self) -> str:
        with self._lock:
            return self._permission_mode

    def authority_revision(self) -> int:
        with self._lock:
            return self._authority_revision

    def set_stopped(self, stopped: bool) -> bool:
        return self._set("_stopped", bool(stopped))

    def is_stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def responses_enabled(self) -> bool:
        with self._lock:
            # Agent/tool availability and ordinary chat replies are separate
            # controls. /agent off only disables task/tool handoff; /stop pauses
            # all replies.
            return not self._stopped

    def set_reply_mode(self, mode: str) -> bool:
        normalized = str(mode or "").strip().casefold()
        if normalized not in {"text", "voice", "both", "auto"}:
            return False
        return self._set("_reply_mode", normalized)

    def reply_mode(self) -> str:
        with self._lock:
            return self._reply_mode

    def set_onlooker_disabled(self, conversation_id: str, disabled: bool) -> bool:
        key = str(conversation_id or "").strip()
        if not key:
            return False
        with self._lock:
            if disabled:
                self._onlooker_disabled.add(key)
            else:
                self._onlooker_disabled.discard(key)
            return self._save()

    def is_onlooker_enabled(self, conversation_id: str) -> bool:
        key = str(conversation_id or "").strip()
        with self._lock:
            return key not in self._onlooker_disabled

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stream_enabled": self._stream_enabled,
                "autonomous_reply_enabled": self._autonomous_reply_enabled,
                "autonomous_reply_groups": dict(sorted(self._autonomous_reply_group_overrides.items())),
                "agent_enabled": self._agent_enabled,
                "permission_mode": self._permission_mode,
                "stopped": self._stopped,
                "reply_mode": self._reply_mode,
                "onlooker_disabled": sorted(self._onlooker_disabled),
            }


class QQCommandInterpreter:
    """Interpret all supported slash commands before the Agent loop."""

    def __init__(
        self,
        state: QQCommandState | None = None,
        *,
        memory_inspector: Callable[..., Mapping[str, Any]] | None = None,
        runtime_snapshot: Callable[[], Mapping[str, Any]] | None = None,
        task_controller: Callable[[str, str, str], Mapping[str, Any] | None] | None = None,
        require_owner: bool = False,
    ) -> None:
        self.state = state or QQCommandState()
        self.memory_inspector = memory_inspector
        self.runtime_snapshot = runtime_snapshot
        self.task_controller = task_controller
        self.require_owner = bool(require_owner)

    def _authorized(self, sender_id: str) -> bool:
        sender = str(sender_id or "").strip()
        if self.require_owner:
            return bool(self.state.owner_ids) and sender in self.state.owner_ids
        return not self.state.owner_ids or sender in self.state.owner_ids

    def execute(
        self,
        text: Any,
        *,
        conversation_id: str = "",
        conversation_type: str = "unknown",
        sender_id: str = "",
    ) -> CommandOutcome | None:
        command = parse_slash_command(text)
        if command is None:
            return None
        if not self._authorized(sender_id):
            return CommandOutcome(status="forbidden", reason="slash_command_forbidden", quiet=True)
        name = command.name
        args = [str(item).strip() for item in command.args if str(item).strip()]
        action = args[0].casefold() if args else "status"

        if name in {"help", "h", "?", "unknown"}:
            return CommandOutcome(reply=self.help_text(), status="help")
        if name in {"status", "state"}:
            return CommandOutcome(
                reply=self._status_text(conversation_id, conversation_type),
                status="status",
            )
        if name == "stream":
            return self._switch_flag(
                args,
                getter=self.state.is_stream_enabled,
                setter=self.state.set_stream_enabled,
                label="流式回复",
                on_text="好，流式回复开了。",
                off_text="好，流式回复关了。",
                status_prefix="stream",
                note="当前独立 Agent 仍先生成完整结果，再由回复宿主投递；这个开关先恢复控制面，后续流式通道会读取它。",
            )
        if name == "at_reply":
            return self._at_reply(args, conversation_id, conversation_type)
        if name == "onlooker":
            return self._onlooker(args, conversation_id, conversation_type)
        if name == "stop":
            self.state.set_stopped(True)
            return CommandOutcome(reply="好，我先安静下来。发 /start 才会恢复回复。", status="stopped")
        if name == "start":
            self.state.set_stopped(False)
            self.state.set_agent_enabled(True)
            return CommandOutcome(reply="嗯，我在。恢复回复了。", status="started")
        if name == "agent":
            return self._agent(args, sender_id=sender_id, conversation_id=conversation_id, conversation_type=conversation_type)
        if name == "reply_mode":
            return self._reply_mode(args)
        if name in {"model", "provider", "character", "mface", "表情包配置"}:
            return self._compatibility_command(name, args)
        if name in {"memory", "timeline", "history", "record"}:
            return self._memory(args, conversation_id, conversation_type, sender_id)
        return CommandOutcome(reply=self.help_text(), status="unknown_command", reason="slash_command_unknown")

    def _switch_flag(
        self,
        args: list[str],
        *,
        getter: Callable[[], bool],
        setter: Callable[[bool], bool],
        label: str,
        on_text: str,
        off_text: str,
        status_prefix: str,
        note: str = "",
    ) -> CommandOutcome:
        action = args[0].casefold() if args else "status"
        if action in {"status", "current", "状态", "当前"}:
            enabled = bool(getter())
            reply = f"{label}现在{'开着' if enabled else '关着'}。"
            if note:
                reply += "\n" + note
            return CommandOutcome(reply=reply, status=f"{status_prefix}_{'on' if enabled else 'off'}")
        if action in {"on", "开", "开启", "打开", "恢复"}:
            saved = setter(True)
            reply = on_text if saved else f"{label}已切换，但状态保存失败，重启后可能恢复。"
            if note:
                reply += "\n" + note
            return CommandOutcome(reply=reply, status=f"{status_prefix}_on")
        if action in {"off", "关", "关闭", "停止"}:
            saved = setter(False)
            reply = off_text if saved else f"{label}已切换，但状态保存失败，重启后可能恢复。"
            if note:
                reply += "\n" + note
            return CommandOutcome(reply=reply, status=f"{status_prefix}_off")
        usage = "/at reply" if status_prefix == "attention_reply" else f"/{status_prefix.replace('_', ' ')}"
        return CommandOutcome(reply=f"用法：{usage} on/off。", status="invalid_arguments")

    def _at_reply(
        self,
        args: list[str],
        conversation_id: str,
        conversation_type: str,
    ) -> CommandOutcome:
        usage = (
            "用法：/at reply on|off|status [群号]；"
            "/at reply default [群号] 恢复该群跟随全局默认。"
            "群聊里不写群号只作用于当前群，私聊里不写群号设置全局默认。"
        )
        if len(args) > 2:
            return CommandOutcome(reply=usage, status="invalid_arguments")

        action = str(args[0] if args else "status").strip().casefold()
        in_group = str(conversation_type or "").casefold() in {"group", "guild"}
        explicit_group_id = str(args[1] if len(args) == 2 else "").strip()
        target_group_id = explicit_group_id or (
            str(conversation_id or "").strip() if in_group else ""
        )
        if target_group_id and (len(target_group_id) > 128 or not target_group_id.isprintable()):
            return CommandOutcome(reply=usage, status="invalid_arguments")

        if action in {"default", "inherit", "reset"}:
            if not target_group_id:
                return CommandOutcome(reply=usage, status="invalid_arguments")
            saved = self.state.clear_autonomous_reply_override(target_group_id)
            effective = self.state.is_autonomous_reply_enabled(target_group_id)
            if not saved:
                return CommandOutcome(
                    reply="群聊自主回复设置未能保存。",
                    status="attention_reply_save_failed",
                )
            label = "这个群" if in_group and not explicit_group_id else f"群 {target_group_id}"
            return CommandOutcome(
                reply=f"{label}已恢复跟随全局默认，现在{'开启' if effective else '关闭'}。",
                status="attention_reply_default",
            )

        if action in {"status", "current", "状态", "当前"}:
            if target_group_id:
                override = self.state.autonomous_reply_override(target_group_id)
                enabled = self.state.is_autonomous_reply_enabled(target_group_id)
                label = "这个群" if in_group and not explicit_group_id else f"群 {target_group_id}"
                detail = (
                    "这是该群单独设置的。"
                    if override is not None
                    else f"该群未单独设置，跟随全局默认（{'开' if enabled else '关'}）。"
                )
            else:
                enabled = self.state.is_autonomous_reply_enabled()
                label = "群聊自主回复的全局默认"
                detail = "有单独设置的群不受这个默认值影响。"
            return CommandOutcome(
                reply=f"{label}现在{'开着' if enabled else '关着'}。{detail}",
                status=f"attention_reply_{'on' if enabled else 'off'}",
            )

        if action not in {"on", "开", "开启", "打开", "恢复", "off", "关", "关闭", "停止"}:
            return CommandOutcome(reply=usage, status="invalid_arguments")
        enabled = action in {"on", "开", "开启", "打开", "恢复"}
        saved = self.state.set_autonomous_reply_enabled(enabled, target_group_id)
        label = (
            "这个群"
            if in_group and not explicit_group_id
            else f"群 {target_group_id}"
            if target_group_id
            else "群聊自主回复的全局默认"
        )
        if not saved:
            return CommandOutcome(
                reply=f"{label}已切换，但状态保存失败，重启后可能恢复。",
                status="attention_reply_save_failed",
            )
        if enabled:
            reply = f"好，{label}的自主回复开了。"
        else:
            reply = f"好，{label}的自主回复关了。直接叫我或者 @ 我，我还是会回。"
        return CommandOutcome(
            reply=reply,
            status=f"attention_reply_{'on' if enabled else 'off'}",
        )

    def _onlooker(self, args: list[str], conversation_id: str, conversation_type: str) -> CommandOutcome:
        if str(conversation_type or "").casefold() != "group":
            return CommandOutcome(reply="/onlooker 只在群聊里生效。", status="onlooker_not_group")
        action = args[0].casefold() if args else "status"
        if action in {"status", "current", "状态", "当前"}:
            enabled = self.state.is_onlooker_enabled(conversation_id)
            return CommandOutcome(
                reply="这个群的旁观入库开着。" if enabled else "这个群的旁观入库关着，@我仍然会回。",
                status=f"onlooker_{'on' if enabled else 'off'}",
            )
        if action in {"on", "开", "开启", "打开", "恢复"}:
            self.state.set_onlooker_disabled(conversation_id, False)
            return CommandOutcome(reply="嗯，这个群的旁观入库恢复了。", status="onlooker_on")
        if action in {"off", "关", "关闭", "停止"}:
            self.state.set_onlooker_disabled(conversation_id, True)
            return CommandOutcome(reply="好，这个群我不再旁观入库了；@我还是照常回。", status="onlooker_off")
        return CommandOutcome(reply="用法：/onlooker on/off。", status="invalid_arguments")

    def _task_control(self, operation: str, task_id: str, sender_id: str, session_id: str):
        if self.task_controller is None:
            return {"ok": False, "reason": "task_control_unavailable"}
        # Preserve embedded legacy 3-argument controllers. The live coordinator
        # accepts a typed session, so QQ cannot leak a private task into a group.
        parameters = inspect.signature(self.task_controller).parameters
        scoped = "session_id" in parameters or any(p.kind == p.VAR_KEYWORD for p in parameters.values())
        return self.task_controller(operation, task_id, sender_id, **({"session_id": session_id} if scoped else {}))

    def _agent(self, args: list[str], *, sender_id: str = "", conversation_id: str = "", conversation_type: str = "unknown") -> CommandOutcome:
        action = args[0].casefold() if args else "status"
        session_id = f"{str(conversation_type or '').casefold()}:{str(conversation_id or '').strip()}"
        if action in {"tasks", "list", "任务", "列表"} or (action in {"status", "状态"} and len(args) > 1):
            operation = "list" if action in {"tasks", "list", "任务", "列表"} else "status"
            task_id = args[1] if operation == "status" else ""
            try:
                result = self._task_control(operation, task_id, str(sender_id or "").strip(), session_id)
            except Exception:
                return CommandOutcome(reply="后台任务查询暂时不可用，没有根据猜测补写状态。", status="agent_task_query_failed")
            if not isinstance(result, Mapping) or not result.get("ok"):
                return CommandOutcome(reply="当前请求者与会话范围内没有找到该后台任务，或查询入口不可用。", status="agent_task_query_rejected")
            if operation == "list":
                tasks = result.get("tasks") or []
                lines = [f"{row['task_id']}：{row['status']}" + ("（取消请求已记录）" if row.get("cancel_requested") and row.get("worker_active") else "") for row in tasks[:10]]
                return CommandOutcome(reply="当前会话的后台任务：\n" + "\n".join(lines) if lines else "当前会话没有后台任务记录。", status="agent_task_list")
            task = result.get("task") or {}
            metadata = task.get("metadata") or {}
            notice = "；取消请求已记录，执行器尚未退出" if task.get("cancel_requested") and task.get("worker_active") else ""
            details = f"；规划轮数 {metadata.get('rounds', 0)}，工具记录 {metadata.get('tool_calls', 0)}"
            budget = metadata.get("budget")
            if isinstance(budget, Mapping):
                details += (f"；请求 {budget.get('requests', 0)}/{budget.get('max_requests', '?')}，"
                            f"Token记账 {budget.get('tokens', 0)}/{budget.get('max_tokens', '?')}，"
                            f"活动秒 {float(budget.get('active_seconds', 0)):.1f}/{budget.get('max_active_seconds', '?')}")
            delivery = f"；交付状态：{metadata.get('delivery_status', 'not_sent')}"
            summary = str(task.get("result_summary") or task.get("error") or "")[:1000]
            return CommandOutcome(reply=f"任务 {task_id} 当前状态：{task.get('status', 'unknown')}{notice}{details}{delivery}。" + (f"\n{summary}" if summary else ""), status="agent_task_status")
        if action in {"status", "状态", "current", "当前"}:
            snap = self.state.snapshot()
            permission_mode = str(snap.get("permission_mode") or "traditional")
            return CommandOutcome(
                reply=(
                    f"工具任务处理当前{'开启' if snap['agent_enabled'] else '关闭'}；"
                    f"权限档位：{AGENT_PERMISSION_LABELS.get(permission_mode, permission_mode)}；"
                    f"回复{'暂停' if snap['stopped'] else '正常'}。"
                ),
                status="agent_status",
            )
        if action in {"permission", "权限", "mode", "模式"}:
            requested = args[1].casefold() if len(args) > 1 else "status"
            if requested in {"status", "current", "当前", "状态"}:
                mode = self.state.permission_mode()
                return CommandOutcome(
                    reply=f"当前 Agent 权限档位：{AGENT_PERMISSION_LABELS[mode]}。",
                    status=f"agent_permission_{mode}",
                )
            aliases = {
                "traditional": "traditional", "legacy": "traditional", "传统": "traditional", "传统模式": "traditional",
                "readonly": "readonly", "read_only": "readonly", "只读": "readonly", "只读模式": "readonly",
                "confirm": "confirm", "ask": "confirm", "询问": "confirm", "确认": "confirm", "调用确认": "confirm",
                "full": "full", "完全访问": "full", "完全": "full",
                "ultra": "ultra", "极限": "ultra",
            }
            mode = aliases.get(requested)
            if mode is None:
                return CommandOutcome(
                    reply="用法：/agent permission traditional|readonly|confirm|full|ultra。",
                    status="invalid_agent_permission_mode",
                )
            if not self.state.set_permission_mode(mode):
                return CommandOutcome(reply="权限档位没有切换成功。", status="agent_permission_change_failed")
            warning = "高权限档位已开启；可用 /agent permission traditional 随时收回。" if mode in {"full", "ultra"} else ""
            return CommandOutcome(
                reply=(
                    f"Agent 权限档位已切换为：{AGENT_PERMISSION_LABELS[mode]}。"
                    f"任务工具当前{'开启' if self.state.is_agent_enabled() else '关闭'}。{warning}"
                ),
                status=f"agent_permission_{mode}",
            )
        if action in {"on", "开", "开启", "打开"}:
            self.state.set_agent_enabled(True)
            self.state.set_stopped(False)
            return CommandOutcome(reply="好，工具任务处理和 Agent 执行开了；日常聊天一直都能正常回复。", status="agent_on")
        if action in {"off", "关", "关闭", "停止", "退出"}:
            self.state.set_agent_enabled(False)
            return CommandOutcome(reply="好，工具任务处理关了；日常聊天仍会正常回复。", status="agent_off")
        if action in {"cancel", "取消", "retry", "重试"}:
            task_id = args[1] if len(args) > 1 else ""
            if not task_id:
                return CommandOutcome(
                    reply=f"命令已识别，但还缺任务号：/agent {action} task-id。",
                    status="agent_task_id_missing",
                )
            if self.task_controller is None:
                return CommandOutcome(
                    reply="命令已识别；当前宿主还没有接入后台任务协调器，不能直接取消或重试这个任务。",
                    status="agent_task_control_unavailable",
                )
            operation = "cancel" if action in {"cancel", "取消"} else "retry"
            try:
                result = self._task_control(operation, task_id, str(sender_id or "").strip(), session_id)
            except Exception as exc:
                return CommandOutcome(
                    reply=f"任务{operation}失败，协调器报错：{type(exc).__name__}。",
                    status="agent_task_control_failed",
                )
            if not isinstance(result, Mapping):
                return CommandOutcome(reply=f"任务{operation}没有返回可读结果。", status="agent_task_control_invalid")
            if not bool(result.get("ok")):
                reason = str(result.get("reason") or "task_not_available").strip()
                if reason in {"fresh_continuation_required", "retry_handler_missing"}:
                    return CommandOutcome(reply="不能直接重放旧工具链。请重新说明要续接的原任务；会根据已有证据和当前权限重新规划，未知的副作用不会自动重试。", status="agent_task_control_rejected")
                return CommandOutcome(reply=f"任务{operation}没有完成：{reason}。", status="agent_task_control_rejected")
            task = result.get("task") if isinstance(result.get("task"), Mapping) else {}
            status = str(task.get("status") or result.get("status") or "updated").strip()
            verb = "记录取消请求，等待执行器到达检查点" if operation == "cancel" and task.get("worker_active") else "取消" if operation == "cancel" else "排队重试"
            return CommandOutcome(
                reply=f"任务 {task_id} 已{verb}，当前状态：{status}。",
                status=f"agent_task_{operation}",
            )
        return CommandOutcome(
            reply="用法：/agent on/off/status；/agent tasks；/agent status task-id；/agent permission traditional|readonly|confirm|full|ultra；或 /agent cancel|retry task-id。",
            status="invalid_arguments",
        )

    def _reply_mode(self, args: list[str]) -> CommandOutcome:
        action = args[0].casefold() if args else "status"
        aliases = {
            "文字": "text", "文本": "text", "text": "text",
            "语音": "voice", "voice": "voice",
            "双发": "both", "文字语音": "both", "both": "both",
            "自动": "auto", "auto": "auto",
        }
        if action in {"status", "current", "当前", "状态"}:
            return CommandOutcome(reply=f"当前回复模式：{self.state.reply_mode()}。", status="reply_mode_current")
        mode = aliases.get(action)
        if mode is None and len(args) > 1 and action in {"set", "设置", "切换"}:
            mode = aliases.get(args[1].casefold())
        if mode is None or not self.state.set_reply_mode(mode):
            return CommandOutcome(reply="回复模式只能是 text、voice、both 或 auto。", status="invalid_reply_mode")
        if mode != "text":
            return CommandOutcome(
                reply=f"已切换回复模式：{mode}。本机语音服务未配置或不可用时仍会发文字。",
                status="reply_mode_changed",
            )
        return CommandOutcome(reply="已切到文字回复模式。", status="reply_mode_changed")

    def _compatibility_command(self, name: str, args: list[str]) -> CommandOutcome:
        if name == "model":
            label = "模型"
            if not args or args[0] in {"status", "current"}:
                snapshot = self.runtime_snapshot() if self.runtime_snapshot is not None else {}
                model = str(snapshot.get("model") or "当前启动时配置的模型") if isinstance(snapshot, Mapping) else "当前启动时配置的模型"
                return CommandOutcome(reply=f"当前{label}：{model}。", status="model_current")
            if args[0] == "list":
                return CommandOutcome(reply="模型列表请在控制中枢刷新；QQ 指令入口已恢复，但不会伪造供应方返回的数据。", status="model_list")
            model = args[-1]
            if not _MODEL_ID_RE.fullmatch(model):
                return CommandOutcome(reply="模型名格式不合法。", status="invalid_model")
            return CommandOutcome(reply=f"已识别切换模型请求：{model}。当前运行实例需要重启或通过控制中枢激活它。", status="model_switch_requested")
        if name == "provider":
            if not args or args[0] in {"status", "current"}:
                snapshot = self.runtime_snapshot() if self.runtime_snapshot is not None else {}
                provider = str(snapshot.get("provider") or "当前启动时配置的供应方") if isinstance(snapshot, Mapping) else "当前启动时配置的供应方"
                return CommandOutcome(reply=f"当前供应方：{provider}。", status="provider_current")
            if args[0] == "list":
                return CommandOutcome(reply="供应方列表请在控制中枢查看；QQ 指令入口已恢复。", status="provider_list")
            provider = args[-1].casefold()
            if not _PROVIDER_ID_RE.fullmatch(provider):
                return CommandOutcome(reply="供应方标识格式不合法。", status="invalid_provider")
            return CommandOutcome(reply=f"已识别切换供应方请求：{provider}。请在控制中枢激活后重启 QQ Agent。", status="provider_switch_requested")
        if name == "character":
            if not args or args[0] in {"status", "current"}:
                return CommandOutcome(reply="当前 QQ 使用启动时固定注入的 Shinku 主人设。", status="character_current")
            return CommandOutcome(reply="已识别角色切换请求；独立版当前固定使用 Shinku 主人设，未接入角色包切换。", status="character_switch_unavailable")
        return CommandOutcome(reply="已识别表情配置指令；独立版当前没有 QQ mface 配置宿主。", status="mface_unavailable")

    def _memory(self, args: list[str], conversation_id: str, conversation_type: str, sender_id: str = "") -> CommandOutcome:
        if self.memory_inspector is None:
            return CommandOutcome(reply="记忆查看入口已识别，但当前没有接入记忆服务。", status="memory_unavailable")
        mode = args[0] if args and args[0] in {"recent", "retrieve", "summary", "search"} else ("retrieve" if args else "recent")
        if mode == "search":
            mode = "retrieve"
        query = " ".join(args[1:] if args and args[0] in {"recent", "retrieve", "summary", "search"} else args).strip()
        try:
            result = self.memory_inspector(
                conversation_id=conversation_id,
                conversation_type=conversation_type,
                requester_id=sender_id,
                query=query,
                mode=mode,
                limit=8,
            )
        except Exception:
            return CommandOutcome(reply="查看记忆时遇到内部错误，我没有根据猜测补写历史。", status="memory_inspection_failed")
        if not isinstance(result, Mapping):
            return CommandOutcome(reply="记忆服务没有返回可读结果。", status="memory_inspection_invalid")
        rendered = str(result.get("followup_context") or result.get("rendered") or "").strip()
        return CommandOutcome(reply=rendered or "这段范围里没有找到可用记录。", status="memory_inspected")

    def _status_text(self, conversation_id: str, conversation_type: str = "") -> str:
        snap = self.state.snapshot()
        onlooker = self.state.is_onlooker_enabled(conversation_id)
        if str(conversation_type or "").casefold() in {"group", "guild"}:
            enabled = self.state.is_autonomous_reply_enabled(conversation_id)
            override = self.state.autonomous_reply_override(conversation_id)
            auto_status = f"群聊自主回复当前群={'开' if enabled else '关'}"
            if override is None:
                auto_status += "（跟随全局默认）"
        else:
            auto_status = f"群聊自主回复全局默认={'开' if snap['autonomous_reply_enabled'] else '关'}"
        return (
            f"状态：工具任务={'开' if snap['agent_enabled'] else '关'}，"
            f"权限={AGENT_PERMISSION_LABELS.get(str(snap.get('permission_mode') or 'traditional'), '传统')}，"
            f"回复={'暂停' if snap['stopped'] else '正常'}，"
            f"流式={'开' if snap['stream_enabled'] else '关'}，"
            f"{auto_status}，"
            f"当前会话旁观入库={'开' if onlooker else '关'}，"
            f"回复模式={snap['reply_mode']}。"
        )

    @staticmethod
    def help_text() -> str:
        return (
            "可用指令：\n"
            "/help、/status\n"
            "/stream on/off/status\n"
            "/at reply on|off|status [群号]；/at reply default [群号]\n"
            "/onlooker on/off/status\n"
            "/stop、/start\n"
            "/agent on/off/status、/agent tasks、/agent status task-id、/agent permission traditional|readonly|confirm|full|ultra、/agent cancel|retry task-id\n"
            "/reply mode text|voice|both|auto\n"
            "/model、/provider、/character\n"
            "/memory recent|retrieve|summary [内容]（/timeline 为摘要别名）\n"
            "/mface config（保留兼容入口）"
        )
