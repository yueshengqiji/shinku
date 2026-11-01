"""独立角色进程的显式启动边界。

check 只读取声明，不导入扩展、不创建库、不调用模型。terminal/serve 先锁定
独占状态根，再启动清洁子进程；子进程继续使用公共 QQ/记忆/Agent 装配。
这隔离配置和状态，不是针对恶意本机代码的操作系统沙箱。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit

from .character import CharacterProfile, load_character


def _json(path: Path):
    def unique(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("instance_duplicate_field")
            result[key] = value
        return result
    raw = path.read_bytes()
    if len(raw) > 65536:
        raise ValueError("instance_file_too_large")
    return json.loads(raw, object_pairs_hook=unique)


_MODEL_KEYS = {f"SHINKU_{channel}_{suffix}" for channel in ("CHAT", "AUX")
               for suffix in ("API_KEY", "BASE_URL", "API_PROTOCOL", "MODEL_NAME")}
_OPTION_KEYS = {
    "SHINKU_LLM_CHAT_TIMEOUT_SECONDS", "SHINKU_LLM_AUX_TIMEOUT_SECONDS",
    "SHINKU_LLM_DEFAULT_MAX_TOKENS", "SHINKU_NAPCAT_BASE_URL",
    "SHINKU_NAPCAT_ACCESS_TOKEN", "SHINKU_NAPCAT_EVENT_TOKEN", "SHINKU_SERVICE_TOKEN",
    "SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST",
    "SHINKU_QQ_AGENT_PEER_IDS",
    "SHINKU_SUPERVISOR_CONFIG",
}


@dataclass(frozen=True)
class InstanceSpec:
    manifest: Path
    character: CharacterProfile
    character_file: Path
    persona_file: Path
    state_root: Path
    owner_id: str
    bot_id: str
    channel: str
    backend_port: int
    extension_module: str = ""
    module_paths: tuple[Path, ...] = ()
    settings: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def binding(self):
        # 改口吻不丢经历；改角色、账号或渠道不能接管既有状态根。
        return dict(version=1, character_id=self.character.character_id,
                    owner_id=self.owner_id, bot_id=self.bot_id, channel=self.channel)

    def validate_root(self, *, allow_lock=False):
        root = self.state_root
        if root.is_symlink() or any(p.is_symlink() for p in root.parents):
            raise ValueError("instance_root_symlink")
        if any((p / ".instance.json").exists() for p in root.parents):
            raise ValueError("instance_nested_root")
        if not root.exists():
            return
        if not root.is_dir():
            raise ValueError("instance_root_not_directory")
        marker = root / ".instance.json"
        if marker.exists():
            if marker.is_symlink() or _json(marker) != self.binding:
                raise ValueError("instance_root_owner_mismatch")
        elif any(p.name != ".instance.lock" or not allow_lock for p in root.iterdir()):
            raise ValueError("instance_root_unbound_nonempty")
        # 状态中的链接和硬链接会绕过根目录归属，启动前拒绝，而不是跟随读取。
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in [*dirs, *files]:
                p = Path(directory) / name
                if p.is_symlink() or p.is_file() and p.stat().st_nlink > 1:
                    raise ValueError("instance_state_link")

    def environment(self, parent=None):
        """只继承操作系统启动必需项；不继承角色、模型、插件或 Python 注入。"""
        parent = os.environ if parent is None else parent
        keys = {"PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "TMPDIR", "SYSTEMROOT",
                "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
                "http_proxy", "https_proxy", "all_proxy", "no_proxy"}
        env = {key: value for key, value in parent.items() if key in keys}
        root = self.state_root
        env.update(self.settings)
        fixed = {
            "ENV_FILE": os.devnull, "CHARACTER_FILE": self.character_file,
            "PERSONA_FILE": self.persona_file, "DATA_ROOT": root / "data",
            "CONFIG_ROOT": root / "config", "LOG_ROOT": root / "logs",
            "MODEL_SERVICES_PATH": root / "model_services.json",
            "MEMORY_MASTER_USER_IDS": self.owner_id, "QQ_BOT_IDS": self.bot_id,
            "QQ_BOT_NAMES": ",".join(self.character.aliases),
            "QQ_AGENT_ENABLED": "true", "QQ_AGENT_SEND_ENABLED": "false",
            "QQ_AGENT_ASYNC_ENABLED": "false", "MEMORY_ENABLED": "true",
            "LLM_HTTP_ATTEMPTS": "1", "LLM_STREAM_ATTEMPTS": "1",
            "ENABLE_NATIVE_TOOL_DECISION": "true",
            "MEMORY_SEMANTIC_ENABLED": "false", "RUNTIME_EXTENSION_MODULE": self.extension_module,
            "MEMORY_DB_PATH": root / "memory.sqlite3",
            "QQ_AGENT_TASKS_DB_PATH": root / "background_tasks.sqlite3",
            "QQ_AGENT_STATE_DB_PATH": root / "agent_tasks.sqlite3",
            "QQ_CONVERSATION_STATE_DB_PATH": root / "conversations.sqlite3",
            "QQ_TRACE_STORE_PATH": root / "traces.json",
            "QQ_COMMAND_STATE_PATH": root / "commands.json",
            "QQ_AGENT_APPROVAL_STORE_PATH": root / "approvals.json",
            "CHAT_METRICS_PATH": root / "chat_metrics.json",
            "QQ_AGENT_PROMPT_CACHE_KEY": "character:" + self.character.character_id,
            "BACKEND_HOST": "127.0.0.1", "BACKEND_PORT": self.backend_port,
            "NAPCAT_WEBHOOK_ENABLED": "true" if self.channel == "qq" else "false",
            "PRIVATE_DATA_ROOT": root / "tools", "PRIVATE_PROFILE_ROOT": self.persona_file.parent,
            "PRIVATE_READ_ROOTS": root / "workspace", "PRIVATE_WRITE_ROOTS": root / "workspace",
            "PRIVATE_DELETE_ROOTS": root / "workspace", "PRIVATE_SHELL_ENABLED": "false",
            "PRIVATE_BROWSER_ENABLED": "false", "PRIVATE_AUDIT_PATH": root / "tools" / "audit.jsonl",
            "PRIVATE_BROWSER_PROFILE": root / "tools" / "browser",
            "PRIVATE_BROWSER_WATCH_PROFILE": root / "tools" / "watch-browser",
            "PRIVATE_REMINDERS_DB": root / "tools" / "reminders.sqlite3",
            "PRIVATE_TWEET_WATCHES_DB": root / "tools" / "watches.sqlite3",
        }
        env.update({"SHINKU_" + key: str(value) for key, value in fixed.items()})
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in (Path(__file__).resolve().parents[1], *self.module_paths))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env


def load_instance(path: str | Path) -> InstanceSpec:
    """严格声明，没有默认角色、默认账号或默认状态路径。"""
    try:
        path = Path(path).expanduser().resolve(strict=True)
        data = _json(path)
        required = {"version", "character_file", "persona_file", "state_root", "settings_file",
                    "owner_id", "bot_id", "channel", "backend_port"}
        if (not isinstance(data, dict) or set(data) - required - {"extension_module", "module_paths"}
                or required - set(data) or type(data["version"]) is not int or data["version"] != 1):
            raise ValueError("instance_fields_invalid")
        def resolve(key, *, exists=True):
            value = data[key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError("instance_path_invalid")
            # 先保存绝对未折叠路径，根目录验证才能发现祖先中的符号链接。
            value = Path(os.path.abspath(path.parent / Path(value).expanduser()))
            if exists:
                return value.resolve(strict=True)
            return value
        character_file, persona_file = resolve("character_file"), resolve("persona_file")
        character = load_character(character_file)
        root, settings_file = resolve("state_root", exists=False), resolve("settings_file")
        if not persona_file.is_file() or not persona_file.stat().st_size:
            raise ValueError("instance_persona_missing")
        if os.name == "posix" and settings_file.stat().st_mode & 0o077:
            raise ValueError("instance_settings_permissions")
        settings = _json(settings_file)
        if (not isinstance(settings, dict) or set(settings) - _MODEL_KEYS - _OPTION_KEYS
                or any(not isinstance(v, str) or not v.strip() or "\0" in v for v in settings.values())):
            raise ValueError("instance_settings_invalid")
        for channel in ("CHAT", "AUX"):
            if any(f"SHINKU_{channel}_{key}" not in settings for key in ("BASE_URL", "MODEL_NAME", "API_PROTOCOL")):
                raise ValueError("instance_model_missing")
            protocol = settings[f"SHINKU_{channel}_API_PROTOCOL"]
            if protocol not in {"openai", "anthropic", "ollama"}:
                raise ValueError("instance_protocol_invalid")
            if protocol != "ollama" and f"SHINKU_{channel}_API_KEY" not in settings:
                raise ValueError("instance_credentials_missing")
        for key, value in settings.items():
            if key.endswith("_BASE_URL"):
                parsed = urlsplit(value)
                if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                    raise ValueError("instance_endpoint_invalid")
        # 复用现有 native-tools 执行器，仅接纳宿主显式声明的精确端点/模型。
        # 声明不等于供应商实测，不能借通配符或继承环境放开未知模型。
        expected_tools = urlsplit(settings["SHINKU_CHAT_BASE_URL"]).netloc.lower() + ":" + settings["SHINKU_CHAT_MODEL_NAME"].lower()
        if settings["SHINKU_CHAT_API_PROTOCOL"] != "openai":
            raise ValueError("instance_execution_protocol_unsupported")
        if ("*" in expected_tools or "," in expected_tools
                or settings.get("SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST", "").lower() != expected_tools):
            raise ValueError("instance_native_tool_declaration_required")
        channel, owner, bot = data["channel"], data["owner_id"], data["bot_id"]
        if channel not in {"terminal", "qq"}:
            raise ValueError("instance_channel_invalid")
        pattern = r"local:[a-zA-Z0-9_-]{1,64}" if channel == "terminal" else r"[1-9][0-9]{4,19}"
        if any(not isinstance(x, str) or not re.fullmatch(pattern, x) for x in (owner, bot)) or owner == bot:
            raise ValueError("instance_accounts_invalid")
        port = data["backend_port"]
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError("instance_port_invalid")
        if channel == "qq" and any(key not in settings for key in (
                "SHINKU_NAPCAT_BASE_URL", "SHINKU_NAPCAT_ACCESS_TOKEN", "SHINKU_NAPCAT_EVENT_TOKEN", "SHINKU_SERVICE_TOKEN")):
            raise ValueError("instance_qq_binding_missing")
        module = data.get("extension_module", "")
        if not isinstance(module, str) or module and not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
            raise ValueError("instance_extension_invalid")
        paths = data.get("module_paths", [])
        if not isinstance(paths, list) or len(paths) > 4 or any(not isinstance(p, str) or not p for p in paths):
            raise ValueError("instance_module_paths_invalid")
        paths = tuple((path.parent / p).resolve(strict=True) for p in paths)
        if any(not p.is_dir() for p in paths):
            raise ValueError("instance_module_paths_invalid")
        # 输入、代码与可写状态分开；根目录不能吞掉配置，更不能指向整个项目。
        if any(p == root or root in p.parents for p in (path, character_file, persona_file, settings_file, Path(__file__).resolve(), *paths)):
            raise ValueError("instance_input_inside_state")
        spec = InstanceSpec(path, character, character_file, persona_file, root, owner, bot,
                            channel, port, module, paths, settings)
        spec.validate_root()
        return spec
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        code = str(exc) if type(exc) is ValueError and re.fullmatch(r"instance_[a-z_]+", str(exc)) else "instance_manifest_invalid"
        raise ValueError(code) from None


@contextmanager
def claim_instance(spec: InstanceSpec):
    """启动器持有进程锁直到子进程退出；第二个进程不能同时装配同一状态根。"""
    import fcntl
    spec.validate_root()
    spec.state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (spec.state_root / ".instance.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("instance_already_running") from None
        spec.validate_root(allow_lock=True)
        marker = spec.state_root / ".instance.json"
        if not marker.exists():
            with marker.open("x", encoding="utf-8") as stream:
                json.dump(spec.binding, stream, ensure_ascii=False)
        (spec.state_root / "workspace").mkdir(exist_ok=True)
        # 子进程继承描述符，启动器异常退出也不能提前释放实例锁。
        yield lock.fileno()


def main(argv=None):
    parser = argparse.ArgumentParser(description="显式角色实例：check 不启动任何运行组件")
    parser.add_argument("action", choices=("check", "terminal", "serve"))
    parser.add_argument("manifest")
    parser.add_argument("--send", action="store_true", help="仅 QQ serve 显式启用发送")
    args = parser.parse_args(argv)
    try:
        spec = load_instance(args.manifest)
        if args.action == "check":
            print(json.dumps(dict(valid=True, character=spec.character.character_id, channel=spec.channel,
                                  state_root=str(spec.state_root), extension=spec.extension_module or None,
                                  native_tools="host_declared_not_probed"), ensure_ascii=False))
            return 0
        if (args.action == "terminal") != (spec.channel == "terminal") or args.send and args.action != "serve":
            raise ValueError("instance_channel_mismatch")
        with claim_instance(spec) as lock_fd:
            env = spec.environment()
            if args.send:
                env["SHINKU_QQ_AGENT_SEND_ENABLED"] = "true"
            return subprocess.call([sys.executable, "-B", "-m", "shinku.instance_runtime", args.action],
                                   env=env, cwd=spec.state_root, pass_fds=(lock_fd,))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
