"""独立终端宿主：显式私用配置 -> 固定发言身份 -> Kernel -> 终端送达回执。

运行 python -m aozora.terminal --config PATH check/chat。check 只读输入文件，
不打开状态库、不连接模型；chat 每行最多一次请求，状态仍由 Store 独占管理。
此入口信任本机操作者，不提供多用户鉴权，也不能作为网络消息的身份认证层。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import stat
import sys
import time
import uuid

from .__main__ import emit, read_json
from .contracts import Event, Identity, KernelError, digest, fields, identifier, text
from .decider import HTTPDecider
from .kernel import Kernel
from .store import Store


@dataclass(frozen=True)
class TerminalSession:
    """仅持有本次启动的配置快照；不追踪真红配置、账号目录或环境变量的变化。"""

    identity: Identity
    database: Path
    actor: str
    scope: str
    decider: HTTPDecider

    def check(self) -> dict:
        # 不调用 Store 构造器，避免「检查」意外绑定新库。已有库的身份在 chat 打开时校验。
        return {"status": "configuration_valid", "name": self.identity.name,
                "identity_hash": digest(self.identity.payload()), "actor": self.actor, "scope": self.scope,
                "database": str(self.database), "database_exists": self.database.exists(),
                "database_binding_checked": False,
                "endpoint": self.decider.endpoint, "model": self.decider.model,
                "credential_configured": bool(self.decider.api_key), "timeout": self.decider.timeout,
                "max_tokens": self.decider.max_tokens, "model_called": False, "database_opened": False}


def load_session(config_path: str | Path) -> TerminalSession:
    """严格加载显式 JSON；相对路径以配置目录为准，先验证凭据/关系再允许初始化数据库。"""
    config_path = Path(config_path).expanduser().resolve()
    value = fields(read_json(str(config_path)),
                   {"version", "identity", "database", "actor", "scope", "model"})
    if type(value["version"]) is not int or value["version"] != 1:
        raise KernelError("unsupported_terminal_config")

    def path(raw: str) -> Path:
        candidate = Path(text(raw, 4096)).expanduser()
        return (config_path.parent / candidate).resolve()

    identity_path, database = path(value["identity"]), path(value["database"])
    identity = Identity.parse(read_json(str(identity_path)))
    actor, scope = identifier(value["actor"]), identifier(value["scope"])
    if actor not in dict(identity.relationships):
        raise KernelError("terminal_actor_unbound")
    model = fields(value["model"], {"endpoint", "name", "api_key_file", "timeout", "max_tokens"})
    protected = [config_path, identity_path]
    api_key = ""
    if model["api_key_file"] is not None:
        key_path = path(model["api_key_file"])
        protected.append(key_path)
        with key_path.open("rb") as stream:
            # 检查已打开文件本身，避免先 stat 再 open 的权限检查竞态。
            import os
            mode = os.fstat(stream.fileno()).st_mode
            if not stat.S_ISREG(mode) or (sys.platform != "win32" and mode & 0o077):
                raise KernelError("credential_file_permissions")
            raw = stream.read(4097)
        if len(raw) > 4096:
            raise KernelError("credential_file_too_large")
        api_key = raw.decode("utf-8").rstrip("\r\n")
        if not api_key:
            raise KernelError("credential_file_empty")
    if any(database == item or (database.exists() and database.samefile(item)) for item in protected):
        raise KernelError("database_overlaps_input")
    # HTTPDecider 构造只校验参数并建立 opener，不发请求、不探测服务、也不切换提供商。
    decider = HTTPDecider(text(model["endpoint"], 4096), model["name"], api_key=api_key,
                          timeout=model["timeout"], max_tokens=model["max_tokens"])
    return TerminalSession(identity, database, actor, scope, decider)


def chat(session: TerminalSession, stdin, stdout, stderr) -> int:
    """每行只产生 message；正文无法声明 actor、scope、观察来源或送达结果。"""
    store = Store(session.database, session.identity)
    kernel = Kernel(store, session.decider)
    interactive = stdin.isatty()
    print(f"{session.identity.name}的本地终端已就绪。每行输入调用一次模型；Ctrl-D 退出。", file=stderr, flush=True)
    while True:
        if interactive:
            print("你：", end="", file=stderr, flush=True)
        line = stdin.readline()
        if not line:
            return 0
        if not line.strip():
            continue
        event = Event(uuid.uuid4().hex, session.scope, session.actor, "message", line.strip(), time.time())
        result = kernel.handle(event)
        if result["status"] != "completed":
            # 保存并输出失败事件 ID，便于核对/显式恢复；不自动重试或伪造角色兜底回复。
            print(json.dumps(result, ensure_ascii=False), file=stderr, flush=True)
            return 1
        if result["delivery"]:
            if interactive:
                print(f"{session.identity.name}：", end="", file=stderr, flush=True)
            emit(store, result["delivery"]["delivery_id"], session.scope, stdout)
        elif interactive:
            print("（本轮没有生成回复。）", file=stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="显式终端配置 JSON")
    parser.add_argument("command", choices=("check", "chat"))
    args = parser.parse_args(argv)
    try:
        session = load_session(args.config)
        if args.command == "check":
            print(json.dumps(session.check(), ensure_ascii=False, indent=2))
            return 0
        return chat(session, sys.stdin, sys.stdout, sys.stderr)
    except KernelError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print('{"error":"terminal_input_or_io_failed"}', file=sys.stderr)
        return 1
    except sqlite3.Error:
        print('{"error":"local_database_failed"}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
