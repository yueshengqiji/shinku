"""独立本地入口：python -m aozora；默认不读取任何身份、模型、凭据或数据库。

CLI 是受信本机宿主接口，不是可直接暴露给 QQ/网络用户的鉴权层。身份文件、scope、
actor 和 observation 的来源由操作者负责。后续通道适配必须在构造 Event 前校验它们。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

from .contracts import Event, Identity, KernelError, digest
from .decider import HTTPDecider
from .demo import run_demo
from .kernel import Kernel
from .store import Store


def read_json(path: str) -> dict:
    with Path(path).open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise KernelError("input_file_too_large")
    return json.loads(raw)


def emit(store: Store, delivery_id: str, scope: str, stream) -> None:
    """先持久占用，再输出/flush，最后记终端回执。中断窗口保留 unknown，禁止重发。"""
    row = store.claim_delivery(delivery_id, scope, "local-terminal")
    stream.write(row["text"] + "\n")
    stream.flush()
    store.resolve_delivery(delivery_id, scope, delivered=True,
                           receipt="terminal-flush:" + digest({"id": delivery_id, "text": row["text"]}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="青空独立事件内核（本地宿主入口）")
    subs = parser.add_subparsers(dest="command", required=True)
    demo = subs.add_parser("demo", help="运行固定合成场景，不调用模型")
    demo.add_argument("--directory", type=Path, required=True)
    for command in ("inspect", "status", "turn", "chat", "cancel", "deliver"):
        sub = subs.add_parser(command)
        sub.add_argument("--identity", required=True, help="用户提供的身份 JSON；没有默认人设")
        sub.add_argument("--db", required=True, help="此个体独占的 SQLite 文件")
        if command != "turn":
            sub.add_argument("--scope", required=True)
        if command in {"turn", "chat"}:
            sub.add_argument("--endpoint", required=True, help="完整 Chat Completions URL，显式配置后才联网")
            sub.add_argument("--model", required=True)
            sub.add_argument("--api-key-env", help="仅读取显式指定的环境变量，不接受命令行明文密钥")
            sub.add_argument("--timeout", type=float, default=30)
            sub.add_argument("--max-tokens", type=int, default=2048)
        if command == "turn":
            sub.add_argument("--event", required=True, help="受信事件 JSON")
            sub.add_argument("--retry", action="store_true", help="显式重试 failed/cancelled 事件；可能再次计费")
        if command == "chat":
            sub.add_argument("--actor", required=True)
        if command in {"cancel", "status"}:
            sub.add_argument("--event-id", required=True)
        if command == "deliver":
            sub.add_argument("--delivery-id", required=True, help="仅输出到本地终端，不发送 QQ")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            result = run_demo(args.directory)
        else:
            identity = Identity.parse(read_json(args.identity))
            decider = None
            if args.command in {"turn", "chat"}:
                api_key = ""
                if args.api_key_env:
                    api_key = os.environ.get(args.api_key_env, "")
                    if not api_key:
                        raise KernelError("api_key_environment_missing")
                decider = HTTPDecider(args.endpoint, args.model, api_key=api_key,
                                      timeout=args.timeout, max_tokens=args.max_tokens)
            store = Store(args.db, identity)
            if args.command == "inspect":
                result = store.snapshot(args.scope)
            elif args.command == "status":
                result = store.status(args.event_id, args.scope)
            elif args.command == "cancel":
                result = store.cancel(args.event_id, args.scope)
            elif args.command == "deliver":
                emit(store, args.delivery_id, args.scope, sys.stdout)
                return 0
            elif args.command == "turn":
                result = Kernel(store, decider).handle(Event.parse(read_json(args.event)), retry=args.retry)
            else:
                kernel = Kernel(store, decider)
                print("本地聊天已就绪。每次输入会调用所配置模型；Ctrl-D 结束。", file=sys.stderr)
                for line in sys.stdin:
                    if not line.strip():
                        continue
                    result = kernel.handle(Event(uuid.uuid4().hex, args.scope, args.actor, "message", line.strip(), time.time()))
                    if result["status"] != "completed":
                        print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
                        return 1
                    if result["delivery"]:
                        emit(store, result["delivery"]["delivery_id"], args.scope, sys.stdout)
                return 0
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("status") in {"failed", "cancelled", "running"} else 0
    except KernelError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print('{"error":"local_input_or_io_failed"}', file=sys.stderr)
        return 1
    except sqlite3.Error:
        print('{"error":"local_database_failed"}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
