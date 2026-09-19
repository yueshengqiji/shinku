"""命令行入口。

两个子命令，各有明确用途：

``shinku doctor``
    把本次解析到的配置**打出来**并建立运行目录。它的价值是"可验证"——
    起服务之前先看一眼它到底连的是哪套配置、哪个数据根、哪个端口。
    （旧项目吃过"以为连的是这台、其实是另一台"的亏。）

``shinku serve --service <name>``
    启动服务。B1 只实现 ``backend``；其余四个宿主在 C 阶段接入。
     未实现的服务**直接报错退出**，不静默降级成一个看似正常的空进程。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from . import __version__
from . import names
from .config import ensure_directories, load_settings
from .qq.config import NapCatConnectionConfig
from .qq.host import NapCatHost
from .qq.napcat import NapCatImageMaterializer, NapCatVisualInputBridge

IMPLEMENTED_SERVICES = ("backend",)


def _cmd_doctor() -> int:
    settings = load_settings()
    created = ensure_directories(settings)

    print(f"project      : {settings.project}")
    print(f"version      : {settings.version}")
    print(f"python       : {sys.version.split()[0]}")
    print(f"executable   : {sys.executable}")
    print()
    print(f"config_root  : {settings.config_root}")
    print(f"data_root    : {settings.data_root}")
    print(f"log_root     : {settings.log_root}")
    print(f"allow_lan    : {settings.allow_lan}")
    print(f"token source : {'env' if settings.service_token else ('file' if settings.service_token_file else 'none')}")
    print()
    print("services:")
    for service in names.SERVICE_NAMES:
        bind = settings.binds[service]
        port = settings.ports[service]
        env_key = names.env_key(service)
        default = names.DEFAULT_PORTS[service]
        marker = "" if port == default else f"  (contract default {default})"
        implemented = "" if service in IMPLEMENTED_SERVICES else "   [not implemented in B1]"
        print(f"  {service:<14} {bind:<10} {port:<6} via {env_key}{marker}{implemented}")
    print()
    print(f"correlation header : {names.CORRELATION_ID_HEADER}")
    print(f"service token hdr  : {names.SERVICE_TOKEN_HEADER}")
    napcat = NapCatConnectionConfig.from_env()
    napcat_diag = napcat.diagnostics()
    print()
    print(
        "napcat webhook    : "
        f"{'enabled' if napcat_diag['webhook_enabled'] else 'disabled'} "
        f"path={napcat_diag['webhook_path']} endpoint={napcat_diag['endpoint'] or '(unset)'}"
    )
    print(
        "napcat credentials : "
        f"action_token={'set' if napcat_diag['token_present'] else 'unset'} "
        f"event_token={'set' if napcat_diag['event_token_present'] else 'unset'}"
    )
    print()
    if created:
        print("created:")
        for path in created:
            print(f"  {path}")
    else:
        print("created: (nothing — all runtime directories already exist)")

    retired = [k for k in names.RETIRED_ENV_KEYS if str(os.environ.get(k, "") or "").strip()]
    if retired:
        print()
        print("WARNING: retired env keys are set and will be IGNORED:")
        for key in retired:
            print(f"  {key}  <- 旧项目命名，新项目不继承。请改用 SHINKU_BACKEND_* / SHINKU_AGENT_*")
    return 0


def _cmd_serve(service: str) -> int:
    settings = load_settings()
    ensure_directories(settings)

    if service not in IMPLEMENTED_SERVICES:
        print(
            f"service {service!r} is not implemented yet (B1 implements: "
            f"{', '.join(IMPLEMENTED_SERVICES)}). Refusing to start a hollow process.",
            file=sys.stderr,
        )
        return 2

    import uvicorn

    from .api import create_app

    napcat_config = NapCatConnectionConfig.from_env()
    napcat_host = None
    if napcat_config.webhook_enabled:
        errors = napcat_config.validate()
        if errors:
            print("invalid enabled NapCat configuration: " + ",".join(errors), file=sys.stderr)
            return 2
        napcat_host = NapCatHost.from_config(
            napcat_config,
            visual_bridge=NapCatVisualInputBridge(
                materialize=NapCatImageMaterializer(allowed_roots=napcat_config.image_roots)
            ),
        )

    app = create_app(
        settings,
        napcat_host=napcat_host,
        napcat_webhook_path=napcat_config.webhook_path,
        napcat_event_token=napcat_config.event_token,
    )
    uvicorn.run(
        app,
        host=settings.binds[service],
        port=settings.ports[service],
        log_level="info",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shinku", description="Shinku 独立实现（洁净室重建）")
    parser.add_argument("--version", action="version", version=f"shinku {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="打印解析到的配置并建立运行目录")

    serve = sub.add_parser("serve", help="启动一个服务")
    serve.add_argument("--service", dest="service", default="backend", choices=names.SERVICE_NAMES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "doctor":
        return _cmd_doctor()
    if args.command == "serve":
        return _cmd_serve(args.service)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
