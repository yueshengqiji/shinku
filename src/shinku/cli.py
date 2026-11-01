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
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from . import __version__
from . import names
from .config import ensure_directories, load_project_env, load_settings
from .qq.config import NapCatConnectionConfig
from .qq.host import NapCatHost
from .qq.napcat import NapCatHttpActionCaller, NapCatImageMaterializer, NapCatVisualInputBridge
from .qq.assembly import QQAgentConfigError, assemble_qq_agent, load_qq_agent_config
from .qq.turns import NapCatTurnDispatcher
from .qq.welcome import GroupWelcomePolicy
from .persona import PersonaLoadError, load_persona
from .providers.catalog import ModelServiceConfigStore
from .tools.execution import ExecutionPolicy
from .supervisor import ShinkuSupervisor, SupervisorConfig

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
    agent_config = load_qq_agent_config()
    print()
    persona_status = "unset"
    if agent_config.persona_file:
        try:
            load_persona(agent_config.persona_file)
            persona_status = "valid"
        except PersonaLoadError as exc:
            persona_status = f"invalid:{exc}"
    print(
        "qq agent          : "
        f"{'enabled' if agent_config.enabled else 'disabled'} "
        f"send={'enabled' if agent_config.send_enabled else 'dry-run'} "
        f"persona={persona_status} "
        f"model={agent_config.chat_model_name or '(unset)'}"
    )
    tool_policy = ExecutionPolicy.from_environment()
    print(
        "tool policy       : "
        f"default_risk={tool_policy.default_risk} "
        f"approval_risks={','.join(sorted(tool_policy.approval_required_by_risk)) or '(none)'} "
        f"overrides={len(tool_policy.risk_by_tool)}"
    )
    if agent_config.enabled:
        errors = list(agent_config.validation_errors())
        if persona_status.startswith("invalid:"):
            errors.append(persona_status.removeprefix("invalid:"))
        print("qq agent config   : " + ("valid" if not errors else ",".join(errors)))
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


def _cmd_serve(service: str, *, agent_bindings=None) -> int:
    # A trusted launcher's context owns injected clients through server exit,
    # including assembly errors. Ordinary CLI launches never invoke a factory.
    with ExitStack() as resources:
        return _serve_backend(service, resources, agent_bindings=agent_bindings)


def _serve_backend(service: str, resources: ExitStack, *, agent_bindings=None) -> int:
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
    from .api.model_runtime import RuntimeModelServiceConfig, RuntimeModelServiceController

    napcat_config = NapCatConnectionConfig.from_env()
    agent_config = load_qq_agent_config()
    model_service_path = os.environ.get("SHINKU_MODEL_SERVICES_PATH", "").strip()
    model_service_store = ModelServiceConfigStore(
        Path(model_service_path)
        if model_service_path
        else settings.data_root / "model_services.json",
        environ=os.environ,
    )
    model_service_config = RuntimeModelServiceConfig.from_agent_config(agent_config)
    model_service_controller = RuntimeModelServiceController(
        store=model_service_store,
        config_module=model_service_config,
    )
    # 已保存的激活项优先于启动环境；没有注册表时保持现有配置不变。
    model_service_controller.load_saved()
    agent_config = load_qq_agent_config()
    if (
        (agent_config.memory_enabled and not agent_config.memory_db_path)
        or not agent_config.command_state_path
        or not agent_config.agent_task_store_path
    ):
        agent_config = replace(
            agent_config,
            **(
                {"memory_db_path": str(settings.data_root / "memory" / "shinku_memory.sqlite3")}
                if agent_config.memory_enabled and not agent_config.memory_db_path
                else {}
            ),
            **(
                {"command_state_path": str(settings.data_root / "qq" / "command_state.json")}
                if not agent_config.command_state_path
                else {}
            ),
            **(
                {"agent_task_store_path": str(settings.data_root / "runtime" / "agent_tasks.sqlite3")}
                if not agent_config.agent_task_store_path
                else {}
            ),
        )
    if agent_config.enabled and not napcat_config.webhook_enabled:
        print(
            "invalid enabled QQ Agent configuration: napcat_webhook_required",
            file=sys.stderr,
        )
        return 2
    if agent_bindings is not None and (not callable(agent_bindings) or not agent_config.enabled):
        print("invalid enabled QQ Agent configuration: host_bindings_require_enabled_agent", file=sys.stderr)
        return 2
    napcat_host = None
    assembly = None
    if napcat_config.webhook_enabled:
        errors = napcat_config.validate()
        if errors:
            print("invalid enabled NapCat configuration: " + ",".join(errors), file=sys.stderr)
            return 2
        assembly_options = {}
        if agent_bindings is not None:
            from .qq.host_bindings import QQAgentHostBindings

            try:
                bindings = resources.enter_context(agent_bindings(agent_config))
                if not isinstance(bindings, QQAgentHostBindings):
                    raise TypeError("host_bindings_required")
                assembly_options = bindings.assembly_options()
            except Exception:
                # A provider constructor may include its URL/key in an error.
                # Do not print arbitrary factory exception messages or fall back.
                print("invalid enabled QQ Agent configuration: host_bindings_invalid", file=sys.stderr)
                return 2
        # CLI显式绑定只读缓存接口；构造caller不联网，图片URL正常时不调用get_image。
        # Docker路径只按受保护配置映射，不能由模型/图片消息指定宿主目录。
        image_caller = NapCatHttpActionCaller(napcat_config.base_url,
            access_token=napcat_config.token_value(), timeout=napcat_config.timeout)
        napcat_host = NapCatHost.from_config(
            napcat_config,
            visual_bridge=NapCatVisualInputBridge(
                materialize=NapCatImageMaterializer(allowed_roots=napcat_config.image_roots,
                    path_mappings=napcat_config.image_path_mappings, media_loader=image_caller.get_image)
            ),
            # Welcome is an external side effect. It is available only when
            # the QQ Agent is explicitly allowed to send; dry-run never sends.
            welcome_policy=(
                GroupWelcomePolicy.from_environment()
                if agent_config.send_enabled
                else GroupWelcomePolicy()
            ),
        )
        if agent_config.enabled:
            try:
                from .qq.file_inputs import QQTextFileReader
                sender = napcat_host.send if agent_config.send_enabled else None
                assembly = assemble_qq_agent(
                    agent_config,
                    **assembly_options,
                    sender=sender,
                    # 本账号缓存范围由宿主配置；收到的文件作为输入，不扩大群成员工具权限。
                    file_reader=QQTextFileReader(image_caller.get_file,
                        path_mappings=napcat_config.file_path_mappings, allowed_roots=napcat_config.file_roots,
                        message_fetcher=image_caller.get_file_message),
                    group_delivery_observer=lambda group_id, recipients, ids, autonomous, sent_at: (
                        napcat_host.ingress.note_group_reply(
                            group_id, recipient_ids=recipients, message_ids=ids,
                            autonomous=autonomous, at=sent_at,
                        )
                    ),
                    group_continuation_observer=napcat_host.ingress.note_group_continuation,
                )
                model_service_controller.attach_runtime(assembly.bridge.runtime)
            except (QQAgentConfigError, ValueError) as exc:
                print(f"invalid enabled QQ Agent configuration: {exc}", file=sys.stderr)
                return 2
            # Memory observes every normalized group message before the reply
            # route decides whether this turn is addressed to Shinku.
            napcat_host.set_passive_observer(assembly.bridge.observe_message)
            # Command switches are live: /at reply and /stop must affect the
            # ingress gate without rebuilding the FastAPI app.
            napcat_host.ingress.set_command_state(assembly.bridge.command_state)
            dispatcher = NapCatTurnDispatcher(
                handler=assembly.bridge,
                flush_batch=napcat_host.flush,
                async_handler=True,
            )
            napcat_host.set_turn_dispatcher(dispatcher)

    app = create_app(
        settings,
        napcat_host=napcat_host,
        napcat_webhook_path=napcat_config.webhook_path,
        napcat_event_token=napcat_config.event_token,
        qq_agent_assembly=assembly,
        napcat_config=napcat_config,
        model_services_controller=model_service_controller,
    )
    # 后端接力入口必须来自显式宿主绑定，不能由事件、网页或模型动态安装。
    if assembly is not None and agent_bindings is not None and bindings.backend_setup is not None:
        app.include_router(resources.enter_context(bindings.backend_setup(assembly)))
    uvicorn.run(
        app,
        host=settings.binds[service],
        port=settings.ports[service],
        log_level="info",
    )
    return 0


def _cmd_supervise(
    service: str,
    *,
    max_restarts: int,
    health_interval: float,
    startup_timeout: float,
    restart_backoff: float,
) -> int:
    """启动并只监督本命令创建的一个后端子进程。"""

    settings = load_settings()
    ensure_directories(settings)
    config = SupervisorConfig(
        service=service,
        max_restarts=max_restarts,
        health_interval_seconds=health_interval,
        startup_timeout_seconds=startup_timeout,
        restart_backoff_seconds=restart_backoff,
    )
    return ShinkuSupervisor(config, settings=settings).run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shinku", description="Shinku 独立实现（洁净室重建）")
    parser.add_argument("--version", action="version", version=f"shinku {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="打印解析到的配置并建立运行目录")

    serve = sub.add_parser("serve", help="启动一个服务")
    serve.add_argument("--service", dest="service", default="backend", choices=names.SERVICE_NAMES)

    supervise = sub.add_parser(
        "supervise",
        help="启动并监督本命令创建的后端子进程（不扫描或杀其他进程）",
    )
    supervise.add_argument("--service", dest="service", default="backend", choices=("backend",))
    supervise.add_argument("--max-restarts", type=int, default=3)
    supervise.add_argument("--health-interval", type=float, default=5.0)
    supervise.add_argument("--startup-timeout", type=float, default=30.0)
    supervise.add_argument("--restart-backoff", type=float, default=2.0)
    return parser


def main(argv: Sequence[str] | None = None, *, agent_bindings=None) -> int:
    """Run the CLI; a trusted backend launcher may inject a binding context.

    ``agent_bindings(finalized_config)`` must return a context manager yielding
    QQAgentHostBindings. It is intentionally unavailable as an argument/env
    module path, and cannot be silently lost across supervisor subprocesses.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if agent_bindings is not None and (args.command != "serve" or args.service != "backend"):
        print("host_bindings_require_backend_serve", file=sys.stderr)
        return 2
    load_project_env()

    if args.command == "doctor":
        return _cmd_doctor()
    if args.command == "serve":
        return _cmd_serve(args.service, agent_bindings=agent_bindings) if agent_bindings is not None else _cmd_serve(args.service)
    if args.command == "supervise":
        return _cmd_supervise(
            args.service,
            max_restarts=args.max_restarts,
            health_interval=args.health_interval,
            startup_timeout=args.startup_timeout,
            restart_backoff=args.restart_backoff,
        )

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
