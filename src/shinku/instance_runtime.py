"""清洁子进程中的公共装配；本地终端和 QQ 使用同一轮次、记忆及执行代码。

生命周期由启动器和 ExitStack 持有。模型客户端按实例分离，执行客户端与聊天
客户端也分离；没有隐式 Angelos 连接。工具由显式扩展注册，未配置则只有公共工具。
"""
from contextlib import contextmanager, ExitStack
import hashlib
import os
from pathlib import Path
import sys
import time
import uuid


@contextmanager
def bindings(config):
    from .llm.runtime import LLMRuntime
    from .agent_runtime.execution import ExecutionProfile
    from .qq.host_bindings import QQAgentHostBindings
    from .extensions import load_runtime_extension
    extension = load_runtime_extension(config.runtime_extension_module)
    # 执行工具注册和聊天只读目录是两道独立边界；只传宿主显式声明，
    # 仍由 HostBindings 校验类型、聊天入口校验账号/权限和真实只读适配器。
    reads = extension.build_chat_read_capabilities(config=config) if extension else {}
    with ExitStack() as resources:
        chat = LLMRuntime(metrics_path=Path(config.metrics_path))
        resources.callback(chat.close)
        # 显式求助配置优先于本地执行器；接收者失联不退回自己执行多步骤任务。
        supervisor = os.environ.get("SHINKU_SUPERVISOR_CONFIG", "").strip()
        if supervisor:
            from .qq.supervisor_handoff import connection_from_file
            from .llm.runtime_expression import RuntimeExpressionAdapter
            from .result_expression import ExpressionProfile
            connection = connection_from_file(supervisor)
            adapter = RuntimeExpressionAdapter(chat)
            expression = ExpressionProfile(adapter, checker=adapter, configuration_id="supervised-character",
                content_policy="selective",
                style_prompt=Path(config.persona_file).read_text(encoding="utf-8"),
                timeout_seconds=60, checker_timeout_seconds=60,
                format_version=2, detail_policy_version=2, paragraph_presentation_version=5, semantic_review_version=1)
            yield QQAgentHostBindings(chat_runtime=chat, chat_read_capabilities=reads,
                expression_profile=expression, delegation=connection)
            return
        execution = LLMRuntime(metrics_path=Path(config.metrics_path).with_name("execution_metrics.json"))
        resources.callback(execution.close)
        if not chat.chat_supports_native_tools() or not execution.chat_supports_native_tools():
            raise ValueError("instance_execution_capability_missing")
        yield QQAgentHostBindings(chat_runtime=chat, chat_read_capabilities=reads,
            execution_profile=ExecutionProfile(execution, profile_id="instance-executor",
                                               runtime_meters_budget=True))


def terminal_loop(bridge, *, input_stream, output_stream, error_stream):
    """仅 stdout 成功 flush 才确认本地传输；不制造 QQ 消息 ID 或已读事实。"""
    from .qq.local_delivery import DesktopReplyDelivery
    from .qq.message import IncomingMessage, MessageSegment
    from .qq.turns import QQTurn
    owners, bots = tuple(bridge.master_user_ids), tuple(bridge.bot_ids)
    if len(owners) != 1 or len(bots) != 1:
        raise ValueError("instance_terminal_identity_ambiguous")
    owner, bot = owners[0], bots[0]
    # 固定定位符支持重启接续；不同实例的数据库和账号绑定独立。
    conversation = "terminal:" + hashlib.sha256((bot + "\0" + owner).encode()).hexdigest()[:32]
    for raw in input_stream:
        text = raw.strip()
        if text == "/exit":
            return 0
        if not text:
            continue
        delivery = DesktopReplyDelivery(bridge, requester=owner, transport="terminal")
        message_id = "terminal_" + uuid.uuid4().hex
        message = IncomingMessage(message_id=message_id, sender_id=owner,
            conversation_id=conversation, conversation_type="desktop",
            segments=(MessageSegment(kind="text", text=text),), timestamp=time.time(),
            raw_event={"self_id": bot})
        turn = QQTurn(conversation_id=conversation, conversation_type="desktop",
            primary_message_id=message_id, message_ids=(message_id,), combined_text=text,
            messages=(message,), local_delivery=delivery.prepare)
        try:
            reply = bridge.handle_turn(turn)
            if reply.final_text:
                output_stream.write(reply.final_text + "\n")
                output_stream.flush()
                delivery.confirm(reply.final_text)
            else:
                delivery.fail()
                error_stream.write("本轮没有可交付的正文。\n")
                return 1
            if reply.agent_status in {"failed", "error"}:
                return 1
        except Exception:
            # 异常可能含提供商凭据，不把任意异常正文印到终端；首错立即停止。
            delivery.fail()
            error_stream.write("instance_turn_failed\n")
            return 1
    return 0


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args not in (["terminal"], ["serve"]):
        return 2
    # 入口只能在显式实例环境下运行，避免直接调用回落到旧默认库。
    required = ("SHINKU_CHARACTER_FILE", "SHINKU_MEMORY_DB_PATH", "SHINKU_QQ_AGENT_STATE_DB_PATH")
    if any(not os.environ.get(key) for key in required):
        print("instance_launcher_required", file=sys.stderr)
        return 2
    if args == ["serve"]:
        from .cli import main as serve
        return serve(["serve", "--service", "backend"], agent_bindings=bindings)
    from .qq.assembly import assemble_qq_agent, load_qq_agent_config
    config = load_qq_agent_config()
    with bindings(config) as host, ExitStack() as resources:
        assembly = assemble_qq_agent(config, **host.assembly_options())
        resources.callback(assembly.bridge.task_store.close)
        resources.callback(assembly.bridge.conversation_runtime.close)
        resources.callback(assembly.task_coordinator.shutdown, wait=True)
        if assembly.bridge.memory_service is not None:
            resources.callback(assembly.bridge.memory_service.store.stop_embedding_worker)
        return terminal_loop(assembly.bridge, input_stream=sys.stdin, output_stream=sys.stdout, error_stream=sys.stderr)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        print("instance_runtime_failed", file=sys.stderr)
        raise SystemExit(1) from None
