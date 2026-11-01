"""独立进程声明、状态所有权及真实公共装配的离线验收，全部使用合成身份。"""
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from shinku.instance import claim_instance, load_instance, main
from shinku.instance_runtime import terminal_loop
from shinku.qq.assembly import assemble_qq_agent, load_qq_agent_config
from shinku.qq.host_bindings import QQAgentHostBindings
from shinku.agent_runtime.execution import ExecutionProfile
from test_qq_owner_access import _SpeechRuntime, _turn


def manifest(tmp_path, name="child"):
    base = tmp_path / name
    base.mkdir()
    (base / "character.json").write_text(json.dumps(dict(version=1, id=name, name=name,
        aliases=[name], relationships={"local:owner": dict(kind="father", address="爸爸")})))
    (base / "persona.md").write_text("这是合成测试角色，不是生产人设。")
    settings = {f"SHINKU_{channel}_{key}": value for channel in ("CHAT", "AUX") for key, value in
                dict(API_KEY="fixture-only", BASE_URL="https://example.invalid/v1",
                     MODEL_NAME="fixture", API_PROTOCOL="openai").items()}
    settings["SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST"] = "example.invalid:fixture"
    (base / "settings.json").write_text(json.dumps(settings))
    (base / "settings.json").chmod(0o600)
    data = dict(version=1, character_file="character.json", persona_file="persona.md",
                state_root="state", settings_file="settings.json", owner_id="local:owner",
                bot_id="local:" + name, channel="terminal", backend_port=18088)
    path = base / "instance.json"
    path.write_text(json.dumps(data))
    return path


def test_check_has_no_runtime_or_storage_side_effects(tmp_path, monkeypatch, capsys):
    path = manifest(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail("check may not launch or create state")
    monkeypatch.setattr("subprocess.call", forbidden)
    assert main(["check", str(path)]) == 0
    assert not (path.parent / "state").exists()
    assert "fixture-only" not in capsys.readouterr().out


@pytest.mark.parametrize("send", [False, True])
def test_qq_launch_requires_explicit_accounts_and_never_inherits_send(tmp_path, monkeypatch, send):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data.update(channel="qq", owner_id="10001", bot_id="10002")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="qq_binding_missing"):
        load_instance(path)
    settings_path = path.parent / "settings.json"
    settings = json.loads(settings_path.read_text())
    settings.update(SHINKU_NAPCAT_BASE_URL="http://127.0.0.1:18001", SHINKU_NAPCAT_ACCESS_TOKEN="fixture",
                    SHINKU_NAPCAT_EVENT_TOKEN="fixture", SHINKU_SERVICE_TOKEN="fixture")
    settings_path.write_text(json.dumps(settings))
    monkeypatch.setenv("SHINKU_QQ_AGENT_SEND_ENABLED", "true")
    seen = []
    def capture(command, *, env, cwd, pass_fds):
        seen.append(command)
        assert command[-1] == "serve" and env["SHINKU_QQ_BOT_IDS"] == "10002"
        assert env["SHINKU_QQ_AGENT_SEND_ENABLED"] == str(send).lower()
        assert env["SHINKU_BACKEND_PORT"] == "18088"
        assert len(pass_fds) == 1 and os.fstat(pass_fds[0])
        assert Path(cwd) == path.parent / "state"
        return 0
    monkeypatch.setattr(subprocess, "call", capture)
    assert main(["serve", str(path), *(["--send"] if send else [])]) == 0
    assert len(seen) == 1


def test_child_keeps_state_lock_after_launcher_handle_closes(tmp_path):
    spec = load_instance(manifest(tmp_path))
    process = None
    try:
        with claim_instance(spec) as fd:
            process = subprocess.Popen([sys.executable, "-B", "-c",
                "import os,sys; os.fstat(int(sys.argv[1])); print('ready',flush=True); sys.stdin.read()", str(fd)],
                pass_fds=(fd,), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            assert process.stdout.readline().strip() == "ready"
        with pytest.raises(ValueError, match="already_running"):
            with claim_instance(load_instance(spec.manifest)):
                pass
    finally:
        if process is not None:
            process.communicate("", timeout=5)
    with claim_instance(load_instance(spec.manifest)):
        pass


def test_environment_does_not_inherit_private_models_or_registry(tmp_path):
    spec = load_instance(manifest(tmp_path))
    env = spec.environment({"SHINKU_CHAT_API_KEY": "wrong", "SHINKU_RUNTIME_EXTENSION_MODULE": "wrong",
                            "SHINKU_MEMORY_DB_PATH": "/wrong", "PYTHONPATH": "/wrong", "HOME": "/home/fixture"})
    assert env["SHINKU_CHAT_API_KEY"] == "fixture-only"
    assert env["SHINKU_RUNTIME_EXTENSION_MODULE"] == ""
    assert env["SHINKU_QQ_AGENT_SEND_ENABLED"] == "false"
    assert env["SHINKU_ENV_FILE"] == os.devnull and "/wrong" not in env.values()
    for key in ("MEMORY_DB_PATH", "MODEL_SERVICES_PATH", "QQ_AGENT_STATE_DB_PATH",
                "QQ_AGENT_TASKS_DB_PATH", "QQ_TRACE_STORE_PATH", "QQ_AGENT_APPROVAL_STORE_PATH"):
        assert Path(env["SHINKU_" + key]).is_relative_to(spec.state_root)
    assert Path(env["SHINKU_PRIVATE_READ_ROOTS"]) == spec.state_root / "workspace"


@pytest.mark.parametrize("declaration", [None, "*:*", "other.invalid:fixture", "example.invalid:other"])
def test_native_tools_require_precise_host_declaration(tmp_path, declaration):
    path = manifest(tmp_path)
    settings_path = path.parent / "settings.json"
    settings = json.loads(settings_path.read_text())
    if declaration is None:
        del settings["SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST"]
    else:
        settings["SHINKU_NATIVE_TOOL_PROVIDER_ALLOWLIST"] = declaration
    settings_path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="native_tool_declaration_required"):
        load_instance(path)


def test_real_clients_receive_explicit_capability_without_requests(tmp_path, monkeypatch):
    import socket
    from shinku.instance_runtime import bindings
    spec = load_instance(manifest(tmp_path))
    def forbidden(*args, **kwargs):
        pytest.fail("client assembly must not issue a request")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    with claim_instance(spec), patch.dict(os.environ, spec.environment(), clear=True):
        config = load_qq_agent_config()
        with bindings(config) as host:
            assert host.chat_runtime is not host.execution_profile.runtime
            assert host.chat_runtime.chat_supports_native_tools()
            assert host.execution_profile.runtime.chat_supports_native_tools()
            assert not host.chat_runtime._tool_profile(host.chat_runtime.chat).verified
            assert host.chat_runtime.runtime_policy.http_attempts == 1


@pytest.mark.parametrize("field,value", [("owner_id", "local:new"), ("bot_id", "local:new")])
def test_restart_refuses_rebound_accounts(tmp_path, field, value):
    path = manifest(tmp_path)
    spec = load_instance(path)
    with claim_instance(spec):
        with pytest.raises(ValueError, match="already_running"):
            with claim_instance(load_instance(path)):
                pass
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="root_owner_mismatch"):
        load_instance(path)


@pytest.mark.parametrize("kind", ["unbound", "symlink", "hardlink", "nested", "input", "credentials", "override", "missing_model"])
def test_invalid_isolation_fails_closed(tmp_path, kind):
    path = manifest(tmp_path)
    spec = load_instance(path)
    if kind in {"symlink", "hardlink", "nested"}:
        with claim_instance(spec):
            pass
    if kind == "unbound":
        spec.state_root.mkdir()
        (spec.state_root / "old.db").write_text("preserve me")
    elif kind == "symlink":
        (spec.state_root / "memory.sqlite3").symlink_to(path.parent / "settings.json")
    elif kind == "hardlink":
        os.link(path.parent / "settings.json", spec.state_root / "memory.sqlite3")
    elif kind in {"nested", "input"}:
        data = json.loads(path.read_text())
        data["state_root"] = "state/child" if kind == "nested" else "."
        path.write_text(json.dumps(data))
    elif kind == "credentials":
        (path.parent / "settings.json").chmod(0o644)
    else:
        settings_path = path.parent / "settings.json"
        data = json.loads(settings_path.read_text())
        if kind == "override":
            data["SHINKU_MEMORY_DB_PATH"] = "/wrong"
        else:
            del data["SHINKU_AUX_MODEL_NAME"]
        settings_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="instance_"):
        load_instance(path)


def make_assembly(spec):
    chat, executor = _SpeechRuntime({"speech": "合成应答"}), _SpeechRuntime({"speech": "合成执行"})
    host = QQAgentHostBindings(chat, ExecutionProfile(executor))
    with patch.dict(os.environ, spec.environment(), clear=True):
        config = replace(load_qq_agent_config(), memory_judge_enabled=False)
        assembly = assemble_qq_agent(config, **host.assembly_options())
    return assembly, chat


def close(assembly):
    assembly.task_coordinator.shutdown(wait=True)
    assembly.bridge.memory_service.store.stop_embedding_worker()
    assembly.bridge.task_store.close()
    assembly.bridge.conversation_runtime.close()


def group_turn(spec, text):
    turn = _turn(spec.owner_id, text, conversation_id="same-group", conversation_type="group")
    return replace(turn, messages=tuple(replace(m, raw_event={**m.raw_event, "self_id": spec.bot_id}) for m in turn.messages))


def event_rows(assembly):
    with assembly.bridge.memory_service.store._connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM memory_events WHERE status='active'")]


def test_two_real_assemblies_same_owner_group_and_restart_are_separate(tmp_path):
    left, right = (load_instance(manifest(tmp_path, name)) for name in ("child", "parent"))
    with claim_instance(left), claim_instance(right):
        a, ar = make_assembly(left)
        b, br = make_assembly(right)
        try:
            for rig in (a, b):
                assert rig.bridge.tool_decision_mode == "task"
                assert rig.bridge.execution_adapter is not None and rig.bridge.sender is None
                assert rig.bridge.memory_service.character == rig.bridge.character
            a.bridge.observe_message(group_turn(left, "蓝晶只属于左边").messages[0])
            assert "蓝晶" not in json.dumps(event_rows(b), ensure_ascii=False)
            out = io.StringIO()
            assert terminal_loop(a.bridge, input_stream=io.StringIO("蓝晶只在下雨时取消\n"), output_stream=out, error_stream=io.StringIO()) == 0
            assert out.getvalue().strip()
            rows = event_rows(a)
            assistant = [json.loads(row["metadata_json"]) for row in rows if json.loads(row["metadata_json"]).get("role") == "assistant"]
            assert assistant and all(row["sender_name"] == "child" for row in assistant)
            assert all(row["delivery_channel"] == "local_terminal" for row in assistant)
            assert all(row["delivery_message_ids"][0].startswith("local-terminal:") for row in assistant)
            a.bridge.command_state.set_permission_mode("readonly")
            assert b.bridge.command_state.permission_mode() == "traditional"
            assert not b.bridge.task_store.candidates(requester="local:owner", session="desktop:same")
            assert not event_rows(b)
        finally:
            close(a)
            close(b)
    with claim_instance(load_instance(left.manifest)):
        restarted, runtime = make_assembly(left)
        try:
            assert restarted.bridge.command_state.permission_mode() == "readonly"
            assert any("蓝晶" in row["text"] for row in event_rows(restarted))
            assert terminal_loop(restarted.bridge, input_stream=io.StringIO("接着说蓝晶的条件\n"), output_stream=io.StringIO(), error_stream=io.StringIO()) == 0
            assert "只在下雨" in json.dumps(runtime.calls, ensure_ascii=False)
        finally:
            close(restarted)


def test_stdout_failure_never_records_assistant_delivery(tmp_path):
    spec = load_instance(manifest(tmp_path))
    class Broken(io.StringIO):
        def flush(self):
            raise OSError("synthetic transport failure")
    with claim_instance(spec):
        assembly, _ = make_assembly(spec)
        try:
            assert terminal_loop(assembly.bridge, input_stream=io.StringIO("你好\n"), output_stream=Broken(), error_stream=io.StringIO()) == 1
            assert not any(json.loads(row["metadata_json"]).get("role") == "assistant" for row in event_rows(assembly))
            assert assembly.bridge.trace_store.snapshot(limit=1)[0]["delivery_status"] == "uncertain"
        finally:
            close(assembly)


@pytest.mark.parametrize("same_message_id", [False, True])
def test_same_group_independent_agents_execute_and_cannot_read_other_results(tmp_path, same_message_id):
    from shinku.agent_runtime.demo import DemoFileTool, DemoRuntime, demo_request
    from shinku.agent_runtime.contracts import AgentBoundaryError
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    from test_qq_tool_decision_boundary import Chat, submit, contract_runtime
    specs = [load_instance(manifest(tmp_path, name)) for name in ("child", "parent")]
    assemblies, tasks = [], []
    try:
        for spec in specs:
            with claim_instance(spec):
                tool = DemoFileTool(spec.state_root / "workspace")
                execution = DemoRuntime(tool)
                request = demo_request(tool)
                contract_runtime(execution, request.contract)
                chat = Chat([submit()])
                with patch.dict(os.environ, spec.environment(), clear=True):
                    config = replace(load_qq_agent_config(), memory_judge_enabled=False)
                    rig = assemble_qq_agent(config,
                        tool_host=ToolHost(ToolRegistry({tool.tool_type: tool})),
                        **QQAgentHostBindings(chat, ExecutionProfile(execution)).assembly_options())
                assemblies.append(rig)
                rig.bridge.command_state.set_permission_mode("ultra")
                turn = group_turn(spec, request.original_request)
                if not same_message_id:
                    turn = replace(turn, primary_message_id=spec.character.character_id + ":message",
                                   message_ids=(spec.character.character_id + ":message",),
                                   messages=tuple(replace(m, message_id=spec.character.character_id + ":message") for m in turn.messages))
                reply = rig.bridge.handle_turn(turn)
                assert reply.agent_status == "completed", reply
                rows = rig.bridge.task_store.candidates(requester="local:owner", session="group:same-group")
                assert len(rows) == 1 and rows[0]["execution_result"]["verification"]["accepted"]
                assert tool.calls == 1 and tool.path.read_text() == "standalone agent verified"
                assert not reply.sent
                tasks.append((turn, rows[0]["task_id"]))
        for index, rig in enumerate(assemblies):
            other_turn, other_id = tasks[1 - index]
            if same_message_id:
                # 局部任务号由原消息派生，可以相同；此时必须返回本库产物，不能串读另一个角色。
                assert other_id == tasks[index][1]
                own = json.dumps(rig.bridge.agent_calls.result(other_turn, other_id))
                assert str(specs[index].state_root / "workspace" / "agent-demo.txt") in own
                assert str(specs[1 - index].state_root) not in own
            else:
                with pytest.raises(AgentBoundaryError, match="task_not_found"):
                    rig.bridge.agent_calls.result(other_turn, other_id)
    finally:
        for rig in assemblies:
            close(rig)
