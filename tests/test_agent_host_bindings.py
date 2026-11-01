"""CLI -> real assembly -> SDK/local adapter checks, without a listening server."""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from shinku import cli
from shinku.agent_runtime.execution import ExecutionProfile
from shinku.qq.host_bindings import QQAgentHostBindings
from shinku.qq.interaction_tools import ChatReadCapability
from shinku.qq.query_continuation import ForegroundQueryPolicy


def runtime():
    return SimpleNamespace(call_chat_json=lambda **kwargs: {"kind": "final", "text": "fixture"})


@pytest.fixture
def isolated(tmp_path):
    persona = tmp_path / "persona.md"
    persona.write_text("Public test persona.", encoding="utf-8")
    env = dict(SHINKU_ENV_FILE=str(tmp_path / "missing.env"),
        SHINKU_DATA_ROOT=str(tmp_path / "data"), SHINKU_CONFIG_ROOT=str(tmp_path / "config"),
        SHINKU_LOG_ROOT=str(tmp_path / "logs"), SHINKU_PERSONA_FILE=str(persona),
        SHINKU_NAPCAT_WEBHOOK_ENABLED="true", SHINKU_NAPCAT_BASE_URL="http://127.0.0.1:18001",
        SHINKU_QQ_AGENT_ENABLED="true", SHINKU_QQ_AGENT_SEND_ENABLED="false",
        SHINKU_MEMORY_ENABLED="false", SHINKU_CHAT_API_KEY="fixture-key",
        SHINKU_CHAT_BASE_URL="https://example.invalid/v1", SHINKU_CHAT_MODEL_NAME="fixture",
        SHINKU_QQ_TRACE_STORE_PATH=str(tmp_path / "trace.json"),
        SHINKU_QQ_CONVERSATION_STATE_DB_PATH=str(tmp_path / "conversation.sqlite"),
        SHINKU_QQ_AGENT_STATE_STORE_PATH=str(tmp_path / "tasks.sqlite"))
    with patch.dict(os.environ, env, clear=True):
        yield env


def capture_real_assembly(monkeypatch):
    original = cli.assemble_qq_agent
    seen = []

    def capture(*args, **kwargs):
        assembly = original(*args, **kwargs)
        seen.append(assembly)
        return assembly

    monkeypatch.setattr(cli, "assemble_qq_agent", capture)
    return seen


def close_assembly(assembly):
    assembly.task_coordinator.shutdown(wait=True)
    assembly.bridge.task_store.close()
    assembly.bridge.conversation_runtime.close()


def test_bindings_snapshot_read_catalog_and_leave_opt_ins_disabled():
    chat, executor = runtime(), runtime()
    cap = ChatReadCapability("test_read", "read", {"type": "object", "properties": {}, "additionalProperties": False})
    catalog = {"test_read": cap}
    bound = QQAgentHostBindings(chat, ExecutionProfile(executor), chat_read_capabilities=catalog)
    catalog.clear()
    options = bound.assembly_options()
    assert options["chat_read_capabilities"] == {"test_read": cap}
    options["chat_read_capabilities"].clear()
    assert bound.chat_read_capabilities == {"test_read": cap}
    assert bound.material_roots == () and bound.expression_profile is None
    assert not bound.foreground_query_policy.enabled and bound.execution_profile.stage_policy is None
    assert not {"sender", "policy", "allowed_tool_names", "task_budget_limits"} & options.keys()


@pytest.mark.parametrize("field,value", [
    ("chat_runtime", None), ("execution_profile", None), ("expression_profile", {}),
    ("material_policy", {}), ("foreground_query_policy", {}),
    ("material_roots", ["root"]), ("material_roots", ("",)),
    ("chat_read_capabilities", {"shell": object()}),
])
def test_invalid_host_dependencies_are_rejected_before_assembly(field, value):
    args = dict(chat_runtime=runtime(), execution_profile=ExecutionProfile(runtime()))
    args[field] = value
    with pytest.raises(ValueError):
        QQAgentHostBindings(**args)


def test_chat_cannot_be_reused_as_the_execution_runtime():
    chat = runtime()
    with pytest.raises(ValueError, match="distinct_execution"):
        QQAgentHostBindings(chat, ExecutionProfile(chat))


@pytest.mark.parametrize("argv", [["doctor"], ["supervise"], ["serve", "--service", "agent"]])
def test_binding_cannot_be_silently_dropped_on_other_commands(monkeypatch, argv):
    def forbidden(*args):
        pytest.fail("not a backend serve operation")
    monkeypatch.setattr(cli, "load_project_env", forbidden)
    assert cli.main(argv, agent_bindings=forbidden) == 2


@pytest.mark.parametrize("state", ["disabled", "no_webhook", "invalid_napcat"])
def test_ineligible_startup_does_not_construct_provider_clients(isolated, monkeypatch, state):
    key, value = {"disabled": ("SHINKU_QQ_AGENT_ENABLED", "false"),
        "no_webhook": ("SHINKU_NAPCAT_WEBHOOK_ENABLED", "false"),
        "invalid_napcat": ("SHINKU_NAPCAT_BASE_URL", "not-a-url")}[state]
    monkeypatch.setenv(key, value)
    def forbidden(*args, **kwargs):
        pytest.fail("ineligible launch must not construct resources or serve")
    monkeypatch.setattr("uvicorn.run", forbidden)
    assert cli.main(["serve"], agent_bindings=forbidden) == 2


@pytest.mark.parametrize("failure", ["wrong_type", "factory_error", "assembly_error", "server_error", "none"])
def test_explicit_binding_context_lifetime_and_no_legacy_fallback(isolated, monkeypatch, failure, capsys):
    events = []
    chat, executor = runtime(), runtime()
    bindings = QQAgentHostBindings(chat, ExecutionProfile(executor))
    seen = capture_real_assembly(monkeypatch)

    @contextmanager
    def factory(config):
        events.append("enter")
        assert config.agent_task_store_path and not config.send_enabled
        try:
            if failure == "factory_error":
                raise ValueError("SECRET_FROM_PROVIDER_CONSTRUCTOR")
            yield object() if failure == "wrong_type" else bindings
        finally:
            events.append("exit")

    def refuse_default(**kwargs):
        pytest.fail("explicit launch cannot instantiate a legacy runtime")
    monkeypatch.setattr("shinku.qq.assembly.LLMRuntime", refuse_default)
    if failure == "assembly_error":
        def reject(*args, **kwargs):
            raise cli.QQAgentConfigError("fixture_assembly_rejected")
        monkeypatch.setattr(cli, "assemble_qq_agent", reject)

    def serve(app, **kwargs):
        events.append("serve")
        bridge = seen[0].bridge
        assert bridge.tool_decision_mode == "task" and bridge.sender is None
        assert bridge.execution_adapter is not None and bridge.result_expression is None
        assert bridge.material_store.store is bridge.task_store
        assert not bridge.foreground_query_policy.enabled
        if failure == "server_error":
            raise RuntimeError("fixture_server_failed")
    monkeypatch.setattr("uvicorn.run", serve)
    try:
        if failure == "server_error":
            with pytest.raises(RuntimeError, match="fixture_server_failed"):
                cli.main(["serve"], agent_bindings=factory)
        else:
            assert cli.main(["serve"], agent_bindings=factory) == (0 if failure == "none" else 2)
        assert events[0] == "enter" and events[-1] == "exit"
        assert ("serve" in events) == (failure in {"server_error", "none"})
        assert "SECRET_FROM_PROVIDER_CONSTRUCTOR" not in capsys.readouterr().err
    finally:
        for assembly in seen:
            close_assembly(assembly)


@pytest.mark.parametrize("permission_mode", ["ultra", "readonly"])
def test_real_cli_assembly_sdk_execution_expression_and_replay(isolated, tmp_path, monkeypatch, permission_mode):
    path = Path(__file__).resolve().parents[1] / "scripts/verify_agent_chat_handoff.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("host_binding_probe", path)
    p = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    case = next(c for c in p.corpus()["cases"] if c["id"] == "file_sync")
    root = tmp_path / "fixture"
    root.mkdir()
    (root / "spec.txt").write_bytes(case["initial"]["spec.txt"].encode())
    ledger = p.JointLedger(tmp_path / "budget.json", p.preview())
    ledger.claim("file_sync:task")
    provider = p.FixtureProvider(case, "task", root)
    file_case = dict(initial=case["initial"], expected=case["expected"])
    reader = p.CompleteFileReader(root, file_case, "read", ledger)
    writer = p.benchmark.FileAdapter(root, file_case, "write", ledger)
    host = p.ToolHost(p.ToolRegistry({t.tool_type: t for t in (reader, writer)}))
    monkeypatch.setattr("shinku.qq.assembly.ToolHost", lambda registry: host)
    seen, lifetimes = capture_real_assembly(monkeypatch), []

    @contextmanager
    def factory(config):
        chat = p.ChatSDKRuntime(ledger, "fixture-key", live=False, forward_transport=httpx.MockTransport(provider))
        execution = p.ExecutionSDKRuntime(SimpleNamespace(root=root, sample=dict(id=case["id"])), ledger,
            "fixture-key", live=False, forward_transport=httpx.MockTransport(provider))
        lifetimes.append("opened")
        try:
            yield QQAgentHostBindings(chat, ExecutionProfile(execution, runtime_meters_budget=True,
                supports_images=False, native_tool_history=True), expression_profile=p.profile(chat),
                material_roots=(root,), foreground_query_policy=ForegroundQueryPolicy(enabled=True))
        finally:
            chat.close()
            execution.close()
            lifetimes.append("closed")

    def serve(app, **kwargs):
        bridge = seen[0].bridge
        assert lifetimes == ["opened"] and bridge.sender is None
        assert bridge.foreground_query_policy.enabled
        bridge.command_state.set_permission_mode(permission_mode)  # Only this isolated synthetic task.
        provider.host = SimpleNamespace(reader=reader, writer=writer, material_store=bridge.material_store)
        turn = p.make_turn(case["request"] + "\n目标文件：" + str(root / "output.txt")
            + "\n原始输入：" + str(root / "spec.txt"), "file_sync:task")
        with ledger.activity():
            reply = bridge(turn)
            tasks = bridge.task_store.candidates(requester=p.SCOPE.requester, session=p.SCOPE.session)
            if permission_mode == "readonly":
                assert not writer.calls and not (root / "output.txt").exists()
                assert (root / "spec.txt").read_bytes() == case["initial"]["spec.txt"].encode()
                assert all(not (task.get("execution_result") or {}).get("verification", {}).get("accepted")
                           for task in tasks)
                assert not reply.sent
                ledger.finish("file_sync:task")
                return
            assert len(tasks) == 1
            result = tasks[0]["execution_result"]
            assert result["verification"]["accepted"] and result["execution"]["status"] == "completed"
            assert (root / "output.txt").read_bytes() == case["expected"]["output.txt"].encode()
            assert (root / "spec.txt").read_bytes() == case["initial"]["spec.txt"].encode()
            assert len(writer.calls) == len(reader.calls) == 1 and not reply.sent
            before = len(ledger.attempts), len(writer.calls)
            p.replay_task_expression(bridge, turn, task_id=tasks[0]["task_id"], context={})
            assert (len(ledger.attempts), len(writer.calls)) == before
            assert any(a["role"] == "execution" for a in ledger.attempts)
            assert any(a["role"] == "expression" for a in ledger.attempts)
            assert all(a["transport_kind"] == "mock_forward" for a in ledger.attempts)
            assert all(not {"benchmark_read", "benchmark_write", "task_material_read"} & set(a["native_tools"])
                       for a in ledger.attempts if a["role"] == "chat")
            with bridge.task_store._transaction() as db:
                ids = db.execute("SELECT expression_id FROM agent_result_expressions").fetchall()
            assert len(ids) == 1
            expression = bridge.result_expression.archive.get(ids[0][0], scope=p.SCOPE)
            assert expression["status"] == "ready" and expression["check"]["accepted"]
            assert not expression["delivery"]["sent"]
        ledger.finish("file_sync:task")
    monkeypatch.setattr("uvicorn.run", serve)
    try:
        assert cli.main(["serve"], agent_bindings=factory) == 0
        assert lifetimes == ["opened", "closed"]
    finally:
        for assembly in seen:
            close_assembly(assembly)
