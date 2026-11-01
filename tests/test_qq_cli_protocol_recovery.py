"""CLI selection must retain transport and never confuse routing with execution."""
import pytest
from types import SimpleNamespace

from shinku.agent.loop import AgentDecision
from shinku.qq.tool_selection import dispatch_diagnostics
from shinku.tools.invocation import ToolInvocation
from test_qq_no_task_route_repair import setup_probe
from test_qq_public_query_fast_track import turn, malformed_native, close_coordinators


def test_exact_native_shell_does_not_need_dispatcher_roundtrip():
    value = {"kind": "tool", "tool_call": {"name": "arbitrary_shell", "arguments": {"command": "fixture"}}}
    host, runtime, probe, shell = setup_probe(value)
    t = turn("测试 CLI")
    decision, _ = host._run_chat_decision(task_id="fixture", context={"qq_turn": t,
        "audience_context": {"role": "master", "is_master": True}}, initial_transcript=host._transcript(t))
    assert decision.invocation.name == "arbitrary_shell"
    assert decision.task_link == {"mode": "new"}
    assert len(runtime.calls) == 1 and not shell.calls and not probe.calls


def test_repair_uses_stream_and_reports_failure_without_requesting_repetition():
    host, runtime, probe, shell = setup_probe()
    host.command_state.set_stream_enabled(True)
    calls = []
    def stream(**options):
        calls.append(options)
        yield {"text": "unverified fragment"}
        if len(calls) == 1:
            return SimpleNamespace(parsed=malformed_native(), error="")
        return SimpleNamespace(parsed=None, error="upstream_connection_failed")
    runtime.stream_chat_json = stream
    runtime.call_chat_json = lambda **kw: pytest.fail("repair must preserve stream transport")
    result = host.handle_turn(turn("用 CLI ping 指定地址"))
    assert len(calls) == 2 and not probe.calls and not shell.calls
    assert result.agent_status == "failed" and result.reason == "tool_protocol_recovery_failed"
    assert "没有执行" in result.final_text and "再说一遍" not in result.final_text
    assert "unverified fragment" not in result.final_text


@pytest.mark.parametrize("name", ["unknown-secret-value", {"secret": "value"}])
def test_diagnostic_does_not_include_unknown_values(name):
    decision = AgentDecision.tool(ToolInvocation("__shinku_tool_dispatch__",
        {"tool_name": name, "arguments": {"password": "hidden"}}))
    result = dispatch_diagnostics(decision, {"arbitrary_shell"})
    assert result == {"status": "rejected", "reason": "unknown_target_tool", "selected_tools": [], "count": 1}


def test_ping_address_is_not_a_file_contract_target():
    from shinku.agent.verification import SmartCompletionVerifier
    gate = SmartCompletionVerifier(allowed_tool_names=[])
    request = "来老婆，我来测试一下你的cli能力，你ping一下221.214.140.66"
    contract = {"deliverables": ["ping 命令结果"], "requirements": [{"id": "r1",
        "effect": "command_run", "targets": [], "checks": {"command": "ping.exe 221.214.140.66", "exit_code": 0}}]}
    assert gate.target_coverage_gaps(contract, [request]) == []
    for text in ['读取 "你ping一下221.214.140.66"', '读取 C:/fixture/221.214.140.66', '读取 221.214.140.66']:
        assert gate.target_coverage_gaps(contract, [text]), text
