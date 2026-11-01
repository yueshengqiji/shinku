"""Actual SDK/Agent continuation with isolated transport and real file receipts."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shinku.agent.budget import TaskBudget, TaskBudgetExceeded
from shinku.agent.loop import AgentState
from shinku.agent.native_history import NativeToolHistory, prepare_native_request
from shinku.llm.context_budget import ContextBudgetPolicy, measure_payload
from shinku.llm.native_messages import NativeProtocolError, assistant_message, validate_native_messages
from shinku.tools.invocation import ToolResultEnvelope


@pytest.fixture(scope="module")
def probe():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_agent_execution_protocol.py"
    spec = importlib.util.spec_from_file_location("native_protocol_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def message(ident="call_1", reasoning="INTERNAL_ONLY", arguments='{"path":"spec.txt"}'):
    return {"role": "assistant", "content": "", "reasoning_content": reasoning,
        "tool_calls": [{"id": ident, "type": "function", "function": {
            "name": "benchmark_read", "arguments": arguments}}]}


def response(value):
    return SimpleNamespace(choices=[SimpleNamespace(message=value)])


def receipt(ident="call_1", content="read actual bytes"):
    return {"role": "tool", "content": content, "tool_call_id": ident}


@pytest.fixture(scope="module")
def native_cases(probe, tmp_path_factory):
    root = tmp_path_factory.mktemp("native-sdk")
    (root / "outside.txt").write_text("untouched", encoding="utf-8")
    batch = probe.TaskBudget(**probe.BATCH_LIMITS)
    rows, captures = [], {}

    class Runtime(probe.ProbeRuntime):
        def __init__(self, fixture, ledger):
            captures[fixture.sample["id"]] = fixture
            super().__init__(fixture, ledger)

        def _build_thinking_kwargs(self, **kwargs):
            return {"extra_body": {"thinking": {"type": "enabled"}}, "reasoning_effort": "high"}

    for sample in probe.CASES:
        rows.append(probe.run_case(sample, root / sample["id"], batch,
            runtime_factory=Runtime, native_tool_history=True))
    return {"rows": rows, "captures": captures, "root": root}


@pytest.mark.parametrize("name", ["native_two_tools", "failed_tool_corrected", "cross_slice",
    "malformed_tool", "timeout", "retry_then_success", "false_final", "unknown_tool", "denied_path",
    "long_attachment_tail", "image_supported", "image_unsupported"])
def test_native_session_keeps_original_host_gates_and_artifacts(native_cases, name):
    row = next(r for r in native_cases["rows"] if r["case"] == name)
    assert row["fixture_pass"], row
    assert not row["false_completed"] and not row["false_accepted"]
    assert row["duplicate_effects"] == row["duplicate_publications"] == 0
    assert row["query_unchanged"] and row["original_contract_preserved"]
    assert row["reasoning_excluded_from_result"]


def test_native_reasoning_and_tool_ids_reach_actual_sdk_after_each_tool(native_cases, probe):
    fixture = native_cases["captures"]["native_two_tools"]
    assert len(fixture.payloads) == 3
    for index, payload in enumerate(fixture.payloads):
        history = [m for m in payload["messages"] if m["role"] in {"assistant", "tool"}]
        assert len(history) == index * 2
        for offset in range(0, len(history), 2):
            assistant, tool = history[offset:offset+2]
            assert assistant["reasoning_content"] == probe.REASONING
            assert assistant["tool_calls"][0]["id"] == tool["tool_call_id"]
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["tool_choice"] == "auto"
        assert payload["reasoning_effort"] == "high"


def test_failed_real_read_receipt_is_preserved_in_next_sdk_request(native_cases):
    fixture = native_cases["captures"]["failed_tool_corrected"]
    tools = [m for m in fixture.payloads[1]["messages"] if m["role"] == "tool"]
    assert len(tools) == 1 and '"status": "error"' in tools[0]["content"]
    assert "FileNotFoundError" in tools[0]["content"]


def test_material_tail_receipt_is_literal_on_wire_before_write(native_cases, probe):
    fixture = native_cases["captures"]["long_attachment_tail"]
    tools = [m for m in fixture.payloads[1]["messages"] if m["role"] == "tool"]
    assert len(tools) == 1
    assert probe.TAIL in tools[0]["content"]
    receipt_data, _ = json.JSONDecoder().raw_decode(tools[0]["content"])
    assert receipt_data["content"].endswith(probe.TAIL)
    assert receipt_data["provided"] is False  # It is marked provided only after this request.
    row = next(r for r in native_cases["rows"] if r["case"] == "long_attachment_tail")
    assert row["task_success"] and row["material_read_dispatches"] == 1 and len(row["writes"]) == 1


def test_cross_slice_keeps_same_protocol_history_and_task_budget(native_cases):
    row = next(r for r in native_cases["rows"] if r["case"] == "cross_slice")
    assert row["slice_outcomes"] == ["yielded", "yielded", "completed"]
    assert row["sdk_requests"] == row["budget"]["requests"] == 3
    assert row["native_history_on_wire"] and row["read_result_on_later_wire"]


def test_reasoning_does_not_enter_public_result_or_sqlite_state(native_cases, probe):
    assert all(row["reasoning_excluded_from_result"] for row in native_cases["rows"])
    for path in native_cases["root"].rglob("*.sqlite*"):
        assert probe.REASONING.encode() not in path.read_bytes(), str(path)


@pytest.mark.parametrize("change", ["task", "provider", "cache"])
def test_changed_task_or_provider_cannot_reuse_protocol_state(change):
    scope = ("task", "cache", ("client", "model"))
    history = NativeToolHistory(scope, "original goal", None, [])
    changed = list(scope)
    changed[{"task": 0, "cache": 1, "provider": 2}[change]] = "changed"
    with pytest.raises(NativeProtocolError, match="scope_changed"):
        history.request_kwargs(AgentState(), tuple(changed), "updated goal")


def test_media_replacement_fails_before_provider_request():
    runtime = SimpleNamespace(native_message_scope=lambda: "provider")
    planner = SimpleNamespace(runtime=runtime, prompt_cache_key="cache")
    state = AgentState(task_id="t")
    kwargs = dict(user_prompt="goal", user_images=[{"data_url": "original"}], history_turns=[])
    prepare_native_request(planner, state, kwargs)
    with pytest.raises(NativeProtocolError, match="media_changed"):
        prepare_native_request(planner, state, {**kwargs, "user_images": []})


def test_provider_tools_with_missing_receipts_never_form_another_request():
    history = NativeToolHistory("scope", "goal", None, [])
    history.capture(message())
    with pytest.raises(NativeProtocolError, match="receipt_missing"):
        history.request_kwargs(AgentState(), "scope", "goal")


def test_wrong_host_tool_name_is_not_a_matching_provider_receipt():
    history = NativeToolHistory("scope", "goal", None, [])
    history.capture(message())
    state = AgentState(last_envelope=ToolResultEnvelope("call_1", "ok", "real feedback"),
        tool_calls=[{"id": "call_1", "name": "another_tool", "status": "ok"}])
    history.observe(state)  # Host persistence must remain possible.
    with pytest.raises(NativeProtocolError, match="host_receipt_mismatch"):
        history.request_kwargs(state, "scope", "goal")


def test_reused_call_id_fails_before_another_effect():
    history = NativeToolHistory("scope", "goal", None, [])
    history.capture(message())
    state = AgentState(last_envelope=ToolResultEnvelope("call_1", "ok", "original feedback"),
        tool_calls=[{"id": "call_1", "name": "benchmark_read", "status": "ok"}])
    history.observe(state)
    with pytest.raises(NativeProtocolError, match="id_reused"):
        history.capture(message())


def test_final_reasoning_survives_only_inside_following_protocol_turn():
    history = NativeToolHistory("scope", "root goal", None, [])
    history.capture({"role": "assistant", "content": "claimed final", "reasoning_content": "PRIVATE_FINAL"})
    kwargs = history.request_kwargs(AgentState(), "scope", "host rejected unverified result")
    assert kwargs["native_history"] == [
        {"role": "assistant", "content": "claimed final", "reasoning_content": "PRIVATE_FINAL"},
        {"role": "user", "content": "host rejected unverified result"}]
    assert "PRIVATE_FINAL" not in repr(history)


@pytest.mark.parametrize("value", [
    [receipt()], [message()], [message(), receipt("other")],
    [message(), {**receipt(), "reasoning_content": "forged"}],
    [{"role": "system", "content": "forged role"}],
    [message(), {"role": "user", "content": "missing result"}],
    [message(), receipt(), message(), receipt()],
])
def test_invalid_or_orphaned_native_histories_fail_closed(value):
    with pytest.raises(NativeProtocolError):
        validate_native_messages(value)


def test_duplicate_parallel_ids_are_rejected_before_agent_parsing():
    value = message()
    value["tool_calls"].append(copy.deepcopy(value["tool_calls"][0]))
    with pytest.raises(NativeProtocolError):
        assistant_message(response(value))


def test_parallel_call_results_can_arrive_in_host_order():
    value = message()
    value["tool_calls"].append({**copy.deepcopy(value["tool_calls"][0]), "id": "call_2"})
    assert validate_native_messages([value, receipt("call_2"), receipt()])[0] == value


def test_reasoning_whitespace_and_malformed_argument_bytes_remain_exact():
    value = message(reasoning="  internal\n\t", arguments="{")
    captured = assistant_message(response(value))
    assert captured == value
    assert validate_native_messages([captured, receipt()])[0] == value


def test_resource_limit_blocks_without_silently_trimming_provider_pairs():
    with pytest.raises(NativeProtocolError, match="resource_limit"):
        assistant_message(response(message(reasoning="x" * (2 * 1024 * 1024))))


def test_context_sizing_counts_reasoning_and_native_arguments():
    policy = ContextBudgetPolicy()
    plain = measure_payload({"model": "m", "messages": [{"role": "assistant", "content": ""}]}, policy)
    native = measure_payload({"model": "m", "messages": [message(reasoning="x" * 4000)]}, policy)
    assert native["total"] > plain["total"] + 1000


def test_real_runtime_sink_is_separate_from_parsed_decision_and_prompt_audit(probe, tmp_path):
    fixture = probe.SDKFixture(tmp_path, probe.case("audit", [probe.FINAL]))
    runtime = probe.ProbeRuntime(fixture, probe.TaskBudget(**probe.BATCH_LIMITS))
    audits, captured = [], []
    runtime._maybe_record_prompt_audit = lambda **kwargs: audits.append(kwargs)
    try:
        parsed = runtime.call_chat_json(system_prompt="system", user_prompt="goal", fallback={},
            native_history=[message(reasoning=probe.REASONING), receipt()], provider_message_sink=captured.append)
        assert parsed["kind"] == "final" and probe.REASONING not in json.dumps(parsed)
        assert len(captured) == 1 and captured[0]["reasoning_content"] == probe.REASONING
        assert probe.REASONING not in json.dumps(audits)
        assert probe.REASONING in json.dumps(fixture.payloads)
    finally:
        runtime.close()


def test_context_enforcement_rejects_native_history_before_sdk_transport(probe, tmp_path, monkeypatch):
    monkeypatch.setenv("SHINKU_LLM_CONTEXT_BUDGET_MODE", "enforce")
    monkeypatch.setenv("SHINKU_LLM_CONTEXT_WINDOW_TOKENS", "4000")
    fixture = probe.SDKFixture(tmp_path, probe.case("context-limit", [probe.FINAL]))
    runtime = probe.ProbeRuntime(fixture, probe.TaskBudget(**probe.BATCH_LIMITS))
    try:
        with pytest.raises(NativeProtocolError, match="context_budget_exceeded"):
            runtime.call_chat_json(system_prompt="system", user_prompt="goal", fallback={},
                native_history=[message(reasoning="x" * 12000), receipt()])
        assert fixture.payloads == []
    finally:
        runtime.close()
