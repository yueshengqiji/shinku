"""P2b preparation probes actual SDK requests; no provider or production config."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def probe():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_agent_execution_protocol.py"
    spec = importlib.util.spec_from_file_location("execution_protocol_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def report(probe, tmp_path_factory):
    return probe.run_suite(tmp_path_factory.mktemp("protocol-parent") / "protocol")


@pytest.mark.parametrize("name", [
    "native_two_tools", "failed_tool_corrected", "cross_slice", "malformed_tool", "timeout",
    "retry_then_success", "false_final", "unknown_tool", "denied_path", "long_attachment_tail",
    "image_supported", "image_unsupported"])
def test_original_host_contract_and_independent_artifact_rules_pass(report, name):
    row = next(r for r in report["rows"] if r["case"] == name)
    assert row["fixture_pass"], row
    assert not row["false_completed"] and not row["false_accepted"]
    assert row["duplicate_effects"] == row["duplicate_publications"] == 0
    assert row["original_contract_preserved"] and row["query_unchanged"]
    assert row["reasoning_excluded_from_result"]


def test_two_actual_native_tool_calls_and_prior_context_reach_sdk(report):
    row = next(r for r in report["rows"] if r["case"] == "native_two_tools")
    assert row["sdk_requests"] == 3 and len(row["writes"]) == 1
    assert row["tool_read_attempts"] == ["spec.txt"]
    assert row["prior_context_on_wire"] and row["confirmed_constraint_on_wire"]
    assert row["budget"]["requests"] == 3
    assert row["budget"]["tokens"] == row["reported_fixture_usage"] == 300


def test_real_missing_file_failure_is_corrected_before_verified_write(report):
    row = next(r for r in report["rows"] if r["case"] == "failed_tool_corrected")
    assert row["tool_read_attempts"] == ["missing.txt", "spec.txt"]
    assert row["task_success"] and len(row["writes"]) == 1


def test_cross_slice_uses_original_receipt_and_same_cumulative_ledger(report):
    row = next(r for r in report["rows"] if r["case"] == "cross_slice")
    assert row["slices"] >= 2 and row["sdk_requests"] == row["budget"]["requests"] == 3
    assert row["read_result_on_later_wire"] and row["task_success"]
    assert row["slice_outcomes"] == ["yielded", "yielded", "completed"]


def test_malformed_native_arguments_fail_closed_before_tool_effect(report):
    row = next(r for r in report["rows"] if r["case"] == "malformed_tool")
    assert row["status"] == "failed" and row["reason"] == "invalid_decision"
    assert row["sdk_requests"] == 1 and row["writes"] == [] and row["budget"]["tools"] == 0


def test_timeout_and_sdk_retry_attempts_remain_distinct_from_exact_usage(report):
    timeout = next(r for r in report["rows"] if r["case"] == "timeout")
    retry = next(r for r in report["rows"] if r["case"] == "retry_then_success")
    assert timeout["sdk_requests"] == timeout["budget"]["requests"] == 2
    assert timeout["budget"]["tokens"] > 0 and timeout["reported_fixture_usage"] == 0
    assert timeout["writes"] == [] and not timeout["task_success"]
    assert retry["sdk_requests"] == retry["budget"]["requests"] == 3
    assert retry["reported_fixture_usage"] == 200
    assert retry["budget"]["tokens"] > retry["reported_fixture_usage"]


def test_actual_original_image_bytes_and_author_are_on_wire_only_when_supported(report):
    supported = next(r for r in report["rows"] if r["case"] == "image_supported")
    unsupported = next(r for r in report["rows"] if r["case"] == "image_unsupported")
    assert supported["image_bytes_on_wire"] and supported["image_source_on_wire"]
    assert unsupported["sdk_requests"] == 0 and unsupported["writes"] == []
    assert unsupported["status"] == "waiting_user" and not unsupported["image_bytes_on_wire"]
    assert unsupported["reason"] == "execution_images_unsupported"


def test_clipped_attachment_is_read_via_real_material_adapter_before_effect(report):
    row = next(r for r in report["rows"] if r["case"] == "long_attachment_tail")
    assert row["attachment_tail_on_wire"] and row["sdk_requests"] == 3
    assert row["budget"]["tools"] == 2 and row["task_success"]
    assert row["material_read_dispatches"] == 1


def test_batch_attempts_include_failed_reads_and_material_adapter_not_verification(report):
    assert report["batch"]["requests"] == sum(r["sdk_requests"] for r in report["rows"]) \
        + report["sse"]["sdk_requests"] + report["history_fields"]["sdk_requests"]
    assert report["batch"]["tools"] == sum(len(r["tool_read_attempts"]) + len(r["writes"])
        + r["material_read_dispatches"] for r in report["rows"])
    assert report["batch_tools_accounting"] == "actual_fixture_file_and_material_dispatches"


def test_protocol_gaps_cannot_be_upgraded_by_successful_file_fixtures(report):
    assert report["fixture_gates_pass"]
    assert set(report["native_thinking_continuation_blockers"]) == {
        "assistant_native_history_preserved", "tool_role_and_id_preserved",
        "reasoning_history_preserved", "provider_message_retained"}
    assert all(not report["history_fields"][key] for key in report["native_thinking_continuation_blockers"])
    assert report["history_fields"]["reasoning_excluded_from_public_response"]
    assert not report["ready_for_thinking_tool_continuation"]
    assert report["provider_acceptance"] == "not_run" and report["provider_cost"] is None
    assert report["model_quality"] == "not_measured"
    assert report["real_model_calls"] == report["real_platform_sends"] == 0
    assert report["enable_production"] is False


def test_sdk_sse_parsing_accounts_usage_and_does_not_emit_reasoning(report):
    stream = report["sse"]
    assert stream["sdk_requests"] == 1 and stream["sent_stream"] and stream["final_json_parsed"]
    assert stream["usage"] == 100 and stream["reasoning_excluded"]
    assert stream["native_thinking_stream_continuity"] == "not_measured"


@pytest.mark.parametrize("limit", ["max_requests", "max_tokens", "max_active_seconds"])
def test_real_sdk_dispatch_stops_at_batch_limit_before_transport(probe, tmp_path, limit):
    limits = dict(probe.BATCH_LIMITS)
    limits[limit] = 1
    batch = probe.TaskBudget(**limits)
    if limit == "max_active_seconds":
        batch.active_seconds = 2  # Controlled elapsed time; no sleep.
    fixture = probe.SDKFixture(tmp_path, probe.case("bounded-sdk", [probe.FINAL]))
    runtime = probe.ProbeRuntime(fixture, batch)
    from shinku.agent.budget import TaskBudgetExceeded
    try:
        with probe.bind_task_budget(probe.TaskBudget(**probe.TASK_LIMITS)):
            if limit == "max_requests":
                runtime.call_chat_json(system_prompt="fixture", user_prompt="fixture", fallback={"kind": "final"})
            with pytest.raises(TaskBudgetExceeded):
                runtime.call_chat_json(system_prompt="fixture", user_prompt="fixture", fallback={"kind": "final"})
        assert len(fixture.payloads) == (1 if limit == "max_requests" else 0)
    finally:
        runtime.close()


def test_batch_tool_limit_refuses_second_actual_file_effect(probe, tmp_path):
    batch = probe.TaskBudget(**{**probe.BATCH_LIMITS, "max_tools": 1})
    sample = probe.case("bounded-tools", [probe.WRITE])
    writer = probe.benchmark.FileAdapter(tmp_path.resolve(), sample, "write", batch)
    path = tmp_path / "output.txt"
    call = {"path": str(path), "content": "verified fixture\n"}
    writer.execute(call=call, context={})
    from shinku.agent.budget import TaskBudgetExceeded
    with pytest.raises(TaskBudgetExceeded):
        writer.execute(call={**call, "content": "unapproved second effect"}, context={})
    assert path.read_text() == "verified fixture\n" and len(writer.calls) == batch.tools == 1


def test_production_provider_environment_is_not_used_by_fixture_runtime(probe, monkeypatch, tmp_path):
    monkeypatch.setenv("SHINKU_CHAT_API_KEY", "production-key-sentinel")
    monkeypatch.setenv("SHINKU_AUX_API_KEY", "production-key-sentinel")
    monkeypatch.setenv("SHINKU_CHAT_BASE_URL", "https://production-sentinel.invalid/v1")
    monkeypatch.setenv("SHINKU_CHAT_MODEL_NAME", "production-model-sentinel")
    fixture = probe.SDKFixture(tmp_path, probe.case("environment-isolation", [probe.FINAL]))
    runtime = probe.ProbeRuntime(fixture, probe.TaskBudget(**probe.BATCH_LIMITS))
    try:
        with probe.bind_task_budget(probe.TaskBudget(**probe.TASK_LIMITS)):
            response = runtime.call_chat_json(system_prompt="fixture", user_prompt="fixture", fallback={"kind": "final"})
        assert response["kind"] == "final" and fixture.payloads[0]["model"] == probe.MODEL
        assert str(runtime.chat.client.base_url) == probe.BASE_URL + "/"
    finally:
        runtime.close()


def candidate(probe):
    return {"model": "candidate-model", "base_url": "https://provider.example/v1", "mode": "thinking",
        "limits": dict(probe.BATCH_LIMITS), "samples": [c["id"] for c in probe.CASES]}


def test_valid_review_draft_does_not_authorize_or_dispatch_a_live_request(probe):
    original = candidate(probe)
    reviewed = probe.validate_candidate(original)
    assert reviewed["draft_only"] and reviewed["live_authorized"] is False
    assert reviewed["real_model_calls"] == 0 and "draft_only" not in original
    assert reviewed["limits"] is not original["limits"]


@pytest.mark.parametrize("mutation", ["api_key", "unknown_field", "userinfo", "query_secret", "fragment",
    "remote_http", "invalid_port", "invalid_model", "unknown_mode", "empty_samples", "duplicate_samples",
    "unknown_sample", "raised_budget", "boolean_count", "fractional_count", "nan_time", "infinite_time"])
def test_invalid_or_credential_bearing_candidate_is_rejected_before_workspace_io(probe, tmp_path, mutation):
    draft = candidate(probe)
    if mutation in {"api_key", "unknown_field"}:
        draft[mutation] = "fixture-secret"
    elif mutation == "userinfo":
        draft["base_url"] = "https://name:fixture-secret@provider.example/v1"
    elif mutation == "query_secret":
        draft["base_url"] += "?api_key=fixture-secret"
    elif mutation == "fragment":
        draft["base_url"] += "#ignored"
    elif mutation == "remote_http":
        draft["base_url"] = "http://provider.example/v1"
    elif mutation == "invalid_port":
        draft["base_url"] = "https://provider.example:99999/v1"
    elif mutation == "invalid_model":
        draft["model"] = "model\nextra"
    elif mutation == "unknown_mode":
        draft["mode"] = "provider-default"
    elif mutation == "empty_samples":
        draft["samples"] = []
    elif mutation == "duplicate_samples":
        draft["samples"].append(draft["samples"][0])
    elif mutation == "unknown_sample":
        draft["samples"].append("production-request")
    elif mutation == "raised_budget":
        draft["limits"]["max_requests"] += 1
    elif mutation == "boolean_count":
        draft["limits"]["max_requests"] = True
    elif mutation == "fractional_count":
        draft["limits"]["max_requests"] = 3.2
    else:
        draft["limits"]["max_active_seconds"] = float("nan" if mutation == "nan_time" else "inf")
    root = tmp_path / "invalid"
    with pytest.raises(ValueError):
        probe.run_suite(root, candidate=draft)
    assert not root.exists()


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:9999/v1", "http://localhost:9999/v1", "http://[::1]:9999/v1"])
def test_explicit_loopback_endpoint_can_be_reviewed_without_network(probe, endpoint):
    draft = candidate(probe)
    draft["base_url"] = endpoint
    assert probe.validate_candidate(draft)["live_authorized"] is False


def test_existing_workspace_and_named_keeper_are_preserved(probe, tmp_path):
    keeper = tmp_path / "keeper.txt"
    keeper.write_text("retain", encoding="utf-8")
    with pytest.raises(FileExistsError):
        probe.run_suite(tmp_path)
    assert keeper.read_text() == "retain"


def test_standalone_cli_blocks_network_qq_persona_private_imports_and_live_options(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/verify_agent_execution_protocol.py"
    code = '''import importlib.abc, runpy, socket, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + '.') for n in ('shinku.qq','shinku.persona','shinku_private')):
            raise AssertionError('Forbidden protocol dependency: ' + fullname)
sys.meta_path.insert(0, Block())
socket.socket.connect = lambda *a: (_ for _ in ()).throw(AssertionError('Network forbidden'))
sys.argv = [sys.argv[1], '--workspace', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "SHINKU_ENV_FILE": str(tmp_path / "missing.env")}
    process = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code, str(script), str(tmp_path / "cli")],
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=40)
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["fixture_gates_pass"] and summary["real_model_calls"] == 0
    forbidden = subprocess.run([sys.executable, "-B", str(script), "--workspace", str(tmp_path / "live"), "--live"],
        env=env, capture_output=True, text=True, encoding="utf-8", timeout=10)
    assert forbidden.returncode == 2 and not (tmp_path / "live").exists()
