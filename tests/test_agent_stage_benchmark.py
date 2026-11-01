"""P5a paired measurement rules and real isolated file/SQLite/event integration."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module")
def benchmark():
    script = Path(__file__).resolve().parents[1] / "scripts/benchmark_agent_stages.py"
    spec = importlib.util.spec_from_file_location("stage_benchmark_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def corpus(benchmark):
    return benchmark.validate_corpus(json.loads(benchmark.CORPUS.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def report(benchmark, tmp_path_factory, corpus):
    return benchmark.run_suite(tmp_path_factory.mktemp("p5") / "paired", corpus=corpus)


@pytest.mark.parametrize("case_id", ["simple_file", "two_modules", "syntax_correction", "early_final_repair",
    "duplicate_effect", "partial_report", "false_completion", "path_escape", "unknown_tool",
    "read_only_grant", "request_exhaustion", "required_material_gap", "untrusted_source",
    "cross_slice_code", "tool_exhaustion"])
def test_every_fixed_sample_has_a_fair_passing_pair(report, case_id):
    pair = [r for r in report["rows"] if r["case"] == case_id]
    assert {r["arm"] for r in pair} == {"react", "staged"}
    assert len({(r["model"], r["case_hash"], r["config_hash"]) for r in pair}) == 1
    for row in pair:
        assert row["fixture_pass"] and row["contract_preserved"] and row["query_unchanged"], row
        assert not row["false_accepted"] and not row["false_completed"]
        assert row["duplicate_effects"] == row["duplicate_publications"] == 0
        assert row["model_calls"] == row["budget"]["requests"] <= row["budget"]["max_requests"]
        assert not row["tokens_are_provider_usage"] and row["provider_cost"] is None
        assert row["real_model_calls"] == row["expression_model_calls"] == row["real_platform_sends"] == 0


def test_simple_samples_have_no_stage_overhead_and_complex_extra_input_is_charged(report):
    simple = [r for r in report["rows"] if r["case"] == "simple_file"]
    assert all(r["stages"] == [] for r in simple)
    assert simple[0]["model_calls"] == simple[1]["model_calls"]
    # Host timestamps/elapsed counters can change JSON byte length. Measure
    # actual added stage input separately instead of assuming byte equality.
    assert simple[0]["scaffold_bytes"] == simple[1]["scaffold_bytes"] == 0
    complex_pair = {r["arm"]: r for r in report["rows"] if r["case"] == "two_modules"}
    assert complex_pair["staged"]["budget"]["tokens"] > complex_pair["react"]["budget"]["tokens"]
    assert complex_pair["staged"]["prompt_bytes"] > complex_pair["react"]["prompt_bytes"]
    assert complex_pair["staged"]["model_calls"] == complex_pair["react"]["model_calls"]
    assert complex_pair["staged"]["stages"][-1][0]["status"] == "completed"
    assert complex_pair["staged"]["stages"][-1][1]["status"] == "completed"


def test_missing_material_and_denied_authority_create_no_file_effect(report):
    for row in report["rows"]:
        if row["case"] in {"path_escape", "unknown_tool", "read_only_grant", "required_material_gap"}:
            assert row["writes"] == []
        if row["case"] == "required_material_gap":
            assert row["model_calls"] == row["publications"] == 0
        if row["case"] == "request_exhaustion":
            assert row["model_calls"] == 1 and len(row["writes"]) == 1
            assert "budget" in row["reason"]
        if row["case"] == "tool_exhaustion":
            assert row["budget"]["tools"] == 1 and len(row["writes"]) == 1
            assert "budget" in row["reason"]


def test_cross_slice_preserves_one_task_contract_and_cumulative_usage(report):
    for row in report["rows"]:
        if row["case"] == "cross_slice_code":
            assert row["model_calls"] == row["budget"]["requests"] == 5
            assert row["budget"]["tools"] == 4 and len(row["writes"]) == 3
            assert row["status"] == "completed" and row["contract_preserved"]


def test_failed_or_uncertain_delivery_is_not_repeated_or_upgraded(report):
    for row in report["rows"]:
        if row["case"] in {"duplicate_effect", "partial_report"}:
            assert row["publications"] == 1
            assert row["delivery"] == (["uncertain"] if row["case"] == "duplicate_effect" else ["failed"])


def test_actual_artifact_disagrees_with_forged_completed_result(benchmark, corpus, tmp_path):
    case = corpus["cases"][0]
    # Even a host/model claim of acceptance cannot override the absent artifact.
    result = {"execution": {"status": "completed"}, "verification": {"accepted": True}}
    row = benchmark.assess(case, result, benchmark.independent_artifacts(tmp_path, case), [], 0,
        outside_unchanged=True, query_unchanged=True)
    assert row["false_completed"] and row["false_accepted"] and not row["fixture_pass"]


def test_accepted_honest_failure_is_not_confused_with_verified_goal_success(benchmark, corpus, tmp_path):
    case = next(c for c in corpus["cases"] if c["id"] == "partial_report")
    result = {"execution": {"status": "partial"}, "verification": {"accepted": True}}
    row = benchmark.assess(case, result, benchmark.independent_artifacts(tmp_path, case), [], 0,
        outside_unchanged=True, query_unchanged=True)
    assert row["fixture_pass"] and not row["task_success"] and not row["false_accepted"]


def test_missing_verification_is_not_treated_as_acceptance(benchmark, corpus, tmp_path):
    case = next(c for c in corpus["cases"] if c["id"] == "request_exhaustion")
    result = {"execution": {"status": "blocked"}, "verification": {}}
    row = benchmark.assess(case, result, benchmark.independent_artifacts(tmp_path, case), [], 0,
        outside_unchanged=True, query_unchanged=True)
    assert row["fixture_pass"] and not row["task_success"] and not row["false_accepted"]


def test_existing_valid_file_without_effect_receipt_does_not_prove_a_write(benchmark, corpus, tmp_path):
    case = corpus["cases"][0]
    for name, content in case["expected"].items():
        (tmp_path / name).write_bytes(content.encode())
    result = {"execution": {"status": "completed"}, "verification": {"accepted": True}}
    row = benchmark.assess(case, result, benchmark.independent_artifacts(tmp_path, case), [], 0,
        outside_unchanged=True, query_unchanged=True)
    assert row["false_completed"] and not row["fixture_pass"]


def test_identical_write_and_second_publication_are_counted_even_on_success(benchmark, corpus, tmp_path):
    case = corpus["cases"][0]
    for name, content in case["expected"].items():
        (tmp_path / name).write_bytes(content.encode())
    result = {"execution": {"status": "completed"}, "verification": {"accepted": True}}
    writes = [{"path": "ack.txt", "sha256": "same"}] * 2
    row = benchmark.assess(case, result, benchmark.independent_artifacts(tmp_path, case), writes, 2,
        outside_unchanged=True, query_unchanged=True)
    assert row["duplicate_effects"] == row["duplicate_publications"] == 1
    assert not row["fixture_pass"]


def test_stage_needs_receipt_and_dependency_and_reverts_after_artifact_change(benchmark, corpus, tmp_path):
    case = next(c for c in corpus["cases"] if c["id"] == "two_modules")
    for name, content in case["expected"].items():
        (tmp_path / name).write_bytes(content.encode())
    assert not any(s["status"] == "completed" for s in benchmark.stage_snapshot(tmp_path, case, []))
    calls = [{"path": n, "sha256": benchmark.digest(c.encode())} for n, c in case["expected"].items()]
    before = benchmark.stage_snapshot(tmp_path, case, calls)
    assert all(s["status"] == "completed" for s in before)
    (tmp_path / "math_ops.py").write_text("changed", encoding="utf-8")
    after = benchmark.stage_snapshot(tmp_path, case, calls, before)
    assert all(s["status"] != "completed" and s["replan_reason"] for s in after)


@pytest.mark.parametrize("mutation", ["cycle", "drop_requirement", "outside_artifact", "raise_budget", "duplicate_id"])
def test_invalid_frozen_plan_or_expanded_budget_is_rejected_before_io(benchmark, corpus, mutation):
    bad = copy.deepcopy(corpus)
    case = next(c for c in bad["cases"] if c["id"] == "two_modules")
    if mutation == "cycle":
        case["stages"][0]["depends"] = ["S02"]
    elif mutation == "drop_requirement":
        case["stages"].pop()
    elif mutation == "outside_artifact":
        case["stages"][0]["artifacts"] = ["../outside.txt"]
    elif mutation == "raise_budget":
        case["budget"] = {"max_requests": bad["budget"]["max_requests"] + 1}
    else:
        bad["cases"][1]["id"] = bad["cases"][0]["id"]
    with pytest.raises(ValueError):
        benchmark.validate_corpus(bad)


def test_missing_duplicate_or_unfair_pairs_cannot_produce_successful_summary(benchmark, corpus, report):
    with pytest.raises(ValueError, match="incomplete_or_duplicate_pairs"):
        benchmark.summarize(corpus, report["rows"][:-1], 1)
    bad = copy.deepcopy(report["rows"])
    bad[0]["config_hash"] = "a-different-host-grant"
    with pytest.raises(ValueError, match="unfair_pair_configuration"):
        benchmark.summarize(corpus, bad, 1)
    bad = copy.deepcopy(report["rows"])
    bad[-1] = bad[0]
    with pytest.raises(ValueError, match="incomplete_or_duplicate_pairs"):
        benchmark.summarize(corpus, bad, 1)


def test_quality_and_cost_claims_remain_unmeasured(report):
    assert report["all_pairs_present"] and report["quality_conclusion"] == "not_measured"
    assert report["enable_production"] is False
    assert all(arm["provider_cost"] is None for arm in report["arms"].values())
    assert len(report["rows"]) == 30


def test_existing_workspace_is_not_overwritten(benchmark, tmp_path):
    (tmp_path / "keeper.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        benchmark.run_suite(tmp_path)
    assert (tmp_path / "keeper.txt").read_text() == "keep"


def test_standalone_cli_has_no_qq_persona_private_or_network_dependency(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = '''import importlib.abc, runpy, socket, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == n or fullname.startswith(n + '.') for n in ('shinku.qq', 'shinku.persona', 'shinku_private')):
            raise AssertionError('Forbidden benchmark dependency: ' + fullname)
sys.meta_path.insert(0, Block())
socket.socket.connect = lambda *a: (_ for _ in ()).throw(AssertionError('Network forbidden'))
sys.argv = [sys.argv[1], '--workspace', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    process = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code,
        str(root / "scripts/benchmark_agent_stages.py"), str(tmp_path / "cli")],
        env={**os.environ, "PYTHONPATH": str(root / "src"), "SHINKU_ENV_FILE": str(tmp_path / "missing.env")},
        capture_output=True, text=True, encoding="utf-8", timeout=40)
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["all_pairs_present"] and summary["quality_conclusion"] == "not_measured"
    assert all(row["fixture_passes"] == 15 for row in summary["arms"].values())
