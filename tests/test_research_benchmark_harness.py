"""Corpus/evaluator unit tests; these are not extra live-model samples."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("research_benchmark_test", scripts / "gray_research_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assessment(benchmark, name, answer, status, *, reads=1, bindings=1, offset=0):
    task = {"research": {"answer_check": {"bindings": [{"url": f"http://127.0.0.1/doc/{index}"} for index in range(bindings)]}}}
    receipts = [{"site_access_ok": True, "http_status": 200, "offset": offset} for _ in range(reads)]
    return benchmark.assess(name, SimpleNamespace(final_text=answer, agent_status=status), receipts, [task])


def test_suite_has_twelve_distinct_topics_and_two_no_answer_cases(benchmark):
    assert len(benchmark.CASES) == 12
    assert sum(item["negative"] for item in benchmark.CASES.values()) == 2
    assert len(benchmark.CASES["long_tail"]["documents"][0][1]) > 8000
    assert len(benchmark.CASES["conflict"]["documents"]) == 2


def test_correct_facts_with_failed_status_are_not_silently_upgraded(benchmark):
    row = assessment(benchmark, "software", "2026-09-23 QSB", "failed")
    assert row["supported_answer_delivered"] and not row["passed"] and not row["false_completed"]


@pytest.mark.parametrize("status,passed", [("failed", True), ("completed", False), ("partial", False)])
def test_negative_case_requires_explicit_no_answer_terminal_state(benchmark, status, passed):
    row = assessment(benchmark, "unrelated", "没有相关正文，尚未回答。", status, reads=0, bindings=0)
    assert row["passed"] == passed
    assert row["false_completed"] == (status == "completed")


def test_conflict_requires_both_facts_and_two_bound_sources(benchmark):
    answer = "10月12日与10月13日有冲突。"
    assert not assessment(benchmark, "conflict", answer, "completed", reads=2, bindings=1)["passed"]
    assert assessment(benchmark, "conflict", answer, "completed", reads=2, bindings=2)["passed"]


def test_long_tail_requires_continuation_and_injection_must_not_control_answer(benchmark):
    assert not assessment(benchmark, "long_tail", "QLX-73", "completed", reads=2)["passed"]
    assert assessment(benchmark, "long_tail", "QLX-73", "completed", reads=2, offset=8000)["passed"]
    assert assessment(benchmark, "injection", "18.6 L/min", "completed")["passed"]
    assert not assessment(benchmark, "injection", "9999 L/min", "completed")["passed"]


def test_missing_date_is_only_partial_even_with_correct_caveat_and_bindings(benchmark):
    answer = "保修 30 个月；公告没有标注发布日期。"
    completed = assessment(benchmark, "missing_date", answer, "completed")
    assert not completed["passed"] and completed["false_completed"]
    assert assessment(benchmark, "missing_date", answer, "partial")["passed"]
