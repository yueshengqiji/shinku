"""成对评估只用合成用量和固定审查，不产生模型请求。"""
from copy import deepcopy
from decimal import Decimal

import pytest

from shinku.agent_runtime.evaluation import compare_reviews, estimate_call_cny, provider_usage, summarize_calls


def test_missing_usage_is_unknown_not_budget_zero():
    assert set(provider_usage({}).values()) == {None}
    usage = provider_usage({'usage': {'prompt_tokens': 20, 'completion_tokens': 8,
        'cache_creation_input_tokens': 20, 'completion_tokens_details': {'reasoning_tokens': 5}}})
    assert usage['reasoning_tokens'] == 5 and usage['cache_miss_tokens'] is None
    assert usage['cache_hit_tokens'] is None


def test_weighted_cache_ratio_and_failed_calls_are_included():
    rows = [{'request_id': str(i), 'status': s, 'elapsed_seconds': 1, 'usage': provider_usage({'usage': u})}
            for i, (s, u) in enumerate([
                ('completed', {'prompt_tokens': 100, 'completion_tokens': 8, 'prompt_cache_hit_tokens': 90}),
                ('failed', {'prompt_tokens': 900, 'completion_tokens': 2, 'prompt_cache_hit_tokens': 0}),
            ])]
    result = summarize_calls(rows)
    assert result['cache_hit_ratio'] == .09 and result['totals']['output_tokens'] == 10
    rows.append({'request_id': 'timeout', 'status': 'timeout', 'usage': {}})
    result = summarize_calls(rows)
    assert result['totals']['input_tokens'] is None and result['known_subtotals']['input_tokens'] == 1000
    assert result['missing_by_field']['input_tokens'] == ['timeout'] and result['sum_call_seconds'] is None
    with pytest.raises(ValueError, match='duplicate'):
        summarize_calls(rows + [rows[0]])


@pytest.mark.parametrize('bad', [True, -1, '2', float('nan')])
def test_invalid_usage_cannot_be_silently_summed(bad):
    with pytest.raises(ValueError):
        summarize_calls([{'request_id': '1', 'status': 'completed', 'usage': {'input_tokens': bad}}])


def test_price_snapshot_no_double_charge_for_reasoning():
    usage = provider_usage({'usage': {'prompt_tokens': 100, 'completion_tokens': 50,
        'prompt_cache_hit_tokens': 80, 'prompt_cache_miss_tokens': 20,
        'completion_tokens_details': {'reasoning_tokens': 40}}})
    price = {'source': 'synthetic-price', 'effective_at': 'fixture', 'model': 'fixture-v1', 'period': 'fixture',
             'hit_cny_per_million': 1, 'miss_cny_per_million': 2, 'output_cny_per_million': 4}
    assert Decimal(estimate_call_cny(usage, price)) == Decimal('0.00032')
    usage['cache_miss_tokens'] = 21
    assert estimate_call_cny(usage, price) is None


def review(**changes):
    return {'reviewer': 'blind-reviewer-A', 'evidence': 'synthetic/answer-and-artifact', 'hard_guards': True,
            'checks': {'answers_request': 'pass', 'evidence': 'pass'}, 'total_cny': '1', **changes}


@pytest.mark.parametrize('candidate,expected', [
    (review(total_cny='.7'), 'adopt_with_recorded_scope'),
    (review(total_cny='.95'), 'no_stable_gain_keep_baseline'),
    (review(total_cny=None), 'cost_unknown_keep_baseline'),
    (review(hard_guards=False), 'insufficient_evidence_or_hard_guard_failure'),
    (review(checks={'answers_request': 'pass', 'evidence': 'fail'}, total_cny='.1'), 'reject_quality_regression'),
])
def test_quality_is_not_bought_with_hidden_regression(candidate, expected):
    before = deepcopy(candidate)
    assert compare_reviews(review(), candidate) == expected
    assert candidate == before


def test_quality_improvement_cost_increase_is_explicit_tradeoff():
    old = review(checks={'answers_request': 'fail', 'evidence': 'pass'})
    assert compare_reviews(old, review(total_cny='1.5')) == 'quality_cost_tradeoff_needs_decision'
    assert compare_reviews(old, review(total_cny='1.03')) == 'adopt_with_recorded_scope'


def test_prepare_freezes_synthetic_inputs_without_calling_models_or_overwriting(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'scripts/evaluate_agent_quality_cost.py'
    spec = importlib.util.spec_from_file_location('quality_cost_fixture_driver', script)
    driver = importlib.util.module_from_spec(spec); spec.loader.exec_module(driver)
    output = tmp_path/'prepared'
    result = driver.prepare(output)
    manifest = json.loads((output/'manifest.json').read_text())
    assert result['external_requests'] == 0 and result['cases'] == 10
    assert manifest['actual_cost_cny'] is None and all(r['status'] == 'not_run' for r in manifest['records'])
    material = (output/'workspace/material.txt').read_text()
    assert len(material) > 16000 and all(v in material for v in ('约束一','约束二','约束三'))
    with pytest.raises(FileExistsError): driver.prepare(output)
