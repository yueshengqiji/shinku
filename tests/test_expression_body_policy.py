"""Versioned body policy preserves full scoped truth and strict review."""
import copy
import json

import pytest

from shinku.agent_runtime.results import seal_result
from shinku.result_expression import ExpressionProfile
from shinku.result_expression.natural import paragraph_blocks
from test_expression_paragraphs import paragraphs, sdk_pair, service, source, STYLE
from test_result_expression import host, SCOPE


def body_source(**changes):
    value = source(); value.pop('result_hash')
    value['delivery'] = dict(status='not_sent', sent=False)
    value.update(changes)
    return seal_result(value)


def body(payload):
    value = paragraphs(payload)
    value['paragraphs'][0] = dict(text='测量和报告已完成，这个目标的验收已通过。', fact_ids=['status', 'verification'])
    value['paragraphs'][1]['text'] = ('离线隔离环境下的响应时间是12.5 ms。'
        '响应时间的记录来源为https://example.invalid/source，条件为离线隔离环境。')
    delivery = next(b for b in payload['fact_blocks'] if b['id'] == 'delivery')
    if not delivery.get('required', True):
        value['paragraphs'][3]['fact_ids'].remove('delivery')
        value['paragraphs'][3]['text'] = value['paragraphs'][3]['text'].replace('送达未知；本次新表达尚未取得完整送达回执。', '')
    return value


def svc(host, model, checker, **options):
    return service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=2, **options)


def test_sdk_body_can_omit_goal_and_initial_delivery_but_checker_sees_full_truth(host):
    value = body_source(); before = copy.deepcopy(value)
    with sdk_pair(body) as (model, checker, calls):
        instance = svc(host, model, checker)
        record = instance.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready', record['fallback_reason']
        assert len(calls) == 2 and record['paragraph_presentation_version'] == 2
        assert '原始目标' not in record['text'] and '送达' not in record['text']
        assert '12.5 ms' in record['text'] and 'https://example.invalid/source' in record['text']
        payload = calls[1][2]
        assert {'goal', 'delivery'}.issubset({b['id'] for b in payload['context_fact_blocks']})
        assert not {'goal', 'delivery'} & {b['id'] for b in payload['fact_blocks']}
        assert '测量响应时间并生成报告，不部署' in json.dumps(payload, ensure_ascii=False)
        assert STYLE not in json.dumps(calls[1]) and 'EXECUTION_MODEL_PRIVATE_SENTINEL' not in json.dumps(calls)
        assert record['canonical_result'] == value == before and record['delivery']['status'] == 'not_sent'
        details = instance.read_details(record['expression_id'], scope=SCOPE, check_current=lambda: '')
        assert details['status'] == 'available' and details['result_hash'] == value['result_hash']
        assert {'goal', 'delivery'}.issubset({b['id'] for b in details['fact_blocks']})
        assert '测量响应时间并生成报告，不部署' in json.dumps(details, ensure_ascii=False)
        assert instance.prepare(value, scope=SCOPE, check_current=lambda: '') == record and len(calls) == 2


@pytest.mark.parametrize('delivery', [
    {'status': 'unknown', 'sent': False}, {'status': 'uncertain', 'sent': False},
    {'status': 'failed', 'sent': False}, {'status': 'sent', 'sent': True},
    {'status': 'not_recorded', 'sent': False}, {'status': 'pending', 'sent': False},
    {'status': 'not_sent', 'sent': False, 'message_ids': ['actual-attempt']},
])
def test_actual_or_unknown_delivery_remains_mandatory(delivery):
    blocks = paragraph_blocks(body_source(delivery=delivery), detail_policy_version=2, presentation_version=2)
    assert next(b for b in blocks if b['id'] == 'delivery')['required']


@pytest.mark.parametrize('extra', [
    {'contract': {'requirements': [{'id': 'send', 'effect': 'send'}]}},
    {'evidence': [{'tool': 'sender', 'execution_status': 'ok', 'receipt': {'delivery_receipt': {'sent': True}}}]},
])
def test_send_contract_or_actual_receipt_prevents_initial_delivery_from_becoming_optional(extra):
    blocks = paragraph_blocks(body_source(**extra), detail_policy_version=2, presentation_version=2)
    assert next(b for b in blocks if b['id'] == 'delivery')['required']


@pytest.mark.parametrize('missing', ['status', 'verification', 'constraints', 'fact_0', 'artifact_0', 'uncertainties', 'deployment'])
def test_critical_body_facts_cannot_be_omitted_even_with_positive_checker(host, missing):
    def build(payload):
        result = body(payload)
        for row in result['paragraphs']:
            row['fact_ids'] = [v for v in row['fact_ids'] if v != missing]
        return result
    with sdk_pair(build) as (model, checker, calls):
        record = svc(host, model, checker).prepare(body_source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_required_facts_missing_or_duplicated'
        assert len(calls) == 1 and record['candidate'] is None


@pytest.mark.parametrize('old,new', [('12.5 ms', '12.6 ms'), ('12.5 ms', '12.5 ms²'),
    ('https://example.invalid/source', 'https://example.invalid/other'), ('fixture-report.txt', 'fixture-report.txt.exe')])
def test_body_policy_does_not_relax_exact_protected_values(host, old, new):
    def build(payload):
        result = body(payload)
        for row in result['paragraphs']:
            row['text'] = row['text'].replace(old, new)
        return result
    with sdk_pair(build) as (model, checker, calls):
        record = svc(host, model, checker).prepare(body_source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_protected_value_missing' and len(calls) == 1


@pytest.mark.parametrize('field,claim', [('goal', '现在已经部署了。'), ('delivery', '这次结果已经发送。')])
def test_omitted_context_still_rejects_contradictory_prose(host, field, claim):
    def build(payload):
        result = body(payload); result['paragraphs'][0]['text'] += claim; return result
    with sdk_pair(build, verdict=dict(supported=False, contradictions=[field])) as (model, checker, calls):
        value = body_source()
        record = svc(host, model, checker).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_neutral_check_rejected' and len(calls) == 2
        assert field in {b['id'] for b in calls[1][2]['context_fact_blocks']}
        assert record['canonical_result'] == value and claim not in record['text']


def test_semantic_constraints_and_remaining_gaps_are_checked_independently(host):
    def build(payload):
        result = body(payload); result['paragraphs'][1]['text'] += '这个结果也适用于任何联网环境。'; return result
    with sdk_pair(build, verdict=dict(supported=False, contradictions=['scope'])) as (model, checker, calls):
        record = svc(host, model, checker).prepare(body_source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_neutral_check_rejected' and len(calls) == 2
    partial = body_source(execution={'status': 'partial', 'reason': 'fixture_gap', 'result_available': True},
        verification={'accepted': False, 'satisfied': [], 'missing': ['报告内容核验']}, remaining=['报告内容核验'])
    blocks = paragraph_blocks(partial, detail_policy_version=2, presentation_version=2)
    assert next(b for b in blocks if b['id'] == 'remaining')['required']
    assert any(a['text'] == '报告内容核验' for a in next(b for b in blocks if b['id'] == 'remaining')['anchors'])


def test_legacy_policy_is_unchanged_and_new_policy_does_not_reuse_old_expression(host):
    value = body_source()
    old_blocks = paragraph_blocks(value, detail_policy_version=2)
    assert next(b for b in old_blocks if b['id'] == 'goal')['required']
    assert next(b for b in old_blocks if b['id'] == 'delivery')['required']
    with sdk_pair() as (model, checker, calls):
        old = service(host, model, checker, detail_policy_version=2)
        record = old.prepare(value, scope=SCOPE, check_current=lambda: '')
        before = len(calls)
        current = svc(host, model, checker)
        assert current.profile.config_hash != old.profile.config_hash
        result = current.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'discarded' and result['fallback_reason'] == 'expression_configuration_changed'
        assert len(calls) == before and old.archive.get(record['expression_id'], scope=SCOPE) == record


@pytest.mark.parametrize('options', [dict(paragraph_presentation_version=True), dict(paragraph_presentation_version=5),
    dict(paragraph_presentation_version=2), dict(paragraph_presentation_version=2, format_version=2)])
def test_new_body_policy_requires_explicit_compatible_profile(options):
    with pytest.raises(ValueError, match='invalid_expression_presentation_policy'):
        ExpressionProfile(**options)
