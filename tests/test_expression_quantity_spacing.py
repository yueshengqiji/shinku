"""Quantity formatting preserves facts without bypassing semantic review."""
import copy

import pytest

from test_expression_paragraphs import sdk_pair, service
from test_expression_receipt_voice import readable, receipt_source
from test_result_expression import SCOPE, host


def quantity_candidate(payload, *, count='5次', measurement='12.5 ms'):
    value = readable(payload)
    for paragraph in value['paragraphs']:
        paragraph['text'] = paragraph['text'].replace('5次', count).replace('12.5 ms', measurement)
    return value


@pytest.mark.parametrize('count,measurement', [
    ('5 次', '12.5 ms'),
    ('5\u00a0次', '12.5 ms'),
    ('5\u3000次', '12.5 ms'),
    ('5\t次', '12.5 ms'),
    ('5次', '12.5ms'),
    ('5次', '12.5\u202fms'),
    ('5 次', '12.5\tms'),
])
def test_horizontal_quantity_separator_survives_sdk_check_and_cache(host, count, measurement):
    source = receipt_source()
    original = copy.deepcopy(source)
    with sdk_pair(lambda payload: quantity_candidate(payload, count=count, measurement=measurement)) as (model, checker, calls):
        svc = service(host, model, checker)
        record = svc.prepare(source, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and record['check']['accepted']
        assert [role for role, _, _ in calls] == ['expression', 'check']
        assert count in record['text'] and measurement in record['text']
        assert calls[1][2]['candidate_text'] == record['text']
        observed = next(b for b in calls[1][2]['fact_blocks'] if b['id'] == 'evidence_1')['receipt_observation']
        assert observed['count'] == 5 and observed['execution_status'] == 'error'
        assert source == original == record['canonical_result']
        assert record['delivery']['status'] == 'not_sent'
        assert svc.prepare(source, scope=SCOPE, check_current=lambda: '') == record
        assert len(calls) == 2


@pytest.mark.parametrize('count,measurement', [
    ('4 次', '12.5 ms'),
    ('15 次', '12.5 ms'),
    ('-5 次', '12.5 ms'),
    ('5 次0', '12.5 ms'),
    ('5\n次', '12.5 ms'),
    ('5次', '12.6 ms'),
    ('5次', '912.5 ms'),
    ('5次', '12.5 msec'),
    ('5次', '12.5 ms²'),
    ('5次', '12.5 m s'),
    ('5次', '12.5\nms'),
])
def test_changed_number_unit_or_line_break_is_rejected_before_checker(host, count, measurement):
    source = receipt_source()
    with sdk_pair(lambda payload: quantity_candidate(payload, count=count, measurement=measurement)) as (model, checker, calls):
        record = service(host, model, checker).prepare(source, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'fallback' and record['candidate'] is None
        assert record['fallback_reason'] == 'expression_protected_value_missing'
        assert len(calls) == 1
        assert record['canonical_result'] == source and record['delivery']['status'] == 'not_sent'


def test_correct_spaced_count_still_needs_independent_limitations_review(host):
    def missing_limitation(payload):
        value = quantity_candidate(payload, count='5 次')
        next(p for p in value['paragraphs'] if p['fact_ids'] == ['evidence_1'])['text'] = 'fixture_read有5 次调用报错。'
        return value

    with sdk_pair(missing_limitation, verdict=dict(supported=False, omitted_limitations=['cause and recovery are unconfirmed'])) as (model, checker, calls):
        source = receipt_source()
        record = service(host, model, checker).prepare(source, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_neutral_check_rejected'
        assert [role for role, _, _ in calls] == ['expression', 'check']
        assert record['candidate'] is None and record['canonical_result'] == source
        assert record['delivery']['status'] == 'not_sent'
