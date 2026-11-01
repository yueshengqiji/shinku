"""Independent prose and SDK regressions for the recorded CP36/37 defects."""
import copy
from dataclasses import replace

import pytest

from shinku.agent_runtime.results import seal_result
from shinku.result_expression import ExpressionProfile, ExpressionService
from shinku.result_expression.natural import paragraph_blocks
from test_expression_paragraphs import sdk_pair, service, source
from test_expression_verified_conditions import file_source, prose as file_prose
from test_result_expression import host, SCOPE


def query_source(**changes):
    value = source(); value.pop('result_hash')
    value.update(task_id='', operation_id='fixture-query-operation', task_identity_missing=True,
        route='public_query', facts=[], artifacts=[], evidence=[],
        original_requests=[{'text': '依已读取的三店来源比较价格，不购买、不写文件、不发送。'}], constraints=[],
        execution={'status': 'partial', 'reason': 'public_query_answer_unverified', 'result_available': True},
        verification={'accepted': False, 'status': 'partial', 'source': 'completion_verifier',
                      'answer_evidence': {'kind': 'public', 'validated': False, 'source_count': 1}},
        remaining=['原问题的答案尚未通过证据核验'], uncertainties=[],
        delivery={'status': 'not_sent', 'sent': False},
        presentation={'schema_version': 1, 'policy_source': 'host_export', 'query_receipt_scope': 'current_operation_export'},
        information={'status': 'complete', 'required_status': 'incomplete', 'reason': 'query_receipt_and_answer_scope',
                     'query_scope': 'current_operation_export', 'export_gaps': [], 'projection': {'gap_count': 0}})
    value.update(changes)
    return seal_result(value)


def rows(value, version=4):
    return {r['id']: r for r in paragraph_blocks(value, detail_policy_version=2, presentation_version=version)}


def grouped_prose(payload):
    return {'schema_version': 2, 'binding_hash': payload['binding_hash'], 'paragraphs': [{
        'text': '这次还不能确认答案，原问题的答案尚未通过证据核验；依据仅限本次实际读取的资料，'
                '未读取的来源尚无完整性凭据。没有部署。', 'fact_ids': payload['required_fact_ids']}]}


def test_recorded_query_code_gets_bound_meaning_without_four_repeated_gap_sentences():
    value = query_source(); before = copy.deepcopy(value); blocks = rows(value)
    assert blocks['reason']['stop_observation']['known'] is True
    assert blocks['reason_audit']['required'] is False
    assert 'public_query_answer_unverified' in blocks['reason_audit']['text']
    assert not blocks['reason']['anchors'] and blocks['reason']['semantic_review_required']
    assert '答案' not in blocks['information']['text']
    assert all(blocks[k].get('required', True) for k in ('status', 'reason', 'remaining', 'information'))
    assert value == before


@pytest.mark.parametrize('code', ['fixture_remaining_condition', 'unrecognized_host_marker'])
def test_unknown_diagnostic_is_audited_while_its_unknown_cause_is_required(code):
    value = file_source(execution={'status': 'waiting_user', 'reason': code, 'result_available': True})
    blocks = rows(value)
    assert blocks['reason']['required'] and blocks['reason']['semantic_review_required']
    assert blocks['reason']['stop_observation']['known'] is False
    assert '未确认' in blocks['reason']['text'] and code not in blocks['reason']['text']
    assert code in blocks['reason_audit']['text']
    assert code in rows(value, 3)['reason']['text']


@pytest.mark.parametrize('change', [dict(task_id='formal-task', task_identity_missing=False),
    dict(route='other'), dict(execution={'status': 'completed', 'reason': 'public_query_answer_unverified'}),
    dict(verification={'accepted': True})])
def test_query_code_without_matching_scope_and_status_cannot_claim_an_explanation(change):
    assert rows(query_source(**change))['reason']['stop_observation']['known'] is False


def test_unknown_original_completion_condition_stays_protected_not_treated_as_a_diagnostic():
    value = file_source(execution={'status': 'partial', 'reason': 'fixture_remaining_condition'},
        verification={'accepted': False, 'source': 'completion_verifier', 'satisfied': [], 'missing': ['unknown_condition']},
        remaining=['unknown_condition'], uncertainties=['测量条件未确认'])
    blocks = rows(value)
    assert blocks['remaining']['required']
    assert any(a['text'] == 'unknown_condition' for a in blocks['remaining']['anchors'])
    assert any(a['text'] == '测量条件未确认' for a in blocks['uncertainties']['anchors'])


@pytest.mark.parametrize('code', ['completion_receipt_missing', 'completion_attempt_stale', 'budget_exhausted', 'task_canceled'])
def test_existing_known_failure_boundaries_are_not_reinterpreted(code):
    value = file_source(execution={'status': 'partial', 'reason': code})
    assert rows(value)['reason'] == rows(value, 3)['reason']
    assert rows(value)['reason_audit'] == rows(value, 3)['reason_audit']


def test_two_complete_handoff_states_form_one_statement_with_full_audit_retained():
    blocks = rows(source())
    assert '本次所需资料已完整交接' in blocks['information']['text']
    assert '不证明已经理解全量内容' in blocks['information']['text']
    assert 'snapshot' in blocks['information_audit']['text']
    assert not blocks['information_audit']['required']


@pytest.mark.parametrize('information', [
    {'status': 'complete', 'required_status': 'unknown'},
    {'status': 'complete', 'required_status': 'complete', 'reason': '必须核对订单20的授权'},
    {'status': 'complete', 'required_status': 'complete', 'gaps': ['missing-material.txt']},
    {'status': 'complete', 'required_status': 'complete', 'snapshot': {'gap_count': 1, 'gaps': ['tail.txt']}}])
def test_material_uncertainties_and_substantive_gaps_cannot_be_compacted_away(information):
    value = source(); value.pop('result_hash'); value['information'] = information; value = seal_result(value)
    assert rows(value)['information'] == rows(value, 3)['information']


@pytest.mark.parametrize('extra,expected', [({'gaps': ['unread-shop.txt']}, 'unread-shop.txt'),
    ({'export_gaps': ['missing-price.txt']}, 'missing-price.txt'),
    ({'snapshot': {'gap_count': 1, 'gaps': ['tail.txt']}}, 'tail.txt'),
    ({'reason': '必须核对订单20的授权'}, '必须核对订单20的授权')])
def test_current_query_scope_does_not_override_additional_material_gaps(extra, expected):
    information = copy.deepcopy(query_source()['information']); information.update(extra)
    block = rows(query_source(information=information))['information']
    assert block['required'] and expected in block['text']
    assert any(a['text'] == expected for a in block['anchors'])


def test_actual_sdk_still_receives_verified_condition_meaning_and_matching_artifact_identity(host):
    value = file_source()
    with sdk_pair(file_prose) as (model, checker, calls):
        result = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=4).prepare(
            value, scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'ready', result['fallback_reason']
        block = next(b for b in calls[1][2]['fact_blocks'] if b['id'] == 'artifact_2')
        assert block['artifact_observation']['read_complete'] is True
        assert block['artifact_observation']['sha256'] == 'b'*64
        assert next(b for b in calls[1][2]['fact_blocks'] if b['id'] == 'verification')['condition_observations']


def test_version_four_does_not_relax_exact_numeric_values_before_the_neutral_check(host):
    def changed(payload):
        required = [b for b in payload['fact_blocks'] if b.get('required', True)]
        return {'schema_version': 2, 'binding_hash': payload['binding_hash'], 'paragraphs': [{
            'text': '\n'.join(b['text'] for b in required).replace('12.5 ms', '13 ms'),
            'fact_ids': [b['id'] for b in required]}]}
    with sdk_pair(changed) as (model, checker, calls):
        result = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=4).prepare(
            source(), scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'fallback' and len(calls) == 1


def test_real_sdk_grouped_prose_retains_all_ids_original_context_and_cache_identity(host):
    value = query_source(); before = copy.deepcopy(value)
    with sdk_pair(grouped_prose) as (model, checker, calls):
        svc = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=4)
        result = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'ready', result['fallback_reason']
        assert result['paragraph_presentation_version'] == 4 and len(calls) == 2
        assert 'public_query_answer_unverified' not in result['text']
        assert result['canonical_result'] == value == before
        assert calls[0][2]['paragraph_presentation_version'] == 4
        assert any(b['id'] == 'reason_audit' for b in calls[1][2]['context_fact_blocks'])
        assert 'stop_observation' in next(b for b in calls[1][2]['fact_blocks'] if b['id'] == 'reason')
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == result and len(calls) == 2


def test_invented_cause_still_requires_and_can_fail_the_neutral_checker(host):
    def invent(payload):
        out = grouped_prose(payload); out['paragraphs'][0]['text'] += '这是网络故障造成的，已经恢复。'
        return out
    with sdk_pair(invent, verdict={'supported': False, 'additional_claims': ['未证明网络故障或恢复']}) as (model, checker, calls):
        svc = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=4)
        result = svc.prepare(query_source(), scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'fallback' and len(calls) == 2
        assert result['check']['accepted'] is False and result['candidate'] is None


def test_missing_gap_id_is_rejected_before_a_positive_neutral_verdict(host):
    def omit(payload):
        out = grouped_prose(payload); out['paragraphs'][0]['fact_ids'].remove('remaining'); return out
    with sdk_pair(omit) as (model, checker, calls):
        result = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=4).prepare(
            query_source(), scope=SCOPE, check_current=lambda: '')
        assert result['status'] == 'fallback' and len(calls) == 1


def test_version_four_cannot_reuse_or_rewrite_a_version_three_expression(host):
    with sdk_pair(grouped_prose) as (model, checker, calls):
        old = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=3)
        first = old.prepare(query_source(), scope=SCOPE, check_current=lambda: '')
        assert first['status'] == 'fallback'  # version three still requires the literal diagnostic code
        upgraded = ExpressionService(old.archive, replace(old.profile, paragraph_presentation_version=4))
        size = len(calls); record = upgraded.prepare(query_source(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_configuration_changed' and len(calls) == size
        assert old.archive.get(first['expression_id'], scope=SCOPE) == first


@pytest.mark.parametrize('options', [dict(paragraph_presentation_version=4),
    dict(paragraph_presentation_version=4, format_version=2)])
def test_version_four_requires_explicit_compatible_host_profile(options):
    with pytest.raises(ValueError, match='invalid_expression_presentation_policy'): ExpressionProfile(**options)
