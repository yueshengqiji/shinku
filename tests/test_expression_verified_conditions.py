"""Verified condition meaning and actual artifact effects reach both SDKs."""
import copy
import json

import pytest

from shinku.agent_runtime.results import seal_result
from shinku.result_expression import ExpressionProfile
from shinku.result_expression.natural import paragraph_blocks
from test_expression_paragraphs import sdk_pair, service, source
from test_result_expression import host, SCOPE


def file_source(**changes):
    result = source(); result.pop('result_hash')
    result.update(original_requests=[{'text': '读取input.txt生成output.txt，保持输入不变，不发送或部署。'}],
        facts=[], constraints=['保持输入不变'], uncertainties=[], delivery={'status': 'not_sent', 'sent': False},
        contract={'requirements': [
            {'id': 'r1', 'description': '完整读取input.txt，读取后输入保持不变。', 'effect': 'read',
             'targets': ['input.txt'], 'checks': {'unchanged': True}},
            {'id': 'r2', 'description': '按输入生成output.txt，只写入一次并核验内容。', 'effect': 'write',
             'targets': ['output.txt'], 'checks': {}}]},
        verification={'accepted': True, 'status': 'completed', 'satisfied': ['r1', 'r2'], 'missing': [],
                      'source': 'completion_verifier'})
    records = [dict(path='input.txt', sha256='a'*64, bytes=8, result_valid=True,
                    completion_targets=['input.txt'], verified_effects=['read'], read_complete=True),
               dict(path='output.txt', sha256='b'*64, bytes=6, result_valid=True,
                    completion_targets=['output.txt'], verified_effects=['write']),
               dict(path='output.txt', sha256='b'*64, bytes=6, result_valid=True,
                    completion_targets=['output.txt'], verified_effects=['read'], read_complete=True)]
    result['evidence'] = [dict(invocation_id='file-call-'+str(i), tool='file_io', execution_status='ok',
        source='executor_receipt', receipt=dict(artifact_receipts=[copy.deepcopy(v)])) for i, v in enumerate(records)]
    result['artifacts'] = [dict(**v, source='executor_receipt', result_version=result['version'],
                                verification='task_passed') for v in records]
    result.update(changes)
    return seal_result(result)


def blocks(result):
    return paragraph_blocks(result, detail_policy_version=2, presentation_version=3)


def prose(payload):
    ids = payload['required_fact_ids']
    # Host-protected paths remain literal. Condition IDs remain attribution
    # metadata, while the independent checker verifies the natural meaning.
    return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=[dict(
        text='输入input.txt已完整读取并保持不变，结果已写到output.txt，按要求只写入一次；'
             '写入后的内容也完成了回读核验，这次目标已通过验收。没有部署，文件核验记录只代表核验当时的内容。',
        fact_ids=ids)])


def test_condition_ids_are_bound_to_verified_meaning_without_becoming_required_prose():
    rows = {r['id']: r for r in blocks(file_source())}
    assert rows['verification']['semantic_review_required']
    assert rows['verification']['condition_observations'][0]['description'] == '完整读取input.txt，读取后输入保持不变。'
    assert not {'r1', 'r2'} & {a['text'] for a in rows['verification']['anchors']}
    assert '保持不变' in rows['verification']['text']
    assert rows['artifact_1']['artifact_observation']['verified_effects'] == ['write']
    assert rows['artifact_2']['artifact_observation']['read_complete'] is True
    assert rows['contract_context']['required'] is False


def test_actual_sdk_checker_receives_condition_meaning_and_current_write_read_receipts(host):
    value = file_source(); before = copy.deepcopy(value)
    with sdk_pair(prose) as (model, checker, calls):
        svc = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=3)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready', record['fallback_reason']
        assert len(calls) == 2 and record['canonical_result'] == before
        assert record['paragraph_presentation_version'] == 3 and 'r1' not in record['text'] and 'r2' not in record['text']
        payload = calls[1][2]
        verifier = next(r for r in payload['fact_blocks'] if r['id'] == 'verification')
        assert verifier['condition_observations'][0]['checks'] == {'unchanged': True}
        assert next(r for r in payload['fact_blocks'] if r['id'] == 'artifact_2')['artifact_observation']['read_complete']
        assert any(r['id'] == 'contract_context' for r in payload['context_fact_blocks'])
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == record and len(calls) == 2


@pytest.mark.parametrize('fault', ['missing_definition', 'duplicate_id', 'wrong_verifier', 'contradictory_status'])
def test_incomplete_or_conflicting_condition_mapping_cannot_invent_a_verified_description(fault):
    value = file_source(); value.pop('result_hash')
    if fault == 'missing_definition': value['contract']['requirements'].pop()
    if fault == 'duplicate_id': value['contract']['requirements'][1]['id'] = 'r1'
    if fault == 'wrong_verifier': value['verification']['source'] = 'execution_model'
    if fault == 'contradictory_status': value['verification']['missing'] = ['r1']
    rows = {r['id']: r for r in blocks(seal_result(value))}
    assert 'condition_observations' not in rows['verification']
    assert {'r1', 'r2'} <= {a['text'] for a in rows['verification']['anchors']}


@pytest.mark.parametrize('fault', ['failed_call', 'wrong_source', 'stale_version', 'changed_sha',
    'invalid_result', 'missing_receipt', 'read_incomplete'])
def test_unproven_or_stale_artifact_cannot_be_described_as_complete_readback(fault):
    value = file_source(); value.pop('result_hash')
    if fault == 'failed_call': value['evidence'][2]['execution_status'] = 'error'
    if fault == 'wrong_source': value['evidence'][2]['source'] = 'execution_model'
    if fault == 'stale_version': value['artifacts'][2]['result_version'] -= 1
    if fault == 'changed_sha': value['artifacts'][2]['sha256'] = 'c'*64
    if fault == 'invalid_result': value['artifacts'][2]['result_valid'] = False
    if fault == 'missing_receipt': value['evidence'][2]['receipt'] = {}
    if fault == 'read_incomplete':
        value['artifacts'][2]['read_complete'] = False
        value['evidence'][2]['receipt']['artifact_receipts'][0]['read_complete'] = False
    row = next(r for r in blocks(seal_result(value)) if r['id'] == 'artifact_2')
    if fault == 'read_incomplete':
        assert row['artifact_observation']['read_complete'] is False
        assert '完整回读' not in row['text']
    else:
        assert 'artifact_observation' not in row


def test_read_receipt_does_not_create_a_write_count_claim():
    rows = blocks(file_source())
    for row in rows:
        if row.get('artifact_observation'):
            assert 'count' not in row['artifact_observation']
    assert '一次' not in next(r['text'] for r in rows if r['id'] == 'artifact_1')


def test_partial_and_missing_conditions_remain_mandatory_and_do_not_promote_completion():
    value = file_source(execution={'status': 'partial', 'reason': 'fixture_gap', 'result_available': True},
        verification={'accepted': False, 'status': 'partial', 'satisfied': ['r1'], 'missing': ['r2'],
                      'source': 'completion_verifier'}, remaining=['r2'])
    rows = {r['id']: r for r in blocks(value)}
    assert rows['remaining']['required'] and rows['remaining']['semantic_review_required']
    assert '只写入一次' in rows['remaining']['text']
    assert '部分完成' in rows['status']['text']
    assert len(rows['verification']['condition_observations']) == 1


def test_new_configuration_cannot_reinterpret_or_reuse_version_two_archive(host):
    value = file_source()
    def catalog(payload):
        rows = [r for r in payload['fact_blocks'] if r.get('required', True)]
        return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=[dict(
            text='现有记录只覆盖本次任务。\n'+'\n'.join(r['text'] for r in rows), fact_ids=[r['id'] for r in rows])])
    with sdk_pair(catalog) as (model, checker, calls):
        old = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=2)
        first = old.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert first['status'] == 'ready', first['fallback_reason']
        before = len(calls)
        new = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=3)
        assert new.profile.config_hash != old.profile.config_hash
        result = new.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert result['fallback_reason'] == 'expression_configuration_changed' and len(calls) == before
        assert old.archive.get(first['expression_id'], scope=SCOPE) == first
        assert old.prepare(value, scope=SCOPE, check_current=lambda: '') == first


@pytest.mark.parametrize('options', [dict(paragraph_presentation_version=3),
    dict(paragraph_presentation_version=3, format_version=2),
    dict(paragraph_presentation_version=3, format_version=1, detail_policy_version=2)])
def test_version_three_requires_an_explicit_compatible_profile(options):
    with pytest.raises(ValueError): ExpressionProfile(**options)


@pytest.mark.parametrize('defect', ['omitted', 'path'])
def test_positive_verdict_cannot_omit_conditions_or_replace_a_protected_path(host, defect):
    def changed(payload):
        result = prose(payload)
        if defect == 'omitted': result['paragraphs'][0]['fact_ids'].remove('verification')
        else: result['paragraphs'][0]['text'] = result['paragraphs'][0]['text'].replace('output.txt', 'other.txt')
        return result
    with sdk_pair(changed) as (model, checker, calls):
        svc = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=3)
        record = svc.prepare(file_source(), scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'fallback' and len(calls) == 1
        assert record['candidate'] is None


def test_independent_rejection_of_an_unproven_claim_still_returns_the_fact_template(host):
    with sdk_pair(prose, verdict={'supported': False, 'additional_claims': ['unproven write count']}) as (model, checker, calls):
        svc = service(host, model, checker, detail_policy_version=2, paragraph_presentation_version=3)
        record = svc.prepare(file_source(), scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'fallback' and record['fallback_reason'] == 'expression_neutral_check_rejected'
        assert len(calls) == 2 and record['candidate'] is None
        assert record['canonical_result']['verification']['accepted'] is True


@pytest.mark.parametrize('key,value,label', [
    ('min_sources', 3, '来源数量下限'),
    ('min_bytes', 24, '字节数量下限'),
    ('contains', 'CP17 isolated artifact.\\n', '必须包含的原文'),
    ('sha256', 'c6b1f85ac7149c7e1d052b74ec08d255ddfc50d17c9f35b473c51738728a729e', 'SHA256内容校验值'),
])
def test_connected_body_explains_known_checks_without_changing_values_or_old_versions(key, value, label):
    result = file_source(); result.pop('result_hash')
    result['contract']['requirements'][1]['checks'] = {key: value}
    result = seal_result(result); before = copy.deepcopy(result)
    older = {b['id']: b for b in paragraph_blocks(result, detail_policy_version=2, presentation_version=4)}
    current = {b['id']: b for b in paragraph_blocks(result, detail_policy_version=2, presentation_version=5)}
    assert label + '：' in current['verification']['text']
    assert key + '：' not in current['verification']['text']
    assert key + '：' in older['verification']['text']
    assert current['verification']['anchors'] == older['verification']['anchors']
    assert current['verification']['condition_observations'] == older['verification']['condition_observations']
    assert current['verification']['condition_observations'][1]['checks'] == {key: value}
    assert result == before


def test_connected_body_keeps_unknown_check_names_and_original_values():
    result = file_source(); result.pop('result_hash')
    result['contract']['requirements'][1]['checks'] = {'unknown_host_condition': 47}
    result = seal_result(result)
    rows = {b['id']: b for b in paragraph_blocks(result, detail_policy_version=2, presentation_version=5)}
    assert 'unknown_host_condition：47' in rows['verification']['text']
    assert rows['verification']['condition_observations'][1]['checks'] == {'unknown_host_condition': 47}
