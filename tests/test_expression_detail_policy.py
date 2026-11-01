"""Observed synonym failures and host audit tiers through real SDK adapters.

Checker fixtures establish gate enforcement, not real semantic model quality.
"""
import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from shinku.agent_runtime.results import export_query_result, export_task_result, seal_result
from shinku.result_expression import ExpressionArchive, ExpressionService
from shinku.result_expression.natural import paragraph_blocks
from test_expression_paragraphs import sdk_pair, service, source
from test_result_expression import SCOPE, canonical, host


def detailed(value=None):
    result = copy.deepcopy(value or source())
    result.pop('result_hash')
    result.setdefault('presentation', dict(schema_version=1, policy_source='host_export',
        optional_success_receipts=True)).update(detail_policy_version=2, material_recovery=False)
    return seal_result(result)


def candidate(payload, changes=None, *, omit=()):
    rows = []
    for block in payload['fact_blocks']:
        if block.get('required', True) and block['id'] not in omit:
            text = (changes or {}).get(block['id'], block['text'])
            rows.append(dict(text='这份记录的范围需要一并考虑。' + text, fact_ids=[block['id']]))
    return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=rows)


@pytest.mark.parametrize('state,accepted,changes', [
    ('partial', False, {'status': '本次部分完成，整项任务未通过验收。'}),
    ('completed', True, {'status': '执行已经完成，当前目标的验收也通过了。'}),
    ('completed', True, {'delivery': '送达未知；本次新表达尚未取得完整送达回执，须按此范围理解。'}),
])
def test_observed_state_synonyms_and_delivery_punctuation_reach_independent_review(host, state, accepted, changes):
    value = canonical(state, accepted=accepted)
    before = copy.deepcopy(value)
    with sdk_pair(lambda p: candidate(p, changes)) as (model, checker, calls):
        record = service(host, model, checker).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and record['candidate']
        assert [r for r, _, _ in calls] == ['expression', 'check']
        reviewed = {b['id']: b for b in calls[1][2]['fact_blocks']}
        assert all(reviewed[k]['semantic_review_required'] and not reviewed[k]['anchors']
                   for k in ('status', 'delivery', 'deployment'))
        assert record['text'] == calls[1][2]['candidate_text']
        assert value == before == record['canonical_result']
        assert record['delivery']['status'] == 'not_sent'


@pytest.mark.parametrize('state', ['partial', 'failed', 'canceled', 'blocked', 'waiting_user', 'running', 'queued', 'unverified'])
def test_wrong_completion_or_cancellation_meaning_is_still_rejected(host, state):
    value = canonical(state, accepted=False)
    with sdk_pair(lambda p: candidate(p, {'status': '任务已全部完成，取消后进程已清理且操作已撤销。'}),
                  verdict=dict(supported=False, contradictions=['unsupported lifecycle change'])) as (model, checker, calls):
        record = service(host, model, checker).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_neutral_check_rejected'
        assert len(calls) == 2 and record['candidate'] is None
        assert record['canonical_result'] == value and not value['verification']['accepted']
        assert '任务已全部完成' not in record['text']


@pytest.mark.parametrize('key,claim', [('delivery', '本次新表达已经送达。'), ('deployment', '服务已经部署并上线。')])
def test_wrong_delivery_and_deployment_meanings_cannot_pass_by_preserving_ids(host, key, claim):
    with sdk_pair(lambda p: candidate(p, {key: claim}),
                  verdict=dict(supported=False, contradictions=['receipt missing'])) as (model, checker, calls):
        record = service(host, model, checker).prepare(detailed(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_neutral_check_rejected' and len(calls) == 2
        assert claim not in record['text'] and record['candidate'] is None
        assert record['delivery']['status'] == 'not_sent'


@pytest.mark.parametrize('key', ['status', 'delivery', 'deployment'])
def test_host_lifecycle_ids_are_still_required_before_review(host, key):
    with sdk_pair(lambda p: candidate(p, omit=(key,))) as (model, checker, calls):
        record = service(host, model, checker).prepare(detailed(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_required_facts_missing_or_duplicated'
        assert len(calls) == 1


def test_semantic_checker_must_cover_lifecycle_fact_even_when_other_values_are_exact(host):
    def missing_status(payload):
        return candidate(payload)
    with sdk_pair(missing_status, verdict=dict(covered_fact_ids=[])) as (model, checker, calls):
        record = service(host, model, checker).prepare(detailed(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_neutral_check_rejected' and len(calls) == 2
        assert not record['check']['neutral_checks']['coverage_exact']


def test_explicit_host_policy_removes_empty_audit_rows_from_body_but_keeps_scoped_details(host):
    value = detailed(); value.pop('result_hash')
    value['verification']['satisfied'] = []
    value['uncertainties'] = []
    value = seal_result(value); before = copy.deepcopy(value)
    with sdk_pair(candidate) as (model, checker, calls):
        svc = service(host, model, checker)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and len(calls) == 2
        ids = {b['id'] for b in calls[1][2]['fact_blocks']}
        assert ids == {'status', 'goal', 'constraints', 'fact_0', 'artifact_0', 'delivery', 'deployment'}
        assert '12.5 ms' in record['text'] and 'fixture-report.txt' in record['text']
        assert all(x not in record['text'] for x in ('未列出', '摘要或哈希', 'projection', '"http_status"'))
        detail = svc.read_details(record['expression_id'], scope=SCOPE, check_current=lambda: '')
        assert detail['status'] == 'available'
        assert svc.archive.get(record['expression_id'], scope=SCOPE)['canonical_result'] == before == value
        assert record['canonical_result'] == value
        assert 'a' * 64 in json.dumps(detail) and 'b' * 64 in json.dumps(detail)
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == record and len(calls) == 2
        assert svc.read_details(record['expression_id'], scope=type(SCOPE)('other', SCOPE.session),
                                check_current=lambda: '')['status'] == 'discarded'
        assert len(calls) == 2


@pytest.mark.parametrize('policy', [None, {}, {'detail_policy_version': True}, {'policy_source': 'model'}])
def test_legacy_or_self_asserted_policy_does_not_lower_audit_requirements(policy):
    value = source(); value.pop('result_hash')
    if policy is None:
        value.pop('presentation')
    else:
        value['presentation'].update(policy)
    blocks = {b['id']: b for b in paragraph_blocks(seal_result(value))}
    assert all(blocks[k].get('required', True) for k in ('remaining', 'uncertainties', 'information'))


def test_nonempty_limits_and_unknown_gap_text_stay_required_and_exact(host):
    value = detailed(canonical('partial', accepted=False)); value.pop('result_hash')
    value['information'] = dict(status='incomplete', required_status='incomplete',
        gaps=['附件尾页未提供，不能确认限制条件。'])
    value = seal_result(value)
    with sdk_pair(lambda p: candidate(p, {'information': '所需资料完整。'})) as (model, checker, calls):
        record = service(host, model, checker).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_protected_value_missing' and len(calls) == 1
        assert '附件尾页未提供' in record['text'] and '没有在线样本' in record['text']
        assert '报告内容核验' in record['text'] and record['canonical_result'] == value


def test_projection_controls_are_audit_but_lost_scope_is_a_required_limitation(host):
    value = detailed(); value.pop('result_hash')
    value['information'] = dict(status='incomplete', required_status='unknown',
        reason='legacy_source_completeness_unknown', snapshot=dict(gap_count=1, gaps=[
            dict(path='/original_requests/0/text', kind='text', total=18000, retained=2400, omitted=15600,
                 reason='text_limit', retrieval='unavailable')]))
    value = seal_result(value)
    with sdk_pair(candidate) as (model, checker, calls):
        svc = service(host, model, checker)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and len(calls) == 2
        info = next(b for b in calls[1][2]['fact_blocks'] if b['id'] == 'information')
        assert info['required'] and info['semantic_review_required']
        assert '/original_requests/0/text' in record['text'] and '未完整保留' in record['text']
        assert '上游资料' in record['text'] and '完整性未知' in record['text']
        assert 'legacy_source_completeness_unknown' not in record['text'] and '15600' not in record['text']
        detail = svc.read_details(record['expression_id'], scope=SCOPE, check_current=lambda: '')
        assert record['canonical_result'] == value and '15600' in json.dumps(detail)


@pytest.mark.parametrize('code', ['completion_receipt_missing', 'completion_attempt_stale', 'unrecognized_stop'])
def test_known_stop_meaning_is_required_while_unknown_diagnostic_is_not_guessed(code):
    value = detailed(canonical('partial', accepted=False)); value.pop('result_hash')
    value['execution']['reason'] = code
    blocks = {b['id']: b for b in paragraph_blocks(seal_result(value))}
    assert blocks['reason']['required']
    if code == 'unrecognized_stop':
        assert code in blocks['reason']['text'] and any(code in a['text'] for a in blocks['reason']['anchors'])
    else:
        assert blocks['reason']['semantic_review_required'] and not blocks['reason']['anchors']
        assert not blocks['reason_audit']['required'] and code in blocks['reason_audit']['text']


def test_host_export_and_fast_query_disable_material_recovery_when_no_package_exists(host):
    task = dict(task_id='fixture-task', requester=SCOPE.requester, session=SCOPE.session, status='partial',
        completion=dict(accepted=False), requests=[dict(text='查询记录')], tool_calls=[])
    for value in (export_task_result(task), export_query_result(scope=SCOPE, request='查询记录', route='public',
            status='partial', completion=dict(accepted=False), operation_id='fixture-query')):
        assert value['presentation']['detail_policy_version'] == 2
        assert value['presentation']['material_recovery'] is False
        with sdk_pair(candidate) as (model, checker, calls):
            def forbidden(*args, **kwargs):
                pytest.fail('No extra material IO is authorized for a receipt-only result')
            base = service(host, model, checker)
            svc = ExpressionService(base.archive, base.profile, material_resolver=forbidden)
            record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
            assert record['status'] == 'ready' and len(calls) == 2
            assert 'agent_query_result' not in record['canonical_result']


def state_result(ready):
    from shinku.agent.self_state import SelfStateTool
    from shinku.agent_runtime.results import build_query_result
    from test_self_state_receipts import checked, context, observe
    reads = []
    tool = SelfStateTool(provider=lambda: reads.append(True) or {'runtime': {'ready': ready}})
    ctx = context(); pair = observe(tool, ctx)
    text, status, evidence = checked(tool, ctx, pair)
    turn = SimpleNamespace(primary_message_id='state', messages=(), conversation_type='private', conversation_id='fixture')
    value = build_query_result(ctx, turn, [pair[1]], route='perception', status=status, text=text, answer_evidence=evidence)
    return value, reads


@pytest.mark.parametrize('ready', [True, False])
def test_real_state_receipt_avoids_duplicate_answer_and_opaque_reference_without_losing_snapshot(host, ready):
    value, reads = state_result(ready); before = copy.deepcopy(value)
    with sdk_pair(candidate) as (model, checker, calls):
        base = service(host, model, checker)
        def forbidden(*args, **kwargs):
            pytest.fail('State presentation must not add a read')
        svc = ExpressionService(base.archive, base.profile, material_resolver=forbidden)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and reads == [True] and len(calls) == 2
        assert 'self_state:' not in record['text'] and '整项目标验收通过' not in record['text']
        assert '所选范围' in record['text'] and '实际部署或消息送达' in record['text']
        assert 'true' in record['text'] if ready else 'false' in record['text']
        ids = calls[0][2]['required_fact_ids']
        assert 'fact_0' in ids and 'answer' not in ids and 'evidence_0' not in ids
        assert not any(b['id'].endswith('_audit') for b in calls[1][2]['fact_blocks'])
        assert 'answer' in calls[0][2]['optional_fact_ids']
        assert value == before == record['canonical_result'] and not value['delivery']['sent']
        detail = svc.read_details(record['expression_id'], scope=SCOPE, check_current=lambda: '')
        assert detail['status'] == 'available' and 'self_state:' in json.dumps(detail)
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == record
        assert len(calls) == 2 and reads == [True]


@pytest.mark.parametrize('tamper', ['reference', 'hash', 'scope', 'invocation', 'fresh'])
def test_state_looking_reference_cannot_be_demoted_without_matching_real_receipt(tamper):
    value, reads = state_result(True); value.pop('result_hash')
    fact = value['facts'][0]
    receipt = value['evidence'][0]['receipt']['self_state_receipt']
    if tamper == 'reference':
        fact['reference'] = 'https://example.invalid/state-source'
    elif tamper == 'hash':
        receipt['snapshot_sha256'] = 'f' * 64
    elif tamper == 'scope':
        receipt['scope'] = 'tools'
    elif tamper == 'invocation':
        fact['invocation_id'] = 'another-read'
    else:
        receipt['fresh'] = False
    blocks = {b['id']: b for b in paragraph_blocks(seal_result(value))}
    assert blocks['answer']['required']
    assert fact['reference'] in [a['text'] for a in blocks['fact_0']['anchors']]
    assert 'fact_0_source_audit' not in blocks and reads == [True]


@pytest.mark.parametrize('old,new', [('12.5 ms', '12.6 ms'), ('fixture-report.txt', 'fixture-report.txt.exe'),
    ('https://example.invalid/source', 'https://example.invalid/source/extra'), ('没有在线样本', '已有在线样本')])
def test_new_detail_policy_still_protects_measurements_addresses_and_substantive_limits(host, old, new):
    def build(payload):
        result = candidate(payload)
        for row in result['paragraphs']:
            row['text'] = row['text'].replace(old, new)
        return result
    with sdk_pair(build) as (model, checker, calls):
        record = service(host, model, checker).prepare(detailed(), scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_protected_value_missing' and len(calls) == 1
        assert new not in record['text'] and record['candidate'] is None


def test_deployment_scope_address_and_measurement_remain_exact_despite_semantic_lifecycle(host):
    value = detailed(); value.pop('result_hash')
    value['deployment'] = dict(status='deployed', source='executor_receipt', scope=['https://example.invalid/app', 2])
    value = seal_result(value)
    with sdk_pair(lambda p: candidate(p, {'deployment': '已部署到https://example.invalid/app/forged，覆盖2项。'})) as (model, checker, calls):
        record = service(host, model, checker).prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['fallback_reason'] == 'expression_protected_value_missing' and len(calls) == 1


def test_explicit_profile_policy_can_present_old_facts_without_editing_canonical_archive(host):
    value = canonical(); value.pop('result_hash'); value['uncertainties'] = []; value = seal_result(value)
    before = copy.deepcopy(value)
    with sdk_pair(candidate) as (model, checker, calls):
        svc = service(host, model, checker, detail_policy_version=2)
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert record['status'] == 'ready' and len(calls) == 2
        assert record['detail_policy_version'] == 2 and record['canonical_result'] == before == value
        assert 'presentation' not in value
        assert 'a' * 64 not in record['text'] and '没有在线样本' not in record['text']
        assert all(x not in calls[0][2]['required_fact_ids'] for x in ('remaining', 'uncertainties'))
        assert 'a' * 64 in json.dumps(svc.read_details(record['expression_id'], scope=SCOPE, check_current=lambda: ''))
        assert svc.prepare(value, scope=SCOPE, check_current=lambda: '') == record and len(calls) == 2


def test_changed_host_policy_does_not_rewrite_or_repurchase_an_existing_expression(host):
    value = canonical()
    with sdk_pair(candidate) as (model, checker, calls):
        svc = service(host, model, checker)
        original = svc.prepare(value, scope=SCOPE, check_current=lambda: '')
        assert original['status'] == 'ready' and len(calls) == 2
        updated = ExpressionService(svc.archive, replace(svc.profile, detail_policy_version=2))
        assert updated.prepare(value, scope=SCOPE, check_current=lambda: '')['fallback_reason'] == 'expression_configuration_changed'
        assert svc.archive.get(original['expression_id'], scope=SCOPE) == original and len(calls) == 2


def test_expired_host_policy_claim_recovers_the_original_tier_without_sdk_or_new_delivery(host):
    from shinku.agent_runtime.contracts import canonical_hash
    from shinku.result_expression.facts import identity, template
    clock = [10.0]; host.clock = lambda: clock[0]
    value = canonical(); value.pop('result_hash'); value['uncertainties'] = []; value = seal_result(value)
    with sdk_pair(candidate) as (model, checker, calls):
        svc = service(host, model, checker, detail_policy_version=2)
        key = canonical_hash(identity(value))
        binding = canonical_hash({**identity(value), 'result_hash': value['result_hash'], 'detail_policy_version': 2})
        svc.archive.claim(key, scope=SCOPE, result=value, config_hash=svc.profile.config_hash, binding_hash=binding,
            format_version=2, detail_policy_version=2, processing_seconds=10)
        clock[0] = 30
        record = svc.prepare(value, scope=SCOPE, check_current=lambda: '', recover_interrupted=True)
        assert record['status'] == 'fallback' and record['fallback_reason'] == 'expression_interrupted'
        assert record['detail_policy_version'] == 2 and record['binding_hash'] == binding
        assert record['text'] == template(paragraph_blocks(value, detail_policy_version=2))
        assert '未列出' not in record['text'] and record['delivery']['status'] == 'not_sent'
        assert not calls and record['canonical_result'] == value


@pytest.mark.parametrize('version', [True, 1, 3, '2', None])
def test_malformed_host_detail_policy_is_rejected_before_transport(host, version):
    with sdk_pair(candidate) as (model, checker, calls):
        with pytest.raises(ValueError, match='invalid_expression_detail_policy'):
            service(host, model, checker, detail_policy_version=version)
        assert not calls
