"""Verified query answers remain mandatory; successful tool audit stays in details."""
import copy

import pytest

from shinku.agent_runtime.results import export_task_result
from shinku.conversation.models import AnswerEvidence, text_digest
from shinku.result_expression.natural import paragraph_blocks
from test_expression_paragraphs import service
from test_expression_semantic_coverage import semantic_sdk
from test_result_expression import SCOPE, host


URL = 'https://query.example/a'
ANSWER = '甲店 CEDAR-MUG 型杯为10元，来源：' + URL


def query_task():
    return dict(task_id='verified-query', requester=SCOPE.requester, session=SCOPE.session,
        status='completed', execution_attempt=1, completion_attempt=1, revision=1,
        requests=[dict(text='查询甲店 CEDAR-MUG 型杯价格并给原始来源。')],
        confirmed_constraints=['不购买，不发送，不部署。'],
        contract=dict(requirements=[dict(id='r1', description='回答甲店价格并给原始来源',
            effect='research_answer', targets=[URL], checks={'min_sources': 1})]),
        completion=dict(accepted=True, satisfied=['r1'], missing=[]),
        material_information=dict(status='complete', required_status='complete', upstream_completeness='unknown'),
        tool_calls=[dict(id='search', name='query_search', status='ok', data={'state_updates': {}}),
            dict(id='read', name='query_read', status='ok', data={'state_updates': {'url': URL, 'http_status': 200}})],
        result_answer_evidence=AnswerEvidence('public', 'verified', True, 2, 1, text_digest(ANSWER)).as_dict())


def prepare(svc, value):
    return svc.prepare(value, scope=SCOPE, check_current=lambda: '')


def test_verified_answer_reaches_sdk_body_without_successful_tool_json_and_retains_details(host):
    task=query_task(); original=copy.deepcopy(task)
    value=export_task_result(task, report=ANSWER)
    with semantic_sdk() as (model, checker, calls):
        svc=service(host, model, checker, detail_policy_version=2,
            paragraph_presentation_version=5, semantic_review_version=1)
        record=prepare(svc, value)
        assert record['status']=='ready' and [r[0] for r in calls]==['expression','check']
        assert ANSWER in record['text'] and '不购买，不发送，不部署' in record['text']
        assert '实际工具记录' not in record['text'] and '"http_status"' not in record['text']
        assert all(not r['id'].startswith('evidence_') for r in calls[1][2]['fact_blocks'])
        details=svc.read_details(record['expression_id'], scope=SCOPE, check_current=lambda: '')
        assert details['status']=='available' and record['canonical_result']['evidence']==value['evidence']
        detail_rows={r['id']:r for r in details['fact_blocks']}
        assert all(k in detail_rows and '实际工具记录' in detail_rows[k]['text']
            for k in ('evidence_0','evidence_1'))
        assert len(value['evidence'])==2 and task==original
        assert value['verification']['accepted'] is True and record['delivery']['sent'] is False
        assert prepare(svc, value)==record and len(calls)==2


@pytest.mark.parametrize('change', ['missing', 'wrong_digest', 'unverified', 'task_kind', 'stale_completion'])
def test_unbound_or_stale_positive_claim_cannot_make_receipts_optional(change):
    task=query_task()
    if change=='missing':task.pop('result_answer_evidence')
    elif change=='wrong_digest':task['result_answer_evidence']['answer_sha256']='a'*64
    elif change=='unverified':task['result_answer_evidence']['validated']=False
    elif change=='task_kind':task['result_answer_evidence']['kind']='task'
    else:task['completion_attempt']=0
    value=export_task_result(task, report=ANSWER)
    rows={r['id']:r for r in paragraph_blocks(value, detail_policy_version=2, presentation_version=5)}
    assert all(rows[k].get('required', True) for k in ('evidence_0','evidence_1'))
    if change=='stale_completion':
        assert value['execution']['status']=='unverified' and value['verification']['accepted'] is False


@pytest.mark.parametrize('state', ['partial','failed','blocked','canceled','running'])
def test_other_lifecycle_states_keep_original_audit_requirement_and_state(state):
    task=query_task(); task['status']=state
    value=export_task_result(task, report=ANSWER)
    rows={r['id']:r for r in paragraph_blocks(value, detail_policy_version=2, presentation_version=5)}
    assert value['execution']['status']==state
    assert all(rows[k].get('required', True) for k in ('evidence_0','evidence_1'))


def test_error_receipt_and_unmet_original_condition_still_require_body_disclosure():
    task=query_task()
    task['tool_calls'].append(dict(id='denied', name='query_read', status='error',
        data={'state_updates': {'url': URL, 'http_status': 403}}))
    task['completion']['missing']=['原始条件仍需确认']
    value=export_task_result(task, report=ANSWER)
    rows={r['id']:r for r in paragraph_blocks(value, detail_policy_version=2, presentation_version=5)}
    assert rows['evidence_2'].get('required', True) and rows['remaining'].get('required', True)
    assert '403' in rows['evidence_2']['text'] and '原始条件仍需确认' in rows['remaining']['text']


def test_optional_receipts_do_not_allow_omitting_verified_answer(host):
    def omit_answer(payload):
        rows=[r for r in payload['fact_blocks'] if r.get('required', True) and r['id']!='answer']
        return dict(schema_version=2, binding_hash=payload['binding_hash'], paragraphs=[
            dict(text='\n'.join(r['text'] for r in rows), fact_ids=[r['id'] for r in rows])])
    with semantic_sdk(build=omit_answer) as (model, checker, calls):
        svc=service(host, model, checker, detail_policy_version=2,
            paragraph_presentation_version=5, semantic_review_version=1)
        record=prepare(svc, export_task_result(query_task(), report=ANSWER))
        assert record['status']=='fallback' and record['fallback_reason']=='expression_required_facts_missing_or_duplicated'
        assert len(calls)==1
