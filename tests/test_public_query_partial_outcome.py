"""A bound partial answer needs no rewrite solely to lower its model label."""
import copy
import json
from dataclasses import replace

import pytest

from test_public_query_source_handoff import evidence, observe
from test_qq_independent_agent import turn
from test_qq_query_continuation import (
    ANSWER, BODY_A, BODY_B, GOAL, PARTIAL, URL_A, URL_B,
    answer_review, drain, query_host, saved_task,
)
from test_research_answer_protocol import sdk_review


def partial_evidence():
    value, _, reader = evidence()
    observe(value, 'read', {'url': URL_A, 'site_access_ok': True, 'http_status': 200,
        'page_text': BODY_A}, reader, url=URL_A)
    payload = value.ledger.review_payload(GOAL, value.state.context, answer=PARTIAL)
    review = answer_review(payload, False)
    review['limitations'][0]['condition_ids'] = ['public_original']
    return value, reader, review


@pytest.mark.parametrize('label', ['completed', ''])
def test_actual_sdk_bound_partial_label_is_lowered_without_rewriting_or_extra_review(label):
    value, _, review = partial_evidence()
    before_sources = copy.deepcopy(value.ledger.sources)
    with sdk_review(json.dumps(review, ensure_ascii=False)) as (runtime, calls):
        value.planner.runtime = runtime
        result = value.finish(PARTIAL, label, admitted=lambda: True)
        assert result == (PARTIAL, 'partial', 'partial', 'public_query_partial_outcome_reconciled')
        assert len(calls) == value.ledger.answer_review_calls == 1
        assert value.correction_calls == 0 and value.corrected is True
        assert value.ledger.sources == before_sources
        assert value.ledger.evaluate_answer(PARTIAL, 'partial').accepted
        assert not value.ledger.evaluate_answer(PARTIAL, 'completed').accepted
        assert not calls[0].get('tools') and not calls[0].get('tool_choice')


def test_second_final_review_remains_available_after_new_sources_without_resetting_count():
    value, reader, review = partial_evidence()
    with sdk_review(json.dumps(review, ensure_ascii=False)) as (front, front_calls):
        value.planner.runtime = front
        assert value.finish(PARTIAL, 'completed', admitted=lambda: True)[1:3] == ('partial', 'partial')
    assert value.ledger.answer_review_calls == 1
    observe(value, 'read', {'url': URL_B, 'site_access_ok': True, 'http_status': 200,
        'page_text': BODY_B}, reader, url=URL_B)
    final = answer_review(value.ledger.review_payload(GOAL, value.state.context, answer=ANSWER), True)
    with sdk_review(json.dumps(final, ensure_ascii=False)) as (runtime, final_calls):
        value.planner.runtime = runtime
        assert value.review(ANSWER, 'completed') == (True, True)
        assert len(front_calls) == len(final_calls) == 1
        assert value.ledger.answer_review_calls == 2 and value.correction_calls == 0
        value.review(ANSWER + ' changed', 'completed')
        assert len(final_calls) == 1 and value.ledger.answer_review_calls == 2
        assert 'answer_review_budget_or_length_exhausted' in value.ledger.review_errors


@pytest.mark.parametrize('fault', ['missing_condition', 'missing_field', 'missing_limitation',
    'forged_source', 'forged_quote', 'unbound_limitation', 'unsupported', 'mixed_contract'])
def test_incomplete_or_invalid_binding_cannot_take_the_status_only_partial_path(fault):
    value, _, review = partial_evidence()
    if fault == 'missing_condition': review['limitations'][0]['condition_ids'] = []
    if fault == 'missing_field': review['limitations'][0]['fields'] = []
    if fault == 'missing_limitation': review['limitations'] = []
    if fault == 'forged_source': review['claims'][0]['source_id'] = 'not_a_read_source'
    if fault == 'forged_quote': review['claims'][0]['quote'] = 'Unobserved source quote.'
    if fault == 'unbound_limitation': review['limitations'][0]['answer_fragment'] = 'not in this answer'
    if fault == 'unsupported': review['verdict'] = 'unsupported'
    if fault == 'mixed_contract':
        value.state.context['agent_completion_contract']['requirements'].append({'id': 'write', 'effect': 'write'})
    with sdk_review(json.dumps(review, ensure_ascii=False)) as (runtime, calls):
        value.planner.runtime = runtime
        result = value.finish(PARTIAL, 'completed', admitted=lambda: True)
        assert result[3] == 'public_query_answer_unverified'
        assert not value.corrected and value.correction_calls == 1
        assert len(calls) == 2 and value.ledger.answer_review_calls == 1
        assert result[0] != PARTIAL and result[1] == 'partial' and result[2] == 'unverified'


@pytest.mark.parametrize('change', ['source_version', 'source_denied', 'authority'])
def test_source_or_authority_change_while_review_waits_cannot_release_the_old_answer(change):
    value, reader, review = partial_evidence()
    allowed = [True]
    def changed(_):
        if change == 'authority':
            allowed[0] = False
        elif change == 'source_denied':
            observe(value, 'read', {'url': URL_A, 'site_access_ok': False, 'http_status': 403}, reader, url=URL_A)
        else:
            observe(value, 'read', {'url': URL_A, 'site_access_ok': True, 'http_status': 200,
                'page_text': 'A changed source version.'}, reader, url=URL_A)
    with sdk_review(json.dumps(review, ensure_ascii=False), after_response=changed) as (runtime, calls):
        value.planner.runtime = runtime
        result = value.finish(PARTIAL, 'completed', admitted=lambda: allowed[0])
        assert not value.corrected and result[0] != PARTIAL
        assert result[2] == 'unverified' and result[1] == 'partial'
        assert result[3] != 'public_query_partial_outcome_reconciled'
        assert len(calls) == (1 if change == 'authority' else 2)
        assert value.ledger.answer_review_calls == 1


@pytest.mark.parametrize('label', ['failed', 'partial'])
def test_explicit_failure_and_partial_labels_keep_their_original_meaning(label):
    value, _, review = partial_evidence()
    with sdk_review(json.dumps(review, ensure_ascii=False)) as (runtime, calls):
        value.planner.runtime = runtime
        result = value.finish(PARTIAL, label, admitted=lambda: True)
        assert result == (PARTIAL, label, 'partial', '')
        assert not value.corrected and value.correction_calls == 0
        assert len(calls) == value.ledger.answer_review_calls == 1


def test_unsupported_body_still_gets_one_real_correction_and_a_second_evidence_review():
    value, _, good = partial_evidence()
    bad = copy.deepcopy(good)
    bad['claims'][0]['answer_fragment'] = '不存在于答复的字段'
    repaired = dict(kind='final', text=PARTIAL, task_outcome='partial')
    with sdk_review([json.dumps(v, ensure_ascii=False) for v in (bad, repaired, good)]) as (runtime, calls):
        value.planner.runtime = runtime
        result = value.finish(f'甲店价格999元。来源：{URL_A}', 'completed', admitted=lambda: True)
        assert result == (PARTIAL, 'partial', 'partial', '')
        assert len(calls) == 3 and value.ledger.answer_review_calls == 2
        assert value.corrected and value.correction_calls == 1
        assert all(not v.get('tools') for v in calls)


@pytest.mark.parametrize('enabled', [False, True])
def test_real_query_worker_preserves_first_partial_and_uses_the_second_review_only_after_new_read(query_host, enabled):
    host = query_host
    host.bridge.foreground_query_policy = replace(host.bridge.foreground_query_policy, enabled=enabled)
    host.chat.replies[-1]['task_outcome'] = 'completed'
    original = host.chat.call_chat_json
    def chat(**options):
        result = original(**options)
        if '答案证据与覆盖复核' in str(options.get('system_extra_blocks')):
            result['limitations'][0]['condition_ids'] = ['public_original']
        return result
    host.chat.call_chat_json = chat
    reply = host.bridge(turn(GOAL))
    front_reviews = [c for c in host.chat.calls if '答案证据与覆盖复核' in str(c.get('system_extra_blocks'))]
    assert len(front_reviews) == 1
    assert not any('仅纠正' in str(c.get('system_extra_blocks')) for c in host.chat.calls)
    if enabled:
        assert reply.background_task_id
        assert drain(host, reply)['status'] == 'completed'
        task = saved_task(host)
        assert task['status'] == 'completed' and task['completion']['accepted'] is True
        assert task['research']['answer_review_calls'] == 2
        assert task['execution_result']['execution_report']['text'] == ANSWER
        assert task['budget']['tools'] == 3
        assert task['budget']['requests'] == len(host.chat.calls) + len(host.execution.calls)
        assert [r['url'] for r in host.reader.calls] == [URL_A, URL_B]
        assert len([r for r in task['tool_calls'] if r.get('phase') == 'foreground']) == 2
    else:
        assert reply.agent_status == 'partial' and not reply.background_task_id
        assert not host.execution.calls and not host.svc.coordinator.list()
        assert [r['url'] for r in host.reader.calls] == [URL_A]
    assert len(host.search.calls) == 1 and not host.mutation.calls
