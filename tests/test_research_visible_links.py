"""Visible citation punctuation must preserve exact source identity."""
import json
from dataclasses import replace

import pytest

from shinku.agent.research import visible_urls
from test_public_query_partial_outcome import partial_evidence
from test_qq_independent_agent import turn
from test_qq_query_continuation import ANSWER, GOAL, PARTIAL, URL_A, URL_B, drain, query_host, saved_task
from test_research_answer_protocol import sdk_review


@pytest.mark.parametrize('wrap', ['（{}）', '{}（原文：已实际读取。）', '「{}」', '『{}』',
    '【{}】', '《{}》', '〈{}〉', '“{}”', '‘{}’', '`{}`', '[来源]({})', '<{}>', '{}。'])
def test_citation_wrappers_leave_the_exact_url_visible(wrap):
    assert visible_urls(wrap.format(URL_A)) == {URL_A}


@pytest.mark.parametrize('url', [URL_A + '/other', URL_A + 'evil', URL_A + '?extra=1',
    'https://forged.example/a', 'https://user:password@source.example/a'])
def test_an_address_cannot_be_prefix_matched_to_a_different_source(url):
    assert URL_A not in visible_urls('（' + url + '）')


def test_unicode_path_and_encoded_punctuation_are_preserved():
    url = 'https://source.example/茶杯?q=%EF%BC%89'
    assert visible_urls('「' + url + '」') == {url}


@pytest.mark.parametrize('wrap', ['（{}）', '{}（原文：已读取。）', '`{}`'])
def test_real_sdk_partial_review_preserves_text_and_final_review_allowance(wrap):
    value, _, review = partial_evidence()
    answer = PARTIAL.replace(URL_A, wrap.format(URL_A))
    # Keep semantic review fragments independently bound to the shown text.
    review['claims'][0]['answer_fragment'] = review['claims'][0]['answer_fragment'].replace(URL_A, wrap.format(URL_A))
    with sdk_review(json.dumps(review, ensure_ascii=False)) as (runtime, calls):
        value.planner.runtime = runtime
        assert value.finish(answer, 'completed', admitted=lambda: True) == (
            answer, 'partial', 'partial', 'public_query_partial_outcome_reconciled')
        assert len(calls) == value.ledger.answer_review_calls == 1
        assert value.correction_calls == 0
        assert value.ledger.evaluate_answer(answer, 'partial').accepted


@pytest.mark.parametrize('enabled', [False, True])
def test_actual_query_worker_with_chinese_citations_keeps_single_task_and_cumulative_review(query_host, enabled):
    host = query_host
    host.bridge.foreground_query_policy = replace(host.bridge.foreground_query_policy, enabled=enabled)
    answer = PARTIAL.replace(URL_A, '（' + URL_A + '）')
    host.chat.replies[-1].update(text=answer, task_outcome='completed')
    call = host.chat.call_chat_json
    def chat(**options):
        value = call(**options)
        if '答案证据与覆盖复核' in str(options.get('system_extra_blocks')):
            value['limitations'][0]['condition_ids'] = ['public_original']
        return value
    host.chat.call_chat_json = chat
    reply = host.bridge(turn(GOAL))
    assert len([c for c in host.chat.calls if '答案证据与覆盖复核' in str(c.get('system_extra_blocks'))]) == 1
    assert not any('仅纠正' in str(c.get('system_extra_blocks')) for c in host.chat.calls)
    if enabled:
        assert drain(host, reply)['status'] == 'completed'
        task = saved_task(host)
        assert task['status'] == 'completed' and task['completion']['accepted'] is True
        assert task['research']['answer_review_calls'] == 2
        assert task['budget']['requests'] == len(host.chat.calls) + len(host.execution.calls)
        assert task['execution_result']['execution_report']['text'] == ANSWER
        assert len(host.svc.task_store.candidates(requester=task['requester'], session=task['session'])) == 1
        assert [r['url'] for r in host.reader.calls] == [URL_A, URL_B]
    else:
        assert reply.agent_status == 'partial' and not reply.background_task_id
        assert not host.execution.calls and [r['url'] for r in host.reader.calls] == [URL_A]
    assert len(host.search.calls) == 1 and not host.mutation.calls
