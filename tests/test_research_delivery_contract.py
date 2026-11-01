"""Research delivery keeps internal fields out of the independently checked answer."""
import copy
import json

import pytest

from shinku.agent.loop import AgentState
from shinku.agent.planner import LLMPlanner
from test_research_answer_protocol import sdk_review, state_for_review
from test_research_evidence_ledger import ANSWER, BODY, review


@pytest.mark.parametrize('tool_feedback', [False, True])
def test_actual_sdk_final_schema_separates_user_answer_and_internal_outcome(tool_feedback):
    state = state_for_review()
    if tool_feedback:
        state.transcript.append(dict(role='tool', name='reader', status='ok', content=BODY))
    sources = copy.deepcopy(state.research.sources)
    final = dict(kind='final', text=ANSWER, task_outcome='completed')
    with sdk_review([json.dumps(final, ensure_ascii=False), json.dumps(review(state.research), ensure_ascii=False)]) as (runtime, calls):
        decision = LLMPlanner(runtime=runtime, system_prompt='isolated-research-delivery')(state=state)
        assert decision.text == ANSWER and decision.outcome == 'completed'
        assert state.research.evaluate_answer(decision.text, decision.outcome).accepted
        assert len(calls) == 2 and not any(c.get('tools') or c.get('tool_choice') for c in calls)
        system = '\n'.join(m['content'] for m in calls[0]['messages'] if m['role'] == 'system')
        example, _ = json.JSONDecoder().raw_decode(system.split('研究交付格式示例：', 1)[1])
        assert set(example) == {'kind', 'text', 'task_outcome'} and example['kind'] == 'final'
        assert example['task_outcome'] == 'completed' and 'task_outcome' not in example['text']
        assert state.research.sources == sources and not state.tool_calls


def test_actual_sdk_unbound_transport_claim_cannot_be_accepted_as_public_source_fact():
    state = state_for_review()
    sources = copy.deepcopy(state.research.sources)
    answer = ANSWER + ' 页面返回 HTTP 200。'
    verdict = review(state.research)
    verdict['verdict'] = 'unsupported'
    verdict['claims'].append(dict(answer_fragment='HTTP 200', source_id='', quote='', date_text='', date_kind='unknown'))
    with sdk_review([json.dumps(dict(kind='final', text=answer, task_outcome='completed'), ensure_ascii=False),
                     json.dumps(verdict, ensure_ascii=False)]) as (runtime, calls):
        decision = LLMPlanner(runtime=runtime, system_prompt='isolated-research-delivery')(state=state)
        assert decision.text == answer
        assert not state.research.evaluate_answer(answer, 'completed').accepted
        assert state.research.answer_check['errors'] == ['claim_1:unbound_text_or_source']
        assert len(calls) == 2 and state.research.sources == sources and not state.tool_calls


def test_ordinary_final_reply_does_not_add_a_research_review_or_change_the_answer():
    state = AgentState(transcript=[dict(role='user', content='你好')])
    with sdk_review(json.dumps(dict(kind='final', text='你好。', task_outcome='completed'), ensure_ascii=False)) as (runtime, calls):
        decision = LLMPlanner(runtime=runtime, system_prompt='isolated-ordinary-delivery')(state=state)
        assert decision.text == '你好。' and decision.outcome == 'completed'
        assert len(calls) == 1 and state.research.answer_review_calls == 0
        assert not state.tool_calls and not calls[0].get('tools')
