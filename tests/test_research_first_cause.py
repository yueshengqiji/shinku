"""An incomplete real SDK response stays distinct from invalid review syntax."""
import copy
import json

import pytest

from shinku.agent.loop import AgentDecision
from shinku.agent.planner import LLMPlanner
from shinku.llm.response_protocol import observe_response
from test_research_answer_protocol import state_for_review
from test_research_incomplete_protocol import incomplete_sdk
from test_research_evidence_ledger import ANSWER, review


@pytest.mark.parametrize('finish,reason', [
    ('length', 'research_review_output_truncated'),
    ('content_filter', 'research_review_output_rejected'),
])
@pytest.mark.parametrize('content_kind', ['empty', 'malformed', 'complete'])
def test_first_cause_is_retained_without_approving_partial_or_retrying(finish, reason, content_kind):
    state = state_for_review()
    valid = json.dumps(review(state.research), ensure_ascii=False)
    content = '' if content_kind == 'empty' else ('{"PRIVATE_FIRST_CAUSE":"unfinished' if content_kind == 'malformed' else valid)
    before = copy.deepcopy(state.research.sources)
    with incomplete_sdk([(content, finish), (valid, 'stop')]) as (runtime, calls):
        planner = LLMPlanner(runtime=runtime, system_prompt='isolated-review')
        planner._review_research_answer(state, AgentDecision.final(ANSWER, outcome='completed'))
        diagnostic = state.progress['answer_review_protocol']
        assert diagnostic['failure_reason'] == reason
        assert diagnostic['finish_reason'] == finish
        assert state.research.answer_check['protocol_failure'] == reason
        assert 'invalid_semantic_review' in state.research.answer_check['errors']
        verdict = state.research.evaluate_answer(ANSWER, 'completed')
        assert not verdict.accepted and reason in verdict.reason
        assert not state.research.satisfies(state.context['agent_completion_contract']['requirements'][0]['checks'])
        assert len(calls) == state.research.answer_review_calls == 1
        assert state.research.sources == before
        assert 'PRIVATE_FIRST_CAUSE' not in json.dumps(diagnostic)
        planner._review_research_answer(state, AgentDecision.final(ANSWER, outcome='completed'))
        assert len(calls) == 1


def test_response_observer_keeps_existing_events_and_nested_calls_isolated():
    valid = '{"fixture":"PRIVATE_FIRST_CAUSE"}'
    with incomplete_sdk([(valid, 'length'), (valid, 'content_filter')]) as (runtime, calls):
        with observe_response() as outer:
            assert runtime.call_chat_json(system_prompt='fixture', user_prompt='data', fallback={}) == json.loads(valid)
            with observe_response() as inner:
                runtime.call_chat_json(system_prompt='fixture', user_prompt='data', fallback={})
            assert outer.finish_reason == 'length' and inner.finish_reason == 'content_filter'
            assert outer.events == inner.events == [{'stage': 'output_incomplete'}, {'stage': 'json_object'}]
            assert 'PRIVATE_FIRST_CAUSE' not in json.dumps(vars(outer))
        assert len(calls) == 2
