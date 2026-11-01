"""Incomplete SDK responses cannot validate research or trigger format recovery."""
import copy
import json
from contextlib import contextmanager

import httpx
from openai import OpenAI
import pytest

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.agent.loop import AgentDecision
from shinku.agent.planner import LLMPlanner
from shinku.llm.policy import RuntimePolicy
from shinku.llm.response_protocol import observe_response, plain_reply_enabled
from test_expression_runtime import FixtureRuntime
from test_research_answer_protocol import state_for_review
from test_research_evidence_ledger import ANSWER, review


@contextmanager
def incomplete_sdk(replies):
    calls = []

    def handle(request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert not payload.get('tools') and not payload.get('tool_choice')
        content, finish = replies[min(len(calls) - 1, len(replies) - 1)]
        return httpx.Response(200, json=dict(id='offline-incomplete-review',
            object='chat.completion', created=0, model='fixture',
            choices=[dict(index=0, finish_reason=finish,
                message=dict(role='assistant', content=content,
                    reasoning_content='PRIVATE_INCOMPLETE_REASONING_SENTINEL'))],
            usage=dict(prompt_tokens=30, completion_tokens=4096 if finish == 'length' else 20,
                total_tokens=4126 if finish == 'length' else 50)))

    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as client:
        with OpenAI(api_key='offline-fixture', base_url='https://research.example.invalid/v1',
                    max_retries=0, http_client=client) as sdk:
            runtime = FixtureRuntime(sdk.chat.completions.create)
            runtime.runtime_policy = RuntimePolicy(http_attempts=1, retry_backoff_seconds=0)
            yield runtime, calls


@pytest.mark.parametrize('finish', ['length', 'content_filter'])
@pytest.mark.parametrize('shape', ['empty', 'partial', 'valid_json'])
def test_default_sdk_incomplete_review_is_terminal_and_cannot_grant_acceptance(finish, shape):
    state = state_for_review()
    valid = json.dumps(review(state.research), ensure_ascii=False)
    first = '' if shape == 'empty' else ('{"verdict":"supported","claims":["PRIVATE_INCOMPLETE_SENTINEL' if shape == 'partial' else valid)
    before = copy.deepcopy(state.research.sources)
    budget = TaskBudget(max_requests=4, max_tokens=100000, max_tools=1, max_active_seconds=30)
    assert not plain_reply_enabled()
    with incomplete_sdk([(first, finish), (valid, 'stop')]) as (runtime, calls), bind_task_budget(budget):
        planner = LLMPlanner(runtime=runtime, system_prompt='isolated-review')
        decision = AgentDecision.final(ANSWER, outcome='completed')
        planner._review_research_answer(state, decision)
        assert not state.research.evaluate_answer(ANSWER, 'completed').accepted
        assert not state.research.satisfies(state.context['agent_completion_contract']['requirements'][0]['checks'])
        assert len(calls) == budget.requests == state.research.answer_review_calls == 1
        assert budget.tools == 0 and state.research.sources == before
        history = state.progress['answer_review_protocol_history']
        assert len(history) == 1 and history[0]['stages'][0] == 'output_incomplete'
        assert history[0]['verdict'] == 'invalid' and history[0]['present_fields'] == []
        assert 'PRIVATE_INCOMPLETE' not in json.dumps(history)
        planner._review_research_answer(state, decision)
        assert len(calls) == 1
    assert not plain_reply_enabled()


@pytest.mark.parametrize('finish', ['length', 'content_filter'])
def test_default_json_compatibility_keeps_value_but_observes_incomplete(finish):
    value = {'legacy': 'PRIVATE_LEGACY_VALUE'}
    with incomplete_sdk([(json.dumps(value), finish)]) as (runtime, calls):
        with observe_response() as observation:
            result = runtime.call_chat_json(system_prompt='legacy', user_prompt='data', fallback={})
            assert result == value and len(calls) == 1
            assert observation.events == [{'stage': 'output_incomplete'}, {'stage': 'json_object'}]
            assert 'PRIVATE_LEGACY' not in json.dumps(observation.events)
    assert not plain_reply_enabled()


@pytest.mark.parametrize('finish', ['length', 'content_filter'])
def test_plain_task_json_still_discards_incomplete_before_parse(finish):
    fallback = {'kind': 'final', 'text': ''}
    with incomplete_sdk([('{"kind":"final","text":"unsafe"}', finish)]) as (runtime, calls):
        with observe_response(allow_plain_reply=True) as observation:
            result = runtime.call_chat_json(system_prompt='task', user_prompt='data', fallback=fallback)
            assert result == fallback and len(calls) == 1
            assert observation.events == [{'stage': 'output_incomplete'}]
    assert not plain_reply_enabled()
