"""Research review uses the real SDK/parser and retains only safe diagnostics."""
import copy
import json
from contextlib import contextmanager

import httpx
from openai import OpenAI
import pytest

from shinku.agent.loop import AgentDecision, AgentState
from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget
from shinku.agent.planner import LLMPlanner
from shinku.llm.policy import RuntimePolicy
from test_expression_runtime import FixtureRuntime
from test_research_evidence_ledger import ANSWER, BODY, URL, Reader, read_ledger, receipt, review
from shinku.tools.invocation import ToolInvocation


@contextmanager
def sdk_review(content, *, after_response=None, native=False):
    calls = []

    def handle(request):
        wire = json.loads(request.content)
        calls.append(wire)
        assert not wire.get('tools') and not wire.get('tool_choice')
        if isinstance(content, Exception):
            raise content
        if after_response is not None:
            after_response(len(calls))
        message = dict(role='assistant', content=content[min(len(calls) - 1, len(content) - 1)]
                       if isinstance(content, list) else content)
        if native:
            message['tool_calls'] = [dict(id='unexpected', type='function',
                function=dict(name='shell', arguments='{"command":"PRIVATE_PROTOCOL_SENTINEL"}'))]
        return httpx.Response(200, json=dict(id='offline-research', object='chat.completion',
            created=0, model='fixture', choices=[dict(index=0, finish_reason='stop',
                message=message)],
            usage=dict(prompt_tokens=30, completion_tokens=20, total_tokens=50)))

    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as client:
        with OpenAI(api_key='offline-fixture', base_url='https://research.example.invalid/v1',
                    max_retries=0, http_client=client) as sdk:
            runtime = FixtureRuntime(sdk.chat.completions.create)
            runtime.runtime_policy = RuntimePolicy(http_attempts=1, retry_backoff_seconds=0)
            yield runtime, calls


def state_for_review():
    return AgentState(research=read_ledger(), context={'agent_completion_contract': {
        'requirements': [{'id': 'release', 'effect': 'research_answer', 'checks': {
            'required_fields': ['发布日期'], 'min_sources': 1, 'require_dates': True}}]}},
        transcript=[{'role': 'user', 'content': 'Cedar 的发布日期？'}])


def test_actual_sdk_research_review_has_valid_json_example_and_safe_parse_diagnostics():
    state = state_for_review()
    value = review(state.research)
    before = copy.deepcopy(state.research.sources)
    with sdk_review(json.dumps(value, ensure_ascii=False)) as (runtime, calls):
        planner = LLMPlanner(runtime=runtime, system_prompt='isolated-review')
        decision = AgentDecision.final(ANSWER, outcome='completed')
        planner._review_research_answer(state, decision)
        assert state.research.evaluate_answer(ANSWER, 'completed').accepted
        assert state.research.satisfies(state.context['agent_completion_contract']['requirements'][0]['checks'])
        system = '\n'.join(message['content'] for message in calls[0]['messages'] if message['role'] == 'system')
        example, _ = json.JSONDecoder().raw_decode(system.split('复核输出格式示例：', 1)[1])
        assert set(example) == {'verdict', 'claims', 'coverage', 'missing_fields', 'limitations'}
        assert example == dict(verdict='no_answer', claims=[], coverage=[], missing_fields=[], limitations=[])
        diagnostics = state.progress['answer_review_protocol']
        assert diagnostics['stages'] == ['json_object'] and diagnostics['verdict'] == 'supported'
        assert diagnostics['response_kind'] == 'object' and diagnostics['review_call'] == 1
        assert diagnostics['present_fields'] == ['claims', 'coverage', 'missing_fields', 'verdict']
        assert state.research.sources == before
        planner._review_research_answer(state, decision)
        assert len(calls) == state.research.answer_review_calls == 1


@pytest.mark.parametrize('fault', ['plain', 'missing_verdict', 'forged_source', 'forged_quote', 'unread_date'])
def test_protocol_correction_cannot_relax_research_evidence(fault):
    state = state_for_review()
    value = review(state.research)
    secret = 'PRIVATE_REVIEW_RESPONSE_SENTINEL'
    if fault == 'plain':
        content = secret + ' 全部正确。'
    else:
        if fault == 'missing_verdict':
            value.pop('verdict')
            value[secret] = 'unknown field with private content'
        elif fault == 'forged_source':
            value['claims'][0]['source_id'] = secret
        elif fault == 'forged_quote':
            value['claims'][0]['quote'] = secret
        else:
            value['claims'][0]['date_text'] = '2099-01-01'
        content = json.dumps(value, ensure_ascii=False)
    with sdk_review(content) as (runtime, calls):
        planner = LLMPlanner(runtime=runtime, system_prompt='isolated-review')
        decision = AgentDecision.final(ANSWER, outcome='completed')
        planner._review_research_answer(state, decision)
        assert not state.research.evaluate_answer(ANSWER, 'completed').accepted
        assert not state.research.satisfies(state.context['agent_completion_contract']['requirements'][0]['checks'])
        expected_calls = 2 if fault in {'plain', 'missing_verdict'} else 1
        assert len(calls) == state.research.answer_review_calls == expected_calls
        diagnostics = state.progress['answer_review_protocol']
        assert secret not in json.dumps(diagnostics)
        assert diagnostics['stages'] == (['parse_invalid'] if fault == 'plain' else ['json_object'])
        assert diagnostics['verdict'] == ('invalid' if fault in {'plain', 'missing_verdict'} else 'supported')
        planner._review_research_answer(state, decision)
        assert len(calls) == expected_calls


def test_source_change_preserves_two_review_limit_with_actual_sdk():
    state = state_for_review()
    with sdk_review(json.dumps(review(state.research))) as (runtime, calls):
        planner = LLMPlanner(runtime=runtime, system_prompt='isolated-review')
        decision = AgentDecision.final(ANSWER, outcome='completed')
        planner._review_research_answer(state, decision)
        state.research.observe(ToolInvocation('reader', {'url': URL}), receipt({'url': URL,
            'page_text': BODY + ' Changed version.', 'http_status': 200, 'site_access_ok': True}), Reader())
        planner._review_research_answer(state, decision)
        assert len(calls) == state.research.answer_review_calls == 2
        assert not state.research.evaluate_answer(ANSWER, 'completed').accepted
        planner._review_research_answer(state, AgentDecision.final(ANSWER + ' Extra.', outcome='completed'))
        assert len(calls) == 2


@pytest.mark.parametrize('fault', ['plain', 'missing_verdict'])
def test_actual_sdk_protocol_only_recovery_rechecks_same_sources_and_answer_once(fault):
    state = state_for_review()
    valid = review(state.research)
    invalid = 'PRIVATE_PROTOCOL_SENTINEL not JSON' if fault == 'plain' else json.dumps({
        k: v for k, v in valid.items() if k != 'verdict'})
    before = copy.deepcopy(state.research.sources)
    with sdk_review([invalid, json.dumps(valid, ensure_ascii=False)]) as (runtime, calls):
        planner = LLMPlanner(runtime=runtime, system_prompt='isolated-review')
        decision = AgentDecision.final(ANSWER, outcome='completed')
        planner._review_research_answer(state, decision)
        assert state.research.evaluate_answer(ANSWER, 'completed').accepted
        assert state.research.sources == before
        assert len(calls) == state.research.answer_review_calls == 2
        first = next(m['content'] for m in calls[0]['messages'] if m['role'] == 'user')
        second = next(m['content'] for m in calls[1]['messages'] if m['role'] == 'user')
        assert first == second
        blocks = '\n'.join(m['content'] for m in calls[1]['messages'] if m['role'] == 'system')
        assert '仅纠正复核协议' in blocks
        assert 'PRIVATE_PROTOCOL_SENTINEL' not in blocks + second
        diagnostics = state.progress['answer_review_protocol_history']
        assert len(diagnostics) == 2 and diagnostics[0]['verdict'] == 'invalid' and diagnostics[1]['verdict'] == 'supported'
        assert 'PRIVATE_PROTOCOL_SENTINEL' not in json.dumps(diagnostics)
        planner._review_research_answer(state, decision)
        assert len(calls) == 2


def test_protocol_correction_remains_within_existing_global_review_allowance():
    state = state_for_review()
    state.research.answer_review_calls = 1
    with sdk_review('invalid review') as (runtime, calls):
        planner = LLMPlanner(runtime=runtime, system_prompt='isolated-review')
        decision = AgentDecision.final(ANSWER, outcome='completed')
        planner._review_research_answer(state, decision)
        assert len(calls) == 1 and state.research.answer_review_calls == 2
        assert not state.research.evaluate_answer(ANSWER, 'completed').accepted
        planner._review_research_answer(state, decision)
        assert len(calls) == 1


@pytest.mark.parametrize('stop', ['request_limit', 'canceled'])
def test_actual_sdk_protocol_correction_cannot_bypass_task_budget_or_cancel(stop):
    state = state_for_review()
    budget = TaskBudget(max_requests=1 if stop == 'request_limit' else 4, max_tokens=100000,
                        max_tools=1, max_active_seconds=10)
    def stop_after_first(number):
        if stop == 'canceled':
            budget.reason = 'task_canceled'
    with sdk_review('invalid review', after_response=stop_after_first) as (runtime, calls), bind_task_budget(budget):
        planner = LLMPlanner(runtime=runtime, system_prompt='isolated-review')
        with pytest.raises(TaskBudgetExceeded, match='task_model_budget_exhausted' if stop == 'request_limit' else 'task_canceled'):
            planner._review_research_answer(state, AgentDecision.final(ANSWER, outcome='completed'))
        assert len(calls) == budget.requests == 1 and budget.tools == 0
        assert state.research.answer_review_calls == 2
        assert not state.research.evaluate_answer(ANSWER, 'completed').accepted


@pytest.mark.parametrize('fault', ['transport', 'native'])
def test_transport_failure_and_unrequested_tool_output_are_not_protocol_corrected(fault):
    state = state_for_review()
    content = httpx.ReadTimeout('PRIVATE_PROTOCOL_SENTINEL') if fault == 'transport' else ''
    with sdk_review(content, native=fault == 'native') as (runtime, calls):
        planner = LLMPlanner(runtime=runtime, system_prompt='isolated-review')
        planner._review_research_answer(state, AgentDecision.final(ANSWER, outcome='completed'))
        assert len(calls) == state.research.answer_review_calls == 1
        assert not state.research.evaluate_answer(ANSWER, 'completed').accepted
        diagnostic = state.progress['answer_review_protocol_history']
        assert diagnostic[-1]['stages'] == ['transport_error' if fault == 'transport' else 'native_call']
        assert 'PRIVATE_PROTOCOL_SENTINEL' not in json.dumps(diagnostic)


@pytest.mark.parametrize('fault', ['forged_source', 'forged_quote', 'unread_date', 'unsupported'])
def test_valid_corrected_protocol_cannot_admit_unbound_or_unsupported_fact(fault):
    state = state_for_review()
    value = review(state.research)
    if fault == 'forged_source':
        value['claims'][0]['source_id'] = 'PRIVATE_PROTOCOL_SENTINEL'
    elif fault == 'forged_quote':
        value['claims'][0]['quote'] = 'PRIVATE_PROTOCOL_SENTINEL'
    elif fault == 'unread_date':
        value['claims'][0]['date_text'] = '2099-01-01'
    else:
        value['verdict'] = 'unsupported'
    with sdk_review(['invalid review', json.dumps(value)]) as (runtime, calls):
        planner = LLMPlanner(runtime=runtime, system_prompt='isolated-review')
        planner._review_research_answer(state, AgentDecision.final(ANSWER, outcome='completed'))
        assert len(calls) == state.research.answer_review_calls == 2
        assert not state.research.evaluate_answer(ANSWER, 'completed').accepted
        assert 'PRIVATE_PROTOCOL_SENTINEL' not in json.dumps(state.progress['answer_review_protocol_history'])


@pytest.mark.parametrize('change_at', [1, 2])
def test_protocol_correction_never_freezes_acceptance_across_source_change(change_at):
    state = state_for_review()
    value = json.dumps(review(state.research))
    def change_source(number):
        if number == change_at:
            state.research.observe(ToolInvocation('reader', {'url': URL}), receipt({'url': URL,
                'page_text': BODY + ' Changed version.', 'http_status': 200, 'site_access_ok': True}), Reader())
    with sdk_review(['invalid review', value], after_response=change_source) as (runtime, calls):
        planner = LLMPlanner(runtime=runtime, system_prompt='isolated-review')
        decision = AgentDecision.final(ANSWER, outcome='completed')
        planner._review_research_answer(state, decision)
        assert len(calls) == change_at
        assert not state.research.evaluate_answer(ANSWER, 'completed').accepted
