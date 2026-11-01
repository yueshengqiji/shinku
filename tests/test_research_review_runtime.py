"""Separate actual SDK review requests without changing execution authority."""
from dataclasses import replace
import json

import pytest

from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget
from shinku.agent.loop import AgentDecision
from shinku.agent.planner import LLMPlanner
from shinku.agent_runtime import AgentBoundaryError, AgentRequest, AgentScope, ExecutionProfile
from shinku.agent_runtime.execution import _MeteredRuntime
from shinku.agent_runtime.research_review import ResearchAnswerReviewRuntime
from test_research_answer_protocol import sdk_review, state_for_review
from test_research_evidence_ledger import ANSWER, review


def separated(primary, checker, key="review-low-v1"):
    return ResearchAnswerReviewRuntime(primary, checker,
        execution_profile_id="executor-high-v1", review_profile_id=key)


def test_actual_sdk_review_uses_separate_transport_and_original_source_checks():
    state = state_for_review()
    value = json.dumps(review(state.research), ensure_ascii=False)
    with sdk_review('{}') as (primary, execution_calls), sdk_review(value) as (checker, review_calls):
        runtime = separated(primary, checker)
        runtime.call_chat_json(system_prompt="execution", user_prompt="goal", fallback={})
        planner = LLMPlanner(runtime=runtime, system_prompt="execution")
        planner._review_research_answer(state, AgentDecision.final(ANSWER, outcome="completed"))
        assert len(execution_calls) == len(review_calls) == 1
        assert state.research.evaluate_answer(ANSWER, "completed").accepted
        assert state.research.satisfies(state.context['agent_completion_contract']['requirements'][0]['checks'])
        wire = review_calls[0]
        assert not wire.get('tools') and not wire.get('tool_choice')
        payload = json.loads(next(m['content'] for m in wire['messages'] if m['role'] == 'user'))
        assert payload['read_sources'] and payload['answer'] == ANSWER
        planner._review_research_answer(state, AgentDecision.final(ANSWER, outcome="completed"))
        assert len(review_calls) == 1


@pytest.mark.parametrize('meter_main,meter_review', [(True, True), (True, False), (False, True), (False, False)])
def test_both_transports_share_one_request_budget_without_double_charging(meter_main, meter_review):
    state = state_for_review()
    budget = TaskBudget(max_requests=2, max_tokens=600000)
    with sdk_review('{}') as (main, main_calls), sdk_review(json.dumps(review(state.research))) as (check, check_calls):
        runtime = separated(main if meter_main else _MeteredRuntime(main),
                            check if meter_review else _MeteredRuntime(check))
        with bind_task_budget(budget):
            runtime.call_chat_json(system_prompt="executor", user_prompt="goal", fallback={})
            planner = LLMPlanner(runtime=runtime, system_prompt="executor")
            planner._review_research_answer(state, AgentDecision.final(ANSWER, outcome="completed"))
            assert budget.requests == 2
            with pytest.raises(TaskBudgetExceeded):
                runtime.call_research_answer_review_json(system_prompt="review", user_prompt="data", fallback={})
        assert len(main_calls) == len(check_calls) == 1
        assert state.research.evaluate_answer(ANSWER, "completed").accepted


def test_changing_review_selection_invalidates_review_cache_within_original_two_call_limit():
    state = state_for_review()
    value = json.dumps(review(state.research))
    with sdk_review('{}') as (main, main_calls), sdk_review(value) as (checker, calls):
        decision = AgentDecision.final(ANSWER, outcome='completed')
        first = LLMPlanner(runtime=separated(main, checker, 'low-v1'), system_prompt='execution')
        first._review_research_answer(state, decision)
        old_key = state.research.answer_check['review_key']
        second = LLMPlanner(runtime=separated(main, checker, 'low-v2'), system_prompt='execution')
        second._review_research_answer(state, decision)
        assert state.research.answer_check['review_key'] != old_key
        assert len(calls) == state.research.answer_review_calls == 2 and main_calls == []
        second._review_research_answer(state, decision)
        assert len(calls) == 2


@pytest.mark.parametrize('field,value', [
    ('native_tools', [{'type': 'function'}]), ('native_tool_choice', 'auto'),
    ('user_images', [{'url': 'data:image/png;base64,AA=='}]),
    ('native_history', [{'role': 'tool', 'content': 'untrusted'}]), ('provider_message_sink', lambda _: None),
])
def test_review_cannot_request_tools_or_inherit_provider_execution_history(field, value):
    with sdk_review('{}') as (main, main_calls), sdk_review('{}') as (checker, review_calls):
        with pytest.raises(AgentBoundaryError, match='research_review_tools_forbidden'):
            separated(main, checker).call_research_answer_review_json(**{field: value})
        assert main_calls == review_calls == []


def test_review_transport_error_does_not_fall_back_to_execution_model():
    class Unavailable:
        def call_chat_json(self, **options):
            raise TimeoutError('PRIVATE_TRANSPORT_SENTINEL')
    state = state_for_review()
    with sdk_review('{}') as (main, calls):
        planner = LLMPlanner(runtime=separated(main, Unavailable()), system_prompt='execution')
        planner._review_research_answer(state, AgentDecision.final(ANSWER, outcome='completed'))
        assert calls == [] and state.research.answer_review_calls == 1
        assert not state.research.evaluate_answer(ANSWER, 'completed').accepted
        assert 'PRIVATE_TRANSPORT_SENTINEL' not in json.dumps(state.progress)


def test_unexpected_native_review_response_is_not_execution_evidence():
    state = state_for_review()
    with sdk_review('{}') as (main, main_calls), sdk_review('{}', native=True) as (checker, calls):
        planner = LLMPlanner(runtime=separated(main, checker), system_prompt='execution')
        planner._review_research_answer(state, AgentDecision.final(ANSWER, outcome='completed'))
        assert main_calls == [] and len(calls) == state.research.answer_review_calls == 1
        assert not state.research.evaluate_answer(ANSWER, 'completed').accepted
        assert 'native_call' in state.progress['answer_review_protocol']['stages']


@pytest.mark.parametrize('fault', ['quote', 'source', 'date'])
def test_separate_reviewer_cannot_supply_forged_source_evidence(fault):
    state = state_for_review()
    data = review(state.research)
    field = {'quote': 'quote', 'source': 'source_id', 'date': 'date_text'}[fault]
    data['claims'][0][field] = '2099-01-01' if fault == 'date' else 'forged'
    with sdk_review('{}') as (main, main_calls), sdk_review(json.dumps(data)) as (checker, calls):
        planner = LLMPlanner(runtime=separated(main, checker), system_prompt='execution')
        planner._review_research_answer(state, AgentDecision.final(ANSWER, outcome='completed'))
        assert not state.research.evaluate_answer(ANSWER, 'completed').accepted
        assert main_calls == [] and len(calls) == 1


def test_real_file_execution_is_unchanged_by_optional_review_transport(tmp_path):
    from shinku.agent_runtime.demo import demo_service, demo_request
    from test_agent_runtime_boundary import settled
    class Unused:
        def call_chat_json(self, **options):
            raise AssertionError('file execution must use primary runtime')
    service, tool, runtime = demo_service(tmp_path)
    try:
        service.executor.profile = replace(service.executor.profile,
            research_review_runtime=Unused(), research_review_profile_id='separate-review')
        client = service.bind(AgentScope('fixture', 'local:fixture'))
        handle = client.submit(demo_request(tool))
        task = settled(client, handle.task_id)
        assert task['status'] == 'completed' and tool.calls == 1 and len(runtime.calls) == 2
        assert client.result(handle.task_id)['verification']['accepted']
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.mark.parametrize('options', [
    {'research_review_profile_id': 'unbound'}, {'research_review_runtime_meters_budget': True},
    {'research_review_runtime': object(), 'research_review_profile_id': 'invalid'},
    {'research_review_runtime_meters_budget': 1},
])
def test_invalid_optional_host_review_configuration_is_rejected(options):
    with pytest.raises(AgentBoundaryError, match='invalid_research_review_profile'):
        ExecutionProfile(None, **options)


def test_caller_cannot_select_research_review_configuration():
    with pytest.raises(TypeError):
        AgentRequest('request', 'goal', {}, research_review_profile_id='caller-low')
