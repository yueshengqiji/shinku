"""Explicit capacity reaches the SDK without weakening facts, budget or replay.

Controlled checker verdicts prove contract wiring, not model semantic accuracy.
"""
import copy
import json
from dataclasses import replace

import httpx
import pytest
from openai import OpenAI

from shinku.agent.budget import TaskBudget, bind_task_budget
from shinku.agent_runtime.contracts import canonical_hash
from shinku.llm.runtime_expression import RuntimeExpressionAdapter
from shinku.result_expression import ExpressionArchive, ExpressionProfile, ExpressionService
from test_expression_runtime import FixtureRuntime
from test_expression_semantic_boundary import sdk_pair
from test_result_expression import SCOPE, canonical, host, prepare

STYLE = 'CAPACITY_PRIVATE_SENTINEL_' + '保留全部风格上下文。' * 1200


@pytest.mark.parametrize('tokens', [4096, 8192, 65536])
def test_full_style_and_explicit_output_reach_actual_sdk_and_keep_host_facts(host, tokens):
    source = canonical(); before = copy.deepcopy(source)
    budget = TaskBudget(max_requests=1, max_tokens=4096, max_tools=1, max_active_seconds=60)
    budget.requests = 1; budget_before = budget.snapshot()
    with sdk_pair('我把结果和限制整理在这里。') as (model, checker, calls):
        profile = ExpressionProfile(model, checker=checker, style_prompt=STYLE,
                                    configuration_id='capacity-fixture', max_output_tokens=tokens)
        svc = ExpressionService(ExpressionArchive(host), profile)
        with bind_task_budget(budget):
            record = prepare(svc, source)
        assert record['status'] == 'ready' and record['check']['accepted']
        assert [role for role, wire, payload in calls] == ['expression', 'check']
        assert calls[0][1]['messages'][0]['content'].endswith(STYLE)
        assert STYLE not in calls[1][1]['messages'][0]['content']
        assert 'CAPACITY_PRIVATE_SENTINEL' not in json.dumps(calls[1], ensure_ascii=False)
        for _, wire, payload in calls:
            assert wire['max_tokens'] == tokens and not wire.get('tools') and not wire.get('tool_choice')
        assert record['usage']['expression']['total_tokens'] == record['usage']['check']['total_tokens'] == 16
        assert source == before and record['canonical_result'] == before
        assert record['delivery']['status'] == 'not_sent'
        budget_after = budget.snapshot()
        assert {k:v for k,v in budget_after.items() if k != 'active_seconds'} == {k:v for k,v in budget_before.items() if k != 'active_seconds'}
        assert budget_after['active_seconds'] >= budget_before['active_seconds']
        assert prepare(svc, source) == record and len(calls) == 2


@pytest.mark.parametrize('values', [
    {'max_output_tokens': True}, {'max_output_tokens': 127}, {'max_output_tokens': 65537},
    {'style_prompt': 'x' * 65537}, {'max_input_chars': 65537},
])
def test_capacity_remains_finite_and_invalid_selection_has_no_transport(values):
    with pytest.raises(ValueError, match='invalid_expression'):
        ExpressionProfile(**values)


def test_default_identity_and_limits_remain_backward_compatible():
    profile = ExpressionProfile()
    assert profile.max_output_tokens == 2048 and profile.max_input_chars == 24000 and profile.style_prompt == ''
    legacy = {'id': 'template-v1', 'style': '', 'runtime': 'NoneType', 'checker': 'NoneType',
              'timeout': 10, 'check_timeout': 10, 'input': 24000, 'output': 2048, 'format': 1}
    assert profile.config_hash == canonical_hash(legacy)
    assert replace(profile, max_output_tokens=8192).config_hash != profile.config_hash
    assert replace(profile, style_prompt=STYLE).config_hash != profile.config_hash


def test_full_style_above_input_budget_is_rejected_without_trimming_or_call(host):
    with sdk_pair('不会被调用。') as (model, checker, calls):
        svc = ExpressionService(ExpressionArchive(host), ExpressionProfile(model, checker=checker,
            style_prompt=STYLE, max_output_tokens=8192, max_input_chars=512))
        source = canonical(); before = copy.deepcopy(source)
        record = prepare(svc, source)
        assert record['fallback_reason'] == 'expression_input_oversized' and not calls
        assert record['canonical_result'] == source == before
        assert '12.5 ms' in record['text'] and '未部署' in record['text']


@pytest.mark.parametrize('tokens', [8192, 65536])
@pytest.mark.parametrize('finish,tools,reason', [
    ('length', None, 'expression_output_truncated'),
    ('content_filter', None, 'expression_output_rejected'),
    ('stop', [{'id':'injected', 'type':'function', 'function':{'name':'shell','arguments':'{}'}}],
     'expression_native_tool_call_forbidden'),
])
def test_larger_output_preserves_terminal_protocol_failure_and_no_retry(host, tokens, finish, tools, reason):
    calls = []
    def handle(request):
        wire = json.loads(request.content); calls.append(wire)
        payload = json.loads(wire['messages'][-1]['content'])
        candidate = {'binding_hash':payload['binding_hash'], 'layout':[{'fact':b['id']} for b in payload['fact_blocks']]}
        return httpx.Response(200, json={'id':'offline','object':'chat.completion','created':0,'model':'offline',
            'choices':[{'index':0,'finish_reason':finish,'message':{'role':'assistant',
                'content':json.dumps(candidate), 'tool_calls':tools}}],
            'usage':{'prompt_tokens':12,'completion_tokens':4,'total_tokens':16}})
    with httpx.Client(transport=httpx.MockTransport(handle), trust_env=False) as http:
        with OpenAI(api_key='offline',base_url='https://capacity.invalid/v1',max_retries=0,http_client=http) as sdk:
            adapter = RuntimeExpressionAdapter(FixtureRuntime(sdk.chat.completions.create))
            svc = ExpressionService(ExpressionArchive(host), ExpressionProfile(adapter, style_prompt=STYLE,
                max_output_tokens=tokens, checker=adapter))
            source = canonical(); before = copy.deepcopy(source)
            record = prepare(svc, source)
            assert record['status'] == 'fallback' and record['fallback_reason'] == reason
            assert len(calls) == 1 and calls[0]['max_tokens'] == tokens
            assert source == record['canonical_result'] == before
            assert record['delivery']['status'] == 'not_sent'
            assert prepare(svc, source) == record and len(calls) == 1


def test_capacity_change_cannot_buy_another_expression_for_cached_result(host):
    with sdk_pair('结果在这里。') as (model, checker, calls):
        profile = ExpressionProfile(model, checker=checker, style_prompt='完整原风格', max_output_tokens=4096)
        first = prepare(ExpressionService(ExpressionArchive(host), profile))
        assert first['status'] == 'ready' and len(calls) == 2
        for altered in (replace(profile,max_output_tokens=8192), replace(profile,style_prompt=STYLE)):
            record = prepare(ExpressionService(ExpressionArchive(host), altered))
            assert record['status'] == 'discarded' and record['fallback_reason'] == 'expression_configuration_changed'
            assert len(calls) == 2 and record['text'] == ''
            stored = ExpressionArchive(host).get(first['expression_id'],scope=SCOPE)
            assert stored == first and stored['delivery']['status'] == 'not_sent'


def test_observed_completion_restatement_with_inconsistent_checker_still_falls_back(host):
    with sdk_pair('任务已按要求完成。', verdict={'supported':True,'additional_claims':['重复完成状态']}) as (model,checker,calls):
        svc = ExpressionService(ExpressionArchive(host), ExpressionProfile(model,checker=checker,style_prompt='简洁中文'))
        source = canonical(); before = copy.deepcopy(source)
        record = prepare(svc, source)
        assert record['fallback_reason'] == 'expression_neutral_check_rejected' and record['candidate'] is None
        assert source == record['canonical_result'] == before and record['delivery']['status'] == 'not_sent'
        assert prepare(svc,source) == record and len(calls) == 2


@pytest.mark.parametrize('status,accepted,verdict,expected', [
    ('completed',True,{},'ready'),
    ('partial',False,{'supported':False,'contradictions':['当前未完成']},'fallback'),
])
def test_same_completion_sentence_uses_contextual_verdict_without_word_exceptions(host, status, accepted, verdict, expected):
    with sdk_pair('任务已按要求完成。',verdict=verdict) as (model,checker,calls):
        source=canonical(status,accepted=accepted); before=copy.deepcopy(source)
        svc=ExpressionService(ExpressionArchive(host),ExpressionProfile(model,checker=checker,style_prompt='简洁中文'))
        record=prepare(svc,source)
        assert record['status']==expected and source==record['canonical_result']==before
        system=calls[1][1]['messages'][0]['content']
        assert 'additional_claims只列事实块无法支持的新增命题' in system
        assert '同义重述不属于新增事实' in system
        assert '有额外断言、矛盾或遗漏限制时supported=false' in system
        assert record['delivery']['status']=='not_sent'
