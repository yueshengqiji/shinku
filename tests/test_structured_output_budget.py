"""真实SDK隔离验证阶段额度、上下文预留、总预算与并发边界；不发外部请求。"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import threading

import httpx
from openai import OpenAI
import pytest

from shinku.llm.output_budget import structured_output_scope, current_output_budget, apply_output_budget
from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget
from shinku.agent_runtime.execution import _MeteredRuntime
from shinku.llm.runtime_types import ModelBundle
from test_expression_runtime import FixtureRuntime


@contextmanager
def sdk(responder=None,finish='stop'):
    calls = []
    def handle(request):
        payload=json.loads(request.content);calls.append(payload)
        body=responder(payload) if responder else {'reply':'合成答复'}
        return httpx.Response(200, json={'id':'offline-stage-budget','object':'chat.completion','created':0,
            'model':'deepseek-flash','choices':[{'index':0,'finish_reason':finish,
            'message':{'role':'assistant','content':json.dumps(body,ensure_ascii=False)}}],
            'usage':{'prompt_tokens':12,'completion_tokens':4,'total_tokens':16}})
    with httpx.Client(transport=httpx.MockTransport(handle),trust_env=False) as http:
        with OpenAI(api_key='synthetic-offline-key',base_url='https://offline.example.invalid/v1',
                    max_retries=0,http_client=http) as client:
            client._shinku_protocol='openai'
            runtime=FixtureRuntime(client.chat.completions.create)
            runtime.chat=ModelBundle(client,'deepseek-flash');runtime.aux=runtime.chat
            runtime._env=lambda key,default='':{'LLM_THINKING_MODE':'enabled','LLM_REASONING_EFFORT':'high'}.get(key,default)
            yield runtime,calls


def call(runtime):
    return runtime.call_chat_json(system_prompt='JSON规则',user_prompt='合成请求',fallback={},temperature=0)


@pytest.mark.parametrize('stage',['preparation','answer_review'])
def test_stage_default_omits_output_limit_with_high_thinking_unchanged(stage):
    with sdk() as (runtime,calls):
        with structured_output_scope(stage):
            assert call(runtime)=={'reply':'合成答复'}
        assert 'max_tokens' not in calls[0]
        assert calls[0]['thinking']=={'type':'enabled'} and calls[0]['reasoning_effort']=='high'
        assert not calls[0].get('tools') and len(calls)==1
        call(runtime)
        assert 'max_tokens' not in calls[-1]  # 不污染后续普通答复的既有默认策略。
    assert current_output_budget() is None


def test_nested_host_ceiling_is_not_raised_by_stage_or_exception():
    original={'model':'fixture','max_tokens':900}
    with structured_output_scope('preparation',maximum=2048):
        with pytest.raises(RuntimeError):
            with structured_output_scope('answer_review'):
                assert current_output_budget().max_tokens==2048
                assert apply_output_budget(original)['max_tokens']==900
                raise RuntimeError('offline')
        assert current_output_budget().max_tokens==2048
    assert current_output_budget() is None and original['max_tokens']==900


@pytest.mark.parametrize('bad',[0,255,32769,True,8192.5,'unlimited','-1'])
def test_invalid_stage_budget_is_refused(bad):
    with pytest.raises(ValueError):
        with structured_output_scope('preparation',maximum=bad):
            pytest.fail('invalid allowance entered')
    assert current_output_budget() is None


def test_environment_is_host_only_and_bounded(monkeypatch):
    monkeypatch.setenv('SHINKU_PREPARATION_MAX_OUTPUT_TOKENS','4096')
    with structured_output_scope('preparation'):
        assert current_output_budget().max_tokens==4096
    monkeypatch.setenv('SHINKU_PREPARATION_MAX_OUTPUT_TOKENS','1000000')
    with pytest.raises(ValueError):
        with structured_output_scope('preparation'):pass


def test_context_reserve_uses_stage_output_without_truncating_current_data(monkeypatch):
    monkeypatch.setenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS','20000')
    monkeypatch.setenv('SHINKU_LLM_OUTPUT_RESERVE_TOKENS','2048')
    monkeypatch.setenv('SHINKU_LLM_CONTEXT_MARGIN_TOKENS','1024')
    monkeypatch.setenv('SHINKU_LLM_CONTEXT_BUDGET_MODE','observe')
    with sdk() as (runtime,calls):
        records=[];runtime._record_context_budget=lambda record:(records.append(record)or record)
        with structured_output_scope('answer_review',maximum=8192):call(runtime)
        assert records[-1]['input_limit_estimate']==20000-8192-1024
        assert calls[0]['messages'][-1]['content']=='合成请求'


def test_real_preparation_host_uses_stage_scope_and_keeps_existing_model(tmp_path):
    from dataclasses import replace
    from shinku.agent_runtime.demo import demo_service
    from shinku.angelos.host import ServiceHost
    service,tool,_=demo_service(tmp_path)
    try:
        with sdk(lambda p: {'disposition':'answer'} if any('输入路线复核器' in m.get('content','')
            for m in p['messages'] if m['role']=='system') else {'reply':'合成答复'}) as (runtime,calls):
            service.executor.profile=replace(service.executor.profile,runtime=runtime,runtime_meters_budget=True)
            result=ServiceHost(service).prepare('你好',[],{'path':str(tmp_path)})
            assert result['reply']=='合成答复' and result['preparation_usage']['requests']==1 and len(calls)==1
            assert 'max_tokens' not in calls[0] and calls[0]['model']=='deepseek-flash'
            assert calls[0]['reasoning_effort']=='high' and tool.calls==0
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()


def test_real_metering_preserves_total_admission_and_known_usage():
    with sdk() as (runtime,calls):
        budget=TaskBudget(max_tokens=9000,max_requests=1)
        with bind_task_budget(budget),structured_output_scope('preparation',maximum=8192):
            call(runtime)
            with pytest.raises(TaskBudgetExceeded):call(runtime)
        assert len(calls)==1 and budget.reported_tokens==16 and budget.requests==1
    with sdk() as (runtime,calls):
        with bind_task_budget(TaskBudget(max_tokens=8000)),structured_output_scope('answer_review',maximum=8192):
            with pytest.raises(TaskBudgetExceeded):call(runtime)
        assert calls==[]


def test_custom_adapter_without_usage_keeps_original_conservative_reservation():
    class Runtime:
        def call_chat_json(self,**kw):return {}
    # 通用自定义适配器未证明支持阶段额度，不能据8192少预留未知输出。
    budget=TaskBudget(max_tokens=100000)
    with bind_task_budget(budget),structured_output_scope('preparation'):
        _MeteredRuntime(Runtime()).call_chat_json(user_prompt='x')
    assert budget.requests==1 and budget.estimated_requests==1 and 32768<budget.tokens<34000


def test_concurrent_scopes_do_not_leak_between_tasks():
    barrier=threading.Barrier(2)
    def worker(limit):
        with structured_output_scope('preparation',maximum=limit):
            barrier.wait(timeout=3);return current_output_budget().max_tokens
    with ThreadPoolExecutor(max_workers=2)as pool:
        a=pool.submit(worker,2048);b=pool.submit(worker,4096)
        assert(a.result(),b.result())==(2048,4096)
    assert current_output_budget() is None
