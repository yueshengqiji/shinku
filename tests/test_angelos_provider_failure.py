"""真实SDK的合成HTTP错误穿过执行宿主到工作台，不调用供应商。"""
from contextlib import contextmanager
import json
import threading

import httpx
from openai import OpenAI
import pytest

from shinku.llm.runtime_types import ModelBundle
from shinku.llm.policy import RuntimePolicy
from shinku.llm.response_protocol import call_failure_metadata
from shinku.llm.provider_failure import provider_failure
from test_task_reply_protocol import ProtocolRuntime
from test_angelos_workspace_tools import workbench, conversation, send
from test_angelos import wait


PRIVATE = 'SYNTHETIC_PRIVATE_ERROR_BODY'


@contextmanager
def provider(status, error, *, read_first=False, final_text=None, read_count=0):
    calls = []
    def handle(request):
        wire = json.loads(request.content); calls.append(wire)
        if len(calls) <= (read_count or int(read_first)):
            message = dict(role='assistant', content=None, tool_calls=[dict(id=f'read-{len(calls)}',type='function',
                function=dict(name='workspace_read_lines',arguments=json.dumps({'path':'a.txt', **({'start_line':len(calls),'line_count':1} if read_count else {})})))])
            finish = 'tool_calls'
        elif final_text is not None:
            message = {'role':'assistant','content':final_text}; finish = 'stop'
        else:
            return httpx.Response(status, json={'error':{'message':PRIVATE, **error}})
        return httpx.Response(200, json=dict(id='offline',object='chat.completion',created=0,model='offline',
            choices=[dict(index=0,finish_reason=finish,message=message)],usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15)))
    with httpx.Client(transport=httpx.MockTransport(handle),trust_env=False) as client, OpenAI(
        api_key='offline-fixture',base_url='https://example.invalid/v1',max_retries=0,http_client=client) as sdk:
        sdk.protocol = 'openai'
        runtime = ProtocolRuntime(sdk.chat.completions.create)
        runtime._swap_lock = threading.RLock()
        runtime.chat = ModelBundle(sdk, 'deepseek-flash'); runtime.aux = runtime.chat
        runtime.chat_supports_native_tools = lambda: True
        runtime.runtime_policy = RuntimePolicy(http_attempts=1,stream_attempts=1)
        yield runtime, calls


@pytest.mark.parametrize('status,error,category,phrase', [
    (402,{'type':'insufficient_balance_error'},'insufficient_balance','余额不足'),
    (402,{'message':'Insufficient Balance (request_id: fixture)'},'insufficient_balance','余额不足'),
    (429,{'code':'insufficient_quota'},'quota_exhausted','额度不足'),
    (429,{'type':'rate_limit_error'},'rate_limit','限流'),
    (401,{},'authentication','身份认证'),
    (403,{},'access_denied','拒绝访问'),
    (400,{'code':'context_length_exceeded'},'context_limit','上下文长度'),
    (413,{},'request_too_large','过大的请求'),
    (500,{},'service_unavailable','暂时不可用'),
    (400,{},'request_rejected','拒绝了本次请求'),
])
def test_http_error_reaches_workbench_without_leaking_body_or_retries(tmp_path,status,error,category,phrase):
    with provider(status,error) as (runtime,calls), workbench(tmp_path,None,runtime=runtime) as (client,service,host,_):
        cid=conversation(client);send(client,cid)
        result=wait(client,cid)['turns'][0]['result']
        assert result['execution']['status']=='failed',json.dumps(result,ensure_ascii=False)
        assert phrase in result['execution_report']['text'],result
        assert len(calls)==1 and not result['evidence']
        task=service.task_store.get(result['task_id'],requester=host.requester,session='angelos:'+cid)
        metadata=task.get('progress', {})['conversation_call_failure']
        assert metadata['category']==category and metadata['http_status']==status
        assert PRIVATE not in json.dumps(result,ensure_ascii=False)+json.dumps(metadata)
        assert task['stop_reason']=='conversation_call_failed'


def test_actual_read_and_original_are_retained_when_balance_runs_out(tmp_path):
    root=tmp_path/'project';root.mkdir();path=root/'a.txt';path.write_text('实际保留的合成原文\n')
    with provider(402,{'type':'insufficient_balance_error'},read_first=True) as (runtime,calls), \
            workbench(tmp_path,None,runtime=runtime) as (client,service,host,_):
        cid=conversation(client,root);send(client,cid)
        result=wait(client,cid)['turns'][0]['result']
        assert result['execution']['status']=='failed' and '余额不足' in result['execution_report']['text'],result
        assert len(calls)==2 and len(result['evidence'])==1
        assert result['evidence'][0]['execution_status']=='ok'
        task=service.task_store.get(result['task_id'],requester=host.requester,session='angelos:'+cid)
        retained=[service.materials.resolve(item['reference'],task=task,still_authorized=lambda:True)
                  for item in task['material_package']['items'] if item['kind']=='executor_result']
        assert retained and '实际保留的合成原文' in json.dumps(retained,ensure_ascii=False)
        assert path.read_text()=='实际保留的合成原文\n'
        assert result['verification'].get('verified') is not True


def test_model_text_cannot_forge_host_failure_category(tmp_path):
    text=json.dumps({'kind':'final','text':'这是模型生成的解释。','task_outcome':'completed',
                     'category':'insufficient_balance','http_status':402})
    with provider(200,{},final_text=text) as (runtime,calls), workbench(tmp_path,None,runtime=runtime) as (client,service,host,_):
        cid=conversation(client);send(client,cid)
        result=wait(client,cid)['turns'][0]['result']
        task=service.task_store.get(result['task_id'],requester=host.requester,session='angelos:'+cid)
        assert not task.get('progress', {}).get('conversation_call_failure')
        assert task['stop_reason']!='conversation_call_failed'


def test_plain_exception_words_are_not_an_account_diagnosis():
    assert provider_failure(RuntimeError('insufficient balance'))=={}
    assert 'category' not in call_failure_metadata('call',RuntimeError('insufficient balance'))


def test_long_investigation_advice_is_append_only_and_not_a_tool_limit(tmp_path):
    root=tmp_path/'project';root.mkdir();(root/'a.txt').write_text(''.join(f'line {i}\n' for i in range(1,11)))
    text=json.dumps({'kind':'final','text':'主管，十行合成资料已经逐行核对。','task_outcome':'completed'})
    with provider(200,{},read_count=10,final_text=text) as (runtime,calls), workbench(tmp_path,None,runtime=runtime) as (client,service,host,_):
        cid=conversation(client,root);send(client,cid)
        result=wait(client,cid)['turns'][0]['result']
        assert result['execution']['status']=='completed',json.dumps(result,ensure_ascii=False)
        assert len(calls)==11 and len(result['evidence'])==10
        assert all(row['execution_status']=='ok' for row in result['evidence'])
        assert '调查阶段整理建议' not in json.dumps(calls[7]['messages'],ensure_ascii=False)
        assert '调查阶段整理建议' in json.dumps(calls[8]['messages'],ensure_ascii=False)
        for old,new in zip(calls,calls[1:]):
            assert new['messages'][:len(old['messages'])]==old['messages']
            assert new['tools']==old['tools']
        assert sum('调查阶段整理建议' in str(m) for m in calls[-1]['messages'])==1
