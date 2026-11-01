"""空返回不能伪装成模型主动失败；真实 SDK -> 同一执行循环 -> 原稿与回执归档。"""
from dataclasses import replace
import json

import pytest
from shinku.agent_runtime import AgentTurnRequest
from shinku.agent_runtime.demo import demo_service
from shinku.angelos.host import ServiceHost
from shinku.llm.response_protocol import observe_response, response_failure_kind
from test_agent_conversation_loop import native_sdk, observed_contract
from test_agent_runtime_boundary import settled
from test_angelos_initiative import seed


@pytest.mark.parametrize('content,reason', [(' '*55,'output_empty'), ('','output_empty'), (None,'output_empty'), ('{"kind":','output_invalid')])
@pytest.mark.parametrize('after_read', [False, True])
def test_invalid_output_preserves_raw_and_receipts_without_fake_draft(tmp_path, content, reason, after_read):
    service, tool, _ = demo_service(tmp_path)
    def responder(payload, index):
        if after_read and index <= 2:
            name, args = ('read_task_records', {'mode':'detail','task_id':'old'}) if index == 1 else ('declare_task_contract', {'task_contract':observed_contract('read_task_records')})
            return {'role':'assistant','content':'','reasoning_content':'PRIVATE_REASONING',
                'tool_calls':[{'id':f'call_{index}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}, 'tool_calls'
        return {'role':'assistant','content':content}, 'stop'
    try:
        with native_sdk(responder) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime, runtime_meters_budget=True, native_tool_history=True)
            host=ServiceHost(service,requester='owner');port=host.port('one')
            if after_read:seed(service,'old')
            handle=port.submit(AgentTurnRequest('explain','解释一下'))
            view=settled(port,handle.task_id);result=port.result(handle.task_id)
            assert view['status']=='failed' and view['reason']=='conversation_'+reason
            assert len(calls)==(3 if after_read else 1)==result['budget']['requests']
            assert tool.calls==0 and not tool.path.exists()
            assert ('空白' if reason=='output_empty' else '无法解析') in result['execution_report']['text']
            assert '暂时无法完成这一步' not in result['execution_report']['text']
            task=service.task_store.get(handle.task_id,requester='owner',session='angelos:one')
            assert not task['progress'].get('answer_drafts')
            outputs=host.reasoning_view.outputs(handle.task_id)
            assert outputs[-1]['text']==(content or '')
            assert outputs[-1]['diagnostics']['failure']==reason
            assert outputs[-1]['diagnostics']['finish_reason']=='stop'
            assert 'PRIVATE_REASONING' not in json.dumps(outputs)
            if after_read:
                assert [r['tool'] for r in result['evidence']]==['read_task_records','declare_task_contract']
                assert all(r['execution_status']=='ok' for r in result['evidence'])
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()


@pytest.mark.parametrize('finish,reason',[('length','output_length'),('content_filter','output_filtered')])
def test_truncation_reason_takes_priority_over_empty(tmp_path, finish, reason):
    service,tool,_=demo_service(tmp_path)
    try:
        with native_sdk(lambda p,n:({'role':'assistant','content':''},finish)) as (runtime,calls):
            service.executor.profile=replace(service.executor.profile,runtime=runtime,native_tool_history=True)
            host=ServiceHost(service,requester='owner');port=host.port('one')
            handle=port.submit(AgentTurnRequest('truncated','解释一下'))
            assert settled(port,handle.task_id)['reason']=='conversation_'+reason
            assert len(calls)==1 and tool.calls==0
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()


def test_legacy_observer_policy_is_unchanged():
    with native_sdk(lambda p,n:({'role':'assistant','content':'   '},'stop')) as (runtime,calls):
        with observe_response() as observation:
            result=runtime.call_chat_json(system_prompt='isolated',user_prompt='data',fallback={'legacy':True})
        assert result=={'legacy':True} and response_failure_kind(observation)==''
        assert {'stage':'empty_response'} in observation.events
        assert len(calls)==1


def test_explanation_scope_guidance_reaches_actual_model_request(tmp_path):
    service,tool,_=demo_service(tmp_path)
    def responder(payload,index):
        instructions='\n'.join(m.get('content','') for m in payload['messages'] if m['role']=='system')
        assert '不要自行把实际运行程序' in instructions
        assert '静态推导须标明是推导' in instructions
        assert '用户明确要求运行、实测或调试' in instructions
        return {'role':'assistant','content':json.dumps({'kind':'final','text':'静态推导：range(3)依次产生0、1、2；未实际运行程序。','task_outcome':'completed'},ensure_ascii=False)},'stop'
    try:
        with native_sdk(responder) as (runtime,calls):
            service.executor.profile=replace(service.executor.profile,runtime=runtime,native_tool_history=True)
            host=ServiceHost(service,requester='owner');port=host.port('one')
            handle=port.submit(AgentTurnRequest('explain-code','解释 for i in range(3): print(i)'))
            assert settled(port,handle.task_id)['status']=='completed'
            result=port.result(handle.task_id)
            assert result['contract'] is None and tool.calls==0 and len(calls)==1
            assert '静态推导' in result['execution_report']['text']
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()


def test_workbench_displays_empty_cause_and_retains_provider_record(tmp_path):
    from fastapi.testclient import TestClient
    from angelos_fixture import fixture
    from test_angelos import new,post,wait
    app,bench,old_host=fixture(tmp_path)
    with native_sdk(lambda p,n:({'role':'assistant','content':' '*55},'stop')) as (runtime,calls):
        service=old_host.service
        service.executor.profile=replace(service.executor.profile,runtime=runtime,native_tool_history=True)
        host=ServiceHost(service,requester=old_host.requester);bench.host=host
        with TestClient(app) as client:
            client.headers['x-angelos-token']=client.get('/angelos/api/bootstrap').json()['token']
            cid=new(client)
            post(client,f'/conversations/{cid}/messages',{'text':'解释一下','request_key':'empty'})
            turn=wait(client,cid)['turns'][-1]
            assert turn['view']['status']=='failed'
            assert turn['view']['reason']=='conversation_output_empty'
            assert '空白内容' in turn['result']['execution_report']['text']
            assert turn['model_outputs'][-1]['diagnostics']['failure']=='output_empty'
            assert host.reasoning_view.outputs(turn['task_id'])[-1]['text']==' '*55
            assert len(calls)==1
