"""面向用户的过程说明：真实SDK往返、任务隔离、无额外调用、最终答复分离。"""
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

from shinku.agent_runtime import AgentTurnRequest
from shinku.agent_runtime.demo import demo_service
from shinku.angelos.host import ServiceHost
from shinku.angelos.runtime_view import ReasoningView
from shinku.angelos.commentary import read_updates
from test_agent_conversation_loop import native_sdk
from test_agent_runtime_boundary import settled


def test_commentary_is_visible_before_final_and_does_not_add_calls(tmp_path):
    service,tool,_=demo_service(tmp_path)
    entered,release=threading.Event(),threading.Event()
    comment='已定位到合成目标文件，接下来写入并检查实际内容。'
    def respond(payload,count):
        if count==1:
            return {'role':'assistant','content':comment,'tool_calls':[{'id':'write-1','type':'function',
                'function':{'name':'demo_file','arguments':json.dumps({'path':str(tool.path)})}}]},'tool_calls'
        entered.set();assert release.wait(10)
        return {'role':'assistant','content':'最终结果：合成文件已写入。'},'stop'
    try:
        with native_sdk(respond) as (runtime,calls):
            service.executor.profile=replace(service.executor.profile,runtime=runtime,
                runtime_meters_budget=True,native_tool_history=True)
            host=ServiceHost(service,requester='owner',reply_delivery=True)
            port=host.port('one');h=port.submit(AgentTurnRequest('progress','写入合成文件并验证。'))
            assert entered.wait(10)
            turn={'id':'progress','task_id':h.task_id,'stage':'executing'}
            for _ in range(3):
                p=host.read_progress('one',turn)
                assert p['updates']['rows'][0]['text']==comment
                assert p['updates']['rows'][0]['is_fact_evidence'] is False
            assert len(calls)==2 and tool.calls==1
            assert host.read_progress('foreign',turn)['available'] is False
            release.set();settled(port,h.task_id)
            p=host.read_progress('one',turn)
            assert len(p['updates']['rows'])==1
            assert '最终结果' in port.result(h.task_id)['execution_report']['text']
    finally:
        release.set();service.coordinator.shutdown(wait=True);service.task_store.close()


def test_projection_excludes_reasoning_final_protocol_and_redacts():
    view=ReasoningView();host=SimpleNamespace(reasoning_view=view,
        profile=SimpleNamespace(runtime=SimpleNamespace(_redact_secrets=lambda s,**kw:s.replace('SECRET','[redacted]'))))
    def add(text,source='provider_message_content',stage='native_call'):
        view.append_output('task',{'text':text,'source':source,'diagnostics':{'events':[{'stage':stage}]},'created':1})
    view.append('task','PRIVATE_REASONING',model='test')
    add('检查配置 SECRET');add('检查配置 SECRET');add('{"kind":"tool"}')
    add('最终答复',stage='json_object');add('适配器对象',source='adapter_return_object')
    rows=read_updates(host,'task')['rows']
    assert len(rows)==1 and rows[0]['text']=='检查配置 [redacted]'
    assert read_updates(host,'other')['rows']==[]
