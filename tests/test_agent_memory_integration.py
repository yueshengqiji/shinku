"""原 Service/AgentLoop/SDK/工作台联测；模型为 MockTransport，无真实模型请求。"""
from dataclasses import replace
import json
import time

import pytest

from shinku.agent_runtime.demo import demo_service
from shinku.angelos.host import ServiceHost
from shinku.angelos.store import WorkbenchStore
from shinku.angelos.workbench import Workbench
from shinku.agent_memory.service import AgentMemoryService
from test_agent_conversation_loop import native_sdk
from agent_memory_fixture import memory_fixture, seed, invoke, allow_memory_contract


def finish(bench, cid):
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        c=bench.refresh(cid);turn=c['turns'][-1]
        if turn['stage'] in {'finished','failed','replied','interrupted'}:return turn
        time.sleep(.01)
    raise AssertionError('synthetic Agent did not finish')


def observed_contract(name):
    return {'goal':'按用户要求处理独立Agent记忆','deliverables':['带来源的记忆结果'],
        'scope':['仅Agent本库，不把历史陈述当当前系统事实'],
        'requirements':[{'id':'memory','description':'实际本库读写',
                         'effect':'memory_manage' if name=='manage_agent_memory' else 'tool:'+name,'targets':[]}]}


def tool_call(name,args,index):
    return {'role':'assistant','content':'','reasoning_content':'SYNTHETIC_REASONING',
        'tool_calls':[{'id':f'memory_call_{index}','type':'function',
                       'function':{'name':name,'arguments':json.dumps(args,ensure_ascii=False)}}]},'tool_calls'


def test_real_sdk_remember_cross_session_recall_and_no_extra_classifier(tmp_path):
    service,file_tool,_=demo_service(tmp_path)
    original=service.authorize
    service.authorize=lambda *a:replace(original(*a),permission_mode='full')
    def respond(payload,index):
        if index==1:return tool_call('declare_task_contract',{'task_contract':observed_contract('manage_agent_memory')},index)
        if index==2:return tool_call('manage_agent_memory',{'operation':'remember','message_id':'remember-sdk',
            'text':'我的白鸽是代理客户端，不是邮件配置。'},index)
        if index==4:return tool_call('read_agent_memory',{'query':'白鸽'},index)
        body={'kind':'final','text':'已保存你的原话。' if index==3 else '你此前说白鸽是代理客户端，不是邮件配置。这是用户陈述，当前安装状态尚未核验。',
              'task_outcome':'completed'}
        if index==5:body['task_contract']=observed_contract('read_agent_memory')
        return {'role':'assistant','content':json.dumps(body,ensure_ascii=False)},'stop'
    bench=None
    try:
        with native_sdk(respond) as (runtime,payloads):
            service.executor.profile=replace(service.executor.profile,runtime=runtime,runtime_meters_budget=True,native_tool_history=True)
            host=ServiceHost(service,requester='owner',memory_root=tmp_path/'memory')
            store=WorkbenchStore(tmp_path/'workbench.sqlite3');bench=Workbench(store,host)
            one=store.conversation('default')
            bench.send(one['id'],{'text':'请记住：我的白鸽是代理客户端，不是邮件配置。','request_key':'remember-sdk'})
            result=finish(bench,one['id'])
            assert result['view']['status']=='completed',result
            assert result.get('memory_index_version') and not result.get('memory_error')
            assert len(payloads)==result['result']['budget']['requests']==3
            two=store.conversation('default')
            bench.send(two['id'],{'text':'我以前说白鸽是什么软件？','request_key':'recall-sdk'})
            result=finish(bench,two['id'])
            assert result['view']['status']=='completed',result
            assert result['result']['budget']['requests']==2 and len(payloads)==5
            assert '代理客户端' in result['result']['execution_report']['text']
            assert file_tool.calls==0
            assert all('max_tokens' not in p for p in payloads)
            first='\n'.join(m.get('content','') or '' for m in payloads[0]['messages'])
            assert 'Agent-only task memory' in first and 'agent_memory_workspace_changed' not in first
            # 每个原生回执仍与assistant调用配对，附加记忆控制块不破坏原链。
            for payload in payloads:
                announced={call['id'] for m in payload['messages'] if m['role']=='assistant' for call in m.get('tool_calls',[])}
                assert all(m['tool_call_id'] in announced for m in payload['messages'] if m['role']=='tool')
    finally:
        if bench:bench.close()
        service.coordinator.shutdown(wait=True);service.task_store.close()


def test_simple_chat_uses_one_model_request_and_zero_memory_tools(tmp_path):
    with memory_fixture(tmp_path) as f:
        service,host,store,_,tool,runtime=f
        runtime.call_chat_json=lambda **kw:{'kind':'final','text':'我们开始工作吧。'}
        bench=Workbench(store,host);c=store.conversation('default')
        try:
            bench.send(c['id'],{'text':'你好','request_key':'hello'})
            turn=finish(bench,c['id'])
            assert turn['view']['status']=='completed'
            assert turn['result']['budget']['requests']==1 and turn['result']['budget']['tools']==0
            assert tool.calls==0
        finally:bench.close()


def test_restart_rebinds_memory_and_user_source_without_global_cache(tmp_path):
    with memory_fixture(tmp_path) as f:
        task,_,ctx=seed(f,'old');allow_memory_contract(f[0],task)
        saved=invoke(f[1],ctx,'manage_agent_memory',operation='remember',text='白鸽',message_id='old-input')
        assert saved['state_updates']['status']=='saved'
        restarted=AgentMemoryService(f[0],'owner',tmp_path/'memory');restarted.bind(f[2])
        assert restarted.recall(ctx,'白鸽')['items'][0]['text']==task['requests'][0]['text']


def test_agent_flow_never_constructs_bot_memory_service_or_router(tmp_path,monkeypatch):
    from shinku.memory.service import MemoryService
    from shinku.memory.router import MemoryRouter
    def forbidden(*a,**kw):raise AssertionError('Bot memory must remain separate')
    monkeypatch.setattr(MemoryService,'__init__',forbidden)
    monkeypatch.setattr(MemoryRouter,'__init__',forbidden)
    with memory_fixture(tmp_path) as f:
        task,_,ctx=seed(f,'one');allow_memory_contract(f[0],task)
        assert invoke(f[1],ctx,'manage_agent_memory',operation='remember',text='白鸽',message_id='one-input')['state_updates']['status']=='saved'
        assert invoke(f[1],ctx,query='白鸽')['state_updates']['items']


def test_retrieval_failure_does_not_fall_back_to_bot_or_change_task_status(tmp_path):
    with memory_fixture(tmp_path) as f:
        task,_,ctx=seed(f,'one');before=f[0].task_store.get('one',requester='owner',session=task['session'])
        # 专用库被替换成外来库，必须失败，不尝试迁移、回退或运行任务。
        f[1].memory.store.path.unlink()
        import sqlite3
        with sqlite3.connect(f[1].memory.store.path) as db:db.execute('CREATE TABLE bot_private(text)')
        f[1].memory.store.path.chmod(0o600)
        value=invoke(f[1],ctx,query='白鸽')
        assert value['status']=='error'
        assert f[0].task_store.get('one',requester='owner',session=task['session'])==before


def test_unconfigured_host_does_not_create_any_memory_database(tmp_path):
    service,_,_=demo_service(tmp_path)
    try:
        host=ServiceHost(service)
        assert host.memory is None and service.executor.tool_host.registry.get('read_agent_memory') is None
        assert not (tmp_path/'memory').exists()
    finally:service.coordinator.shutdown(wait=True);service.task_store.close()
