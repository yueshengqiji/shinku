"""校园网故障抽象为合成路径配置：真实文件读取与原生SDK接线，不改主机网络。

模型响应为确定性夹具，仅证明指导/历史/回执进入同一循环；真实模型是否学会
选择有效对照仍须单独实测，不能用此测试的预设答案宣称质量提升。
"""
from dataclasses import replace
import json
import pytest
from shinku.agent_runtime import AgentTurnRequest
from shinku.agent_runtime.demo import demo_service
from shinku.angelos.host import ServiceHost
from shinku.tools.workspace import build_workspace_tools
from test_agent_conversation_loop import native_sdk
from test_agent_runtime_boundary import settled


@pytest.mark.parametrize('blocked',[False,True])
def test_followup_inspects_previous_change_and_actual_path_without_extra_review(tmp_path,blocked):
    root=tmp_path/'workspace';root.mkdir()
    files={'previous.txt':'建议：将服务地址设为直连；未实际验证。',
           'actual.txt':'已应用直连；邻居解析失败；默认网关仍可达；门户仍公布同一地址。'}
    for name,text in files.items():(root/name).write_text(text)
    service,write_tool,_=demo_service(tmp_path/'state')
    for handler in build_workspace_tools(root,authorize=lambda ctx,write:not blocked and not write):
        service.executor.tool_host.registry.register(handler)
    original=service.authorize
    service.authorize=lambda *args:replace(original(*args),allowed_tools=frozenset({'workspace_read_lines'}),
        policy=replace(original(*args).policy,risk_by_tool={'workspace_read_lines':'low'}))
    system_prefix=[]
    def respond(payload,count):
        system='\n'.join(m['content'] for m in payload['messages'] if m['role']=='system')
        assert '先回查上一建议' in system and '实际生效路径' in system
        assert '不能覆盖原场景失败' in system and '部署/回滚记录' in system
        assert 'current_step填写steps中的步骤id' in system
        assert '不把用户再次确认说成能授予root' in system
        system_prefix.append(system)
        if count==1 or count==2 and not blocked:
            if count==2:
                assert '未实际验证' in json.dumps(payload,ensure_ascii=False)
            return {'role':'assistant','content':'','tool_calls':[{'id':f'read-{count}','type':'function',
                'function':{'name':'workspace_read_lines','arguments':json.dumps({'path':'previous.txt' if count==1 else 'actual.txt'})}}]},'tool_calls'
        tools=[m for m in payload['messages'] if m['role']=='tool']
        if blocked:
            assert len(tools)==1 and 'workspace_access_failed' in tools[0]['content']
            assert '邻居解析失败' not in json.dumps(payload,ensure_ascii=False)
            text='实际读取被拒绝，尚未确认当前路径；用户确认不能代替系统权限。'
        else:
            assert len(tools)==2 and '邻居解析失败' in tools[-1]['content']
            text='上一建议的直连前提不成立；当前失败不能证明门户改址。需验证经网关的路径；尚未修改或修好。'
        return {'role':'assistant','content':json.dumps({'kind':'final','text':text,'task_outcome':'partial'})},'stop'
    try:
        with native_sdk(respond) as (runtime,calls):
            service.executor.profile=replace(service.executor.profile,runtime=runtime,runtime_meters_budget=True,native_tool_history=True)
            host=ServiceHost(service,requester='owner')
            host.prepare=lambda *a:pytest.fail('no separate review/preparation')
            port=host.port('one');handle=port.submit(AgentTurnRequest('followup','按你的方案配置后还是不通，检查上一建议和实际路径。'))
            view=settled(port,handle.task_id);result=port.result(handle.task_id)
            assert view['status']=='partial'
            assert len(calls)==(2 if blocked else 3)
            assert result['budget']['tools']==(1 if blocked else 2)
            assert all(s==system_prefix[0] for s in system_prefix)
            assert result['verification']['layers']['goal_completion']=='not_verified'
            assert write_tool.calls==0
            assert all((root/n).read_text()==v for n,v in files.items())
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()
