"""两个实际进程之间的HTTP委托与重启只读恢复；不调用真实供应商。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from shinku.agent_runtime import AgentScope, AgentTurnRequest, ExecutionGrant
from shinku.agent_runtime.delegation_http import RemoteAgentClient, authority_payload
from shinku.tools.execution import ExecutionPolicy
from test_agent_delegation import SECRET, SCOPE


SERVER = r'''
from pathlib import Path
import json,os,socket,sys
from fastapi import FastAPI
import uvicorn
from shinku.agent_runtime import AgentExecutor,AgentScope,AgentService,ExecutionGrant,ExecutionProfile
from shinku.agent_runtime.demo import DemoFileTool,DemoRuntime,demo_request
from shinku.agent_runtime.delegation import DelegationGateway,DelegationPrincipal
from shinku.agent_runtime.materials import TaskMaterialStore
from shinku.agent.task_state import AgentTaskStore
from shinku.hosts.tool_host import ToolHost
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.registry import ToolRegistry
from shinku.tools.execution import ExecutionPolicy
from shinku.api.agent_delegation import build_delegation_router
root=Path(sys.argv[1]);ready=Path(sys.argv[2]);audit=Path(sys.argv[3])
tool=DemoFileTool(root);runtime=DemoRuntime(tool);base=runtime.call_chat_json;declared=False
def decide(**kwargs):
    global declared
    if not declared:
        declared=True;runtime.calls.append(kwargs)
        return {'kind':'tool','tool_call':{'name':'declare_task_contract','arguments':{'task_contract':demo_request(tool).contract}}}
    return base(**kwargs)
runtime.call_chat_json=decide
grant=ExecutionGrant('process-fixture',frozenset({tool.tool_type}),policy=ExecutionPolicy(risk_by_tool={tool.tool_type:'low'}),
    budget_limits={'max_requests':12,'max_tokens':600000,'max_tools':8,'max_active_seconds':30},resume_safe=True)
store=AgentTaskStore(root/'remote.sqlite3');coordinator=AgentTaskCoordinator(state_path=root/'workers.sqlite3',max_workers=1)
materials=TaskMaterialStore(store,approved_roots=(root,))
service=AgentService(executor=AgentExecutor(ToolHost(ToolRegistry({tool.tool_type:tool})),ExecutionProfile(runtime)),
    authorize=lambda s,t,o:grant,task_store=store,coordinator=coordinator,material_store=materials,recovery_enabled=True)
gateway=DelegationGateway(service,{'shinku':DelegationPrincipal('synthetic-delegation-connection-secret-000000000',lambda s,t,o:grant)})
app=FastAPI();app.include_router(build_delegation_router(gateway))
@app.middleware('http')
async def record_counts(request,call_next):
    response=await call_next(request)
    audit.write_text(json.dumps({'model_calls':len(runtime.calls),'tool_calls':tool.calls,'pid':os.getpid()}))
    return response

sock=socket.socket();sock.bind(('127.0.0.1',0));sock.listen(128)
ready.write_text(json.dumps({'endpoint':'http://127.0.0.1:'+str(sock.getsockname()[1]),'pid':os.getpid()}))
try:uvicorn.Server(uvicorn.Config(app,log_level='error',lifespan='off',ws='none')).run(sockets=[sock])
finally:
    coordinator.shutdown(wait=True);store.close()
    audit.write_text(json.dumps({'model_calls':len(runtime.calls),'tool_calls':tool.calls,'pid':os.getpid()}))
'''


def test_real_process_restart_retains_result_and_duplicate_does_not_repeat_effect(tmp_path):
    script=tmp_path/'server.py';script.write_text(SERVER)
    request=AgentTurnRequest('original-message','Create the isolated demo file and verify its contents.')
    grant=ExecutionGrant('process-fixture',frozenset({'demo_file'}),
        policy=ExecutionPolicy(risk_by_tool={'demo_file':'low'}),
        budget_limits={'max_requests':12,'max_tokens':600000,'max_tools':8,'max_active_seconds':30})
    stored_hash=None
    stored_messages=None
    for index in range(2):
        ready,audit=tmp_path/f'ready-{index}.json',tmp_path/f'audit-{index}.json'
        with (tmp_path/f'server-{index}.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-B',str(script),str(tmp_path),str(ready),str(audit)],
                cwd=tmp_path,env={**os.environ,'SHINKU_ENV_FILE':'/dev/null'},stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+6
                while not ready.exists():
                    assert process.poll() is None and time.monotonic()<deadline
                    time.sleep(.01)
                info=json.loads(ready.read_text());assert info['pid']!=os.getpid()
                client=RemoteAgentClient(info['endpoint'],'shinku',SECRET,
                    authority=lambda s,o:authority_payload(grant))
                handle=client.port(SCOPE).submit(request)
                deadline=time.monotonic()+5
                while True:
                    result=client.port(SCOPE).result(handle.task_id)
                    if result['execution']['result_available']:break
                    assert time.monotonic()<deadline
                    time.sleep(.01)
                assert result['execution']['status']=='completed',result['execution']
                if stored_hash is not None:assert result['result_hash']==stored_hash
                stored_hash=result['result_hash']
                handoff=client.port(SCOPE).handoff(handle.task_id)
                assert client.port(SCOPE).details(handoff)==result
                messages=client.port(SCOPE).messages(handle.task_id)
                if stored_messages is not None:assert messages==stored_messages
                stored_messages=messages
                assert sum(r['kind']=='submit' for r in messages['messages'])==1
                assert (tmp_path/'agent-demo.txt').read_text()=='standalone agent verified'
            finally:
                process.terminate()
                process.wait(timeout=6)
        assert process.returncode in (0,-15)
        counts=json.loads(audit.read_text())
        assert counts['tool_calls']==(1 if index==0 else 0)
        if index==1:assert counts['model_calls']==0
