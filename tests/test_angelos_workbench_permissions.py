"""四档真实授权与一次性批准；确定性模型和隔离文件，不访问付费服务。"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from shinku.agent_runtime import AgentScope, AgentTurnRequest
from shinku.agent_runtime.demo import demo_service, demo_request
from shinku.angelos.host import ServiceHost
from shinku.angelos.permissions import PermissionControls, ControlledTool
from shinku.angelos.app import create_angelos_app


def installed(tmp_path, level):
    service, tool, runtime = demo_service(tmp_path)
    controls = PermissionControls(tmp_path/'permission.json', default=level, approval_timeout=1)
    host = ServiceHost(service, permissions=controls)
    return service, tool, runtime, controls, host


def pending(controls):
    until = time.monotonic()+4
    while time.monotonic()<until:
        rows=controls.snapshot()['pending']
        if rows:
            return rows[0]
        time.sleep(.01)
    pytest.fail('missing approval')


def finish(port, handle):
    until = time.monotonic()+4
    while time.monotonic()<until:
        value=port.status(handle.task_id)
        if not value.get('worker_active') and value['status'] not in {'queued','running'}:
            return value
        time.sleep(.01)
    pytest.fail(str(value))


@pytest.mark.parametrize('level,execute', [('zayin',False),('he',False),('aleph',True)])
def test_actual_file_execution_gated(tmp_path,level,execute):
    service,tool,_,controls,host=installed(tmp_path,level)
    try:
        port=host.port('test'); handle=port.submit(demo_request(tool))
        finish(port,handle)
        assert bool(tool.calls) is execute
        assert tool.path.exists() is execute
        assert controls.snapshot()['pending']==[]
    finally:
        service.coordinator.shutdown(wait=True)


@pytest.mark.parametrize('decision,execute',[('approve',True),('deny',False)])
def test_waw_exact_call_approval_before_real_write(tmp_path,decision,execute):
    service,tool,_,controls,host=installed(tmp_path,'waw')
    try:
        port=host.port('test'); handle=port.submit(demo_request(tool))
        row=pending(controls)
        assert row['taskId']==handle.task_id and row['arguments']['path']==str(tool.path)
        assert not tool.path.exists() and tool.calls==0
        controls.decide(row['id'],row['revision'],decision)
        finish(port,handle)
        assert bool(tool.calls) is execute
        with pytest.raises(ValueError,match='approval_expired'):
            controls.decide(row['id'],row['revision'],'approve')
    finally:
        service.coordinator.shutdown(wait=True)


def test_revocation_invalidates_waiting_call_and_persists(tmp_path):
    service,tool,_,controls,host=installed(tmp_path,'waw')
    try:
        port=host.port('test');handle=port.submit(demo_request(tool));row=pending(controls)
        controls.select('he',controls.revision)
        with pytest.raises(ValueError,match='approval_expired'):
            controls.decide(row['id'],row['revision'],'approve')
        finish(port,handle)
        assert tool.calls==0
        assert PermissionControls(controls.path).level=='he'
    finally:
        service.coordinator.shutdown(wait=True)


def test_zayin_no_internal_or_business_tool_schema(tmp_path):
    service,_,runtime,_,host=installed(tmp_path,'zayin')
    seen=[]
    def answer(**kwargs):
        seen.append(kwargs)
        return {'kind':'final','text':'我们可以先讨论方案。','task_outcome':'completed'}
    runtime.call_chat_json=answer
    try:
        port=host.port('chat');handle=port.submit(AgentTurnRequest('only-chat','你好'))
        assert finish(port,handle)['status']=='completed'
        assert seen and all(not v.get('native_tools') for v in seen)
    finally:
        service.coordinator.shutdown(wait=True)


def test_readonly_rejects_read_label_with_write_effect(tmp_path):
    service,tool,_,controls,host=installed(tmp_path,'he')
    tool.agent_permission_class='read'
    grant=service._grant(AgentScope(host.requester,'angelos:test'),{},'submit')
    assert tool.tool_type not in grant.allowed_tools
    service.coordinator.shutdown(wait=True)


def test_permission_api_token_revision_and_bot_isolation(tmp_path):
    service,_,_,controls,host=installed(tmp_path,'he')
    try:
        with TestClient(create_angelos_app(tmp_path/'ui',host=host)) as client:
            data={'level':'aleph','revision':controls.revision}
            assert client.post('/angelos/api/permissions',json=data).status_code==403
            client.headers['x-angelos-token']=client.get('/angelos/api/bootstrap').json()['token']
            assert client.post('/angelos/api/permissions',json=data,headers={'origin':'https://evil.invalid'}).status_code==403
            assert client.post('/angelos/api/permissions',json=data).status_code==200
            assert controls.level=='aleph'
            assert client.post('/angelos/api/permissions',json={'level':'he','revision':data['revision']}).status_code==409
            assert not (tmp_path/'qq').exists()
    finally:
        service.coordinator.shutdown(wait=True)


def test_approval_timeout_does_not_execute(tmp_path):
    controls=PermissionControls(tmp_path/'permission.json',default='waw',approval_timeout=.01)
    service,tool,_=demo_service(tmp_path)
    result=ControlledTool(tool,controls).execute(call={'path':str(tool.path)},context={
        'angelos_permission_revision':controls.revision,'agent_task_state':{
            'task_id':'t','session':'angelos:c','execution_attempt':1}})
    assert result['code']=='approval_timed_out' and not result['state_updates']['executed']
    assert tool.calls==0
    service.coordinator.shutdown(wait=True)


@pytest.mark.skipif(sys.platform!='linux' or os.geteuid()==0,reason='Linux non-root privilege boundary')
def test_no_new_privileges_inherited_by_child():
    code="from shinku.angelos.permissions import require_unprivileged; require_unprivileged(); import subprocess; print(subprocess.check_output(['cat','/proc/self/status'],text=True))"
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
    assert 'NoNewPrivs:\t1' in result.stdout


def test_task_budget_expiry_while_awaiting_approval_cannot_write(tmp_path):
    from shinku.agent.budget import TaskBudget, bind_task_budget
    controls=PermissionControls(tmp_path/'permissions.json',default='waw',approval_timeout=1)
    service,tool,_=demo_service(tmp_path)
    try:
        with bind_task_budget(TaskBudget(max_active_seconds=.01)):
            result=ControlledTool(tool,controls).execute(call={'path':str(tool.path)},context={
                'angelos_permission_revision':controls.revision,'agent_task_state':{
                    'task_id':'t','session':'angelos:c','execution_attempt':1}})
        assert result['code']=='approval_canceled'
        assert tool.calls==0 and not tool.path.exists()
        assert not controls.snapshot()['pending']
    finally:
        service.coordinator.shutdown(wait=True)
