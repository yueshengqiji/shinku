"""显式连接配置与默认关闭；合成凭据不会打开生产模型或平台发送。"""
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from shinku.agent_runtime import AgentBoundaryError, AgentScope
from shinku.agent_runtime.delegation_http import authority_payload
from shinku.agent_runtime.demo import demo_service
from shinku.angelos.app import create_angelos_app
from shinku.angelos.delegation_config import install_from_file
from test_agent_delegation import SECRET, SCOPE


@pytest.fixture
def configured(tmp_path):
    service, tool, runtime = demo_service(tmp_path)
    host = SimpleNamespace(service=service, delegation=None, permissions=None, ready=lambda:True)
    token = tmp_path/'connection.token'; token.write_text(SECRET); token.chmod(0o600)
    path = tmp_path/'hosts.json'
    config = {'schema_version':1,'hosts':[{'id':'shinku','token_file':'connection.token',
        'requesters':['fixture'],'sessions':['private:'],'allowed_tools':[tool.tool_type],
        'budget_limits':{'max_requests':12,'max_tokens':600000,'max_tools':8,'max_active_seconds':30}}]}
    path.write_text(json.dumps(config));path.chmod(0o600)
    yield host, tool, runtime, path, config
    service.coordinator.shutdown(wait=True);service.task_store.close()


def test_config_installs_only_scoped_existing_capabilities(configured):
    host,tool,runtime,path,config=configured
    gateway=install_from_file(host,path)
    grant=host.service.authorize(SCOPE,{},'inspect')
    # 未绑定任务的宿主授权不从启动器配置凭空成为已提交的请求。
    assert grant is not None  # 原服务默认授权不受安装影响；HTTP仍须专用认证。
    base=host.service.authorize(AgentScope('local-angelos','angelos:fixture'),{},'inspect')
    payload={'scope':{'requester':'fixture','session':'private:fixture'},'authority':authority_payload(base),
             'operation':'capabilities','arguments':{},'nonce':'a'*48}
    caps=gateway.dispatch('shinku',payload)
    assert [r['function']['name'] for r in caps['tools']]==[tool.tool_type]
    payload['scope']['requester']='someone-else'
    with pytest.raises(AgentBoundaryError,match='requester_denied'):
        gateway.dispatch('shinku',payload)
    assert not runtime.calls


@pytest.mark.parametrize('fault',['config_permissions','token_permissions','unknown_tool','duplicate_id','unknown_field'])
def test_config_fails_before_any_model_or_effect(configured,fault):
    host,tool,runtime,path,config=configured
    if fault=='config_permissions':path.chmod(0o644)
    elif fault=='token_permissions':(path.parent/'connection.token').chmod(0o644)
    else:
        if fault=='unknown_tool':config['hosts'][0]['allowed_tools']=['does_not_exist']
        elif fault=='duplicate_id':config['hosts'].append(dict(config['hosts'][0]))
        else:config['hosts'][0]['root']=True
        path.write_text(json.dumps(config))
    with pytest.raises(ValueError):install_from_file(host,path)
    assert host.delegation is None and not runtime.calls and tool.calls==0


def test_standalone_page_does_not_open_host_protocol_by_default(tmp_path):
    with TestClient(create_angelos_app(tmp_path)) as client:
        assert client.post('/angelos/host/v1',json={}).status_code==404
