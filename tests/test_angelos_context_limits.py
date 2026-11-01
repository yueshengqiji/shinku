"""容量接线和规划诊断回归；合成路由/原文，不调用真实模型。"""
import copy
import json
from types import SimpleNamespace as N

import pytest
from fastapi.testclient import TestClient

from test_angelos_controls import controls
from shinku.angelos.models import ModelControls
from shinku.llm.model_limits import model_limits
from shinku.agent.loop import AgentLoop
from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.agent_runtime.context import build_execution_prompt


def attach(controls):
    controls.selection = {'providerId':'deepseek','modelId':'deepseek-fixture','effort':'auto'}
    bundle=N(model='deepseek-fixture',client=N(base_url='http://localhost:1234/v1',protocol='openai'))
    controls.controller.runtime.chat=bundle
    return bundle


def test_refresh_rebinds_persists_and_does_not_inherit_missing_capacity(controls):
    bundle=attach(controls)
    controls.probe=lambda settings:[{'id':settings.chat_model,'contextWindow':240000,'maxOutputTokens':12000}]
    assert controls.refresh('deepseek')['contextLimits']['window_tokens']==240000
    assert model_limits(bundle)['source']=='provider_metadata'
    saved=json.loads(controls.path.read_text())
    assert saved['models']['deepseek'][0]['contextWindow']==240000
    controls.probe=lambda settings:[{'id':settings.chat_model}]
    assert controls.refresh('deepseek')['contextLimits']['window_tokens']==0
    assert model_limits(bundle)['source']=='unknown'


def test_override_exact_route_reset_restore_and_no_shared_config_change(controls):
    bundle=attach(controls);controls.refresh('deepseek')
    original=controls.controller.store.path.read_bytes()
    data={'providerId':'deepseek','modelId':'deepseek-fixture','contextWindow':180000}
    result=controls.set_context_window(data)
    assert result['contextLimits']['source']=='angelos_override'
    assert result['contextLimits']['window_tokens']==180000
    assert result['selected']==controls.selection
    assert controls.controller.store.path.read_bytes()==original
    other=ModelControls(controls.controller,controls.path);other.restore();other.bind_runtime_limits()
    assert model_limits(bundle)['window_tokens']==180000
    # 相同模型名在另一供应商无权继承这个覆盖值。
    assert other._metadata('openai','deepseek-fixture')['contextWindow']==0
    other.set_context_window({**data,'contextWindow':0})
    assert model_limits(bundle)['window_tokens']==0


@pytest.mark.parametrize('value',[True,-1,2000001,1.5,'100000',None])
def test_invalid_capacity_preserves_state(controls,value):
    attach(controls);before=copy.deepcopy(controls.context_overrides)
    with pytest.raises(ValueError,match='invalid_context_window'):
        controls.set_context_window({'providerId':'deepseek','modelId':'deepseek-fixture','contextWindow':value})
    assert controls.context_overrides==before and not controls.path.exists()


def test_save_failure_restores_runtime_capacity(controls,monkeypatch):
    bundle=attach(controls);controls.refresh('deepseek')
    before=controls.path.read_bytes()
    def fail():raise OSError('fixture disk failure')
    monkeypatch.setattr(controls,'_persist',fail)
    with pytest.raises(OSError):controls.set_context_window(
        {'providerId':'deepseek','modelId':'deepseek-fixture','contextWindow':180000})
    assert model_limits(bundle)['window_tokens']==0 and controls.path.read_bytes()==before


def test_capacity_api_checks_origin_token_and_busy_tasks(tmp_path,controls):
    from angelos_fixture import fixture
    from shinku.angelos.host import ServiceHost
    from test_angelos import new,post
    app,bench,fixture_host=fixture(tmp_path)
    bench.host=ServiceHost(fixture_host.service,models=controls)
    data={'providerId':'deepseek','modelId':'deepseek-fixture','contextWindow':180000}
    with TestClient(app) as client:
        token=client.get('/angelos/api/bootstrap').json()['token']
        assert client.post('/angelos/api/models/context-window',json=data).status_code==403
        client.headers['x-angelos-token']=token
        assert client.post('/angelos/api/models/context-window',json=data,headers={'origin':'https://foreign.invalid'}).status_code==403
        cid=new(client)
        bench.store.change('conversation',cid,lambda c:c['turns'].append({'id':'pending','text':'等待','stage':'preparing'}))
        for endpoint,body in [('context-window',data),('refresh',{'providerId':'deepseek'})]:
            response=client.post('/angelos/api/models/'+endpoint,json=body)
            assert response.status_code==409 and response.json()['reason']=='model_change_busy'
        bench.store.change('conversation',cid,lambda c:c['turns'].clear())
        assert post(client,'/models/context-window',data)['models']['available']


@pytest.mark.parametrize('error,detail',[
    (AgentBoundaryError('execution_context_oversized'),'execution_context_oversized'),
    (AgentBoundaryError('conversation_authorization_changed'),'conversation_authorization_changed'),
    (AgentBoundaryError('secret fixture error body'),'AgentBoundaryError'),
    (RuntimeError('secret fixture error body'),'RuntimeError'),
])
def test_planner_error_keeps_code_and_location_without_exception_body(error,detail):
    def planner(*,state):raise error
    result=AgentLoop(planner=planner,handlers={}).run(task_id='fixture',context={})
    assert result.status=='failed' and result.reason=='planner_error:'+detail
    diagnostic=result.progress['planner_failure']
    assert diagnostic['stage']=='planning' and diagnostic['frames'][-1]['function']=='planner'
    assert 'secret fixture' not in json.dumps(diagnostic)


def test_native_unknown_window_uses_resource_bound_without_losing_source():
    task={'requests':[{'text':'保留原始要求'}], 'source_data':[{'content':'原始资料'*15000}]}
    state=N(context={'agent_task_state':task,'agent_context_compaction_supported':True},transcript=[])
    original=copy.deepcopy(task)
    parts=build_execution_prompt(state,native_rules=True)
    assert task==original and task['source_data'][0]['content'] in parts.user_prompt
    with pytest.raises(AgentBoundaryError,match='execution_context_oversized'):
        build_execution_prompt(state,native_rules=False)
    state.context['agent_context_compaction_supported']=False
    with pytest.raises(AgentBoundaryError,match='execution_context_oversized'):
        build_execution_prompt(state,native_rules=True)


def test_known_window_still_rejects_oversized_protected_input():
    state=N(context={'agent_task_state':{'requests':[{'text':'原始资料'*15000}]},
        'agent_context_compaction_supported':True,'agent_context_window_tokens':1000},transcript=[])
    with pytest.raises(AgentBoundaryError,match='execution_context_oversized'):
        build_execution_prompt(state,native_rules=True)


def test_confirmed_route_capacity_reaches_ninety_percent_compactor(controls,tmp_path,monkeypatch):
    from test_context_compaction import Summarizer,payload
    from shinku.llm.context_compaction import ContextCompactor
    monkeypatch.delenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS',raising=False)
    monkeypatch.delenv('SHINKU_LLM_MODEL_CONTEXT_WINDOWS_JSON',raising=False)
    bundle=attach(controls);controls.refresh('deepseek')
    p=payload(6800);original=copy.deepcopy(p);model=Summarizer()
    host=ContextCompactor(model,tmp_path/'compactions.sqlite',aware=True)
    assert host.project(p,bundle,2,'fixture')==p and not model.calls
    assert host.status('fixture')['automatic_compaction'] is False
    controls.set_context_window({'providerId':'deepseek','modelId':'deepseek-fixture','contextWindow':2100})
    out=host.project(p,bundle,2,'fixture')
    assert model.calls and out['messages'][-1]==p['messages'][-1] and p==original
    status=host.status('fixture')
    assert status['automatic_compaction'] is True and status['capacity_source']=='angelos_override'
    assert status['projected_tokens']<2100*.9
    assert 'max_tokens' not in out and 'max_completion_tokens' not in out


def test_background_worker_blocks_capacity_update(controls,tmp_path):
    from shinku.angelos.host import ServiceHost
    from shinku.agent_runtime.demo import demo_service
    service,_,_=demo_service(tmp_path)
    try:
        host=ServiceHost(service,models=controls)
        service._sessions['fixture']='ignored'
        service.coordinator.get=lambda tid:{'status':'running','worker_active':True}
        with pytest.raises(ValueError,match='model_change_busy'):
            host.update_model_capacity({'providerId':'deepseek','modelId':'deepseek-fixture','contextWindow':100000})
        assert not controls.context_overrides
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()
