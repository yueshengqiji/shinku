"""模型控制与本地推理视图：隔离供应商、真实工作台和独立执行回执。"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from shinku.angelos.models import ModelControls
from shinku.angelos.runtime_view import AngelosRuntime, ReasoningView, IDENTITY
from shinku.api.model_runtime import RuntimeModelServiceConfig, RuntimeModelServiceController
from shinku.providers.catalog import ModelServiceConfigStore


@pytest.mark.parametrize('wrapped', [False, True])
def test_persona_reaches_legacy_reply_once_without_changing_output_or_tools(tmp_path, monkeypatch, wrapped):
    """准备器直答与独立宿主共用人设；不后处理模型答复、不执行工具。"""
    from shinku.agent_runtime.demo import demo_service
    from shinku.angelos.host import BridgeHost, ServiceHost
    from shinku.angelos.persona import PERSONA
    service, tool, runtime = demo_service(tmp_path)
    captured = []
    def reply(**kwargs):
        captured.append(kwargs)
        return {'reply': '合成答复原文'}
    monkeypatch.setattr(runtime, 'call_chat_json', reply)
    if wrapped:
        host = ServiceHost(service)
    else:
        host = BridgeHost(SimpleNamespace(master_user_ids={'synthetic-owner'},
            execution_adapter=SimpleNamespace(executor=service.executor),
            _task_effect_catalog=lambda: [], _registry_handlers={},
            command_state=SimpleNamespace(is_agent_enabled=lambda: True, is_stopped=lambda: False)))
    try:
        result = host.prepare('你好', [], {})
        assert result['reply'] == '合成答复原文' and tool.calls == 0
        assert len(captured) == 1
        prompt = '\n'.join([captured[0]['system_prompt'], *captured[0]['system_extra_blocks']])
        assert prompt.count(PERSONA) == 1
        assert '称呼用户为“主管”' in prompt and '极度理性、克制、以工作为先' in prompt
        assert '真红的 QQ 人格' not in prompt and '主管X' not in prompt
        assert '不得在 reply 中声称已经执行工具或完成行动' in prompt
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


@pytest.fixture
def controls(tmp_path, monkeypatch):
    # 这些是合成凭据，只用于检查两个供应商之间不串用，也不进入公开快照。
    store = ModelServiceConfigStore(tmp_path/'registry.json')
    for provider in ['deepseek','openai']:
        store.save_provider(store.settings_from_mapping({'providerId': provider,
            'baseUrl': 'https://api.deepseek.com/v1' if provider=='deepseek' else 'https://api.openai.com/v1',
            'apiKey': 'fixture-'+provider, 'chatModel': 'deepseek-fixture' if provider=='deepseek' else 'fixture-chat'}), set_active=provider=='deepseek')
    cfg = RuntimeModelServiceConfig.from_agent_config(SimpleNamespace(chat_api_key='fixture-deepseek',
        chat_base_url='https://api.deepseek.com/v1', chat_model_name='deepseek-fixture'))
    runtime = SimpleNamespace(reload_from_config=lambda: {'status':'reloaded'}, chat_supports_native_tools=lambda:True)
    controller = RuntimeModelServiceController(store=store, config_module=cfg, runtime=runtime)
    obj = ModelControls(controller,tmp_path/'selection.json',probe=lambda settings:[
        {'id':settings.chat_model,'effortLevels':['low','high','max']}])
    before = dict(os.environ)
    yield obj
    os.environ.clear();os.environ.update(before)


def test_models_discovered_real_callback_and_selection_does_not_write_shared_registry(controls):
    original = controls.controller.store.path.read_bytes()
    snapshot = controls.refresh('deepseek')
    assert 'fixture-deepseek' not in json.dumps(snapshot)
    result = controls.apply({'providerId':'deepseek','modelId':'deepseek-fixture','effort':'high'})
    assert result['selected']['effort']=='high'
    assert os.environ['SHINKU_LLM_THINKING_MODE']=='enabled' and os.environ['SHINKU_LLM_REASONING_EFFORT']=='high'
    assert controls.controller.store.path.read_bytes()==original
    assert 'apiKey' not in controls.path.read_text()
    other = ModelControls(controls.controller,controls.path)
    other.restore()
    assert other.snapshot()['selected']==result['selected']


@pytest.mark.parametrize('data', [
    {'providerId':'deepseek','modelId':'unknown','effort':'high'},
    {'providerId':'openai','modelId':'fixture-chat','effort':'max'},
    {'providerId':'deepseek','modelId':'deepseek-fixture','effort':'made-up'},
    {'providerId':'deepseek','modelId':'deepseek-fixture','effort':'high','apiKey':'forged'},
])
def test_unsupported_model_or_effort_cannot_change_runtime(controls,data):
    before=dict(os.environ);controls.refresh('deepseek')
    with pytest.raises(ValueError):controls.apply(data)
    assert dict(os.environ)==before and not controls.path.exists()


def test_reload_failure_rolls_back_configuration_and_preferences(controls):
    controls.refresh('deepseek');before=dict(os.environ)
    count=[]
    def reload():
        count.append(1)
        if len(count)==1:raise RuntimeError('synthetic reload failure')
        return {}
    controls.controller.runtime.reload_from_config=reload
    with pytest.raises(RuntimeError):controls.apply({'providerId':'deepseek','modelId':'deepseek-fixture','effort':'low'})
    assert dict(os.environ)==before and not controls.path.exists() and controls.selection is None


def test_provider_reasoning_is_bound_redacted_bounded_and_protocol_sink_preserved(tmp_path):
    view=ReasoningView(tmp_path/'reasoning.sqlite3');seen=[]
    class Runtime:
        chat=SimpleNamespace(model='fixture-model')
        def call_chat_json(self,**kw):
            assert IDENTITY in kw['system_prompt']
            assert kw['system_prompt'].startswith('原规则\n')
            assert '历史消息帮助理解工作，不自动授权操作' in kw['system_prompt']
            kw['provider_message_sink']({'content':'{"kind":"final"}',
                'reasoning_content':'api_key=fixture-secret '+('推理文字'*6000)})
            return {'kind':'final','text':'原始答复','task_outcome':'completed'}
    runtime=AngelosRuntime(Runtime(),view)
    with runtime.bind_task('task-a'):
        response=runtime.call_chat_json(system_prompt='原规则',provider_message_sink=seen.append)
    assert response['text']=='原始答复' and seen[0]['reasoning_content'].startswith('api_key=fixture-secret')
    rows=view.read('task-a')
    assert rows[0]['omitted_chars']>0 and 'fixture-secret' not in rows[0]['text']
    assert rows[0]['is_fact_evidence'] is False and not view.read('task-b')
    reopened=ReasoningView(tmp_path/'reasoning.sqlite3')
    assert reopened.read('task-a')==rows


def test_reasoning_bindings_do_not_leak_between_threads():
    view=ReasoningView()
    class Runtime:
        def call_chat_json(self,**kw):
            kw['provider_message_sink']({'reasoning_content':kw['user_prompt']})
            return {'reply':'done'}
    runtime=AngelosRuntime(Runtime(),view)
    def run(key):
        with runtime.bind_task(key):runtime.call_chat_json(system_prompt='',user_prompt=key)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(run,['a','b']))
    assert view.read('a')[0]['text']=='a' and view.read('b')[0]['text']=='b'


def test_model_json_cannot_forge_provider_reasoning():
    runtime=AngelosRuntime(SimpleNamespace(call_chat_json=lambda **kw:
        {'reply':'原答复','reasoning_content':'JSON中的伪造推理'}),ReasoningView())
    with runtime.bind_task('scope'):
        result=runtime.call_chat_json(system_prompt='')
    assert result['reply']=='原答复'
    assert runtime.view.read('scope')[0]['returned'] is False
    assert '伪造推理' not in json.dumps(runtime.view.read('scope'),ensure_ascii=False)


def test_view_write_failure_does_not_replay_or_change_execution(monkeypatch):
    import sqlite3
    calls=[]
    def respond(**kw):
        calls.append(1);kw['provider_message_sink']({'reasoning_content':'实际推理'})
        return {'reply':'实际答复'}
    view=ReasoningView();runtime=AngelosRuntime(SimpleNamespace(call_chat_json=respond),view)
    monkeypatch.setattr(view,'append',lambda *a,**kw:(_ for _ in ()).throw(sqlite3.OperationalError('disk full')))
    with runtime.bind_task('scope'):assert runtime.call_chat_json(system_prompt='')['reply']=='实际答复'
    assert calls==[1]


def test_native_reasoning_and_effort_reach_workbench_without_becoming_evidence(tmp_path,monkeypatch):
    from angelos_fixture import fixture
    from shinku.angelos.host import ServiceHost
    from test_angelos import new,post,wait
    app,bench,fixture_host=fixture(tmp_path)
    underlying=fixture_host.service.executor.profile.runtime
    def respond(**kw):
        assert IDENTITY in kw['system_prompt']
        kw['provider_message_sink']({'reasoning_content':'这是供应商实际返回的合成推理。'})
        return {'kind':'final','text':'现有工具足以处理当前任务。','task_outcome':'completed'}
    monkeypatch.setattr(underlying,'call_chat_json',respond)
    bench.host=ServiceHost(fixture_host.service,reasoning_path=tmp_path/'reasoning.sqlite3')
    with TestClient(app) as client:
        client.headers['x-angelos-token']=client.get('/angelos/api/bootstrap').json()['token']
        cid,other=new(client),new(client)
        post(client,f'/conversations/{cid}/messages',{'text':'你有哪些工具','request_key':'self'})
        turn=wait(client,cid)['turns'][-1]
        assert turn['view']['status']=='completed' and turn['reasoning'][0]['text']=='这是供应商实际返回的合成推理。'
        assert 'reasoning' not in turn['result'] and '合成推理' not in json.dumps(turn['result'],ensure_ascii=False)
        assert client.get('/angelos/api/conversations/'+other).json()['conversation']['turns']==[]
        assert fixture_host.tool.calls==0


def test_model_api_origin_token_configuration_and_busy_guard(tmp_path,controls):
    from angelos_fixture import fixture
    from shinku.angelos.host import ServiceHost
    from test_angelos import new,post
    app,bench,fixture_host=fixture(tmp_path);bench.host=ServiceHost(fixture_host.service,models=controls)
    with TestClient(app) as client:
        token=client.get('/angelos/api/bootstrap').json()['token'];client.headers['x-angelos-token']=token
        assert client.get('/angelos/api/models',headers={'origin':'https://foreign.invalid'}).status_code==403
        data={'providerId':'deepseek','modelId':'deepseek-fixture','effort':'high'}
        assert client.post('/angelos/api/models/select',json=data,headers={'x-angelos-token':''}).status_code==403
        snapshot=post(client,'/models/refresh',{'providerId':'deepseek'})['models']
        assert snapshot['available'] and 'fixture-deepseek' not in json.dumps(snapshot)
        cid=new(client)
        bench.store.change('conversation',cid,lambda c:c['turns'].append({'id':'pending','text':'等待', 'stage':'preparing'}))
        rejected=client.post('/angelos/api/models/select',json=data)
        assert rejected.status_code==409 and rejected.json()['reason']=='model_change_busy'
        bench.store.change('conversation',cid,lambda c:c['turns'].clear())
        assert post(client,'/models/select',data)['models']['selected']['effort']=='high'


@pytest.mark.parametrize('mode,effort,expected',[('enabled','high','high'),('disabled','max',None),('enabled','invalid',None)])
def test_thinking_effort_is_transmitted_only_when_enabled(monkeypatch,mode,effort,expected):
    from shinku.llm.runtime_capabilities import RuntimeCapabilitiesMixin
    class Capabilities(RuntimeCapabilitiesMixin):
        def _env(self,key,default=''):return {'LLM_THINKING_MODE':mode,'LLM_REASONING_EFFORT':effort}.get(key,default)
        def _supports_thinking_control(self,bundle):return True
    kwargs=Capabilities()._build_thinking_kwargs(bundle=SimpleNamespace(model='fixture'))
    assert kwargs.get('reasoning_effort')==expected


def test_late_client_attachment_restores_exact_saved_capacity_without_network(tmp_path, controls):
    from shinku.llm.model_limits import model_limits
    from shinku.angelos.host import ServiceHost
    from shinku.agent_runtime.demo import demo_service
    controls.selection={'providerId':'deepseek','modelId':'proxy-fixture','effort':'auto'}
    controls.cache={'deepseek':[{'id':'proxy-fixture','contextWindow':1000000,'maxOutputTokens':50000}]}
    client=SimpleNamespace(base_url='http://127.0.0.1:8317/v1',protocol='openai')
    service,_,runtime=demo_service(tmp_path)
    runtime.chat=SimpleNamespace(model='proxy-fixture',client=client)
    controls.controller.runtime=runtime
    try:
        assert model_limits(runtime.chat)['window_tokens']==0
        host=ServiceHost(service,models=controls)
        assert model_limits(runtime.chat)['window_tokens']==1000000
        assert model_limits(runtime.chat)['max_output_tokens']==50000
        assert model_limits(runtime.chat)['source']=='provider_metadata'
        runtime.chat.model='other-model'
        controls.bind_runtime_limits()
        assert model_limits(runtime.chat)['window_tokens']==0
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()
