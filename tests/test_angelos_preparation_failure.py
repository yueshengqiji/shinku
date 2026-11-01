"""准备截断/失败不得提交半份契约；工作台保存有用原稿和用量，不保存推理。"""
import json
from pathlib import Path
import time

import pytest
from fastapi.testclient import TestClient

from angelos_fixture import fixture
from shinku.angelos.host import ServiceHost,PreparationFailure


class LegacyPreparationHost(ServiceHost):
    """只验证旧BridgeHost准备协议；正式工作台同一循环另有整链回归。"""
    single_loop = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.single_loop = False

from shinku.angelos.store import WorkbenchStore
from shinku.agent_runtime.demo import demo_service
from shinku.llm.response_protocol import note_incomplete_response
from test_angelos import new,post,wait


@pytest.mark.parametrize('finish',['length','content_filter'])
def test_prepare_rejects_even_parseable_reply_on_incomplete_finish(tmp_path,monkeypatch,finish):
    service,tool,runtime=demo_service(tmp_path)
    calls=[]
    def truncated(**kw):
        calls.append(kw)
        kw['provider_message_sink']({'content':'{"task_contract":{"goal":"部分原稿', 'reasoning_content':'REASONING_MUST_NOT_BE_ARCHIVED'})
        note_incomplete_response(finish)
        return {'reply':'看起来完整，但不能在输出截断时交付'}
    monkeypatch.setattr(runtime,'call_chat_json',truncated)
    host=ServiceHost(service)
    try:
        with pytest.raises(PreparationFailure) as exc:
            host.prepare('读取 input.txt 并分析',[],{'path':str(tmp_path)})
        error=exc.value
        assert error.code=='input_preparation_'+('output_length' if finish=='length' else 'output_filtered')
        assert error.failure['text']=='{"task_contract":{"goal":"部分原稿'
        assert not error.failure['validated'] and not error.failure['complete']
        assert 'REASONING' not in json.dumps(error.failure) and len(calls)==1 and tool.calls==0
        assert error.usage['requests']==1
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()


def test_workbench_retains_failed_preparation_across_restart_without_task_submission(tmp_path,monkeypatch):
    app,bench,fixture_host=fixture(tmp_path)
    service=fixture_host.service;calls=[]
    def truncated(**kw):
        calls.append(kw)
        kw['provider_message_sink']({'content':'{"task_contract":"DRAFT_TAIL"', 'reasoning_content':'PRIVATE_REASONING'})
        note_incomplete_response('length')
        return {}
    monkeypatch.setattr(service.executor.profile.runtime,'call_chat_json',truncated)
    bench.host=LegacyPreparationHost(service,requester=fixture_host.requester)
    with TestClient(app) as client:
        client.headers['x-angelos-token']=client.get('/angelos/api/bootstrap').json()['token']
        cid=new(client)
        post(client,f'/conversations/{cid}/messages',{'text':'读取 input.txt 并分析','request_key':'truncated'})
        turn=wait(client,cid)['turns'][-1]
        assert turn['stage']=='failed' and turn['error']=='input_preparation_output_length'
        assert not turn.get('task_id') and 'DRAFT_TAIL' in turn['preparation_failure']['text']
        assert not turn['preparation_failure']['validated'] and turn['preparation_usage']['requests']==1
        saved=WorkbenchStore(bench.store.path).read('conversation',cid)['turns'][-1]
        assert saved['preparation_failure']==turn['preparation_failure']
        assert 'PRIVATE_REASONING' not in json.dumps(saved['preparation_failure'])
        assert len(calls)==1 and fixture_host.tool.calls==0


def test_preparation_error_never_publishes_secret_or_executes_contract(tmp_path,monkeypatch):
    service,tool,runtime=demo_service(tmp_path)
    def fail(**kw):
        kw['provider_message_sink']({'content':'api_key=sk-syntheticsecret123456'})
        raise RuntimeError('sensitive transport detail must not escape')
    monkeypatch.setattr(runtime,'call_chat_json',fail)
    try:
        with pytest.raises(PreparationFailure) as exc:
            ServiceHost(service).prepare('分析 input.txt',[],{})
        assert exc.value.code=='input_preparation_call_failed'
        data=json.dumps(exc.value.failure)
        assert 'syntheticsecret' not in data and 'sensitive transport' not in data and tool.calls==0
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()


@pytest.mark.parametrize('response,expected',[
    ({'task_contract': {}}, {'code':'requirements_missing_or_oversized','field_path':'requirements','phase':'validation'}),
    ({'unexpected': 'synthetic'}, {'code':'contract_missing','field_path':'task_contract','phase':'validation'}),
])
def test_contract_validation_detail_and_body_survive_workbench_restart(tmp_path,monkeypatch,response,expected):
    """原始协议错误可定位字段；不自动修复、重试或执行模型契约。"""
    app,bench,host=fixture(tmp_path)
    raw=json.dumps(response,ensure_ascii=False,indent=2)
    def invalid(**kw):
        kw['provider_message_sink']({'content':raw,'reasoning_content':'PRIVATE_REASONING'})
        return response
    monkeypatch.setattr(host.service.executor.profile.runtime,'call_chat_json',invalid)
    bench.host=LegacyPreparationHost(host.service,requester=host.requester)
    with TestClient(app) as client:
        client.headers['x-angelos-token']=client.get('/angelos/api/bootstrap').json()['token']
        cid=new(client)
        post(client,f'/conversations/{cid}/messages',{'text':'给出一个普通建议','request_key':'invalid-contract'})
        turn=wait(client,cid)['turns'][-1]
        assert turn['stage']=='failed' and turn['error']=='contract_preparation_rejected'
        assert turn['preparation_failure']['detail']==expected
        assert turn['preparation_failure']['text']==raw and not turn.get('task_id')
        saved=WorkbenchStore(bench.store.path).read('conversation',cid)['turns'][-1]
        assert saved['preparation_failure']==turn['preparation_failure']
        assert 'PRIVATE_REASONING' not in json.dumps(saved['preparation_failure'])
        assert host.tool.calls==0 and turn['preparation_usage']['requests']==1


def test_successful_preparation_body_retained_without_reasoning(tmp_path,monkeypatch):
    """正文保留供应商格式；敏感值脱敏，展示许可与验证结果仍独立。"""
    app,bench,host=fixture(tmp_path)
    raw=' { "reply": "合成建议 api_key=sk-syntheticsecret123456" } '
    def reply(**kw):
        kw['provider_message_sink']({'content':raw,'reasoning_content':'PRIVATE_REASONING'})
        return json.loads(raw)
    monkeypatch.setattr(host.service.executor.profile.runtime,'call_chat_json',reply)
    bench.host=LegacyPreparationHost(host.service,requester=host.requester)
    with TestClient(app) as client:
        client.headers['x-angelos-token']=client.get('/angelos/api/bootstrap').json()['token']
        cid=new(client)
        post(client,f'/conversations/{cid}/messages',{'text':'给出一个普通建议','request_key':'reply-body'})
        turn=wait(client,cid)['turns'][-1]
        assert turn['stage']=='replied' and not turn.get('task_id')
        output=turn['preparation_output']
        assert output['source']=='preparation_model_output' and not output['validated']
        assert output['text'].startswith(' { "reply": ') and output['text'].endswith(' } ')
        assert 'syntheticsecret' not in output['text'] and 'PRIVATE_REASONING' not in json.dumps(output)
        assert WorkbenchStore(bench.store.path).read('conversation',cid)['turns'][-1]['preparation_output']==output
        assert host.tool.calls==0


def test_successful_contract_preparation_retains_body(tmp_path,monkeypatch):
    from shinku.agent_runtime.demo import demo_request
    service,tool,runtime=demo_service(tmp_path)
    response={'task_contract':demo_request(tool).contract}
    raw=json.dumps(response,ensure_ascii=False,indent=2)
    def prepare(**kw):
        kw['provider_message_sink']({'content':raw})
        return response
    monkeypatch.setattr(runtime,'call_chat_json',prepare)
    try:
        result=ServiceHost(service).prepare('生成合成文件',[],{})
        assert result['preparation_output']['text']==raw and result['contract']
        assert tool.calls==0
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()
