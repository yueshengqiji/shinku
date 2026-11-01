"""工作台问答真实API；状态来自隔离执行器，HTTP回答不扩大权限。"""
from types import SimpleNamespace
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from shinku.api.angelos import build_angelos_router
from shinku.angelos.host import ServiceHost
from test_agent_recovery import service_at, Runtime, close
from test_agent_questions import WAIT, FINAL
from test_angelos import new, post, wait


@pytest.fixture
def env(tmp_path):
    service, _, _, runtime = service_at(tmp_path, Runtime([WAIT, FINAL]), materials=True)
    host = ServiceHost(service, requester='owner', reply_delivery=True)
    router, bench = build_angelos_router(SimpleNamespace(data_root=tmp_path), host=host)
    app = FastAPI(); app.include_router(router)
    with TestClient(app) as client:
        client.headers['x-angelos-token'] = client.get('/angelos/api/bootstrap').json()['token']
        try: yield client, bench, host, runtime
        finally: bench.close(); close(service)


def question(client):
    cid = new(client)
    post(client, '/conversations/'+cid+'/messages', {'text':'整理材料，语言由我决定','request_key':'ask'})
    t = wait(client,cid)['turns'][0]
    assert t['question']['text'] == WAIT['text']
    data = {k:t['view'][k] for k in ('attempt','version')}
    return cid, t, dict(data,question_id=t['question']['id'],text='中文')


def test_defer_refresh_answer_keeps_original_and_single_execution(env):
    client, bench, host, runtime = env
    cid,t,data = question(client)
    url = f'/conversations/{cid}/turns/{t["id"]}/'
    assert client.post('/angelos/api'+url+'resume',json=data).json()['reason'] == 'question_answer_required'
    c = post(client,url+'defer',data)['conversation']
    assert c['turns'][0]['question']['deferred'] and len(runtime.calls) == 1
    listing = client.get('/angelos/api/bootstrap').json()['conversations'][0]
    assert listing['waiting'][0]['deferred'] and listing['activity'] is None
    assert post(client,url+'undefer',data)['conversation']['turns'][0]['question']['deferred'] is False
    for _ in range(3):
        assert client.get('/angelos/api/conversations/'+cid).json()['conversation']['turns'][0]['question']
    assert len(runtime.calls) == 1
    post(client,url+'answer',data)
    result = wait(client,cid)['turns'][0]
    assert result['view']['status'] == 'completed'
    assert result['questions'][0]['text'] == WAIT['text']
    assert result['supplements'][0]['text'] == '中文' and not result['question']
    post(client,url+'answer',data) # HTTP响应丢失后的重试，旧UI版本不能制造第二次执行
    assert len(runtime.calls) == 2


def test_stale_cancel_scope_and_origin(env):
    client, bench, host, runtime = env
    cid,t,data = question(client); url=f'/conversations/{cid}/turns/{t["id"]}/'
    stale=client.post('/angelos/api'+url+'answer',json={**data,'version':0})
    assert stale.status_code == 409 and stale.json()['reason'] == 'task_version_conflict'
    assert client.post('/angelos/api'+url+'answer',json=data,headers={'origin':'https://invalid.example'}).status_code==403
    other=new(client)
    assert client.post('/angelos/api'+url.replace(cid,other)+'answer',json=data).status_code==409
    post(client,url+'cancel',data)
    assert client.post('/angelos/api'+url+'answer',json=data).json()['reason']=='question_canceled'
    assert len(runtime.calls)==1


def test_new_active_task_prevents_answer_from_starting_second_worker(env):
    client, bench, host, runtime = env
    cid,t,data=question(client)
    bench.store.change('conversation',cid,lambda c:c['turns'].append({'id':'new','stage':'preparing','text':'合成新任务','attachments':[]}))
    r=client.post(f'/angelos/api/conversations/{cid}/turns/{t["id"]}/answer',json=data)
    assert r.json()['reason']=='conversation_busy' and len(runtime.calls)==1


def test_pending_answer_cannot_be_deleted_or_moved_to_new_workspace(env,tmp_path):
    client,bench,host,runtime=env
    cid,t,data=question(client)
    r=client.post(f'/angelos/api/conversations/{cid}/delete',json={'confirm':True})
    assert r.json()['reason']=='conversation_waiting_answer'
    w=post(client,'/workspaces',{'name':'另一工作区','path':str(tmp_path),'description':'新的约束'})['workspace']
    post(client,'/conversations/'+cid,{'workspaceId':w['id']})
    r=client.post(f'/angelos/api/conversations/{cid}/turns/{t["id"]}/answer',json=data)
    assert r.json()['reason']=='task_workspace_changed' and len(runtime.calls)==1


def test_resume_refusal_preserves_answer_and_specific_error(env):
    from dataclasses import replace
    client,bench,host,runtime=env
    cid,t,data=question(client)
    original=host.service.authorize
    host.service.authorize=lambda s,t,o:replace(original(s,t,o),resume_safe=False) if o=='resume' else original(s,t,o)
    url=f'/conversations/{cid}/turns/{t["id"]}/answer'
    r=client.post('/angelos/api'+url,json=data)
    assert r.json()['reason']=='resume_effects_unconfirmed'
    row=client.get('/angelos/api/conversations/'+cid).json()['conversation']['turns'][0]
    assert row['question']['answer_received'] and row['question']['answer_text']=='中文'
    assert row['question_error']=='resume_effects_unconfirmed' and len(runtime.calls)==1
    host.service.authorize=original
    post(client,url,data)
    row=wait(client,cid)['turns'][0]
    assert row['view']['status']=='completed' and row['question_error']=='' and len(runtime.calls)==2
