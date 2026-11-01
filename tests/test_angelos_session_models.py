"""会话模型与任务提醒：合成配置、真实工作台/执行派发，无供应商网络。"""
import json
import os
from pathlib import Path
import subprocess
import time
import threading
from types import SimpleNamespace
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from test_angelos_controls import controls
from shinku.angelos.host import ServiceHost
from shinku.angelos.app import create_angelos_app
from shinku.agent_runtime.demo import demo_service

A={'providerId':'deepseek','modelId':'deepseek-fixture','effort':'auto'}
B={'providerId':'openai','modelId':'fixture-chat','effort':'auto'}


def test_real_runtime_clients_and_thinking_do_not_mutate_environment(controls, monkeypatch):
    controls.cache['deepseek']=[{'id':A['modelId'],'effortLevels':['high']}]
    monkeypatch.setenv('SHINKU_ANGELOS_NATIVE_TOOL_PROVIDER_ALLOWLIST','api.deepseek.com:deepseek-fixture,api.openai.com:fixture-chat')
    before=dict(os.environ)
    a=controls.create_runtime({**A,'effort':'high'});b=controls.create_runtime(B)
    try:
        assert dict(os.environ)==before
        assert a.chat.client is not b.chat.client
        assert a.chat.model==A['modelId'] and b.chat.model==B['modelId']
        assert a._build_thinking_kwargs(bundle=a.chat)['reasoning_effort']=='high'
        assert 'reasoning_effort' not in b._build_thinking_kwargs(bundle=b.chat)
        assert a._env('CHAT_API_KEY')=='fixture-deepseek' and b._env('CHAT_API_KEY')=='fixture-openai'
        os.environ['SHINKU_CHAT_MODEL_NAME']='unrelated'
        assert a._env('CHAT_MODEL_NAME')==A['modelId'] and b._env('CHAT_MODEL_NAME')==B['modelId']
    finally:a.close();b.close()


@pytest.fixture
def bench(tmp_path,controls,monkeypatch):
    service,tool,default=demo_service(tmp_path)
    from shinku.tasks.coordinator import AgentTaskCoordinator
    service.coordinator.shutdown(wait=True)
    service.coordinator=AgentTaskCoordinator(max_workers=2)
    service.task_views.coordinator=service.coordinator
    controls.selection=dict(A)
    controls.cache={'deepseek':[{'id':A['modelId'],'effortLevels':['high']}],'openai':[{'id':B['modelId']} ]}
    made=[];gates={}
    class Runtime:
        def __init__(self,data):self.selection=dict(data);self.chat=SimpleNamespace(model=data['modelId'],client=SimpleNamespace());self.calls=[];self.closed=False
        def chat_supports_native_tools(self):return True
        def call_chat_json(self,**kwargs):
            self.calls.append(kwargs)
            gate=gates.get(self.selection['modelId'])
            if gate:assert gate.wait(5),'synthetic gate timed out'
            return {'kind':'final','text':'合成回答 '+self.selection['modelId'],'task_outcome':'completed'}
        def close(self):self.closed=True
    def create(data):
        controls.validate(data);r=Runtime(data);made.append(r);return r
    monkeypatch.setattr(controls,'create_runtime',create)
    host=ServiceHost(service,models=controls)
    host.test_gates=gates
    app=create_angelos_app(tmp_path/'ui',host=host)
    with TestClient(app) as client:
        client.headers['x-angelos-token']=client.get('/angelos/api/bootstrap').json()['token']
        yield app.state.angelos,host,client,made
    service.coordinator.shutdown(wait=True);host.session_models.close();service.task_store.close()


def new(client):return client.post('/angelos/api/conversations',json={'workspaceId':'default'}).json()['conversation']['id']


def test_selection_persisted_per_conversation_busy_is_local_and_guarded(bench):
    w,h,c,made=bench;a=new(c);b=new(c)
    assert c.post(f'/angelos/api/conversations/{b}/model',json=B).json()['models']['selected']==B
    assert w.store.read('conversation',a)['modelSettings']==A
    assert w.store.read('conversation',b)['modelSettings']==B
    w.store.change('conversation',a,lambda v:v['turns'].append({'id':'busy','stage':'preparing','text':'合成'}))
    assert c.post(f'/angelos/api/conversations/{a}/model',json=B).json()['reason']=='conversation_busy'
    assert c.post(f'/angelos/api/conversations/{b}/model',json=A).status_code==200
    assert c.post(f'/angelos/api/conversations/{b}/model',json=B,headers={'x-angelos-token':'bad'}).status_code==403
    assert c.post(f'/angelos/api/conversations/{b}/model',json={**B,'apiKey':'do-not-store'}).status_code==409
    from shinku.angelos.store import WorkbenchStore
    assert WorkbenchStore(w.store.path).read('conversation',b)['modelSettings']==A
    assert h.models.selection==A
    assert 'apiKey' not in json.dumps(w.store.list('conversation'))


def test_tasks_pin_distinct_executors_and_default_delegation_is_unchanged(bench):
    from shinku.agent_runtime import AgentScope
    w,h,c,made=bench;a=new(c);b=new(c)
    c.post(f'/angelos/api/conversations/{b}/model',json=B)
    for cid in (a,b):
        r=c.post(f'/angelos/api/conversations/{cid}/messages',json={'text':'你好','request_key':cid,'attachments':[]})
        assert r.status_code==200,r.text
    until=time.monotonic()+5
    while time.monotonic()<until:
        rows=[w.refresh(cid) for cid in (a,b)]
        if all(row['turns'][-1]['stage']=='finished' for row in rows):break
        time.sleep(.01)
    assert all(row['turns'][-1]['stage']=='finished' for row in rows),rows
    sessions=[h.service._sessions[row['turns'][-1]['task_id']] for row in rows]
    assert sessions[0].executor is not sessions[1].executor
    assert [s.executor.profile.runtime.chat.model for s in sessions]==[A['modelId'],B['modelId']]
    assert all(s.executor.profile.runtime.runtime.calls for s in sessions)
    assert rows[0]['turns'][-1]['model_settings']==A and rows[1]['turns'][-1]['model_settings']==B
    assert h.session_models.for_task(AgentScope('other','remote'),{}) is h.session_models.base
    projection=c.get('/angelos/api/bootstrap').json()['conversations']
    assert all(row['latestTask']['status']=='completed' for row in projection)


def test_notification_state_machine_uses_receipts_and_deduplicates(tmp_path):
    source=Path(__file__).parents[1]/'frontend/angelos/sessions.js'
    script=tmp_path/'notifications.cjs'
    script.write_text('''const fs=require('fs'),vm=require('vm'),assert=require('assert');
vm.runInThisContext(fs.readFileSync(process.argv[2],'utf8'));
let saved;const persist=x=>{saved=JSON.parse(JSON.stringify(x))};
let n=new AngelosTaskNotices(null,persist);
const row=(status,attempt=1)=>({id:'a',title:'包含失败字样但已完成',latestTask:{id:'t',attempt,status}});
n.observe([row('failed')]);assert.equal(n.unread().length,0);
n.observe([row('running',2)]);n=new AngelosTaskNotices(saved,persist);
n.observe([row('completed',2)]);assert.equal(n.unread()[0].label,'任务完成');
n.observe([row('completed',2)]);assert.equal(n.unread().length,1);
n.observe([row('running',3)]);n.observe([row('failed',3)]);assert.equal(n.unread()[0].label,'任务失败');
n.readAll();n.observe([row('failed',3)]);assert.equal(n.unread().length,0);
n.observe([row('partial',4)]);assert.equal(n.unread()[0].label,'本轮已结束');
n.observe([]);assert.equal(n.unread().length,0);
n.observe([{id:'b',title:'快速任务',taskStates:[{id:'one',attempt:1,status:'completed',stage:'executing'}]}]);assert.equal(n.unread().length,0);
n.observe([{id:'b',title:'快速任务',taskStates:[{id:'one',attempt:1,status:'completed',stage:'finished'},{id:'two',attempt:1,status:'failed',stage:'finished'}]}]);assert.equal(n.unread().length,2);
''')
    r=subprocess.run(['node',str(script),str(source)],capture_output=True,text=True,timeout=10)
    assert r.returncode==0,r.stdout+r.stderr


def test_two_conversations_really_overlap_and_switch_does_not_touch_active_runtime(bench):
    w,h,c,made=bench;a=new(c);b=new(c)
    gate=threading.Event();h.test_gates[A['modelId']]=gate
    try:
        r=c.post(f'/angelos/api/conversations/{a}/messages',json={'text':'你好','request_key':'overlap-a','attachments':[]})
        assert r.status_code==200
        deadline=time.monotonic()+3
        while time.monotonic()<deadline and not any(r.calls for r in made):time.sleep(.01)
        a_runtime=next(r for r in made if r.calls)
        assert c.post(f'/angelos/api/conversations/{b}/model',json=B).status_code==200
        assert c.post(f'/angelos/api/conversations/{a}/model',json=B).json()['reason']=='conversation_busy'
        c.post(f'/angelos/api/conversations/{b}/messages',json={'text':'你好','request_key':'overlap-b','attachments':[]})
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            rb=w.refresh(b)
            if rb['turns'][-1]['stage']=='finished':break
            time.sleep(.01)
        assert rb['turns'][-1]['stage']=='finished'
        assert w.refresh(a)['turns'][-1]['stage']=='executing'
        assert not a_runtime.closed and a_runtime.selection==A
        assert rb['turns'][-1]['model_settings']==B
    finally:gate.set()


def test_failed_selection_does_not_replace_saved_or_running_client(bench,monkeypatch):
    w,h,c,made=bench;cid=new(c)
    assert c.post(f'/angelos/api/conversations/{cid}/model',json=B).status_code==200
    old=h.session_models.entries[cid][1]
    before=w.store.read('conversation',cid)
    def fail(*args,**kwargs):raise OSError('synthetic disk failure')
    monkeypatch.setattr(w.store,'change',fail)
    assert c.post(f'/angelos/api/conversations/{cid}/model',json=A).status_code==409
    assert w.store.read('conversation',cid)==before
    assert h.session_models.entries[cid][1] is old and not old.profile.runtime.runtime.closed
    assert made[-1].closed
