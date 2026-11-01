"""会话删除的真实 API 回归：工作区和审计保留，运行中拒绝，记忆不复活。"""
from types import SimpleNamespace
import pytest
from test_angelos import env, new, post, wait
from shinku.agent_memory.store import AgentMemoryStore


def test_delete_scope_and_memory(env, tmp_path):
    client, bench, host = env
    cid, other = new(client), new(client)
    post(client, f'/conversations/{cid}/messages', {'text':'生成文件', 'request_key':'write'})
    turn = wait(client, cid)['turns'][0]
    post(client, f'/conversations/{cid}/attachments', {'name':'x.txt', 'data':'eA=='})
    memory = AgentMemoryStore(tmp_path/'memory')
    host.memory = SimpleNamespace(store=memory, requester=host.requester)
    source={'session':'angelos:'+cid,'task_id':turn['task_id']}
    row=memory.put(principal=host.requester, workspace='w', key='key', kind='user_statement', text='删除内容', source=source)
    memory.put(principal=host.requester,workspace='w',key='other',kind='user_statement',text='保留内容',source={'session':'angelos:'+other,'task_id':'other'})
    workspaces=bench.store.list('workspace'); calls=host.tool.calls
    assert post(client, f'/conversations/{cid}/delete', {'confirm':True})['removed']['conversationId']==cid
    assert client.get('/angelos/api/conversations/'+cid).status_code==409
    assert bench.store.read('conversation',other)
    assert bench.store.list('workspace')==workspaces
    assert bench.store.list('attachment')==[]
    assert host.tool.calls==calls and host.tool.path.exists()
    assert host.port(cid).result(turn['task_id']) # 审计独立保留
    assert all(x['id'] != row['id'] for x in memory.search(host.requester,'w','删除内容'))
    with memory.transaction() as db:
        assert db.execute('SELECT deleted,payload FROM memories WHERE id=?',(row['id'],)).fetchone()==(1,'{}')
        assert not db.execute('SELECT 1 FROM memory_fts WHERE id=?',(row['id'],)).fetchone()
    assert memory.search(host.requester,'w','保留内容')
    assert memory.put(principal=host.requester,workspace='w',key='late',kind='task_experience',text='删除内容',source=source)['status']=='forgotten'


@pytest.mark.parametrize('stage',['preparing','submitting','executing'])
def test_active_delete_rejected(env,stage):
    client,bench,_=env;cid=new(client)
    bench.store.change('conversation',cid,lambda c:c['turns'].append({'stage':stage,'id':'busy'}))
    r=client.post('/angelos/api/conversations/'+cid+'/delete',json={'confirm':True})
    assert r.status_code==409 and r.json()['reason']=='conversation_busy'
    assert bench.store.read('conversation',cid)


def test_delete_guards(env):
    client,bench,_=env;cid=new(client);url='/angelos/api/conversations/'+cid+'/delete'
    assert client.post(url,json={'confirm':True},headers={'origin':'https://bad.invalid'}).status_code==403
    assert client.post(url,json={'confirm':True},headers={'x-angelos-token':''}).status_code==403
    assert client.post(url,json={'confirm':False}).status_code==409
    assert client.post(url,json={'confirm':True,'all':True}).status_code==409
    assert bench.store.read('conversation',cid)


def test_expired_task_allows_deletion_but_other_errors_do_not(env, monkeypatch):
    from shinku.agent_runtime.contracts import AgentBoundaryError
    client,bench,host=env;cid=new(client)
    bench.store.change('conversation',cid,lambda c:c['turns'].append({'stage':'finished','id':'old','task_id':'expired','text':'old'}))
    def reject(code):
        def status(_):raise AgentBoundaryError(code)
        return SimpleNamespace(status=status)
    monkeypatch.setattr(host,'port',lambda _:reject('host_authorization_denied'))
    assert client.post('/angelos/api/conversations/'+cid+'/delete',json={'confirm':True}).status_code==409
    assert bench.store.read('conversation',cid)
    monkeypatch.setattr(host,'port',lambda _:reject('task_not_found'))
    assert post(client,'/conversations/'+cid+'/delete',{'confirm':True})['removed']['conversationId']==cid
