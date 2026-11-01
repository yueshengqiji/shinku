"""项目约定真实文件读取、重读及模型上下文注入，不调用付费模型。"""
from types import SimpleNamespace
from test_angelos import env, post, wait
from shinku.angelos.workspace_instructions import read_instructions
from shinku.agent_runtime.context import build_execution_prompt


def test_parent_order_changes_and_nested_not_global(tmp_path):
    root=tmp_path/'project';root.mkdir();sub=root/'child';sub.mkdir()
    (tmp_path/'AGENTS.md').write_text('上层约定')
    (root/'AGENTS.md').write_text('项目约定')
    (sub/'AGENTS.md').write_text('仅子目录')
    r=read_instructions({'path':str(root)})
    assert [f['content'] for f in r['files']][-2:]==['上层约定','项目约定']
    assert '仅子目录' not in str(r)
    (root/'AGENTS.md').write_text('新约定')
    assert read_instructions({'path':str(root)})['files'][-1]['content']=='新约定'
    assert read_instructions({'path':''})['status']=='unbound'


def test_unsafe_and_large_not_silently_truncated(tmp_path):
    file=tmp_path/'AGENTS.md';file.symlink_to('/etc/passwd')
    r=read_instructions({'path':str(tmp_path)})
    assert r['status']=='partial' and not any(f['path']==str(file) for f in r['files'])
    file.unlink();file.write_text('x'*32769)
    r=read_instructions({'path':str(tmp_path)})
    assert r['status']=='partial' and r['issues'][-1]['reason']=='instructions_too_large'


def test_entry_and_send_snapshot(env,tmp_path):
    client,bench,_=env
    (tmp_path/'AGENTS.md').write_text('合成项目：先验证后报告')
    w=post(client,'/workspaces',{'name':'说明测试','path':str(tmp_path)})['workspace']
    url='/angelos/api/workspaces/'+w['id']+'/instructions'
    assert client.get(url).json()['instructions']['files'][-1]['content']=='合成项目：先验证后报告'
    assert client.get(url,headers={'origin':'https://bad.invalid'}).status_code==403
    c=post(client,'/conversations',{'workspaceId':w['id']})['conversation']
    (tmp_path/'AGENTS.md').write_text('发送前更新约定')
    post(client,'/conversations/'+c['id']+'/messages',{'text':'你好','request_key':'one'})
    t=wait(client,c['id'])['turns'][0]
    assert t['workspace_instructions']['files'][-1]['content']=='发送前更新约定'


def test_prompt_contains_host_instructions_without_contract():
    rules={'files':[{'path':'/synthetic/AGENTS.md','content':'使用项目既有测试'}], 'status':'loaded'}
    state=SimpleNamespace(context={'agent_task_state':{'requests':[]},'angelos_workspace_instructions':rules},transcript=[])
    parts=build_execution_prompt(state)
    assert '使用项目既有测试' in parts.user_prompt
    assert 'workspace_instructions' in parts.user_prompt
    assert '使用项目既有测试' not in str(state.context['agent_task_state'].get('contract'))


def test_real_service_first_model_input_has_rules(tmp_path):
    from test_angelos_workspace_tools import workbench, conversation, send, final
    root=tmp_path/'project';root.mkdir();(root/'AGENTS.md').write_text('PROJECT_RULE_297: use existing tests')
    with workbench(tmp_path,lambda kw,n,service:final()) as (client,service,host,calls):
        cid=conversation(client,root);send(client,cid)
        wait(client,cid)
        assert calls and 'PROJECT_RULE_297' in calls[0]['user_prompt']
        task=next(iter(service._sessions.values())).context['agent_task_state']
        assert not any('PROJECT_RULE_297' in str(s) for s in task.get('source_data',[]))
