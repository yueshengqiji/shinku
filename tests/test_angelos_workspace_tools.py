"""真实工作台/文件适配器的离线验收；模型仅提供确定性选择，不使用外部服务。"""
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.demo import demo_service
from shinku.agent_runtime.materials import TaskMaterialStore
from shinku.angelos.app import create_angelos_app
from shinku.angelos.host import ServiceHost
from shinku.angelos.permissions import PermissionControls
from test_angelos import post, wait
from test_angelos_workbench_permissions import pending


@contextmanager
def workbench(root, responder, *, level='he', runtime=None, blocked=(), source_mode='ask'):
    service, _, base = demo_service(root)
    service.materials = TaskMaterialStore(service.task_store)
    original = service.authorize
    service.authorize = lambda *a: replace(original(*a), permission_mode=source_mode,
        policy=replace(original(*a).policy, blocked=frozenset(blocked)),
        budget_limits={'max_requests': 32, 'max_tokens': 2000000, 'max_tools': 24, 'max_active_seconds': 30})
    calls = []
    def answer(**kw):
        calls.append(kw)
        return responder(kw, len(calls), service)
    base.call_chat_json = answer
    if runtime:
        service.executor.profile = replace(service.executor.profile, runtime=runtime, runtime_meters_budget=True, native_tool_history=True)
    controls = PermissionControls(root/'permissions.json', default=level, approval_timeout=3)
    host = ServiceHost(service, permissions=controls)
    try:
        with TestClient(create_angelos_app(root/'ui', host=host)) as client:
            client.headers['x-angelos-token'] = client.get('/angelos/api/bootstrap').json()['token']
            yield client, service, host, calls
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def conversation(client, root=None, name='fixture'):
    wid = post(client, '/workspaces', {'name': name, 'path': str(root)})['workspace']['id'] if root else 'default'
    return post(client, '/conversations', {'workspaceId': wid})['conversation']['id']


def send(client, cid, key='turn'):
    post(client, '/conversations/'+cid+'/messages', {'request_key': key, 'text': '核实所选项目的实际文件，并按请求交付结果。'})


def final():
    return {'kind': 'final', 'text': '主管，已取得所选文件的实际结果。', 'task_outcome': 'completed'}


def call(name, **args):
    return {'tool_call': {'type': 'workspace_'+name, **args}}


def task_for(service):
    return next(iter(service._sessions.values())).context['agent_task_state']


@pytest.mark.parametrize('level', ['he', 'waw', 'aleph', 'zayin'])
def test_workbench_catalog_and_actual_line_read(tmp_path, level):
    root = tmp_path/'project'; root.mkdir(); (root/'a.py').write_text('first\nmiddle\nlast\n')
    def respond(kw, count, service):
        return call('read_lines', path='a.py', start_line=2, line_count=1) if count == 1 and level != 'zayin' else final()
    with workbench(tmp_path, respond, level=level) as (client, service, host, calls):
        cid = conversation(client, root); send(client, cid)
        result = wait(client, cid)['turns'][0]['result']
        task = service.task_store.get(result['task_id'], requester=host.requester, session='angelos:'+cid)
        grant = service._grant(AgentScope(host.requester, 'angelos:'+cid), task, 'inspect')
        workspace = {n for n in grant.allowed_tools if n.startswith('workspace_')}
        assert workspace == (set() if level == 'zayin' else {'workspace_info', 'workspace_search', 'workspace_read_lines', 'workspace_preview_patch'} | ({'workspace_apply_patch'} if level != 'he' else set()))
        assert not host.permissions.snapshot()['pending']
        assert result['execution']['status'] == 'completed', result
        if level != 'zayin':
            assert 'middle' in calls[1]['user_prompt'] and '"start_line": 2' in calls[1]['user_prompt']
            assert result['evidence'][0]['execution_status'] == 'ok'
            assert len(calls) == 2
        else:
            assert not calls[0].get('native_tools') and len(calls) == 1
        assert (root/'a.py').read_text() == 'first\nmiddle\nlast\n'


def test_default_workspace_has_info_but_no_invented_file_scope(tmp_path):
    def respond(kw, count, service):
        return call('info') if count == 1 else final()
    with workbench(tmp_path, respond) as (client, service, host, calls):
        cid = conversation(client); send(client, cid)
        result = wait(client, cid)['turns'][0]['result']
        assert result['execution']['status'] == 'completed', result
        assert 'process_working_directory' in calls[1]['user_prompt']
        ctx = next(iter(service._sessions.values())).context
        grant = service._grant(AgentScope(host.requester, 'angelos:'+cid), task_for(service), 'inspect')
        assert {n for n in grant.allowed_tools if n.startswith('workspace_')} == {'workspace_info'}
        handler = host.registry_handlers()['workspace_read_lines']
        forged = {**ctx, 'angelos_workspace_binding': {'id':'default','root':str(tmp_path)}}
        assert handler.execute(call={'path':'any'}, context=forged)['status'] == 'error'


@pytest.mark.parametrize('decision', ['aleph', 'approve', 'deny', 'revoke'])
def test_real_preview_apply_and_waw_boundary(tmp_path, decision):
    root = tmp_path/'project'; root.mkdir(); path = root/'a.py'; path.write_text('value=1\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    def respond(kw, count, service):
        if count == 1: return call('preview_patch', path='a.py', expected_sha256=digest, edits=[{'old':'value=1','new':'value=2'}])
        if count == 2:
            # 读取真实工具保存的预览凭据，不以模型生成的随机ID伪造成功。
            task = task_for(service)
            preview = task['tool_calls'][-1]['data']['state_updates']['preview_id']
            return call('apply_patch', path='a.py', expected_sha256=digest, preview_id=preview)
        return final()
    with workbench(tmp_path, respond, level='aleph' if decision == 'aleph' else 'waw') as (client, service, host, calls):
        cid = conversation(client, root); send(client, cid)
        if decision != 'aleph':
            row = pending(host.permissions)
            assert row['tool'] == 'workspace_apply_patch' and path.read_text() == 'value=1\n'
            if decision == 'revoke': host.permissions.select('he', host.permissions.revision)
            else: host.permissions.decide(row['id'], row['revision'], decision)
        result = wait(client, cid)['turns'][0]['result']
        assert path.read_text() == ('value=2\n' if decision in {'aleph','approve'} else 'value=1\n'), result
        if decision in {'aleph','approve'}:
            assert result['execution']['status'] == 'completed' and result['evidence'][-1]['execution_status'] == 'ok', result


@pytest.mark.parametrize('case', ['outside', 'symlink', 'changed_root', 'revoked', 'blocked'])
def test_selected_workspace_does_not_bypass_scope_or_denials(tmp_path, case):
    root = tmp_path/'project'; root.mkdir(); outside = tmp_path/'outside'; outside.mkdir()
    (outside/'secret').write_text('OUTSIDE_SECRET'); (root/'safe').write_text('inside')
    (root/'link').symlink_to(outside/'secret')
    def respond(kw, count, service):
        if count == 1: return call('read_lines', path='safe')
        return final()
    with workbench(tmp_path, respond) as (client, service, host, _):
        cid = conversation(client, root); send(client, cid); wait(client, cid)
        ctx = next(iter(service._sessions.values())).context
        handler = host.registry_handlers()['workspace_read_lines']
        path = {'outside':str(outside/'secret'), 'symlink':'link'}.get(case, 'safe')
        if case == 'changed_root':
            wid = host.workspace_tools.store.read('conversation',cid)['workspaceId']
            host.workspace_tools.store.workspace({'name':'fixture','path':str(outside)}, key=wid)
        elif case == 'revoked': host.permissions.select('zayin', host.permissions.revision)
        elif case == 'blocked':
            original = service.authorize
            service.authorize = lambda *a: replace(original(*a), policy=replace(original(*a).policy, blocked=frozenset({'workspace_read_lines'})))
        result = handler.execute(call={'path':path}, context=ctx)
        assert result.get('status') in {'error','blocked'} and 'OUTSIDE_SECRET' not in str(result)


def test_source_host_explicit_deny_is_preserved(tmp_path):
    root = tmp_path/'project'; root.mkdir()
    with workbench(tmp_path, lambda *a: final(), blocked=['workspace_search']) as (client, service, host, _):
        cid = conversation(client,root); send(client,cid); wait(client,cid)
        task = task_for(service)
        grant = service._grant(AgentScope(host.requester,'angelos:'+cid),task,'inspect')
        assert 'workspace_search' not in grant.allowed_tools
        assert 'workspace_read_lines' in grant.allowed_tools


def test_two_task_roots_search_receipts_and_preview_tokens_stay_separate(tmp_path):
    roots=[tmp_path/'one',tmp_path/'two']
    for index,root in enumerate(roots):
        root.mkdir();(root/'a.txt').write_text(f'project {index}\ncommon=1\n')
    def respond(kw,count,service):
        return call('search',query='project',glob='*.txt') if count%2 else final()
    with workbench(tmp_path,respond,level='aleph') as (client,service,host,calls):
        contexts=[]
        for index,root in enumerate(roots):
            cid=conversation(client,root,str(index));send(client,cid)
            result=wait(client,cid)['turns'][0]['result']
            assert result['execution']['status']=='completed',result
            assert f'project {index}' in calls[-1]['user_prompt']
            assert f'project {1-index}' not in calls[-1]['user_prompt']
            contexts.append(service._sessions[result['task_id']].context)
        handlers=host.registry_handlers()
        sha=hashlib.sha256((roots[0]/'a.txt').read_bytes()).hexdigest()
        preview=handlers['workspace_preview_patch'].execute(call={'path':'a.txt','expected_sha256':sha,
            'edits':[{'old':'common=1','new':'common=2'}]},context=contexts[0])['state_updates']
        args={'path':'a.txt','expected_sha256':sha,'preview_id':preview['preview_id']}
        cross=handlers['workspace_apply_patch'].execute(call=args,context=contexts[1])
        assert cross['status']=='error'
        (roots[0]/'a.txt').write_text('external edit\n')
        changed=handlers['workspace_apply_patch'].execute(call=args,context=contexts[0])
        assert changed['status']=='error' and (roots[0]/'a.txt').read_text()=='external edit\n'
        assert (roots[1]/'a.txt').read_text()=='project 1\ncommon=1\n'


def test_existing_host_tool_and_grant_are_not_replaced(tmp_path):
    from shinku.tools.workspace import build_workspace_tools
    service,_,_=demo_service(tmp_path)
    fixed=build_workspace_tools(tmp_path,authorize=lambda ctx,write:not write)[0]
    service.executor.tool_host.registry.register(fixed)
    try:
        host=ServiceHost(service)
        assert host.registry_handlers()[fixed.tool_type] is fixed
        grant=service._grant(AgentScope(host.requester,'angelos:unbound'),{},'inspect')
        assert fixed.tool_type not in grant.allowed_tools
        assert host.workspace_tools.store is None
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()


def test_legacy_task_resume_keeps_original_tool_ceiling(tmp_path):
    from shinku.agent_runtime import TaskHandle
    root=tmp_path/'project';root.mkdir()
    def respond(kw,count,service):
        if count==1:return {'kind':'final','text':'合成外部依赖暂不可用。','task_outcome':'failed'}
        return final()
    with workbench(tmp_path,respond) as (client,service,host,calls):
        cid=conversation(client,root);send(client,cid)
        result=wait(client,cid)['turns'][0]['result']
        assert result['execution']['status']=='failed',result
        task=service.task_store.get(result['task_id'],requester=host.requester,session='angelos:'+cid)
        # 合成一个加载前的持久化授权快照，旧记录没有任何workspace工具。
        old=task['host_grant']
        old['allowed_tools']=[n for n in old['allowed_tools'] if not n.startswith('workspace_')]
        old['policy']['allowed']=[n for n in old['policy']['allowed'] if not n.startswith('workspace_')]
        service.task_store.save(task)
        grant=service._grant(AgentScope(host.requester,'angelos:'+cid),task,'resume')
        assert not {n for n in grant.allowed_tools if n.startswith('workspace_')}
        port=host.port(cid);view=port.status(task['task_id'])
        handle=port.resume(TaskHandle(task['task_id'],view['attempt'],view['version']),request_key='resume-fixture')
        from test_angelos_workbench_permissions import finish
        assert finish(port,handle)['status']=='completed'
        assert len(calls)==2


def test_local_aleph_does_not_expand_readonly_source_host(tmp_path):
    root=tmp_path/'project';root.mkdir()
    with workbench(tmp_path,lambda *a:final(),level='aleph',source_mode='readonly') as (client,service,host,_):
        cid=conversation(client,root);send(client,cid);wait(client,cid)
        grant=service._grant(AgentScope(host.requester,'angelos:'+cid),task_for(service),'inspect')
        assert 'workspace_read_lines' in grant.allowed_tools
        assert 'workspace_apply_patch' not in grant.allowed_tools


def test_self_state_reports_only_the_selected_task_workspace(tmp_path):
    root=tmp_path/'project';root.mkdir()
    def respond(kw,count,service):
        if count==1:return {'tool_call':{'type':'read_self_state','scope':'tools'}}
        return final()
    with workbench(tmp_path,respond) as (client,service,host,calls):
        cid=conversation(client,root);send(client,cid)
        result=wait(client,cid)['turns'][0]['result']
        assert result['execution']['status']=='completed',result
        assert 'workspace_access' in calls[1]['user_prompt'] and str(root) in calls[1]['user_prompt']
        assert 'workspace_read_lines' in calls[1]['user_prompt']
        assert result['evidence'][0]['execution_status']=='ok'
