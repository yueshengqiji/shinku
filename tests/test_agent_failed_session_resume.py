"""已认证失败命令可以接续；未知执行和篡改仍阻止恢复，旧命令不重放。"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from shinku.agent_runtime import AgentScope
from shinku.agent_runtime.recovery import TaskRecoveryGate
from shinku.tools.workspace import tool_result
from test_agent_recovery import ledger
from test_angelos import post, wait
from test_angelos_workspace_tools import workbench, conversation, send, final


class Executor:
    tool_type='fixture_session'
    agent_permission_class='change'
    agent_completion_effects=('command_run',)
    agent_progress_class='change'

    def __init__(self,status='failed'):
        self.starts=0
        self.session={'session_id':'synthetic-session','command_sha256':hashlib.sha256(b'fixture command').hexdigest(),
                      'status':status,'cursor':0,'returncode':2 if status=='failed' else 125,'cleanup_status':'stopped'}

    def tool_metadata(self):
        return SimpleNamespace(input_schema={'type':'object','properties':{},'additionalProperties':False})

    def normalize_call(self,call):return {}

    def execute(self,*,call,context):
        self.starts+=1
        self.owner={k:context.get('agent_task_state',{}).get(k) for k in ('task_id','requester','session')}
        self.attempt=context.get('agent_task_state',{}).get('execution_attempt')
        return tool_result(self.tool_type,{'execution_session':deepcopy(self.session),'verified_effects':[]},error='command_failed')

    def agent_execution_event(self,*,receipt,context):
        if receipt.get('execution_session')!=self.session:return None
        task=context.get('agent_task_state',{})
        if any(task.get(k)!=v for k,v in self.owner.items()) or task.get('execution_attempt')!=self.attempt:return None
        return {'kind':{'failed':'execution_error','timed_out':'deadline_exceeded','cancelled':'cancelled','interrupted':'interrupted'}[self.session['status']],
                'session_id':self.session['session_id'],'command_sha256':self.session['command_sha256'],'cursor':self.session['cursor']}


def failure_task(ledger,status='failed'):
    store,task,_,clock=ledger;executor=Executor(status)
    executor.owner={k:task[k] for k in ('task_id','requester','session')};executor.attempt=1
    binding={**executor.owner,'attempt':1,'work_id':task['task_id'],'work_attempt':1}
    row={'name':executor.tool_type,'id':'failed-command','arguments':{},'executed':True,'status':'error',
         'execution_binding':binding,'data':{'state_updates':{'execution_session':deepcopy(executor.session),'verified_effects':[]}}}
    task['tool_calls']=[row];task=store.save(task)
    return store,task,executor,row


def review(store,task,executor):
    return TaskRecoveryGate(store,{executor.tool_type:executor}).review(task,context={},still_authorized=lambda:True)


@pytest.mark.parametrize('status',['failed','timed_out','cancelled','interrupted'])
def test_authenticated_failed_exit_is_preserved_not_unknown(ledger,status):
    store,task,executor,row=failure_task(ledger,status)
    before=deepcopy(task);result=review(store,task,executor)
    assert result['status']=='continuable',result
    assert result['checks'][0]['status']=='preserve_effect'
    assert executor.starts==0
    assert store.get(task['task_id'],requester=task['requester'],session=task['session'])==before
    assert row['status']=='error' and not row['data']['state_updates']['verified_effects']


def test_running_observation_can_use_same_attempt_authenticated_terminal(ledger):
    store,task,executor,row=failure_task(ledger,'timed_out')
    running=deepcopy(row);running.update(id='earlier-running',status='ok')
    running['data']['state_updates']['execution_session'].update(status='running',returncode=None,cleanup_status='running')
    task['tool_calls']=[running,row];task=store.save(task)
    result=review(store,task,executor)
    assert result['status']=='continuable',result
    assert all(r['receipt_invocation_id']=='failed-command' for r in result['checks'])


@pytest.mark.parametrize('tamper',['missing_session','missing_binding','wrong_task','wrong_attempt','wrong_hash','wrong_exit','unknown_executed','wrong_status','unknown_handler'])
def test_unknown_or_forged_failure_stays_blocked(ledger,tamper):
    store,task,executor,row=failure_task(ledger)
    if tamper=='missing_session':row['data']['state_updates'].pop('execution_session')
    elif tamper=='missing_binding':row.pop('execution_binding')
    elif tamper=='wrong_task':row['execution_binding']['task_id']='other'
    elif tamper=='wrong_attempt':row['execution_binding']['attempt']=2
    elif tamper=='wrong_hash':row['data']['state_updates']['execution_session']['command_sha256']='other'
    elif tamper=='wrong_exit':row['data']['state_updates']['execution_session']['returncode']=0
    elif tamper=='unknown_executed':row.pop('executed')
    elif tamper=='wrong_status':row['status']='unknown'
    else:executor.agent_execution_event=None
    task['tool_calls']=[row];task=store.save(task)
    assert review(store,task,executor)['status']=='review_required'
    assert executor.starts==0


@pytest.mark.parametrize('field',['task_id','requester','session','attempt','work_id','work_attempt'])
def test_other_binding_cannot_supply_terminal_receipt(ledger,field):
    store,task,executor,row=failure_task(ledger)
    earlier=deepcopy(row);earlier.update(id='earlier-running',status='ok')
    earlier['data']['state_updates']['execution_session'].update(status='running',returncode=None)
    row['execution_binding'][field]=99 if field in {'attempt','work_attempt'} else 'other'
    task['tool_calls']=[earlier,row];task=store.save(task)
    result=review(store,task,executor)
    assert result['status']=='review_required'
    assert result['checks'][0]['receipt_invocation_id']=='earlier-running'


def test_workbench_transport_failure_resume_preserves_error_without_rerunning_tool(tmp_path):
    executor=Executor()
    def respond(kw,count,service):
        if count==1:return {'tool_call':{'type':executor.tool_type}}
        if count==2:raise RuntimeError('synthetic model unavailable')
        assert executor.starts==1
        return final()
    with workbench(tmp_path,respond,level='aleph') as (client,service,host,calls):
        service.recovery_enabled=True
        service.executor.tool_host.registry.register(executor)
        original=service.authorize
        service.authorize=lambda *a:replace(original(*a),allowed_tools=original(*a).allowed_tools|{executor.tool_type},
            policy=replace(original(*a).policy,allowed=original(*a).policy.allowed|{executor.tool_type},
                           blocked=original(*a).policy.blocked-{executor.tool_type},risk_by_tool={executor.tool_type:'low'}))
        cid=conversation(client);send(client,cid)
        first=wait(client,cid)['turns'][0]
        assert first['view']['status']=='failed' and executor.starts==1,first
        before=service.task_store.get(first['task_id'],requester=host.requester,session='angelos:'+cid)
        result=post(client,f'/conversations/{cid}/turns/{first["id"]}/resume',{
            'attempt':first['view']['attempt'],'version':first['view']['version'],'request_key':'retry'})
        after=wait(client,cid)['turns'][0]
        assert after['view']['status']=='completed',after
        stored=service.task_store.get(first['task_id'],requester=host.requester,session='angelos:'+cid)
        assert stored['tool_calls']==before['tool_calls'] and executor.starts==1 and len(calls)==3
        assert stored['budget']['requests']>before['budget']['requests']
        assert stored['recovery_history'][0]['original_stop_reason']==before['stop_reason']


def test_page_explains_recovery_failure_instead_of_unknown_operation():
    from pathlib import Path
    import re,shutil,subprocess
    node=shutil.which('node')
    if not node:pytest.skip('Node.js unavailable')
    source=(Path(__file__).resolve().parents[1]/'frontend/angelos/app.js').read_text()
    prefix=source[:source.index('let token=')]
    renderer=re.search(r'const messageError=.*?;',source).group()
    code=prefix+'\n'+renderer+"\nprocess.stdout.write(JSON.stringify(['recovery_effect_unknown','recovery_process_identity_unavailable'].map(messageError)))"
    result=json.loads(subprocess.check_output([node,'-e',code],text=True))
    assert all('旧' in text and '操作未完成' not in text for text in result)


@pytest.mark.parametrize('recoverable',[True,False])
def test_model_switch_resumes_only_with_fresh_recovery_and_keeps_original_identity(tmp_path,recoverable):
    def respond(kw,count,service):
        if count==1:raise RuntimeError('synthetic original provider unavailable')
        return final()
    with workbench(tmp_path,respond) as (client,service,host,calls):
        service.recovery_enabled=recoverable
        selected={'providerId':'source-a','modelId':'model-a','effort':'auto'}
        host.model='model-a';host.model_options=lambda:{'selected':dict(selected)}
        cid=conversation(client);send(client,cid)
        first=wait(client,cid)['turns'][0]
        assert first['view']['status']=='failed'
        selected.update(providerId='source-b',modelId='model-b');host.model='model-b'
        response=client.post(f'/angelos/api/conversations/{cid}/turns/{first["id"]}/resume',json={
            'attempt':first['view']['attempt'],'version':first['view']['version'],'request_key':'change-provider'})
        if not recoverable:
            assert response.status_code==409 and response.json()['reason']=='task_model_changed'
            assert len(calls)==1
            return
        assert response.status_code==200,response.text
        after=wait(client,cid)['turns'][0]
        assert after['view']['status']=='completed' and len(calls)==2
        assert after['submitted_model']=='model-a' and after['model']=='model-b'
        assert after['submitted_model_settings']['providerId']=='source-a'
        assert after['model_transitions'][0]['to']['attempt']==2


def test_direct_plain_reply_preserves_tail_without_inventing_completion():
    from shinku.llm.runtime_json_call import recover_agent_plain_text
    text='合成分析正文。'*600+'END_OF_ORIGINAL'
    fallback={'kind':'final','task_outcome':'failed','_agent_reply_delivery':True}
    got=recover_agent_plain_text(text,fallback=fallback)
    assert got['text']==text and got['task_outcome']=='partial'
    assert '_agent_reply_delivery' not in got
    strict=recover_agent_plain_text(text,fallback={'kind':'final','task_outcome':'failed'})
    assert strict['task_outcome']=='failed' and len(strict['text'])==4000
    assert recover_agent_plain_text('{invalid',fallback=fallback) is None
