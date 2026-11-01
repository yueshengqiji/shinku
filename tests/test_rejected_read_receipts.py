"""Rejected dispatches cannot erase an independently refreshed file read."""
from dataclasses import replace
from pathlib import Path
import copy
import importlib
import json
import time

import pytest

from shinku.agent.loop import AgentState
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent.verification import SmartCompletionVerifier
from shinku.agent_runtime import AgentScope, ExecutionGrant, ExecutionProfile
from shinku.agent_runtime.demo import demo_request, demo_service
from shinku.agent_runtime.materials import MaterialPolicy, MaterialSource, TaskMaterialSession, TaskMaterialStore
from shinku.agent_runtime.material_session import MaterialReadTool
from shinku.hosts.tool_host import ToolHost
from shinku.tools.execution import ExecutionPolicy, execute_invocation
from shinku.tools.invocation import ToolInvocation
from shinku.tools.registry import ToolRegistry

SCOPE = AgentScope('dispatch-fixture', 'private:dispatch-fixture')


@pytest.fixture
def actual_reader(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    joint = importlib.import_module('verify_agent_chat_handoff')
    return joint.CompleteFileReader(tmp_path, {'initial':{}, 'expected':{'output.txt':'output'}}, 'read')


def test_rejected_parameters_attest_no_handler_dispatch(actual_reader, tmp_path):
    call = ToolInvocation(actual_reader.tool_type, {'path':str(tmp_path/'attachment.txt')})
    result, envelope = execute_invocation(call, handlers={actual_reader.tool_type:actual_reader})
    assert result is None and envelope.status == 'error' and envelope.data['code'] == 'bad_args'
    assert actual_reader.calls == []
    assert getattr(envelope, 'executed', None) is False


@pytest.mark.parametrize('raises', [False, True])
def test_started_handler_is_distinct_from_success_or_failure(actual_reader, tmp_path, monkeypatch, raises):
    path=tmp_path/'output.txt';path.write_bytes(b'output')
    entered=[];original=actual_reader.execute
    def execute(**kwargs):
        entered.append(True)
        if raises:raise OSError('isolated read failure')
        result=original(**kwargs)
        result['executed']=False
        result['state_updates']['executed']=False
        return result
    monkeypatch.setattr(actual_reader,'execute',execute)
    _, envelope=execute_invocation(ToolInvocation(actual_reader.tool_type,{'path':str(path)}),
        handlers={actual_reader.tool_type:actual_reader})
    assert entered == [True]
    assert getattr(envelope,'executed',None) is True
    assert envelope.status == ('error' if raises else 'ok')


def rejected_loop_record(reader, path):
    class Runtime:
        calls=0
        def chat_supports_native_tools(self):return True
        def call_chat_json(self,**kwargs):
            self.calls+=1
            if self.calls==1:return {'tool_call':{'type':reader.tool_type,'path':str(path)}}
            return {'kind':'final','text':'The parameter check rejected this call.'}
    observed=[]
    ToolHost(ToolRegistry({reader.tool_type:reader})).run(runtime=Runtime(),system_prompt='Offline dispatch receipt.',
        policy=ExecutionPolicy(risk_by_tool={reader.tool_type:'low'}),completion_gate=lambda state,text:True,
        state_observer=lambda state,status:observed.append(state))
    assert reader.calls == [] and len(observed[-1].tool_calls)==1
    return observed[-1].tool_calls[0]


def test_loop_and_sqlite_preserve_validation_provenance(actual_reader, tmp_path):
    record=rejected_loop_record(actual_reader,tmp_path/'attachment.txt')
    assert record['status']=='error' and record.get('executed') is False
    assert record['data']['code']=='bad_args'
    store=AgentTaskStore(tmp_path/'receipt.sqlite')
    try:
        task=new_task(task_id='rejected',requester=SCOPE.requester,session=SCOPE.session,
            request='Read a scoped attachment.',message_id='rejected',contract=None)
        task['tool_calls']=[record];saved=store.save(task)
        restored=store.get(saved['task_id'],requester=SCOPE.requester,session=SCOPE.session)
        assert restored['tool_calls']==[record]
    finally:store.close()


@pytest.fixture
def complete_read(tmp_path, actual_reader):
    store=AgentTaskStore(tmp_path/'proof.sqlite',clock=lambda:1000.0)
    task=new_task(task_id='read',requester=SCOPE.requester,session=SCOPE.session,
        request='Read the attachment without changing it.',message_id='read',contract=None)
    task.update(status='running',execution_attempt=1);task=store.save(task)
    source=tmp_path/'attachment.txt';source.write_bytes(b'a'*80+b'TAIL\n')
    materials=TaskMaterialStore(store,approved_roots=(tmp_path,),policy=MaterialPolicy(inline_chars=8,page_chars=64))
    task['material_package']=materials.package(task,[MaterialSource('attachment','attachment',local_path=source)],authorization_ref='offline')
    task=store.save(task);session=TaskMaterialSession(materials,task,still_authorized=lambda:True)
    item=session.prompt_package()['items'][0];calls=[]
    for offset in range(0,item['content_chars'],64):
        receipt=materials.read(item['reference'],task=session._bound_task(),still_authorized=session.still_authorized,
            offset=offset,length=64,provided=True)
        calls.append({'name':MaterialReadTool.tool_type,'id':receipt['read_id'],'status':'ok',
            'arguments':{'reference':item['reference'],'offset':offset,'length':64},
            'data':{'state_updates':{'material_read_receipt':{k:v for k,v in receipt.items() if k!='content'}}}})
    contract={'requirements':[{'id':'read','description':'Complete source read unchanged','effect':'read',
        'targets':[str(source)],'checks':{'unchanged':True,'contains':'TAIL'}}]}
    state=AgentState(task_id=task['task_id'],tool_calls=calls,context={'agent_completion_contract':contract})
    session._attest_file_reads(state)
    verifier=SmartCompletionVerifier(tool_capabilities={
        MaterialReadTool.tool_type:{'effects':('read',),'target_arguments':()},
        actual_reader.tool_type:{'effects':('read',),'target_arguments':('path',)}},
        result_handlers={MaterialReadTool.tool_type:MaterialReadTool(session),actual_reader.tool_type:actual_reader})
    assert verifier.evaluate(state,'The complete attachment was read.',outcome='completed').accepted
    yield state,verifier,session,source,actual_reader
    store.close()


def test_known_rejected_read_cannot_supersede_the_verified_material_read(complete_read):
    state,verifier,_,source,reader=complete_read
    record=rejected_loop_record(reader,source)
    # Also exercises the verifier independently of the dispatch producer.
    record['executed']=False
    state.tool_calls.append(record)
    verdict=verifier.evaluate(state,'The attachment was read without changes.',outcome='completed')
    assert verdict.accepted and verdict.satisfied==('read',) and not verdict.missing


@pytest.mark.parametrize('execution_fact',[True,None,0,'false'])
def test_executed_or_unknown_failure_still_supersedes_a_read(complete_read,execution_fact):
    state,verifier,_,source,reader=complete_read
    failure={'name':reader.tool_type,'id':'failed','arguments':{'path':str(source)},'status':'error',
        'data':{'state_updates':{'executed':False}}}
    if execution_fact is not None:failure['executed']=execution_fact
    state.tool_calls.append(failure)
    assert not verifier.evaluate(state,'The attachment was read.',outcome='completed').accepted


@pytest.mark.parametrize('change',['changed','revoked','permission','canceled'])
def test_rejected_dispatch_does_not_bypass_current_source_or_authority(complete_read,change):
    state,verifier,session,source,reader=complete_read
    record=rejected_loop_record(reader,source);record['executed']=False;state.tool_calls.append(record)
    if change=='changed':source.write_bytes(b'changed source')
    elif change=='revoked':session.store.revoke(session.prompt_package()['items'][0]['reference'],task=session._bound_task())
    elif change=='permission':session.still_authorized=lambda:False
    else:
        task=copy.deepcopy(session.task);task['status']='canceled';session.store.store.save(task)
    assert not verifier.evaluate(state,'The attachment was read.',outcome='completed').accepted


def test_independent_worker_recovers_after_real_rejected_read(actual_reader, tmp_path):
    service,writer,_=demo_service(tmp_path)
    service.executor.tool_host.registry.register(actual_reader)
    source=tmp_path/'attachment.txt';content='a'*80+'TAIL: unchanged.';source.write_bytes(content.encode())
    service.materials=TaskMaterialStore(service.task_store,approved_roots=(tmp_path,),policy=MaterialPolicy(inline_chars=8,page_chars=64))
    grant=ExecutionGrant('offline-rejected-read',frozenset({writer.tool_type,actual_reader.tool_type}),
        policy=ExecutionPolicy(risk_by_tool={writer.tool_type:'low',actual_reader.tool_type:'low'}),permission_mode='ultra',
        budget_limits={'max_requests':16,'max_tokens':600000,'max_tools':12,'max_active_seconds':30},still_authorized=lambda:True,
        resume_safe=True,material_sources=(MaterialSource('attachment','attachment',local_path=source),))
    service.authorize=lambda scope,task,operation:grant
    request=demo_request(writer,key='real-rejected-read');contract=copy.deepcopy(request.contract)
    contract['requirements'].insert(0,{'id':'read','description':'Read original attachment unchanged',
        'effect':'read','targets':[str(source)],'checks':{'unchanged':True,'contains':'TAIL'}})
    request=replace(request,contract=contract)
    class Runtime:
        calls=0;rejected=False
        def chat_supports_native_tools(self):return True
        def call_chat_json(self,**kwargs):
            self.calls+=1
            data=json.loads(kwargs['user_prompt'].split('\n',1)[1].split('\n[Task materials',1)[0])
            package=data['task_materials'];reference=package['items'][0]['reference']
            receipts=service.materials.read_receipts({'task_id':package['task_id'],'requester':SCOPE.requester,'session':SCOPE.session})
            offset=max((r['offset']+r['returned_chars'] for r in receipts if r.get('provided')),default=0)
            if offset<len(content):return {'tool_call':{'type':'task_material_read','reference':reference,'offset':offset,'length':64}}
            if not writer.calls:return {'tool_call':{'type':writer.tool_type,'path':str(writer.path)}}
            if not self.rejected:
                self.rejected=True
                return {'tool_call':{'type':actual_reader.tool_type,'path':str(source)}}
            return {'kind':'final','text':'The complete source remained unchanged and the output was verified.','task_outcome':'completed'}
    runtime=Runtime();service.executor.profile=ExecutionProfile(runtime,max_rounds=8)
    try:
        client=service.bind(SCOPE);handle=client.submit(request);deadline=time.monotonic()+6
        while time.monotonic()<deadline:
            view=client.status(handle.task_id)
            if not view['worker_active'] and view['status'] not in {'running','queued'}:break
            time.sleep(.005)
        result=client.result(handle.task_id)
        assert view['status']=='completed' and result['verification']['accepted'] is True,json.dumps({
            'status':view['status'],'verification_reason':result['verification']['reason'],
            'missing':result['verification']['missing'],'writer_calls':writer.calls,'reader_calls':len(actual_reader.calls)})
        assert writer.calls==1 and writer.path.read_bytes()==b'standalone agent verified'
        assert actual_reader.calls==[] and source.read_bytes()==content.encode()
        saved=service.task_store.get(handle.task_id,requester=SCOPE.requester,session=SCOPE.session)
        errors=[c for c in saved['tool_calls'] if c['name']==actual_reader.tool_type]
        assert len(errors)==1 and errors[0]['executed'] is False and errors[0]['data']['code']=='bad_args'
        before=(runtime.calls,writer.calls,saved['budget'],service.materials.read_receipts(saved))
        assert client.submit(request).task_id==handle.task_id
        client.status(handle.task_id);client.result(handle.task_id)
        assert before==(runtime.calls,writer.calls,saved['budget'],service.materials.read_receipts(saved))
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()
