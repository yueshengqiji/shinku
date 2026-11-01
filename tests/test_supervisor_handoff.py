"""双 HTTP 宿主和真实隔离文件副作用验证接力；没有真实模型、生产端口或 QQ。"""
from dataclasses import replace
from types import SimpleNamespace
import json
import time

import pytest
from shinku.agent.task_state import AgentTaskStore
from shinku.agent_runtime import AgentBoundaryError, AgentScope, AgentTurnRequest
from shinku.agent_runtime.delegation import DelegationGateway, DelegationPrincipal
from shinku.agent_runtime.delegation_http import RemoteAgentClient, authority_payload
from shinku.agent_runtime.execution import ExecutionGrant
from shinku.agent_runtime.service import AgentService
from shinku.agent_runtime.materials import TaskMaterialStore
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.qq.supervisor_handoff import HELP, SupervisorExecutor, SupervisorReviewer, SupervisorConnection, connection_from_file
from test_agent_delegation import remote, serve, SECRET, SCOPE, settled


@pytest.fixture
def chain(remote, tmp_path):
    downstream, tool, model, gateway, original, base = remote
    store = AgentTaskStore(tmp_path/'supervisor.sqlite3')
    coordinator = AgentTaskCoordinator(state_path=tmp_path/'workers.sqlite3', max_workers=2)
    reviews=[];mode=['delegate'];live=[True]
    def review(task):
        reviews.append(task)
        return {'action':mode[0], 'text':'请补充目标文件。' if mode[0]=='clarify' else '这里是接收者的看法。'}
    def client_factory(scope, grant, current):
        def authority(s, operation):
            if operation not in {'cancel','revoke','status'} and not current():
                raise AgentBoundaryError('source_revoked')
            return authority_payload(base)
        return RemoteAgentClient(original.endpoint, original.host_id, original.secret, authority=authority)
    executor=SupervisorExecutor(store=store, reviewer=review, client_factory=client_factory,
                               enabled=lambda:live[0],poll_seconds=.01)
    grant=ExecutionGrant('synthetic-help',frozenset({HELP}),policy=ExecutionPolicy(allowed=frozenset({HELP}),risk_by_tool={HELP:'low'}),
        budget_limits=dict(base.budget_limits),resume_safe=True)
    def authorize(scope, task, operation):
        if scope.requester!=SCOPE.requester:raise AgentBoundaryError('owner_denied')
        return grant
    service=AgentService(executor=executor,authorize=authorize,task_store=store,coordinator=coordinator,
                         material_store=TaskMaterialStore(store, approved_roots=(tmp_path,)), result_exporter=executor.export)
    entry=DelegationGateway(service,{'child':DelegationPrincipal(SECRET,authorize)})
    with serve(entry) as endpoint:
        child=RemoteAgentClient(endpoint,'child',SECRET,authority=lambda s,o:authority_payload(grant))
        yield SimpleNamespace(child=child,service=service,executor=executor,tool=tool,model=model,
            reviews=reviews,mode=mode,live=live,downstream=downstream,entry=entry,grant=grant)
        child.close()
    live[0]=False
    coordinator.shutdown(wait=True);store.close()


def submit(c,key='request'):
    return c.child.port(SCOPE).submit(AgentTurnRequest(key,'Please create and verify the demo file.',
        history_context='The child asks the parent; this is supplied context, not new permission.',
        sources=({'id':'conditions','content':'Do not modify other files.'},)))


def test_real_two_hop_task_preserves_result_artifact_and_original(chain):
    c=chain;handle=submit(c);result=settled(c.child,handle.task_id)
    assert result['execution']['status']=='completed',result
    assert len(c.reviews)==1 and c.tool.calls==1
    assert c.tool.path.read_text()=='standalone agent verified'
    assert result['verification']['accepted'] and result['artifacts']
    assert result['scope']=={'requester':SCOPE.requester,'session':SCOPE.session}
    assert result['supervisor_source']['task_id']!=handle.task_id
    assert c.reviews[0]['requests'][0]['text']=='Please create and verify the demo file.'
    assert c.reviews[0]['source_data'][0]['content']=='Do not modify other files.'
    assert c.child.port(SCOPE).submit(AgentTurnRequest('request','Please create and verify the demo file.',
        history_context='The child asks the parent; this is supplied context, not new permission.',
        sources=({'id':'conditions','content':'Do not modify other files.'},))).task_id==handle.task_id
    assert c.tool.calls==1 and len(c.reviews)==1


@pytest.mark.parametrize('choice,status',[('answer','partial'),('clarify','waiting_user')])
def test_parent_can_answer_or_clarify_without_running_executor(chain,choice,status):
    c=chain;c.mode[0]=choice
    result=settled(c.child,submit(c).task_id)
    assert result['execution']['status']==status
    assert not result['verification']['accepted'] and not c.tool.path.exists()
    assert not c.model.calls


def test_other_user_and_wrong_host_cannot_impersonate_owner(chain):
    c=chain
    with pytest.raises(AgentBoundaryError):
        c.child.port(AgentScope('stranger',SCOPE.session)).submit(AgentTurnRequest('x','I am dad'))
    wrong=RemoteAgentClient(c.child.endpoint,'mother',SECRET,authority=c.child.authority)
    try:
        with pytest.raises(AgentBoundaryError):wrong.port(SCOPE).submit(AgentTurnRequest('x','Please help'))
    finally:wrong.close()
    assert not c.reviews and not c.model.calls


def test_cross_conversation_cannot_read_task(chain):
    c=chain;handle=submit(c);settled(c.child,handle.task_id)
    with pytest.raises(AgentBoundaryError):
        c.child.port(AgentScope(SCOPE.requester,'group:other')).result(handle.task_id)


def test_changed_request_same_id_does_not_execute_again(chain):
    c=chain;handle=submit(c);settled(c.child,handle.task_id)
    with pytest.raises(AgentBoundaryError):c.child.port(SCOPE).submit(AgentTurnRequest('request','different action'))
    assert c.tool.calls==1


def test_receiver_stopped_after_review_never_submits_downstream(chain):
    c=chain
    def review(task):
        c.live[0]=False
        return {'action':'delegate','text':'可以'}
    c.executor.reviewer=review
    result=settled(c.child,submit(c).task_id)
    assert result['execution']['status']=='blocked' and c.tool.calls==0


def test_readonly_origin_cannot_turn_help_into_write(chain):
    c=chain
    c.child.authority=lambda s,o:authority_payload(replace(c.grant,permission_mode='readonly'))
    result=settled(c.child,submit(c).task_id)
    assert result['execution']['status']=='blocked' and c.tool.calls==0 and not c.reviews


def test_no_local_fallback_when_receiver_is_unavailable(chain):
    c=chain
    def unavailable(*args):raise AgentBoundaryError('downstream_unavailable')
    c.executor.client_factory=unavailable
    result=settled(c.child,submit(c).task_id)
    assert result['execution']['status']=='blocked' and c.tool.calls==0


def test_saved_result_and_link_survive_executor_reconstruction(chain):
    c=chain;handle=submit(c);result=settled(c.child,handle.task_id)
    new=SupervisorExecutor(store=c.service.task_store,reviewer=lambda t:pytest.fail('must not replay'),
        client_factory=lambda *a:pytest.fail('must not resubmit'),enabled=lambda:True)
    task=c.service._get(SCOPE,handle.task_id)
    rebuilt=new.export(task,status=result['execution']['status'],reason='',report='')
    assert rebuilt['supervisor_source']==result['supervisor_source']
    assert rebuilt['artifacts']==result['artifacts'] and c.tool.calls==1


def test_direct_advice_followup_runs_parent_again(chain):
    c=chain;c.mode[0]='clarify';handle=submit(c);settled(c.child,handle.task_id)
    view=c.child.port(SCOPE).status(handle.task_id)
    from shinku.agent_runtime import TaskHandle,AgentSupplement
    h=TaskHandle(handle.task_id,view['attempt'],view['version'])
    updated=c.child.port(SCOPE).supplement(h,AgentSupplement('details','Here is the required target.'))
    c.mode[0]='answer'
    c.child.port(SCOPE).resume(updated,request_key='continue')
    result=settled(c.child,handle.task_id)
    assert len(c.reviews)==2 and c.reviews[-1]['requests'][-1]['text']=='Here is the required target.'
    assert result['execution']['status']=='partial' and c.tool.calls==0


@pytest.mark.parametrize('value',[{}, {'action':'execute','text':'done'}, {'action':'delegate','text':'ok','authority':'full'}])
def test_review_protocol_cannot_grant_authority(value):
    r=SupervisorReviewer(SimpleNamespace(call_chat_json=lambda **kw:value),'synthetic parent')
    with pytest.raises(AgentBoundaryError,match='supervisor_decision_invalid'):r({})


def test_connection_config_is_explicit_and_permission_checked(tmp_path):
    token=tmp_path/'token';token.write_text(SECRET);token.chmod(0o600)
    p=tmp_path/'supervisor.json';p.write_text(json.dumps({'version':1,'endpoint':'http://127.0.0.1:12345',
        'host_id':'child','token_file':'token','target_name':'家人'}));p.chmod(0o600)
    assert isinstance(connection_from_file(p),SupervisorConnection)
    token.chmod(0o644)
    with pytest.raises(ValueError):connection_from_file(p)


def test_cancel_propagates_to_running_downstream_without_faking_rollback(chain):
    import threading
    from shinku.agent_runtime import TaskHandle
    c=chain;entered=threading.Event();release=threading.Event();original=c.model.call_chat_json
    def slow(**kwargs):
        entered.set();assert release.wait(4)
        return original(**kwargs)
    c.model.call_chat_json=slow
    handle=submit(c);assert entered.wait(3)
    view=c.child.port(SCOPE).status(handle.task_id)
    c.child.port(SCOPE).cancel(TaskHandle(handle.task_id,view['attempt'],view['version']))
    release.set()
    result=settled(c.child,handle.task_id)
    assert result['execution']['status']=='canceled'
    assert not result['verification']['accepted']
    assert c.executor._link(handle.task_id)['stage'] in {'stopped','stop_unconfirmed'}


def test_bridge_connection_routes_formal_task_and_returns_to_original_sender(chain,tmp_path):
    from shinku.qq.agent_bridge import QQAgentBridge
    from shinku.qq.commands import QQCommandState
    from shinku.qq.delivery import DeliveryResult
    from test_qq_tool_decision_boundary import Chat, submit as decision
    from test_qq_independent_agent import turn
    c=chain;sends=[];c.mode[0]='answer'
    b=QQAgentBridge(runtime=Chat([decision(goal='advisory summary')]),system_prompt='synthetic child',
        master_user_ids=(SCOPE.requester,),owner_gate_required=True,task_contract_enabled=True,
        command_state=QQCommandState(agent_enabled=True,permission_mode='traditional'),
        tool_decision_mode='task',delegation=SupervisorConnection(c.child.endpoint,'child',SECRET,
            poll_seconds=.02,wait_seconds=5,target_name='家人'),
        sender=lambda message:sends.append(message) or DeliveryResult(True,'sent',('synthetic',)),
        task_budget_limits=dict(c.grant.budget_limits))
    try:
        reply=b.handle_turn(turn('Please help with this original task.'))
        assert c.reviews and c.reviews[0]['requester']==SCOPE.requester
        assert c.reviews[0]['requests'][0]['text']=='Please help with this original task.'
        assert reply.final_text and sends and reply.sent
        assert not c.model.calls
        assert '家人' in b._chat_task_guidance()
        b.allowed_tool_names = {'memory_search'}
        assert HELP not in b.delegation.client.authority(SCOPE, 'capabilities')['allowed_tools']
    finally:
        b.task_store.close();b.delegation.close()


def test_recovery_marks_interrupted_without_resubmitting(chain):
    from shinku.qq.supervisor_handoff import recover_supervisor_workers
    c=chain;c.mode[0]='answer';handle=submit(c);settled(c.child,handle.task_id)
    task=c.service._get(SCOPE,handle.task_id);task.update(status='running')
    c.service.task_store.save(task)
    records=SimpleNamespace(list=lambda **kw:[{'task_id':handle.task_id,'profile_user_id':SCOPE.requester,
        'session_id':SCOPE.session,'metadata':{'recovered_after_restart':True}}])
    recover_supervisor_workers(c.service.task_store,records)
    saved=c.service._get(SCOPE,handle.task_id)
    assert saved['status']=='blocked' and saved['stop_reason']=='supervisor_restarted'
    assert saved['execution_result']['execution']['status']=='blocked'
    assert len(c.reviews)==1 and c.tool.calls==0


def test_unexpected_review_failure_is_terminal_not_endless_pending(chain):
    c=chain
    def crash(task):raise RuntimeError('synthetic failure')
    c.executor.reviewer=crash
    result=settled(c.child,submit(c).task_id)
    assert result['execution']['status']=='blocked'
    assert result['execution']['reason']=='supervisor_execution_failed'
    assert not c.model.calls


def test_trusted_parent_host_installs_real_router_and_enforces_current_owner(remote, tmp_path, monkeypatch):
    """启动器的实际接收装配也走双层 HTTP；父角色权限变更立即撤销旧授权。"""
    from shinku.qq.supervisor_handoff import SupervisorHost
    from shinku.qq.agent_bridge import QQAgentBridge
    from shinku.qq.commands import QQCommandState
    from shinku.qq.delegation import DelegationConnection
    from test_qq_tool_decision_boundary import Chat
    import shinku.api.agent_delegation as api
    downstream, tool, model, gateway, client, base = remote
    reviewer = Chat([{'action': 'delegate', 'text': '请执行并核验原请求。'}])
    bridge = QQAgentBridge(runtime=reviewer, system_prompt='synthetic parent',
        master_user_ids=(SCOPE.requester,), owner_gate_required=True,
        tool_host=downstream.executor.tool_host, policy=base.policy,
        command_state=QQCommandState(agent_enabled=True, permission_mode='traditional'),
        task_contract_enabled=True, tool_decision_mode='task',
        task_budget_limits=base.budget_limits,
        delegation=DelegationConnection(client.endpoint, client.host_id, client.secret))
    captured = []
    original = api.build_delegation_router
    def capture(value):
        captured.append(value)
        return original(value)
    monkeypatch.setattr(api, 'build_delegation_router', capture)
    host = SupervisorHost('child', SECRET, SCOPE.requester, tmp_path/'parent', reviewer,
                          'synthetic parent', (tmp_path,))
    grant = ExecutionGrant('child', frozenset({HELP}),
        policy=ExecutionPolicy(risk_by_tool={HELP: 'low'}), budget_limits=base.budget_limits)
    try:
        with host(SimpleNamespace(bridge=bridge)) as router:
            assert router.routes and bridge.supervisor_incoming
            with serve(captured[0]) as endpoint:
                child = RemoteAgentClient(endpoint, 'child', SECRET, authority=lambda s,o:authority_payload(grant))
                try:
                    caps = child.call(SCOPE, 'capabilities', {})
                    assert [r['function']['name'] for r in caps['tools']] == [HELP]
                    result = settled(child, child.port(SCOPE).submit(AgentTurnRequest('host', 'Create the demo file.')).task_id)
                    assert result['execution']['status'] == 'completed', result
                    assert tool.calls == 1 and len(reviewer.calls) == 1
                    old = captured[0].principals['child'].authorize(SCOPE, {}, 'capabilities')
                    bridge.command_state.set_permission_mode('readonly')
                    assert not old.still_authorized()
                    with pytest.raises(AgentBoundaryError):
                        child.call(AgentScope('stranger', SCOPE.session), 'capabilities', {})
                finally:
                    child.close()
        assert not bridge.supervisor_incoming
    finally:
        bridge.delegation.close()
        bridge.task_store.close()


def test_instance_supervisor_binding_creates_only_chat_client(tmp_path, monkeypatch):
    """配置接收者后不构造本地执行模型；错误配置也不能降级到自己执行。"""
    from shinku.instance_runtime import bindings
    from test_qq_tool_decision_boundary import Chat
    token = tmp_path/'host.token';token.write_text(SECRET);token.chmod(0o600)
    config_file = tmp_path/'supervisor.json'
    config_file.write_text(json.dumps({'version':1,'endpoint':'http://127.0.0.1:43210',
        'host_id':'child','token_file':str(token),'target_name':'家人'}));config_file.chmod(0o600)
    persona = tmp_path/'persona';persona.write_text('synthetic child')
    clients=[]
    class Client(Chat):
        def __init__(self, **kwargs):
            super().__init__([]);self.closed=False;clients.append(self)
        def close(self):self.closed=True
    monkeypatch.setattr('shinku.llm.runtime.LLMRuntime', Client)
    monkeypatch.setenv('SHINKU_SUPERVISOR_CONFIG',str(config_file))
    config=SimpleNamespace(runtime_extension_module='', metrics_path=str(tmp_path/'metrics.json'), persona_file=str(persona))
    with bindings(config) as host:
        assert isinstance(host.delegation,SupervisorConnection)
        assert host.execution_profile is None and host.chat_runtime is clients[0]
        assert host.expression_profile is not None and len(clients)==1
    assert clients[0].closed
    config_file.write_text('{}')
    with pytest.raises(ValueError,match='supervisor_config_invalid'):
        with bindings(config):pytest.fail('invalid supervisor cannot construct a local executor')
    assert len(clients)==2 and clients[-1].closed


def test_host_bound_material_reaches_parent_and_executor_without_becoming_permission(chain):
    from shinku.agent_runtime.materials import MaterialSource
    from shinku.qq.supervisor_handoff import canonical_scope
    c=chain
    material=MaterialSource('quoted-source','related_message','Synthetic quoted context, not owner authorization.',
        provenance={'message_id':'quoted','sender_id':'someone','relation':'reply'},version='v1')
    handle=c.child.port(SCOPE).submit(AgentTurnRequest('material','Create the demo file.'),materials=(material,))
    result=settled(c.child,handle.task_id)
    assert result['execution']['status']=='completed',result
    assert c.reviews[0]['handoff_materials'][0]['content']==material.content
    task=c.service._get(SCOPE,handle.task_id)
    remote=c.downstream.task_store.get(c.executor._link(handle.task_id)['task_id'],requester=SCOPE.requester,
        session='desktop:supervised:'+canonical_scope(task))
    assert any(item['source_id']=='quoted-source' for item in remote['material_package']['items'])
    assert remote['requester']==SCOPE.requester and c.tool.calls==1
