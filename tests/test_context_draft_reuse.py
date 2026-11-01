"""原稿复提只节省生成请求；真实临时文件与原完成门继续决定交付。"""
from types import SimpleNamespace

import pytest

from shinku.agent.answer_protocol import retain_answer_draft, draft_after_contract
from shinku.agent_runtime.conversation import ConversationTurn
from shinku.agent_runtime.context_tools import ContextCompactionTool
from shinku.agent_runtime.contracts import AgentBoundaryError
from shinku.angelos.runtime_view import AngelosRuntime, ReasoningView
from shinku.tools.execution import ExecutionPolicy
from test_analysis_acceptance_layers import case, REPORT
from test_context_compaction import Summarizer, bundle, window


def attach_turn(state, reader, task):
    # 只模拟进程凭据查询；读取、哈希及完成核验使用真实实现。
    turn = ConversationTurn.__new__(ConversationTurn)
    turn.handlers = {reader.tool_type: reader}
    turn.current = lambda context: (task, SimpleNamespace(policy=ExecutionPolicy()))
    state.context['_agent_conversation_turn'] = turn
    retain_answer_draft(state, {'kind':'final', 'text':REPORT, 'task_outcome':'completed'})
    state.tool_calls.append({'name':'declare_task_contract','status':'ok'})
    turn.declaration = SimpleNamespace(agent_permission_class='read')
    return turn


def test_readonly_original_is_offered_once_and_gate_still_decides(case):
    state, gate, reader, _, task = case
    turn = attach_turn(state, reader, task)
    gate.result_handlers['declare_task_contract'] = turn.declaration
    draft = draft_after_contract(state)
    assert draft['text'] == REPORT and not draft['validated']
    assert gate.evaluate(state, draft['text'], outcome=draft['outcome']).accepted
    assert draft_after_contract(state) is None
    assert not state.progress['draft_contract_reuse']['is_fact_evidence']


@pytest.mark.parametrize('change', ['changed','revoked','new_request','attempt','scope','write','failed_declaration','tampered_draft'])
def test_reuse_cannot_cross_evidence_or_task_boundaries(case, change):
    state, _, reader, path, task = case
    attach_turn(state, reader, task)
    if change == 'changed': path.write_text('外部更新')
    if change == 'revoked': reader.allowed = False
    if change == 'new_request': task['requests'].append({'text':'新要求','message_id':'new'})
    if change == 'attempt': task['execution_attempt'] += 1
    if change == 'scope': task['session'] = 'other'
    if change == 'write': task['contract']['requirements'].append({'id':'write','effect':'write','targets':[str(path)]})
    if change == 'failed_declaration': state.tool_calls[-1]['status'] = 'error'
    if change == 'tampered_draft': state.progress['answer_drafts'][-1]['text'] += '伪造'
    assert draft_after_contract(state) is None


def test_reoffering_does_not_soften_execution_claim_gate(case):
    state, gate, reader, _, task = case
    turn = attach_turn(state, reader, task)
    gate.result_handlers['declare_task_contract'] = turn.declaration
    retain_answer_draft(state, {'kind':'final','text':REPORT+'\n我已部署服务。','task_outcome':'completed'})
    draft = draft_after_contract(state)
    assert draft is not None
    verdict = gate.evaluate(state,draft['text'],outcome=draft['outcome'])
    assert not verdict.accepted and '实际' in verdict.reason
    assert draft_after_contract(state) is None


@pytest.mark.parametrize('denied', [False, True, 'revoked'])
def test_context_tool_rechecks_host_permission_and_never_grants_effect(denied):
    requests = []
    grant = SimpleNamespace(policy=ExecutionPolicy(blocked=frozenset({'request_context_compaction'}) if denied else frozenset()))
    def current(context):
        if denied == 'revoked': raise AgentBoundaryError('agent_permission_changed')
        return {'task_id':'bound-task'}, grant
    runtime = SimpleNamespace(request_context_compaction=lambda binding,reason:
        requests.append((binding,reason)) or {'status':'requested','grants_permissions':False,'is_execution_evidence':False})
    turn = SimpleNamespace(current=current, service=SimpleNamespace(executor=SimpleNamespace(profile=SimpleNamespace(runtime=runtime))))
    result = ContextCompactionTool(turn).execute(call={'reason':'调查结束，进入成稿'},context={})
    assert len(requests) == (0 if denied else 1)
    assert 'is_execution_evidence' in str(result) and 'False' in str(result)


def test_runtime_rejects_cross_binding_and_unknown_window(monkeypatch,tmp_path):
    from shinku.llm.native_messages import NativeProtocolError
    base=Summarizer();base.chat=bundle()
    runtime=AngelosRuntime(base,ReasoningView(tmp_path/'reasoning.sqlite3'),context_awareness=True)
    window(monkeypatch,100000)
    with runtime.bind_task('one'):
        with pytest.raises(NativeProtocolError,match='binding_changed'): runtime.request_context_compaction('two','整理')
        assert runtime.request_context_compaction('one','整理')['status']=='requested'
        monkeypatch.delenv('SHINKU_LLM_CONTEXT_WINDOW_TOKENS')
        with pytest.raises(NativeProtocolError,match='window_unknown'): runtime.request_context_compaction('one','整理')


@pytest.mark.parametrize('late_contract,isolated',[(False,False),(True,False),(False,True)])
def test_actual_sdk_preserves_draft_history_and_avoids_duplicate_generation(tmp_path,monkeypatch,late_contract,isolated):
    from dataclasses import replace
    import json
    from test_agent_conversation_loop import native_sdk
    from test_agent_runtime_boundary import settled
    from test_analysis_acceptance_layers import Reader
    from shinku.agent_runtime import AgentTurnRequest
    from shinku.agent_runtime.demo import demo_service
    from shinku.angelos.host import ServiceHost
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    window(monkeypatch,1000000)
    source=tmp_path/'source.txt';source.write_text('当前入口负责组装服务，扩展负责提供可选能力。')
    reader=Reader(source)
    contract={'goal':'分析项目','deliverables':['结构与建议'],'scope':['只读'],'requirements':[
        {'id':'read','effect':'read','description':'真实读取','targets':[str(source)],'checks':{'unchanged':True}},
        {'id':'answer','effect':'local_answer','description':'分析','targets':[],
         'checks':{'required_fields':['项目结构'],'min_sources':1}}]}
    answer={'kind':'final','text':REPORT,'task_outcome':'completed'}
    service, business_tool, _=demo_service(tmp_path)
    names=service.executor.tool_host.registry.view()
    service.executor.tool_host=ToolHost(ToolRegistry({**names,reader.tool_type:reader}))
    authorize=service.authorize
    def authorized(*args):
        grant=authorize(*args)
        return replace(grant,allowed_tools=grant.allowed_tools|{reader.tool_type},
            tool_context={**grant.tool_context,'agent_isolated_draft_required':isolated},
            policy=replace(grant.policy,risk_by_tool={**grant.policy.risk_by_tool,reader.tool_type:'low'}))
    service.authorize=authorized
    def responder(payload,index):
        if index==1: name,args=reader.tool_type,{'path':str(source)}
        elif not late_contract or index==3:
            name,args='declare_task_contract',{'task_contract':contract}
        else:
            assert index==2
            return {'role':'assistant','content':json.dumps(answer,ensure_ascii=False)},'stop'
        if not late_contract and index==3:
            return {'role':'assistant','content':json.dumps(answer,ensure_ascii=False)},'stop'
        assert index<=3, '原稿应直接复提，不能第四次请求重复生成'
        return {'role':'assistant','content':'','reasoning_content':'SYNTHETIC_REASONING',
            'tool_calls':[{'id':f'call_{index}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]},'tool_calls'
    try:
        with native_sdk(responder) as (base,calls):
            service.executor.profile=replace(service.executor.profile,runtime=base,runtime_meters_budget=True,native_tool_history=True)
            # 原稿复提属于旧契约宿主；直接交付模式不强制补声明或重新成稿。
            host=ServiceHost(service,requester='owner',reply_delivery=False);port=host.port('one')
            handle=port.submit(AgentTurnRequest('read','分析项目 '+str(source)+'，给出结构与建议。'))
            status=settled(port,handle.task_id);result=port.result(handle.task_id)
            assert status['status']=='completed', json.dumps(result,ensure_ascii=False)
            assert len(calls)==result['budget']['requests']==3 and reader.calls==1 and business_tool.calls==0
            assert result['execution_report']['text']==REPORT
            assert all('max_tokens' not in p and 'max_completion_tokens' not in p for p in calls)
            # 两次续接保留系统、根请求、旧预算和真实原生回执；只在尾部追加新状态。
            for previous,current in zip(calls,calls[1:] if not isolated else calls[1:2]):
                assert current['messages'][:len(previous['messages'])]==previous['messages']
                assert current['messages'][0]==calls[0]['messages'][0]
            if isolated:
                assert calls[-1]['messages'][:len(calls[1]['messages'])]!=calls[1]['messages']
                assert 'local_evidence' in str(calls[-1]['messages'])
            else:
                assert sum('Host context budget advisory' in str(m.get('content','')) for m in calls[-1]['messages'])>=2
            task=service.task_store.get(handle.task_id,requester='owner',session='angelos:one')
            drafts=task.get('progress',{}).get('answer_drafts',[])
            assert len(drafts)==1 and drafts[0]['text']==REPORT
            if late_contract:
                assert task['progress']['draft_contract_reuse']['status']=='submitted_to_original_gate'
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()
