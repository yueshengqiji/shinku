"""真红对话选择→真实本机HTTP→独立执行→SDK表达/逐句检查→原宿主投递。"""
from dataclasses import replace
import hashlib
import json
import time

import pytest

from shinku.agent_runtime import AgentBoundaryError
from shinku.agent.task_state import AgentTaskStore
from shinku.hosts.tool_host import ToolHost
from shinku.qq.agent_bridge import QQAgentBridge
from shinku.qq.commands import QQCommandState
from shinku.qq.delegation import DelegationConnection
from shinku.qq.delivery import DeliveryResult
from shinku.result_expression import ExpressionProfile
from shinku.tasks.coordinator import AgentTaskCoordinator
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.registry import ToolRegistry
from test_agent_delegation import remote, SECRET, SCOPE
from test_expression_semantic_coverage import semantic_sdk, full_body
from test_qq_independent_agent import turn
from test_qq_tool_decision_boundary import Chat, submit


class CatalogueOnly:
    """聊天宿主只有能力目录；任何本地执行都让测试失败。"""
    def __init__(self, tool): self.tool = tool
    def __getattr__(self, name): return getattr(self.tool, name)
    def execute(self, **kwargs): raise AssertionError('delegated work must not execute in Bot')


def bridge_for(tmp_path, remote, replies, *, expression=None, asynchronous=False, master='fixture'):
    service, tool, runtime, gateway, client, base = remote
    chat, sent = Chat(replies), []
    commands = QQCommandState(agent_enabled=True, permission_mode='ultra')
    coordinator = AgentTaskCoordinator(max_workers=1)
    store = AgentTaskStore(tmp_path/'bot-tasks.sqlite3')
    connection = DelegationConnection(client.endpoint, 'shinku', SECRET, poll_seconds=.02, wait_seconds=5)
    bridge = QQAgentBridge(runtime=chat, system_prompt='SYNTHETIC_SHINKU_PERSONA',
        tool_host=ToolHost(ToolRegistry({tool.tool_type:CatalogueOnly(tool)})),
        command_state=commands, task_store=store, task_coordinator=coordinator,
        master_user_ids=(master,), owner_gate_required=True,
        task_contract_enabled=True, tool_decision_mode='task', delegation=connection,
        async_agent_enabled=asynchronous, expression_profile=expression,
        policy=ExecutionPolicy(risk_by_tool={tool.tool_type:'low'}),
        sender=lambda message: sent.append(message) or DeliveryResult(True,'sent',('synthetic-message',)),
        task_budget_limits=dict(base.budget_limits))
    bridge.delegation_publications = []
    deliver = bridge._deliver_text
    def recorded_delivery(*args, **kwargs):
        bridge.delegation_publications.append((args, kwargs))
        return deliver(*args, **kwargs)
    bridge._deliver_text = recorded_delivery
    return bridge, chat, sent, store, coordinator


def close(store, coordinator):
    coordinator.shutdown(wait=True)
    store.close()


def test_direct_reply_crosses_http_and_expression_without_claiming_goal_verified(remote, tmp_path):
    """新独立宿主经同一真红交接链回传，原文、事实和真实投递回执仍分别绑定。"""
    from shinku.angelos.host import ServiceHost
    service, tool, runtime, gateway, client, base = remote
    ServiceHost(service, requester='owner')
    with semantic_sdk() as (model, checker, calls):
        profile = ExpressionProfile(model, checker=checker, format_version=2, style_prompt='SYNTHETIC_SHINKU_STYLE')
        bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [submit()], expression=profile)
        try:
            reply = bridge(turn('创建 ' + str(tool.path) + ' 并说明结果。'))
            assert reply.agent_status == 'completed', (reply.reason, reply.final_text)
            assert tool.calls == 1 and len(runtime.calls) == 3
            assert sent and len(bridge.delegation_publications) == 1
            row = bridge.delegation._rows(SCOPE)[0]
            result = client.port(SCOPE).result(row['task_id'])
            assert result['verification']['layers']['policy'] == 'agent_reply_delivery'
            assert result['verification']['layers']['goal_completion'] == 'not_verified'
            assert result['execution_report']['text'] in reply.final_text
            assert '当前目标验收通过' not in reply.final_text
            assert '整项目标未做独立验收' in reply.final_text
            assert [role for role, _, _ in calls] == ['expression', 'check']
        finally:
            close(store, coordinator)


def test_model_chosen_submit_reaches_independent_executor_and_gemini_expression_once(remote, tmp_path):
    service, tool, runtime, gateway, client, base = remote
    with semantic_sdk() as (model, checker, expression_calls):
        profile = ExpressionProfile(model, checker=checker, format_version=2, style_prompt='SYNTHETIC_SHINKU_STYLE')
        bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [submit(goal='仅为建议摘要')], expression=profile)
        try:
            original = '  请在 '+str(tool.path)+' 创建示例文件并验证内容。\n'
            answer = bridge(turn(original))
            assert answer.agent_status == 'completed', (answer.reason, len(chat.calls), answer.final_text)
            assert tool.calls == 1 and sent and len(bridge.delegation_publications) == 1 and len(chat.calls) == 1
            assert [role for role,_,_ in expression_calls] == ['expression','check']
            timing = bridge.trace_store.snapshot()[0]['performance']
            phases = {r['phase']: r for r in timing['phases']}
            assert {'routing', 'delegation_wait', 'expression', 'expression_check', 'delivery'} <= phases.keys()
            assert phases['expression']['requests'] == 1 and phases['expression_check']['requests'] == 1
            assert bridge.delegation.expression.profile.semantic_review_version == 1
            assert bridge.delegation.expression.profile.paragraph_presentation_version == 7
            row = bridge.delegation._rows(SCOPE)[0]
            remote_task = service.task_store.get(row['task_id'], requester=SCOPE.requester, session=SCOPE.session)
            assert remote_task['requests'][0]['text'] == original
            assert '仅为建议摘要' not in remote_task['requests'][0]['text']
            assert not store.candidates(requester=SCOPE.requester, session=SCOPE.session)
            assert 'SYNTHETIC_SHINKU_PERSONA' not in json.dumps(runtime.calls, ensure_ascii=False)
            assert SECRET not in json.dumps(chat.calls, ensure_ascii=False)
            assert row['task_id'] in json.dumps(row)
            assert bridge.execution_adapter is None
        finally:
            close(store, coordinator)


def test_direct_chat_does_not_submit_and_delegation_choice_has_no_category_gate(remote, tmp_path):
    service, tool, runtime, gateway, client, base = remote
    bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [
        {'kind':'final','text':'可以，我们先讨论你的想法。'}, submit()])
    try:
        answer = bridge(turn('先聊聊思路'))
        assert answer.final_text == '可以，我们先讨论你的想法。', (answer.reason, len(chat.calls))
        assert not runtime.calls and not bridge.delegation._rows(SCOPE)
        # 不根据“简单查询”字样改变模型的委托选择；执行仍必须符合实际原请求。
        answer = bridge(turn('这不是简单查询：请在 '+str(tool.path)+' 创建文件并验证。', message_id='second'))
        assert answer.agent_status == 'completed', (answer.reason, len(chat.calls), answer.final_text)
        assert tool.calls == 1 and len(chat.calls) == 2
        rules = '\n'.join(chat.calls[-1]['system_extra_blocks'])
        assert '关键词' in rules
        assert '正式操作或多步任务只提交完整目标' not in rules
        assert '宿主提供的当前Angelos能力' in chat.calls[-1]['user_prompt']
        assert 'demo_file' in chat.calls[-1]['user_prompt']
        assert 'demo_file' not in rules
    finally:
        close(store, coordinator)


def test_duplicate_message_and_status_read_do_not_repeat_execution_or_expression(remote, tmp_path):
    service, tool, runtime, gateway, client, base = remote
    with semantic_sdk() as (model, checker, expression_calls):
        profile = ExpressionProfile(model, checker=checker, format_version=2, style_prompt='SYNTHETIC_STYLE')
        bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [submit()], expression=profile)
        try:
            message = turn('创建 '+str(tool.path)+' 并验证。')
            answer = bridge(message)
            assert answer.agent_status == 'completed', (answer.reason, len(chat.calls), answer.final_text)
            row = bridge.delegation._rows(SCOPE)[0]
            before = tool.calls, len(runtime.calls), len(expression_calls)
            chat.replies.append(submit())
            bridge(message)
            chat.replies.append({'kind':'tool','tool_call':{'name':'agent_task_result','arguments':{'task_id':row['task_id']}}})
            bridge(turn('把结果给我看看', message_id='result-read'))
            assert before == (tool.calls, len(runtime.calls), len(expression_calls))
        finally:
            close(store, coordinator)


def test_foreign_requester_cannot_submit_or_claim_aleph_from_workbench(remote, tmp_path):
    service, tool, runtime, gateway, client, base = remote
    bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [submit()], master='someone-else')
    try:
        reply = bridge(turn('创建 '+str(tool.path)))
        assert reply.agent_status in {'blocked','failed'}
        assert not runtime.calls and tool.calls == 0
    finally:
        close(store, coordinator)


def test_async_delegation_uses_host_worker_only_for_wait_and_publishes_once(remote, tmp_path):
    service, tool, runtime, gateway, client, base = remote
    bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [submit()], asynchronous=True)
    try:
        answer = bridge(turn('创建 '+str(tool.path)+' 并验证。'))
        assert answer.background_task_id
        deadline = time.monotonic()+5
        while True:
            state = coordinator.get(answer.background_task_id)
            if not state['worker_active'] and state['status'] not in {'queued','running'}: break
            assert time.monotonic()<deadline
            time.sleep(.01)
        assert tool.calls == 1 and sent and len(bridge.delegation_publications) == 1
        assert state['status'] == 'completed', state
        trace = bridge.trace_store.snapshot()[0]
        assert trace['status'] == 'completed' and trace['sent']
        assert trace['performance']['delegatedTaskIds']
        phases = {r['phase']: r for r in trace['performance']['phases']}
        assert phases['delegation_wait']['measured'] == 1
        assert phases['delivery']['measured'] == 1
    finally:
        close(store, coordinator)


def test_transport_unavailable_does_not_run_local_executor(remote, tmp_path):
    service, tool, runtime, gateway, client, base = remote
    bridge, chat, sent, store, coordinator = bridge_for(tmp_path, remote, [submit()])
    def unavailable(*args,**kwargs):
        raise ConnectionError('synthetic disconnected')
    bridge.delegation.client.session.post = unavailable
    try:
        result = bridge(turn('创建 '+str(tool.path)))
        assert result.agent_status == 'blocked'
        assert tool.calls == 0 and not runtime.calls
        row = bridge.delegation._rows(SCOPE)[0]
        assert row['status'] == 'submitting' and row['task_id']
    finally:
        close(store, coordinator)


def test_full_long_fact_tail_reaches_sdk_checker_and_reply_via_host_reference(remote, tmp_path):
    def with_references(payload):
        refs=set(payload['host_reference_fact_ids'])
        normal=[r for r in payload['fact_blocks'] if r.get('required',True) and r['id'] not in refs]
        return {'schema_version':2,'binding_hash':payload['binding_hash'],'paragraphs':[
            {'text':'本次的结果连同原条件如下。\n'+'\n'.join(r['text'] for r in normal),
             'fact_ids':[r['id'] for r in normal]}, *({'fact':key} for key in refs)]}
    service, tool, runtime, gateway, client, base = remote
    long_text='原始资料内容。'*1400+'TAIL_LIMIT_DO_NOT_DROP'
    execute=tool.execute
    def full_result(**kwargs):
        result=execute(**kwargs)
        result['state_updates']['result_facts']=[{'label':'完整原文','value':long_text,
            'condition':'只适用于本次隔离样本','reference':'https://example.invalid/original'}]
        return result
    tool.execute=full_result
    with semantic_sdk(build=with_references) as (model,checker,calls):
        profile=ExpressionProfile(model,checker=checker,format_version=2,style_prompt='SYNTHETIC_STYLE',max_input_chars=65536)
        bridge,chat,sent,store,coordinator=bridge_for(tmp_path,remote,[submit()],expression=profile)
        try:
            reply=bridge(turn('创建 '+str(tool.path)+' 并验证，完整保留返回的原文及限制。'))
            assert reply.agent_status=='completed',reply.reason
            assert long_text in reply.final_text
            assert len(calls)==2 and long_text in calls[1][2]['candidate_text']
            assert calls[0][2]['host_reference_fact_ids']
            assert '只适用于本次隔离样本' in reply.final_text
            assert 'https://example.invalid/original' in reply.final_text
            row=bridge.delegation._rows(SCOPE)[0]
            row.pop('expression_text');row.pop('expression_result_hash')
            bridge.delegation._save(SCOPE,row)
            chat.replies.append({'kind':'tool','tool_call':{'name':'agent_task_result','arguments':{'task_id':row['task_id']}}})
            reread=bridge(turn('再给我完整结果',message_id='read-without-expression-cache'))
            assert long_text in reread.final_text and '只适用于本次隔离样本' in reread.final_text
            assert len(calls)==2  # 只读结果不再次调用表达或执行模型。
        finally:close(store,coordinator)


@pytest.mark.parametrize('revoke',[False,True])
def test_lease_keeps_running_while_expression_waits_and_revocation_suppresses_delivery(remote,tmp_path,revoke):
    service,tool,runtime,gateway,client,base=remote
    gateway.lease_seconds=.4
    controls={'revoke':revoke}
    def build(payload):
        time.sleep(.8)  # 超过服务端单租约；独立续租必须覆盖表达等待。
        if controls.get('revoke'):controls['commands'].set_agent_enabled(False)
        return full_body(payload)
    with semantic_sdk(build=build) as (model,checker,calls):
        profile=ExpressionProfile(model,checker=checker,format_version=2,style_prompt='SYNTHETIC_STYLE')
        bridge,chat,sent,store,coordinator=bridge_for(tmp_path,remote,[submit()],expression=profile)
        controls['commands']=bridge.command_state
        try:
            reply=bridge(turn('创建 '+str(tool.path)+' 并验证。'))
            assert reply.agent_status=='completed'
            assert tool.calls==1
            if revoke:
                assert not sent and reply.delivery_status=='suppressed' and len(calls)==1
            else:
                assert reply.sent and len(calls)==2
        finally:close(store,coordinator)


def test_natural_status_question_still_reaches_main_model_and_current_remote_view(remote,tmp_path):
    service,tool,runtime,gateway,client,base=remote
    bridge,chat,sent,store,coordinator=bridge_for(tmp_path,remote,[submit()])
    try:
        bridge(turn('创建 '+str(tool.path)+' 并验证。'))
        task_id=bridge.delegation._rows(SCOPE)[0]['task_id']
        chat.replies.append({'kind':'tool','tool_call':{'name':'agent_task_status','arguments':{'task_id':task_id}}})
        reply=bridge(turn('进度怎么样',message_id='natural-status'))
        assert len(chat.calls)==2 and reply.reason=='delegation_status'
        assert task_id in chat.calls[-1]['user_prompt']
        assert tool.calls==1
    finally:close(store,coordinator)


def test_supplement_preserves_originals_and_result_read_labels_old_version(remote,tmp_path,monkeypatch):
    from shinku.agent_runtime import AgentSupplement, TaskHandle
    from shinku.agent_runtime.materials import MaterialSource
    import shinku.qq.task_materials as materials
    service,tool,runtime,gateway,client,base=remote
    original_sources=materials.sources_for_turn
    monkeypatch.setattr(materials,'sources_for_turn',lambda *args:([],[
        {'source_id':'related_context','reason':'material_context_unavailable','required':True}]))
    bridge,chat,sent,store,coordinator=bridge_for(tmp_path,remote,[submit()])
    try:
        first=bridge(turn('创建 '+str(tool.path)+' 并验证。'))
        assert first.agent_status=='waiting_user' and tool.calls==0
        row=bridge.delegation._rows(SCOPE)[0];task_id=row['task_id']
        old=client.port(SCOPE).result(task_id)
        monkeypatch.setattr(materials,'sources_for_turn',lambda *args:([
            MaterialSource('supplement-attachment','document','NEW_ORIGINAL_TAIL',required=False)],[]))
        chat.replies.append({'kind':'tool','tool_call':{'name':'agent_task_supplement','arguments':{'task_id':task_id}}})
        added=bridge(turn('补充当前文件的具体内容',message_id='supplement'))
        assert added.reason=='delegation_supplement'
        task=service.task_store.get(task_id,requester=SCOPE.requester,session=SCOPE.session)
        assert task['requests'][0]['text'].startswith('创建 ')
        assert task['requests'][-1]['text']=='补充当前文件的具体内容'
        assert any(r['source_id']=='supplement-attachment' for r in task['material_package']['items'])
        assert task['execution_result']['result_hash']==old['result_hash']
        chat.replies.append({'kind':'tool','tool_call':{'name':'agent_task_result','arguments':{'task_id':task_id}}})
        reply=bridge(turn('当前结果是什么',message_id='result-after-supplement'))
        assert '上一执行版本' in reply.final_text and tool.calls==0
        # 继续沿用原端口版本检查；取消与恢复也不创建新任务。
        view=client.port(SCOPE).status(task_id)
        with pytest.raises(AgentBoundaryError,match='version_conflict'):
            client.port(SCOPE).cancel(TaskHandle(task_id,view['attempt'],view['version']-1))
        cancelled=client.port(SCOPE).cancel(TaskHandle(task_id,view['attempt'],view['version']))
        assert cancelled['stopped'] and cancelled['files_rolled_back'] is False
        view=client.port(SCOPE).status(task_id)
        resumed=client.port(SCOPE).resume(TaskHandle(task_id,view['attempt'],view['version']),request_key='resume-existing')
        assert resumed.task_id==task_id and resumed.attempt==view['attempt']+1
    finally:close(store,coordinator)



def test_existing_positional_expression_binding_keeps_its_original_meaning():
    from shinku.qq.host_bindings import QQAgentHostBindings
    from shinku.agent_runtime import ExecutionProfile
    chat, executor = Chat([]), Chat([])
    profile = ExpressionProfile(chat, checker=executor, format_version=2)
    bound = QQAgentHostBindings(chat, ExecutionProfile(executor), profile)
    assert bound.expression_profile is profile and bound.delegation is None



def test_original_text_preserves_indentation_and_whitespace_segments():
    from shinku.qq.delegation import original_message_text
    from shinku.qq.message import IncomingMessage
    message=IncomingMessage.from_event({'message_type':'private','user_id':'fixture','message_id':'text',
        'message':[{'type':'text','data':{'text':v}} for v in ('  def run():\n','    ','return 1\n')]})
    assert original_message_text(message)=='  def run():\n    return 1\n'



def test_replayed_resume_message_cannot_start_another_attempt(remote,tmp_path,monkeypatch):
    import shinku.qq.task_materials as materials
    service,tool,runtime,gateway,client,base=remote
    monkeypatch.setattr(materials,'sources_for_turn',lambda *args:([],[
        {'source_id':'related_context','reason':'material_context_unavailable','required':True}]))
    bridge,chat,sent,store,coordinator=bridge_for(tmp_path,remote,[submit()])
    try:
        assert bridge(turn('创建 '+str(tool.path)+' 并验证。')).agent_status=='waiting_user'
        task_id=bridge.delegation._rows(SCOPE)[0]['task_id']
        decision={'kind':'tool','tool_call':{'name':'agent_task_resume','arguments':{'task_id':task_id}}}
        message=turn('继续这个任务',message_id='same-resume-message')
        chat.replies.append(decision);bridge(message)
        assert client.port(SCOPE).status(task_id)['attempt']==2
        chat.replies.append(decision);repeated=bridge(message)
        assert repeated.reason=='delegation_duplicate'
        assert client.port(SCOPE).status(task_id)['attempt']==2 and tool.calls==0
    finally:close(store,coordinator)
