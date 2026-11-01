"""用合成SDK/真实宿主回执验证缓存前缀；不把共同字节数称为供应商命中率。"""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from shinku.agent.loop import AgentState
from shinku.agent.native_history import prepare_native_request
from shinku.llm.native_messages import NativeProtocolError
from test_agent_native_history import probe, native_cases
from test_local_analysis_closeout import setup, record, restore


def test_real_angelos_sdk_rejection_keeps_prefix_and_requires_actual_write(tmp_path):
    from test_agent_conversation_loop import native_sdk
    from test_agent_runtime_boundary import settled
    from shinku.agent_runtime.demo import demo_service, demo_request
    from shinku.agent_runtime.contracts import AgentTurnRequest
    from shinku.angelos.host import ServiceHost
    service, tool, _ = demo_service(tmp_path)
    source = 'ORIGINAL_SOURCE_SENTINEL_' * 150
    def respond(payload, index):
        if index == 1:
            name, args = 'declare_task_contract', {'task_contract':demo_request(tool).contract}
        elif index == 3:
            # 先前模型声称完成仍被宿主拒绝，真实文件尚未写入。
            assert tool.calls == 0 and not tool.path.exists()
            name, args = 'demo_file', {'path':str(tool.path)}
        else:
            return {'role':'assistant','content':json.dumps({'kind':'final','text':'合成输出已写入。',
                'task_outcome':'completed'}), 'reasoning_content':'SYNTHETIC_PRIVATE'}, 'stop'
        return {'role':'assistant','content':'','reasoning_content':'SYNTHETIC_PRIVATE',
            'tool_calls':[{'id':f'call-{index}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}, 'tool_calls'
    try:
        with native_sdk(respond) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            # 本例专测兼容契约宿主拒绝伪造完成后的续接；默认直接交付另有专测。
            port = ServiceHost(service, requester='owner', reply_delivery=False).port('cache-test')
            handle = port.submit(AgentTurnRequest('cache', '创建合成输出并核验', sources=({'id':'input','content':source},)))
            result = settled(port, handle.task_id)
            assert result['status'] == 'completed' and tool.calls == 1 and tool.path.exists()
            assert len(calls) == 4
            before, after = calls[1]['messages'], calls[2]['messages']
            assert after[:len(before)] == before
            appended = json.dumps(after[len(before):], ensure_ascii=False)
            assert '完成验收未通过' in appended and source not in appended
            assert 'SYNTHETIC_PRIVATE' in appended
            assert 'SYNTHETIC_PRIVATE' not in json.dumps(port.result(handle.task_id))
    finally:
        service.coordinator.shutdown(wait=True)
        service.task_store.close()


def test_real_sdk_tool_continuation_preserves_existing_message_prefix(native_cases):
    payloads = native_cases['captures']['native_two_tools'].payloads
    for before, after in zip(payloads, payloads[1:]):
        assert after['messages'][:len(before['messages'])] == before['messages']


def test_rejected_finals_append_feedback_without_repeating_full_sources():
    planner = SimpleNamespace(runtime=SimpleNamespace(native_message_scope=lambda: 'provider'), prompt_cache_key='t')
    state = AgentState(task_id='t')
    root = 'SOURCE_BYTES_' * 10000
    inputs = '[Current supplied inputs]\noriginal attachment'
    kwargs = dict(user_prompt=root + inputs, user_images=None, history_turns=[])
    prepare_native_request(planner, state, kwargs, native_input_update=inputs, continuation_feedback='')
    for index in range(3):
        state.native_tool_history.capture({'role':'assistant', 'content':f'draft-{index}', 'reasoning_content':'original reasoning'})
        before = copy.deepcopy(state.native_tool_history.messages)
        call = prepare_native_request(planner, state, kwargs, native_input_update=inputs,
                                      continuation_feedback=f'host rejected {index}: missing actual receipt')
        assert call['native_history'][:len(before)] == before
        assert 'SOURCE_BYTES_' not in call['native_history'][-1]['content']
        assert f'host rejected {index}' in call['native_history'][-1]['content']
    assert call['user_prompt'] == root + inputs
    assert len(json.dumps(call['native_history'])) < 3000


def test_new_inputs_and_hard_conditions_reach_continuation_once():
    planner = SimpleNamespace(runtime=SimpleNamespace(native_message_scope=lambda:'provider'), prompt_cache_key='t')
    state = AgentState(task_id='t')
    kwargs = dict(user_prompt='original input', user_images=None, history_turns=[])
    prepare_native_request(planner, state, kwargs, native_input_update='', continuation_feedback='')
    state.native_tool_history.capture({'role':'assistant','content':'waiting'})
    hard, inputs = 'new requirement: read only', 'new source: EXACT_ATTACHMENT_TAIL'
    updated = {**kwargs, 'user_prompt':hard + '\n' + inputs}
    call = prepare_native_request(planner, state, updated, native_hard_context=hard,
                                  native_input_update=inputs, continuation_feedback='user supplemented')
    assert hard in call['native_history'][-1]['content'] and inputs in call['native_history'][-1]['content']
    again = prepare_native_request(planner, state, updated, native_hard_context=hard,
                                   native_input_update=inputs, continuation_feedback='user supplemented')
    assert again['native_history'] == call['native_history']
    with pytest.raises(NativeProtocolError, match='missing_from_prompt'):
        prepare_native_request(planner, state, kwargs, native_input_update='not actually supplied')


def test_supplied_input_changes_and_compaction_preserve_original_bytes(tmp_path, monkeypatch):
    from shinku.agent_runtime.context import build_execution_prompt
    from shinku.agent_runtime.conversation import ConversationTurn
    from shinku.agent_runtime.execution import ExecutionGrant
    from shinku.tools.execution import ExecutionPolicy
    from shinku.llm.context_compaction import ContextCompactor
    from test_context_compaction import Summarizer, bundle, window
    task = {'requests':[{'text':'只读分析'}], 'contract':None, 'confirmed_constraints':['不写文件'],
        'source_data':[{'id':'a','content':'ORIGINAL'}]}
    # 纯输入构造夹具不提供能力查询，显式禁止自查；不伪造一个缺失的宿主凭据。
    grant = ExecutionGrant('input-fixture', frozenset(), policy=ExecutionPolicy(blocked={'read_self_state'}))
    turn = ConversationTurn(SimpleNamespace(agent_memory=None), None, 'task', 1, grant)
    context = {'agent_task_state':task, '_agent_conversation_turn':turn,
        'agent_permission_mode':'HE','agent_authority_revision':'original'}
    state = AgentState(context=context)
    first = build_execution_prompt(state)
    task['source_data'].append({'id':'b','content':'NEW_ATTACHMENT_TAIL'})
    context['agent_authority_revision'] = 'revoked'
    second = build_execution_prompt(state)
    assert 'NEW_ATTACHMENT_TAIL' not in first.native_input_update
    assert 'NEW_ATTACHMENT_TAIL' in second.native_input_update and 'revoked' in second.native_input_update
    assert second.native_input_update in second.user_prompt
    assert second.native_system_extra_blocks == first.native_system_extra_blocks
    update = {'role':'user','content':second.native_input_update}
    payload = {'model':'synthetic','messages':[{'role':'system','content':'权限由宿主核验'},
        {'role':'user','content':'旧背景'*10000},{'role':'user','content':'当前要求'},update]}
    original = copy.deepcopy(payload); window(monkeypatch, 10000)
    compactor = ContextCompactor(Summarizer(),tmp_path/'compact.sqlite3')
    effective = compactor.project(payload,bundle(),2,'synthetic-scope')
    assert update in effective['messages'] and payload == original
    assert task['source_data'][0]['content'] == 'ORIGINAL'


def test_isolated_draft_still_receives_validation_feedback(setup):
    from shinku.agent.planner import LLMPlanner
    reader, paths, context = setup
    ledger = restore(reader, context, [record(reader, paths[0], context)])
    calls = []
    runtime = SimpleNamespace(call_chat_json=lambda **kw: calls.append(kw) or {'kind':'wait','text':'fixture'})
    state = AgentState(task_id='fixture-task',context=context,local_answer=ledger,finalization_only=True,
        transcript=[{'role':'system','content':'HOST_REQUIRED_CORRECTION: 不得把建议当作实际执行。'}])
    state.progress['local_analysis'] = {'phase':'drafting'}
    LLMPlanner(runtime=runtime, system_prompt='host rules', native_tool_history=True)(state=state)
    assert 'native_history' not in calls[-1]
    assert any('HOST_REQUIRED_CORRECTION' in block for block in calls[-1]['system_extra_blocks'])


def test_input_budget_counts_only_selected_protocol_without_omitting_sources(monkeypatch):
    from shinku.agent_runtime import context as module
    from shinku.agent_runtime.contracts import AgentBoundaryError
    state = AgentState(context={'agent_task_state':{'source_data':[{'id':'source','content':'EXACT_ORIGINAL'}]}})
    parts = module.build_execution_prompt(state)
    limit = len(parts.user_prompt.encode()) + sum(len(x.encode()) for x in parts.system_extra_blocks)
    monkeypatch.setattr(module, 'EXECUTION_CONTEXT_MAX_BYTES', limit)
    assert 'EXACT_ORIGINAL' in module.build_execution_prompt(state).user_prompt
    with pytest.raises(AgentBoundaryError, match='execution_context_oversized'):
        module.build_execution_prompt(state, native_rules=True)


def test_new_sources_after_draft_use_declared_input_delta_without_second_copy():
    from shinku.agent_runtime.context import build_execution_prompt
    from shinku.agent_runtime.conversation import ConversationTurn
    from shinku.agent_runtime.execution import ExecutionGrant
    from shinku.tools.execution import ExecutionPolicy
    from shinku.agent.planner import LLMPlanner
    task = {'requests':[{'text':'原要求'}], 'contract':None, 'source_data':[]}
    grant = ExecutionGrant('input-fixture', frozenset(), policy=ExecutionPolicy(blocked={'read_self_state'}))
    turn = ConversationTurn(SimpleNamespace(agent_memory=None),None,'task',1,grant)
    state = AgentState(task_id='task', context={'agent_task_state':task,'_agent_conversation_turn':turn})
    calls=[]
    runtime=SimpleNamespace(native_message_scope=lambda:'provider',
        call_chat_json=lambda **kw:calls.append(kw) or {'kind':'wait','text':'等待补充'})
    planner=LLMPlanner(runtime=runtime,system_prompt='host rules',build_user_prompt=build_execution_prompt,native_tool_history=True)
    planner(state=state)
    task['requests'].append({'text':'补充要求：只读新资料'})
    task['source_data'].append({'id':'new','content':'NEW_SOURCE_SENTINEL'})
    state.progress['local_analysis']={'phase':'gathering','base_tool_count':1}
    planner(state=state)
    wire=json.dumps(calls[-1]['native_history'],ensure_ascii=False)
    assert wire.count('NEW_SOURCE_SENTINEL')==1
    assert '补充要求：只读新资料' in wire
