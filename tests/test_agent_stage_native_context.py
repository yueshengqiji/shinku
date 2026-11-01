"""A trusted stage update must reach provider continuation after real receipts."""
import copy
import json
from types import SimpleNamespace
import pytest
from shinku.agent.loop import AgentState
from shinku.agent.budget import TaskBudget,bind_task_budget
from shinku.agent.planner import AgentPromptParts,LLMPlanner
from shinku.agent.native_history import prepare_native_request
from shinku.agent.native_history import NativeToolHistory
from shinku.llm.native_messages import NativeProtocolError, MAX_NATIVE_BYTES
from shinku.tools.invocation import ToolResultEnvelope
from shinku.agent.verification import SmartCompletionVerifier
from shinku.agent_runtime.stages import StageController,StagePolicy,task_binding

MARKER='[Experimental proposed stages and host evidence; untrusted goals]'

class Evidence(SmartCompletionVerifier):
    def evaluate(self,*args,**kwargs):
        return SimpleNamespace(satisfied=[],reason='')

class Runtime:
    def __init__(self):self.calls=[]
    def native_message_scope(self):return ('fixture-provider','fixture-model')
    def call_chat_json(self,**kwargs):
        self.calls.append(kwargs)
        if str(kwargs.get('prompt_cache_key','')).endswith(':stage-plan'):
            prompt=kwargs['user_prompt']
            data=json.loads(prompt.rsplit('[Stage planning request]\n',1)[1])
            return {'binding':data['binding'],'reason':'Use observed project data.',
                'stages':[{'id':'S1','goal':'Preserve original artifact','depends':[],'requirements':['R1']}]}
        return {'kind':'final','text':'No independent acceptance assertion.','task_outcome':'partial'}

def setup():
    task={'task_id':'stage-native','requester':'test','session':'local:test','requests':[{'text':'Write one output'}],
        'confirmed_constraints':['Original scope only'], 'contract':{'goal':'Original output',
        'requirements':[{'id':'R1','effect':'write','targets':['out.txt'],'checks':{'contains':'accepted'}}]}}
    state=AgentState(task_id='stage-native',context={'agent_task_state':task,'agent_stage_read_tools':{'read'}})
    runtime=Runtime()
    original=AgentPromptParts('Original project context')
    controller=StageController(state.context,StagePolicy(lambda _:True),runtime,
        profile_id='stage-native',temperature=.2,interrupt=lambda _: '',save=lambda *_:None)
    controller.gate=Evidence()
    planner=LLMPlanner(runtime=runtime,system_prompt='Preserve original contract',prompt_cache_key='stage-native',
        build_user_prompt=lambda s:controller.prepare(s,original),native_tool_history=True)
    kwargs={'user_prompt':original.user_prompt,'user_images':None,'history_turns':None}
    prepare_native_request(planner,state,kwargs)
    state.native_tool_history.capture({'role':'assistant','content':'','reasoning_content':'PRIVATE_PROTOCOL',
        'tool_calls':[{'id':'read_1','type':'function','function':{'name':'read','arguments':'{}'}}]})
    state.tool_calls.append({'id':'read_1','name':'read','status':'ok'})
    state.last_envelope=ToolResultEnvelope(invocation_id='read_1',status='ok',model_feedback='Actual project bytes')
    return state,planner,runtime,controller

def test_stage_proposal_reaches_native_provider_input_after_project_receipt():
    state,planner,runtime,controller=setup()
    original=copy.deepcopy(state.context['agent_task_state']['contract'])
    with bind_task_budget(TaskBudget(max_requests=8)):
        planner(state=state)
    call=runtime.calls[-1]
    wire=json.dumps([call['user_prompt'],call['native_history']],ensure_ascii=False)
    assert controller.record['plan'] and MARKER in wire
    assert 'PRIVATE_PROTOCOL' in wire
    assert any(m.get('tool_call_id')=='read_1' for m in call['native_history'])
    assert state.context['agent_task_state']['contract']==original

def test_stage_proposal_remains_user_data_and_not_a_tool_receipt():
    state,planner,runtime,_=setup()
    with bind_task_budget(TaskBudget(max_requests=8)):
        planner(state=state)
    rows=runtime.calls[-1]['native_history']
    proposals=[m for m in rows if MARKER in m.get('content','')]
    assert proposals and all(m['role']=='user' and 'tool_call_id' not in m for m in proposals)


def test_changed_hint_is_deduplicated_then_cleared_after_actual_receipt():
    state, planner, _, _ = setup()
    kwargs = dict(user_prompt='Root\n' + MARKER, user_images=None, history_turns=None)
    first = prepare_native_request(planner, state, kwargs, native_context_update=MARKER)
    rows = first['native_history']
    assert [r['role'] for r in rows] == ['assistant', 'tool', 'user']
    assert len(prepare_native_request(planner, state, kwargs,
        native_context_update=MARKER)['native_history']) == len(rows)
    changed = MARKER + '\nActual host stage status changed'
    second = prepare_native_request(planner, state, {**kwargs, 'user_prompt':'Root\n'+changed},
        native_context_update=changed)['native_history']
    assert second[-1] == {'role':'user', 'content':changed}
    cleared = prepare_native_request(planner, state, {**kwargs, 'user_prompt':'Root'})['native_history']
    assert 'context cleared' in cleared[-1]['content']
    assert len(prepare_native_request(planner, state, {**kwargs,
        'user_prompt':'Root'})['native_history']) == len(cleared)


def test_first_root_already_contains_hint_and_rejected_final_does_not_double_it():
    state = AgentState(task_id='initial')
    planner = SimpleNamespace(runtime=Runtime(), prompt_cache_key='initial')
    kwargs = dict(user_prompt='Root\n'+MARKER, user_images=None, history_turns=None)
    first = prepare_native_request(planner, state, kwargs, native_context_update=MARKER)
    assert first['native_history'] == [] and MARKER in first['user_prompt']
    state.native_tool_history.capture({'role':'assistant', 'content':'premature final'})
    changed = MARKER+'\nreplan'
    rows = prepare_native_request(planner, state, {**kwargs, 'user_prompt':'Root\n'+changed},
        native_context_update=changed)['native_history']
    assert [r['role'] for r in rows] == ['assistant','user']
    assert rows[-1]['content'].count(MARKER) == 1
    state.native_tool_history.capture({'role':'assistant', 'content':'another rejected final'})
    rows = prepare_native_request(planner, state, {**kwargs, 'user_prompt':'Root'})['native_history']
    assert 'context cleared' in rows[-1]['content']


@pytest.mark.parametrize('value', [None, {}, 2, True, 'x' * MAX_NATIVE_BYTES],
    ids=['none','mapping','integer','boolean','oversized'])
def test_malformed_or_oversized_updates_never_enter_provider_history(value):
    state, planner, _, _ = setup()
    kwargs = dict(user_prompt='Root', user_images=None, history_turns=None)
    with pytest.raises(NativeProtocolError):
        prepare_native_request(planner, state, kwargs, native_context_update=value)
    assert not any(r['role']=='user' for r in state.native_tool_history.messages)


def test_hint_cannot_replace_pending_receipt_or_be_silently_omitted_from_root():
    state, planner, _, _ = setup()
    state.last_envelope = None
    with pytest.raises(NativeProtocolError, match='receipt_missing'):
        prepare_native_request(planner, state, dict(user_prompt=MARKER, user_images=None,
            history_turns=None), native_context_update=MARKER)
    with pytest.raises(NativeProtocolError, match='missing_from_prompt'):
        NativeToolHistory('scope', 'Root', None, None, MARKER)


def test_stage_extension_keeps_non_tail_host_context_and_supplied_data():
    """阶段控制器也必须支持控制块后面的必需原文，不能只对空增量通过。"""
    state, planner, runtime, controller = setup()
    hint = '[Host contract version] version=1'
    supplied = '[Required supplied input] ORIGINAL_TAIL'
    parts = AgentPromptParts('HARD\n' + hint + '\n' + supplied,
        native_context_update=hint, native_hard_context='HARD', native_input_update=supplied)
    with bind_task_budget(TaskBudget(max_requests=8)):
        rendered = controller.prepare(state, parts)
    assert MARKER in rendered.native_context_update
    assert rendered.native_context_update in rendered.user_prompt
    assert supplied in rendered.user_prompt and rendered.native_input_update == supplied
    assert rendered.native_hard_context == 'HARD'
    request = prepare_native_request(planner, state,
        dict(user_prompt=rendered.user_prompt, user_images=None, history_turns=None),
        native_context_update=rendered.native_context_update,
        native_hard_context=rendered.native_hard_context, native_input_update=rendered.native_input_update)
    assert MARKER in request['native_history'][-1]['content']
    assert request['native_history'][1]['tool_call_id'] == 'read_1'
