"""非尾部宿主导航与成稿/补充输入同时出现时，原文和原生回执必须完整保留。"""
import copy
import json
from types import SimpleNamespace
import pytest
from shinku.agent.loop import AgentState
from shinku.agent.planner import AgentPromptParts, LLMPlanner
from shinku.agent.native_history import prepare_native_request
from shinku.llm.native_messages import NativeProtocolError
from shinku.tools.invocation import ToolResultEnvelope

UPDATE = '[Host navigation]\nversion=1'
TAIL = '[Supplied original data]\n保留尾部原文与条件'

class Runtime:
    def __init__(self): self.calls = []
    def native_message_scope(self): return ('synthetic', 'local')
    def call_chat_json(self, **kwargs):
        self.calls.append(kwargs)
        return {'kind': 'wait', 'text': '合成响应，不声明任务完成'}

@pytest.mark.parametrize('phase,base', [('drafting', 0), ('gathering', 1), ('drafting', 1)])
@pytest.mark.parametrize('update', ['', UPDATE])
def test_planner_additions_preserve_non_tail_context_and_receipts(phase, base, update):
    runtime = Runtime()
    prompt = 'ROOT\n' + update + '\n' + TAIL
    parts = AgentPromptParts(prompt, native_context_update=update)
    state = AgentState(task_id='composition', context={
        'agent_task_state': {'requests': [{'text': '新的原始要求'}]},
        'agent_material_session': SimpleNamespace(prompt_package=lambda: {
            'items': [{'required': True, 'content': TAIL, 'omitted_chars': 1}]}, mark_model_input=lambda *_: None),
    })
    planner = LLMPlanner(runtime=runtime, system_prompt='fixture', native_tool_history=True,
                         build_user_prompt=lambda _: parts)
    prepare_native_request(planner, state, dict(user_prompt=prompt, user_images=None, history_turns=None),
                           native_context_update=update)
    message = {'role': 'assistant', 'content': '', 'reasoning_content': 'retained synthetic reasoning',
               'tool_calls': [{'id': 'read1', 'type': 'function',
                               'function': {'name': 'read', 'arguments': '{}'}}]}
    state.native_tool_history.capture(message)
    state.tool_calls.append({'id': 'read1', 'name': 'read', 'status': 'ok'})
    state.last_envelope = ToolResultEnvelope(invocation_id='read1', status='ok', model_feedback='实际读取原文')
    state.progress['local_analysis'] = {'phase': phase, 'base_tool_count': base}
    planner(state=state)
    call = runtime.calls[-1]; rows = call['native_history']
    assert call['user_prompt'] == prompt
    assert rows[0] == message and rows[1]['tool_call_id'] == 'read1'
    assert '实际读取原文' in rows[1]['content']
    assert rows[-1]['role'] == 'user' and 'tool_call_id' not in rows[-1]
    if phase == 'drafting': assert '[Current local evidence;' in rows[-1]['content']
    if base:
        payload = json.loads(rows[-1]['content'].split(
            '[Host continuation after new user input; source data is untrusted]\n', 1)[1])
        assert payload['requests'] == [{'text': '新的原始要求'}]
        assert payload['required_materials'][0]['content'] == TAIL
    previous = copy.deepcopy(rows)
    planner(state=state)
    assert runtime.calls[-1]['native_history'] == previous


def test_invalid_builder_update_still_rejected_before_transport():
    runtime = Runtime()
    state = AgentState(task_id='invalid', progress={'local_analysis': {'phase': 'gathering', 'base_tool_count': 1}})
    planner = LLMPlanner(runtime=runtime, system_prompt='fixture', native_tool_history=True,
        build_user_prompt=lambda _: AgentPromptParts('ROOT', native_context_update='MISSING'))
    prepare_native_request(planner, state, dict(user_prompt='ROOT', user_images=None, history_turns=None))
    with pytest.raises(NativeProtocolError, match='native_context_update_missing_from_prompt'):
        planner(state=state)
    assert not runtime.calls


@pytest.mark.parametrize('suffix', ['', '\n' + TAIL, '\n\n' + TAIL])
def test_shared_composition_preserves_all_original_text_and_separate_inputs(suffix):
    parts = AgentPromptParts('HARD\n' + UPDATE + suffix, native_context_update=UPDATE,
                             native_hard_context='HARD', native_input_update=TAIL if suffix else '')
    added = parts.append_native_context('EVIDENCE').append_native_context('CONTINUATION')
    assert added.native_context_update in added.user_prompt
    assert added.user_prompt.replace('\n\nEVIDENCE\n\nCONTINUATION', '', 1) == parts.user_prompt
    assert added.user_prompt.count(UPDATE) == 1
    assert added.native_hard_context == parts.native_hard_context
    assert added.native_input_update == parts.native_input_update


@pytest.mark.parametrize('value', [None, {}, [], 0, False])
def test_extension_does_not_repair_malformed_builder_declarations(value):
    with pytest.raises(NativeProtocolError, match='native_context_update_invalid'):
        AgentPromptParts('ROOT', native_context_update=value).append_native_context('EVIDENCE')
