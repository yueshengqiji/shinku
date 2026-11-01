"""固定效果kind、长资料原稿丢失和上下文越界反例，经过真实文件与原完成门。"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from shinku.agent.loop import AgentDecision, AgentLoop
from shinku.agent.planner import LLMPlanner
from shinku.agent.answer_protocol import normalize_answer_effect
from shinku.agent.task_state import new_task, AgentTaskStore
from shinku.agent.verification import SmartCompletionVerifier
from shinku.agent_runtime.context import execution_context, build_execution_prompt
from shinku.agent_runtime.results import export_task_result
from shinku.tools.workspace import build_workspace_tools
from shinku.tools.invocation import ToolInvocation


ANSWER = '源码定义当前配置。建议先确认独立配置的使用范围；只有要求共同取值时才考虑统一。未运行程序，运行影响尚未核实。'


@pytest.fixture
def analysis(tmp_path):
    path = tmp_path / 'input.txt';path.write_text('当前配置的数值为10，未给出单位；两份配置独立。')
    task = new_task(task_id='protocol', requester='fixture', session='local:fixture', message_id='input',
        request='读取 input.txt，分析配置及维护建议，不修改文件。', contract={
            'goal': '分析配置', 'deliverables': ['配置分析与建议'], 'scope': ['只读'], 'requirements': [
                {'id': 'read', 'description': '读取输入', 'effect': 'read', 'targets': [str(path)], 'checks': {'unchanged': True}},
                {'id': 'answer', 'description': '分析', 'effect': 'local_answer', 'targets': [], 'checks': {'min_sources': 1, 'required_fields': ['配置现状', '维护建议']}}]})
    task['execution_attempt'] = 1
    ctx = execution_context(task, SimpleNamespace(tool_context={}, permission_mode='read', authorization_ref='fixture-grant'), attempt=1)
    allowed = [True]
    tools = {t.tool_type: t for t in build_workspace_tools(tmp_path, authorize=lambda context, write: allowed[0] and not write)}
    gate = SmartCompletionVerifier(result_handlers=tools, tool_capabilities={
        'workspace_read_lines': {'effects': ['read'], 'target_arguments': ['path']}})
    return task, ctx, tools, gate, path, allowed


class Runtime:
    def __init__(self, response, *, failure=None):
        self.response, self.failure, self.calls = response, failure, []
    def chat_supports_native_tools(self):
        return True
    def call_chat_json(self, **kw):
        self.calls.append(kw)
        if len(self.calls) == 1:
            return {'tool_call': {'name': 'workspace_read_lines', 'arguments': {'path': 'input.txt'}}}
        if self.failure:
            from shinku.llm.response_protocol import note_incomplete_response
            note_incomplete_response(self.failure)
        return deepcopy(self.response) if len(self.calls) == 2 else {}


def run(analysis, response, failure=None):
    task, ctx, tools, gate, *_ = analysis
    runtime = Runtime(response, failure=failure)
    loop = AgentLoop(planner=LLMPlanner(runtime=runtime, system_prompt='fixture', build_user_prompt=build_execution_prompt),
                     handlers=tools, completion_gate=gate, sliced=True, max_rounds=8)
    result = loop.run(task_id=task['task_id'], context=ctx)
    return result, runtime, loop


@pytest.mark.parametrize('effect', ['local_answer', 'research_answer', 'history_answer', 'capability_answer'])
@pytest.mark.parametrize('outcome', ['completed', 'partial', 'failed', 'blocked'])
def test_effect_alias_is_text_only_and_never_upgrades_outcome(effect, outcome):
    raw = {'kind': effect, 'text': ANSWER, 'task_outcome': outcome};original = deepcopy(raw)
    decision = AgentDecision.from_value(raw)
    assert decision.kind == 'final' and decision.text == ANSWER and decision.outcome == outcome
    assert raw == original


@pytest.mark.parametrize('extra', [{'tool_call': {'name': 'write'}}, {'tool_calls': [{'name': 'send'}]},
    {'invocation': {'name': 'delete'}}, {'effect': 'write'}, {'authorization': 'full'}, {'text': {'not': 'text'}}])
def test_effect_alias_does_not_hide_tools_or_unknown_shapes(extra):
    assert AgentDecision.from_value({'kind': 'local_answer', 'text': ANSWER, **extra}) is None


def test_local_answer_alias_reaches_real_reader_and_original_gate(analysis):
    result, runtime, loop = run(analysis, {'kind': 'local_answer', 'text': ANSWER, 'task_outcome': 'completed'})
    assert result.status == 'completed' and result.completion['accepted']
    assert result.completion['layers']['answer_evidence'] == 'unverified'
    assert len(result.tool_envelopes) == 1 and len(runtime.calls) == 2  # 一次读、一次成稿，无额外质量审查
    assert result.progress['answer_drafts'][0]['text'] == ANSWER
    blocks = '\n'.join(runtime.calls[1]['system_extra_blocks'])
    assert '不是消息kind' in blocks and '"kind":"final"' in blocks


@pytest.mark.parametrize('response', [
    {'kind': 'unknown_answer', 'text': ANSWER, 'task_outcome': 'completed'},
    {'kind': 'local_answer', 'text': ANSWER, 'extra': 'unsupported'},
])
def test_invalid_draft_is_retained_and_labelled_without_research_budget_or_acceptance(analysis, response, tmp_path):
    result, runtime, _ = run(analysis, response)
    assert result.status == 'partial' and result.reason == 'local_analysis_invalid_decision'
    assert not result.completion['accepted'] and ANSWER in result.final_text and '未核实的模型原稿' in result.final_text
    assert '候选来源' not in result.final_text and len(runtime.calls) == 2
    task = analysis[0];task.update(progress=result.progress, completion=result.completion, status=result.status)
    # 新开SQLite实例仍保留完整原稿，展示与事实回执分离。
    store = AgentTaskStore(tmp_path/'draft.sqlite');store.save(task);store.close()
    store = AgentTaskStore(tmp_path/'draft.sqlite');saved=store.get(task['task_id'], requester=task['requester'], session=task['session']);store.close()
    exported = export_task_result(saved, status=result.status, reason=result.reason, report=result.final_text)
    assert saved['progress']['answer_drafts'][0]['text'] == ANSWER
    assert exported['answer_draft']['text'] == ANSWER and not exported['answer_draft']['validated']
    assert 'verified_answer' not in exported and not exported['verification']['accepted']
    saved['execution_attempt'] += 1
    assert 'answer_draft' not in export_task_result(saved, status='partial', reason=result.reason)


@pytest.mark.parametrize('reason', ['length', 'content_filter'])
def test_incomplete_output_cannot_be_normalized_into_completion(analysis, reason):
    result, runtime, _ = run(analysis, {'kind': 'local_answer', 'text': ANSWER, 'task_outcome': 'completed'}, failure=reason)
    assert result.status == 'partial' and not result.completion['accepted']
    assert result.reason == ('local_analysis_output_length' if reason == 'length' else 'local_analysis_output_filtered')
    assert ANSWER in result.final_text and len(runtime.calls) == 2


@pytest.mark.parametrize('defect', ['changed', 'revoked', 'missing_material', 'new_request'])
def test_failed_draft_never_exposes_old_body_after_source_or_scope_changes(analysis, defect):
    task, ctx, tools, gate, path, allowed = analysis
    class Mutating(Runtime):
        def call_chat_json(self, **kw):
            value = super().call_chat_json(**kw)
            if len(self.calls) == 2:
                if defect == 'changed': path.write_text('已变化')
                if defect == 'revoked': allowed[0] = False
                if defect == 'missing_material': ctx['agent_material_session'] = SimpleNamespace(information=lambda: {'required_status': 'incomplete'})
                if defect == 'new_request': task['requests'].append({'message_id': 'new', 'text': '新的任务'})
            return value
    runtime = Mutating({'kind': 'unknown_answer', 'text': ANSWER})
    loop = AgentLoop(planner=LLMPlanner(runtime=runtime, system_prompt='fixture', build_user_prompt=build_execution_prompt), handlers=tools, completion_gate=gate, sliced=True)
    result = loop.run(context=ctx)
    assert result.status == 'failed' and not result.completion['accepted'] and ANSWER not in result.final_text
    assert result.progress['answer_drafts'][0]['text'] == ANSWER


def test_kind_write_is_not_an_answer_alias_and_claimed_write_stays_unverified(analysis):
    assert normalize_answer_effect({'kind':'write','text':'已写入文件'}) is None
    result, runtime, _ = run(analysis, {'kind':'local_answer','text':'我已经修改 input.txt。','task_outcome':'completed'})
    assert result.status != 'completed' and not result.completion.get('accepted',False)
    assert analysis[4].read_text() == '当前配置的数值为10，未给出单位；两份配置独立。'


def test_forged_canonical_path_cannot_restore_read_of_another_resource(analysis):
    from shinku.agent.local_answer import LocalAnswerLedger
    task, ctx, tools, _, path, _ = analysis
    result, _, loop = run(analysis, {'kind':'local_answer','text':ANSWER,'task_outcome':'completed'})
    assert result.status == 'completed'
    calls = deepcopy(loop._checkpoint.state.tool_calls)
    receipt = calls[0]['data']['state_updates'];other = str(path.parent/'other.txt')
    receipt['source_path'] = other;receipt['completion_targets'].append(other)
    ledger = LocalAnswerLedger();ledger.configure(ctx);ledger.restore_reads(calls,tools,ctx)
    assert not ledger.sources


def test_recovery_retains_all_old_drafts_without_restoring_display_or_acceptance(analysis):
    result, _, _ = run(analysis, {'kind':'unknown_answer','text':ANSWER})
    task, ctx, tools, gate, _, allowed = analysis
    original = deepcopy(result.progress['answer_drafts'])
    task['progress'] = result.progress;task['execution_attempt'] += 1
    ctx = execution_context(task,SimpleNamespace(tool_context={},permission_mode='read',authorization_ref='fixture-grant'),attempt=2)
    # 没有新读取、没有可用原稿；恢复只保留旧资料，不把旧许可带进新轮次。
    loop = AgentLoop(planner=lambda **kw:{'kind':'invalid'},handlers=tools,completion_gate=gate,sliced=True)
    recovered=loop.run(context=ctx)
    assert recovered.progress['answer_drafts']==original
    assert not recovered.progress.get('answer_draft_presentation',{}).get('allowed')
    assert recovered.status != 'completed' and ANSWER not in recovered.final_text


@pytest.mark.parametrize('progress',[None,{'answer_drafts':None},{'answer_drafts':[None,42,'bad']}, {'answer_drafts':[], 'answer_draft_presentation':True}])
def test_corrupt_or_missing_draft_metadata_does_not_break_export_or_mint_answer(analysis,progress):
    task=deepcopy(analysis[0]);task['progress']=progress
    result=export_task_result(task,status='failed',reason='local_analysis_invalid_decision')
    assert 'answer_draft' not in result and 'verified_answer' not in result


def test_effect_alias_casing_is_normalized_without_changing_original():
    value={'kind':' LOCAL_ANSWER ','text':ANSWER,'task_outcome':'failed'}
    original=deepcopy(value)
    assert AgentDecision.from_value(value).outcome=='failed' and value==original
