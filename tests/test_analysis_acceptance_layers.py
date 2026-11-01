"""真实临时文件贯穿分层验收、宿主循环、结果导出与只读视图，严格路径作反例。"""
from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest

from shinku.agent.analysis_acceptance import AnalysisDeliveryPolicy, ordinary_analysis
from shinku.agent.local_answer import LocalAnswerLedger
from shinku.agent.loop import AgentState, AgentLoop, AgentDecision, _call_signature
from shinku.agent.planner import LLMPlanner
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent.verification import SmartCompletionVerifier
from shinku.agent_runtime.context import execution_context
from shinku.agent_runtime.contracts import AgentScope
from shinku.agent_runtime.results import export_task_result
from shinku.agent_runtime.task_view import TaskViewReader
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope


REPORT = ('项目结构：当前入口负责组装服务，扩展负责提供可选能力。\n'
          '建议：增加装配诊断和规则命中原因，先观察再调整默认行为。\n'
          '未核实范围：这是一份静态分析，没有运行性能基准，也没有宣称完成修改或部署。')


class Reader:
    tool_type = 'fixture_source'
    agent_permission_class = 'read'
    agent_local_answer_source = True
    agent_completion_effects = ('read',)
    agent_completion_target_arguments = ('path',)

    def __init__(self, path):
        self.path, self.allowed = path, True
        self.calls = 0

    def tool_metadata(self):
        return SimpleNamespace(description='Read one isolated source.', input_schema={
            'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']})

    def normalize_call(self, call):
        return dict(call) if call.get('path') == str(self.path) else None

    def execute(self, *, call, context):
        self.calls += 1
        text = self.path.read_text()
        return {'tool_type': self.tool_type, 'followup_context': text, 'state_updates': {
            'path': str(self.path), 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'read_complete': True,
            'verified_effects': ['read'], 'completion_targets': [str(self.path)]}}

    def agent_verify_result(self, *, receipt, context):
        if not self.allowed:
            raise PermissionError('fixture denied')
        text = self.path.read_text()
        digest = hashlib.sha256(text.encode()).hexdigest()
        return {'result_valid': digest == receipt['sha256'], 'sha256': digest, 'file_text': text}


@pytest.fixture
def case(tmp_path):
    path = tmp_path / 'source.txt'
    path.write_text('当前入口负责组装服务，扩展负责提供可选能力。')
    contract = {'goal': '分析项目', 'deliverables': ['项目结构及优化建议'], 'scope': ['只读'],
        'requirements': [{'id': 'read', 'description': '完整读取输入', 'effect': 'read', 'targets': [str(path)], 'checks': {'unchanged': True}},
                         {'id': 'answer', 'description': '输出分析', 'effect': 'local_answer', 'targets': [],
                          'checks': {'required_fields': ['项目结构', '优化建议', '未核实范围'], 'min_sources': 13}}]}
    task = new_task(task_id='fixture-analysis', requester='fixture', session='local:fixture',
        request='分析这个项目，给出优化建议', message_id='input', contract=contract)
    task['execution_attempt'] = 1
    grant = SimpleNamespace(tool_context={}, permission_mode='read', authorization_ref='fixture-grant')
    context = execution_context(task, grant, attempt=1)
    reader = Reader(path)
    invocation = ToolInvocation(reader.tool_type, {'path': str(path)})
    receipt = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'read_complete': True, 'verified_effects': ['read'], 'completion_targets': [str(path)]}
    call = {'name': invocation.name, 'id': invocation.id, 'arguments': invocation.arguments,
        'executed': True, 'status': 'ok', 'data': {'state_updates': receipt},
        'call_fingerprint': hashlib.sha256(_call_signature(invocation).encode()).hexdigest(),
        'execution_binding': {'task_id': task['task_id'], 'requester': task['requester'], 'session': task['session'], 'attempt': 1}}
    ledger = LocalAnswerLedger();ledger.configure(context)
    ledger.observe(invocation, ToolResultEnvelope(invocation.id, 'ok', '', call['data'], executed=True), reader, context)
    state = AgentState(context=context, local_answer=ledger, tool_calls=[call])
    gate = SmartCompletionVerifier(result_handlers={reader.tool_type: reader},
        tool_capabilities={reader.tool_type: {'effects': ['read'], 'target_arguments': ['path']}})
    return state, gate, reader, path, task


@pytest.mark.parametrize('error', ['invalid_quote', 'invalid_fragment_selector', 'invalid_review_schema', 'unbound_numeric_fact', 'review_runtime_error'])
def test_local_review_failure_delivers_analysis_without_verifying_its_facts(case, error):
    state, gate, *_ = case
    state.local_answer.answer_check = {'valid': False, 'binding_error': error}
    verdict = gate.evaluate(state, REPORT, outcome='completed')
    assert verdict.accepted and verdict.status == 'completed'
    assert verdict.satisfied == ('read',) and not verdict.missing
    assert verdict.layers['answer_evidence'] == 'unverified' and verdict.layers['diagnostic'] == error
    assert verdict.advisory[0]['strict_satisfied'] is False


@pytest.mark.parametrize('change', ['changed', 'deleted', 'revoked', 'partial', 'failed', 'foreign_target', 'no_sources'])
def test_real_read_requirements_remain_hard(case, change):
    state, gate, reader, path, _ = case
    if change == 'changed': path.write_text('修改后的正文')
    if change == 'deleted': path.unlink()
    if change == 'revoked': reader.allowed = False
    if change == 'partial': state.tool_calls[0]['data']['state_updates']['read_complete'] = False
    if change == 'failed': state.tool_calls[0]['status'] = 'error'
    if change == 'foreign_target': state.tool_calls[0]['data']['state_updates']['completion_targets'] = [str(path.parent/'other')]
    if change == 'no_sources': state.local_answer.sources.clear()
    assert not gate.evaluate(state, REPORT, outcome='completed').accepted


@pytest.mark.parametrize('user_text', ['分析表格并统计总数', '精确分析每项金额', '分析项目并运行测试',
    '分析项目并修改配置', '审计项目所有结论', '分析项目，每个结论逐条提供证据', 'Analyze and calculate the sum', 'Analyze this repo and run tests'])
def test_explicit_strict_or_action_requests_never_enter_ordinary_mode(case, user_text):
    state, gate, _, _, task = case
    task['requests'][0]['text'] = user_text
    grant = SimpleNamespace(tool_context={}, permission_mode='read', authorization_ref='fixture-grant')
    state.context = execution_context(task, grant, attempt=1)
    assert not ordinary_analysis(state)
    assert not gate.evaluate(state, REPORT, outcome='completed').accepted


@pytest.mark.parametrize('user_text', [
    '读取 source.txt，分析项目入口和状态归属。只分析，不修改文件。',
    '读取 source.txt，整理关键约束，逐条解释影响并给出行动顺序。不写文件或发送消息。',
])
def test_explicit_readonly_analysis_is_delivered_with_real_reads(case, user_text):
    state, gate, _, _, task = case
    task['requests'][0]['text'] = user_text
    grant = SimpleNamespace(tool_context={}, permission_mode='read', authorization_ref='fixture-grant')
    state.context = execution_context(task, grant, attempt=1)
    assert ordinary_analysis(state)
    verdict = gate.evaluate(state, REPORT, outcome='completed')
    assert verdict.accepted and verdict.layers['execution_results'] == 'passed'


@pytest.mark.parametrize('change', ['model_dict', 'model_note', 'new_request', 'mixed_contract', 'scope'])
def test_model_cannot_opt_in_or_reuse_policy_after_host_input_changes(case, change):
    state, gate, _, _, task = case
    if change == 'model_dict': state.context['agent_analysis_delivery_policy'] = {'mode': 'ordinary'}
    if change == 'model_note':
        state.context.pop('agent_analysis_delivery_policy');task['notes']['mode'] = 'ordinary'
    if change == 'new_request': task['requests'].append({'message_id': 'second', 'text': '请精确审计'})
    if change == 'mixed_contract': task['contract']['requirements'].append({'id': 'write', 'effect': 'write', 'targets': ['/fixture/output']})
    if change == 'scope': state.context['agent_request_scope'] = None
    assert not ordinary_analysis(state)
    assert not gate.evaluate(state, REPORT, outcome='completed').accepted


@pytest.mark.parametrize('claim', ['我已删除文件。', '我已修改配置。', '我已发送报告。', '我已运行测试，全部通过。', '部署已完成。',
    '文件已修改。', '已创建文件。', '所有测试全部通过。', 'I have deployed the service.', '执行结果：我已修改配置。'])
def test_execution_success_claims_cannot_become_soft_warnings(case, claim):
    state, gate, *_ = case
    assert not gate.evaluate(state, REPORT+'\n'+claim, outcome='completed').accepted


def test_non_read_or_forged_effect_call_cannot_use_analysis_delivery(case):
    state, gate, *_ = case
    state.tool_calls.append({'name': 'fixture_command', 'status': 'ok', 'executed': True, 'data': {}})
    assert not gate.evaluate(state, REPORT, outcome='completed').accepted
    state.tool_calls.pop()
    state.tool_calls[0]['data']['state_updates']['verified_effects'].append('write')
    assert not gate.evaluate(state, REPORT, outcome='completed').accepted


@pytest.mark.parametrize('text,accepted', [('完成了。',False), ('项目入口按职责拆分；建议增加装配诊断。未实测性能收益。',True),
    ('先执行已审批调用，再进入规划循环；达到轮数上限时切片模式保存断点。',True),
    ('源码使用已绑定的回执，再将状态写入内部账本。建议观察该模块的错误诊断。',True)])
def test_short_useful_analysis_does_not_need_arbitrary_length_or_field_labels(case, text, accepted):
    state, gate, *_ = case
    assert gate.evaluate(state,text,outcome='completed').accepted is accepted


def test_review_is_bounded_and_keeps_original_failure_diagnostic(case):
    state, _, *_ = case
    class Runtime:
        calls = 0
        def call_chat_json(self, **kwargs):
            self.calls += 1
            return {}
    runtime = Runtime();planner = LLMPlanner(runtime=runtime, system_prompt='fixture')
    state.local_answer.answer_check = {'valid': False, 'binding_error': 'invalid_quote'}
    planner._review_local_answer(state, AgentDecision.final(REPORT))
    diagnostic = copy.deepcopy(state.local_answer.answer_check)
    planner._review_local_answer(state, AgentDecision.final(REPORT+'\n补充说明。'))
    assert runtime.calls == 0 and state.local_answer.answer_check == diagnostic


def test_loop_delivers_once_after_bad_review_and_exports_layers_after_database_reload(case, tmp_path):
    state, gate, reader, _, task = case
    state.context['agent_restored_tool_calls'] = state.tool_calls
    calls = []
    def planner(*, state):
        calls.append(state)
        state.local_answer.apply_answer_review(REPORT, {}, state.context)
        return AgentDecision.final(REPORT, outcome='completed')
    loop = AgentLoop(planner=planner, handlers={reader.tool_type: reader}, sliced=True,
        policy=ExecutionPolicy(default_risk='low'), completion_gate=gate)
    result = loop.run(context=state.context)
    assert result.status == 'completed' and result.final_text == REPORT and len(calls) == 1
    task.update(status=result.status, tool_calls=calls[-1].tool_calls, completion=result.completion, completion_attempt=1)
    store = AgentTaskStore(tmp_path/'tasks.sqlite');task=store.save(task);store.close()
    store=AgentTaskStore(tmp_path/'tasks.sqlite')
    saved=store.get(task['task_id'], requester=task['requester'], session=task['session'])
    exported=export_task_result(saved, report=result.final_text)
    view=TaskViewReader(store).get(AgentScope(task['requester'],task['session']), task['task_id'])
    assert exported['execution']['status'] == view['status'] == 'completed'
    assert not exported['remaining'] and not view['verification']['missing']
    assert exported['verification']['layers']['answer_evidence'] == 'unverified'
    assert view['verification']['layers'] == exported['verification']['layers']
    assert 'verified_answer' not in exported and exported['execution_report']['text'] == REPORT
    from shinku.conversation.models import AnswerEvidence, text_digest
    saved['result_answer_evidence']=AnswerEvidence('local','verified',True,1,1,text_digest(REPORT)).as_dict()
    assert 'verified_answer' not in export_task_result(saved, report=REPORT)
    # 旧尝试的分层回执不能替新尝试排除未满足条件。
    saved['execution_attempt']=2
    stale=export_task_result(saved, report=REPORT)
    assert stale['execution']['status'] == 'unverified' and stale['remaining']
    store.close()


def test_verified_content_retains_strict_evidence_status_when_all_checks_match(case):
    state, gate, _, _, task = case
    checks=task['contract']['requirements'][-1]['checks'];checks['min_sources']=1;checks['required_fields']=['项目结构']
    answer='项目结构：当前入口负责组装服务，扩展负责提供可选能力。\n'+REPORT
    source=next(iter(state.local_answer.sources))
    state.local_answer.apply_answer_review(answer, {'verdict':'supported','claims':[{'answer_fragment':answer.splitlines()[0],
        'evidence':[{'source_id':source,'quote':'当前入口负责组装服务，扩展负责提供可选能力。'}]}],
        'coverage':[{'field':'项目结构','supported':True,'claim_indices':[0]}],'missing_fields':[]}, state.context)
    verdict=gate.evaluate(state,answer,outcome='completed')
    assert verdict.accepted and verdict.layers['answer_evidence']=='verified' and not verdict.layers['warning']


@pytest.mark.parametrize('missing_material', [False, True])
def test_independent_service_reaches_real_read_and_exports_delivery_without_review_retry(case, missing_material):
    import time
    from shinku.agent_runtime import AgentRequest, ExecutionGrant, ExecutionProfile
    from shinku.agent_runtime.execution import AgentExecutor
    from shinku.agent_runtime.service import AgentService
    from shinku.agent_runtime.materials import MaterialPolicy, MaterialSource, TaskMaterialStore
    from shinku.hosts.tool_host import ToolHost
    from shinku.tasks.coordinator import AgentTaskCoordinator
    from shinku.tools.registry import ToolRegistry
    _, _, reader, _, task = case
    class Runtime:
        calls = 0
        def chat_supports_native_tools(self): return True
        def call_chat_json(self, **kwargs):
            self.calls += 1
            if any('本地答案证据复核' in block for block in kwargs.get('system_extra_blocks') or []): return {}
            if not reader.calls: return {'tool_call': {'type': reader.tool_type, 'path': str(reader.path)}}
            return {'kind': 'final', 'text': REPORT, 'task_outcome': 'completed'}
    runtime = Runtime()
    grant = ExecutionGrant('fixture-grant', frozenset({reader.tool_type}), permission_mode='read',
        policy=ExecutionPolicy(risk_by_tool={reader.tool_type:'low'}),
        material_sources=(MaterialSource('required','document','完整必需资料'*100),) if missing_material else ())
    service = AgentService(executor=AgentExecutor(ToolHost(ToolRegistry({reader.tool_type:reader})),ExecutionProfile(runtime)),
        authorize=lambda *_:grant, task_store=AgentTaskStore(), coordinator=AgentTaskCoordinator(max_workers=1))
    if missing_material:
        service.materials=TaskMaterialStore(service.task_store, policy=MaterialPolicy(inline_chars=8,page_chars=64))
    try:
        client=service.bind(AgentScope(task['requester'], task['session']))
        handle=client.submit(AgentRequest('fixture-input',task['requests'][0]['text'], task['contract']))
        deadline=time.monotonic()+5
        while True:
            view=client.status(handle.task_id)
            if not view['worker_active'] and view['status'] not in {'queued','running','planning','yielded'}:break
            assert time.monotonic()<deadline
            time.sleep(.005)
        result=client.result(handle.task_id)
        if missing_material:
            assert view['status']!='completed' and result['information']['required_status']=='incomplete'
        else:
            assert view['status']=='completed', (view,result)
            assert reader.calls==1 and runtime.calls==2
            assert result['verification']['layers']['quality_review']=='not_requested'
            assert result['verification']['layers']['answer_evidence']=='unverified'
            assert not result['remaining'] and result['execution_report']['text']==REPORT
            assert 'verified_answer' not in result
    finally:
        service.coordinator.shutdown(wait=True);service.task_store.close()
