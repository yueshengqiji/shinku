"""用真实临时文件和宿主循环验证恢复、跨切片收尾及证据拒绝边界。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from shinku.agent.local_answer import LocalAnswerLedger
from shinku.agent.loop import AgentDecision, AgentLoop, AgentState, _call_signature
from shinku.agent.planner import LLMPlanner
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent.verification import SmartCompletionVerifier
from shinku.tools.execution import ExecutionPolicy
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Reader:
    """合成文件适配器；执行与验证分别读取磁盘，撤权后不得取得正文。"""
    tool_type = 'fixture_read'
    agent_progress_class = 'observe'
    agent_permission_class = 'read'
    agent_completion_effects = ('read',)
    agent_completion_target_arguments = ('path',)
    agent_local_answer_source = True

    def __init__(self, root):
        self.root, self.calls, self.checks, self.allowed = root, 0, 0, True

    def normalize_call(self, call):
        return dict(call) if isinstance(call.get('path'), str) else None

    def read(self, path):
        path = Path(path).resolve()
        if not self.allowed or path.parent != self.root:
            raise PermissionError('fixture denied')
        return path.read_text()

    def execute(self, *, call, context):
        self.calls += 1
        text = self.read(call['path'])
        return {'tool_type': self.tool_type, 'followup_context': text, 'state_updates': {
            'path': call['path'], 'sha256': digest(text), 'read_complete': True,
            'completion_targets': [call['path']], 'verified_effects': ['read']}}

    def agent_verify_result(self, *, receipt, context):
        self.checks += 1
        text = self.read(receipt['path'])
        return {'sha256': digest(text), 'file_text': text,
                'result_valid': receipt['sha256'] == digest(text)}


@pytest.fixture
def setup(tmp_path):
    reader = Reader(tmp_path)
    paths = [tmp_path / f'source_{i}.txt' for i in range(40)]
    for path in paths:
        path.write_text('当前入口按职责分离，优化建议需以源码为依据。')
    context = {'agent_completion_contract': {'goal': '分析项目', 'deliverables': ['分析'], 'scope': ['只读'],
        'requirements': [{'id': 'input', 'effect': 'read', 'description': '读取输入',
                          'targets': [str(paths[0])], 'checks': {'unchanged': True}},
                         {'id': 'answer', 'effect': 'local_answer', 'description': '交付建议', 'targets': [],
                          'checks': {'required_fields': ['优化方向'], 'min_sources': 1}}]},
        'agent_task_state': {'task_id': 'fixture-task', 'requester': 'fixture', 'session': 'local:fixture',
                             'execution_attempt': 2}}
    return reader, paths, context


def record(reader, path, context):
    invocation = ToolInvocation(reader.tool_type, {'path': str(path)})
    data = reader.execute(call=invocation.arguments, context=context)
    return {'name': reader.tool_type, 'id': invocation.id, 'arguments': invocation.arguments,
            'status': 'ok', 'executed': True, 'data': data,
            'call_fingerprint': digest(_call_signature(invocation)),
            'execution_binding': {**{k: context['agent_task_state'][k] for k in ('task_id', 'requester', 'session')},
                                  'attempt': 1, 'work_id': '', 'work_attempt': 1}}


def restore(reader, context, records):
    ledger = LocalAnswerLedger()
    ledger.configure(context)
    ledger.restore_reads(records, {reader.tool_type: reader}, context)
    return ledger


def gate(reader):
    return SmartCompletionVerifier(result_handlers={reader.tool_type: reader},
        tool_capabilities={reader.tool_type: {'effects': ['read'], 'target_arguments': ['path']}})


def bind(ledger, context, answer='当前入口按职责分离。'):
    identifier = next(i for i, r in ledger.sources.items() if r['valid'])
    ledger.apply_answer_review(answer, {'verdict': 'supported', 'missing_fields': [],
        'claims': [{'answer_fragment': answer, 'evidence': [{'source_id': identifier, 'quote': '当前入口按职责分离'}]}],
        'coverage': [{'field': '优化方向', 'supported': True, 'claim_indices': [0]}]}, context)
    return answer


def test_sqlite_receipts_rebuild_current_evidence_without_tool_replay(setup, tmp_path):
    reader, paths, context = setup
    records = [record(reader, p, context) for p in paths[:2]]
    # 使用项目自己的持久化存储，关闭后新建实例模拟进程恢复。
    db = tmp_path / 'tasks.sqlite'
    store = AgentTaskStore(db)
    task = new_task(task_id='fixture-task', requester='fixture', session='local:fixture', request='分析项目', message_id='fixture-input', contract=context['agent_completion_contract'])
    task.update(tool_calls=records, execution_attempt=2, contract=context['agent_completion_contract'])
    store.save(task)
    store.close()
    store = AgentTaskStore(db)
    saved = store.get('fixture-task', requester='fixture', session='local:fixture')
    store.close()
    before = copy.deepcopy(saved['tool_calls'])
    ledger = restore(reader, context, saved['tool_calls'])
    assert reader.calls == 2 and len(ledger.sources) == 2
    assert not ledger.answer_check and not ledger.evaluate_answer('旧答案', 'completed').accepted
    answer = bind(ledger, context)
    state = AgentState(context=context, local_answer=ledger, tool_calls=saved['tool_calls'])
    verdict = gate(reader).evaluate(state, answer, outcome='completed')
    assert verdict.accepted and verdict.status == 'completed'
    assert saved['tool_calls'] == before


@pytest.mark.parametrize('corruption', ['changed', 'deleted', 'permission', 'foreign_task', 'foreign_session',
    'future_attempt', 'zero_attempt', 'not_executed', 'unbound', 'fingerprint', 'partial', 'receipt_path', 'not_reader'])
def test_restore_rejects_unusable_or_unbound_receipts(setup, corruption):
    reader, paths, context = setup
    row = record(reader, paths[0], context)
    if corruption == 'changed': paths[0].write_text('已变更')
    if corruption == 'deleted': paths[0].unlink()
    if corruption == 'permission': reader.allowed = False
    if corruption == 'foreign_task': row['execution_binding']['task_id'] = 'other'
    if corruption == 'foreign_session': row['execution_binding']['session'] = 'other'
    if corruption == 'future_attempt': row['execution_binding']['attempt'] = 3
    if corruption == 'zero_attempt': row['execution_binding']['attempt'] = 0
    if corruption == 'not_executed': row['executed'] = False
    if corruption == 'unbound': row.pop('execution_binding')
    if corruption == 'fingerprint': row['call_fingerprint'] = 'forged'
    if corruption == 'partial': row['data']['state_updates']['read_complete'] = False
    if corruption == 'receipt_path': row['data']['state_updates']['path'] = str(paths[1])
    if corruption == 'not_reader': reader.agent_permission_class = 'write'
    ledger = restore(reader, context, [row])
    assert not ledger.sources and not ledger.answer_check and reader.calls == 1


def test_later_failed_receipt_invalidates_restored_source_and_review_budget_survives(setup):
    reader, paths, context = setup
    row = record(reader, paths[0], context)
    failed = copy.deepcopy(row)
    failed.update(status='error', data={})
    context['agent_task_state']['progress'] = {'local_answer_review_calls': 2}
    ledger = restore(reader, context, [row, failed])
    assert not any(r['valid'] for r in ledger.sources.values())
    assert ledger.answer_review_calls == 2


def test_oversized_sources_do_not_displace_valid_small_sources(setup):
    reader, paths, context = setup
    rows = []
    for path in paths[:33]:
        path.write_text('长' * 120001)
        rows.append(record(reader, path, context))
    rows.append(record(reader, paths[-1], context))
    ledger = restore(reader, context, rows)
    assert len(ledger.sources) == 1
    assert next(iter(ledger.sources.values()))['path'] == str(paths[-1])


def test_changed_source_reread_requires_new_answer_review(setup):
    reader, paths, context = setup
    row = record(reader, paths[0], context)
    ledger = restore(reader, context, [row])
    answer = bind(ledger, context)
    paths[0].write_text('新的完整源码')
    fresh = record(reader, paths[0], context)
    ledger.restore_reads([fresh], {reader.tool_type: reader}, context)
    assert next(iter(ledger.sources.values()))['text'] == '新的完整源码'
    assert not ledger.evaluate_answer(answer, 'completed').accepted


def test_sliced_analysis_stops_expanding_and_delivers_verified_answer(setup):
    reader, paths, context = setup
    seen = []
    def planner(*, state):
        seen.append(state.finalization_only)
        if state.finalization_only:
            return AgentDecision.final(bind(state.local_answer, context), outcome='completed')
        return AgentDecision.tool(ToolInvocation(reader.tool_type, {'path': str(paths[len(state.tool_calls)])}))
    loop = AgentLoop(planner=planner, handlers={reader.tool_type: reader}, sliced=True, max_rounds=2,
        policy=ExecutionPolicy(default_risk='low'),
        completion_gate=gate(reader))
    result = loop.run(context=context, initial_transcript=[{'role': 'user', 'content': '只读分析项目并给优化方向'}])
    for _ in range(15):
        if result.status != 'yielded': break
        result = loop.run(checkpoint=result.checkpoint)
    assert result.status == 'completed' and result.final_text == '当前入口按职责分离。', (result.reason, result.completion)
    assert reader.calls == 12 and seen[-1] is True and result.completion['accepted']


def test_recovered_analysis_drafts_immediately_and_cannot_restore_blocked_tool(setup):
    reader, paths, context = setup
    rows = [record(reader, p, context) for p in paths[:24]]
    context['agent_restored_tool_calls'] = rows
    def planner(*, state):
        assert state.finalization_only
        return AgentDecision.final(bind(state.local_answer, context), outcome='completed')
    loop = AgentLoop(planner=planner, handlers={reader.tool_type: reader}, sliced=True, max_rounds=1,
        policy=ExecutionPolicy(default_risk='low'),
        completion_gate=gate(reader))
    result = loop.run(context=context)
    assert result.status == 'completed' and reader.calls == 24, (result.reason, result.completion)
    captured = []
    blocked = AgentLoop(planner=lambda **kw: captured.append(kw['state']) or AgentDecision.final('已阻止', outcome='failed'),
        handlers={reader.tool_type: reader}, policy=ExecutionPolicy(blocked=frozenset({reader.tool_type})))
    blocked.run(context=context)
    assert not captured[0].local_answer.sources


def test_drafting_rejected_answer_gets_bounded_repair_across_slice(setup):
    reader, paths, context = setup
    context['agent_restored_tool_calls'] = [record(reader, p, context) for p in paths[:24]]
    seen = []
    def planner(*, state):
        seen.append(state)
        return AgentDecision.final('没有依据的完成声明', outcome='completed')
    loop = AgentLoop(planner=planner, handlers={reader.tool_type: reader}, sliced=True, max_rounds=1,
        completion_gate=gate(reader))
    result = loop.run(context=context)
    assert result.status == 'yielded'
    result = loop.run(checkpoint=result.checkpoint)
    assert result.status == 'yielded'
    result = loop.run(checkpoint=result.checkpoint)
    assert result.status == 'failed' and result.reason == 'completion_no_progress'
    assert len(seen) == 3 and reader.calls == 24


def test_mutating_contract_does_not_enter_readonly_closeout(setup):
    reader, paths, context = setup
    context['agent_restored_tool_calls'] = [record(reader, p, context) for p in paths[:24]]
    context['agent_completion_contract']['requirements'].append(
        {'id': 'write', 'effect': 'write', 'targets': [str(paths[-1])]})
    seen = []
    loop = AgentLoop(planner=lambda **kw: seen.append(kw['state']) or AgentDecision.final('尚未执行写入', outcome='failed'),
        handlers={reader.tool_type: reader}, sliced=True)
    loop.run(context=context)
    assert not seen[0].finalization_only


def test_drafting_uses_verified_evidence_and_retains_native_history_without_old_previews(setup):
    reader, paths, context = setup
    ledger = restore(reader, context, [record(reader, paths[0], context)])
    class Runtime:
        def __init__(self): self.calls = []
        def native_message_scope(self): return ('fixture',)
        def call_chat_json(self, **kwargs):
            self.calls.append(kwargs)
            return {'kind': 'wait', 'text': 'fixture'}
    runtime = Runtime()
    planner = LLMPlanner(runtime=runtime, system_prompt='fixture', native_tool_history=True,
        tool_specs=[{'type': 'function', 'function': {'name': reader.tool_type, 'parameters': {}}}])
    state = AgentState(task_id='fixture-task', context=context, local_answer=ledger,
        transcript=[{'role': 'tool', 'content': 'OLD_PREVIEW_MUST_NOT_ENTER_ANSWER'}])
    planner(state=state)
    original = state.native_tool_history.user_prompt
    state.finalization_only = True
    state.progress['local_analysis'] = {'phase': 'drafting'}
    planner(state=state)
    call = runtime.calls[-1]
    assert state.native_tool_history.user_prompt == original
    assert 'OLD_PREVIEW_MUST_NOT_ENTER_ANSWER' in original
    assert 'OLD_PREVIEW_MUST_NOT_ENTER_ANSWER' not in call['user_prompt']
    assert call['native_tools'] is None and call['history_turns'] is None
    assert 'native_history' not in call and not state.native_tool_history.messages
    assert '当前入口按职责分离' in call['user_prompt']
    assert json.loads(call['user_prompt'])['contract'] == context['agent_completion_contract']
    planner(state=state)
    assert state.native_tool_history.user_prompt == original
    assert 'native_history' not in runtime.calls[-1]
    state.finalization_only = False
    state.progress['local_analysis'] = {'phase': 'gathering', 'base_tool_count': 1}
    context['agent_task_state']['requests'] = [{'message_id': 'new', 'text': '核实新增来源后继续'}]
    planner(state=state)
    history = runtime.calls[-1]['native_history']
    assert len(history) == 1 and '核实新增来源后继续' in history[-1]['content']
    planner(state=state)
    assert runtime.calls[-1]['native_history'] == history


def test_missing_required_read_prevents_early_draft_and_cannot_complete_at_limit(setup):
    reader, paths, context = setup
    context['agent_completion_contract']['requirements'][0]['targets'] = [str(paths[-1])]
    seen = []
    def planner(*, state):
        seen.append(state.finalization_only)
        if state.finalization_only:
            return AgentDecision.final(bind(state.local_answer, context), outcome='completed')
        return AgentDecision.tool(ToolInvocation(reader.tool_type, {'path': str(paths[len(state.tool_calls)])}))
    loop = AgentLoop(planner=planner, handlers={reader.tool_type: reader}, sliced=True, max_rounds=1,
        policy=ExecutionPolicy(default_risk='low'), completion_gate=gate(reader))
    result = loop.run(context=context)
    for _ in range(30):
        if result.status != 'yielded': break
        result = loop.run(checkpoint=result.checkpoint)
    assert reader.calls == 24 and not any(seen[:24]) and seen[24]
    assert result.status == 'failed' and 'input' in result.completion['missing']


def test_cancel_or_revoked_authority_precedes_restoration_reads(setup):
    reader, paths, context = setup
    context['agent_restored_tool_calls'] = [record(reader, paths[0], context)]
    loop = AgentLoop(planner=lambda **kw: pytest.fail('canceled work must not plan'),
        handlers={reader.tool_type: reader}, interruption_check=lambda state: 'agent_permission_changed')
    result = loop.run(context=context)
    assert result.status == 'blocked' and reader.checks == 0


@pytest.mark.parametrize('change', ['none', 'authority', 'expired', 'foreign', 'attempt', 'revision', 'revoked', 'origin'])
def test_recovery_material_check_never_exposes_canceled_task_content(setup, tmp_path, change):
    from shinku.agent_runtime import AgentBoundaryError
    from shinku.agent_runtime.materials import MaterialSource, TaskMaterialStore
    reader, paths, context = setup
    clock = [1000.0]
    store = AgentTaskStore(tmp_path / 'material.sqlite', clock=lambda: clock[0])
    task = new_task(task_id='material-task', requester='fixture', session='local:fixture',
                    request='只读分析', message_id='fixture-input', contract=None)
    task.update(execution_attempt=1, cancel_requested=False)
    task = store.save(task)
    materials = TaskMaterialStore(store, approved_roots=(tmp_path,))
    package = materials.package(task, [MaterialSource('fixture-source', 'document', '保留的完整原文',
        local_path=paths[0])] if change == 'origin' else [MaterialSource('fixture-source', 'document', '保留的完整原文')],
        authorization_ref='fixture-grant')
    task['material_package'] = package
    task.update(status='canceled', cancel_requested=True)
    task = store.save(task)
    reference = package['items'][0]['reference']
    with pytest.raises(AgentBoundaryError, match='material_task_canceled'):
        materials.resolve(reference, task=task, still_authorized=lambda: True)
    if change == 'expired': clock[0] += 3601
    if change == 'foreign': task['session'] = 'other'
    if change == 'attempt': task['execution_attempt'] += 1
    if change == 'revision': task['revision'] -= 1
    if change == 'revoked': materials.revoke(reference, task=task)
    if change == 'origin': paths[0].write_text('已经变更')
    if change == 'none':
        before = store.get(task['task_id'], requester='fixture', session='local:fixture')
        assert materials.check_retained_for_recovery(reference, task=task, still_authorized=lambda: True) is None
        assert store.get(task['task_id'], requester='fixture', session='local:fixture') == before
        assert materials.limits(task)['reads'] == 0
    else:
        with pytest.raises(AgentBoundaryError):
            materials.check_retained_for_recovery(reference, task=task, still_authorized=lambda: change != 'authority')
    store.close()


def test_invalidated_sources_release_text_budget_and_keep_prior_receipt(setup):
    reader, paths, context = setup
    for path in paths[:10]: path.write_text('旧' * 12000)
    rows = [record(reader, path, context) for path in paths[:10]]
    ledger = restore(reader, context, rows)
    assert len(ledger.sources) == 10
    for path in paths[:10]: path.write_text('内容已变化')
    ledger.refresh({reader.tool_type: reader}, context)
    ledger.restore_reads([record(reader, paths[-1], context)], {reader.tool_type: reader}, context)
    assert sum(row['valid'] for row in ledger.sources.values()) == 1
    assert len(ledger.sources) == 11
    assert rows[0]['data']['state_updates']['sha256'] == digest('旧' * 12000)


def test_invalid_cached_review_can_repair_once_without_expanding_review_budget(setup):
    reader, paths, context = setup
    ledger = restore(reader, context, [record(reader, paths[0], context)])
    answer = '当前入口按职责分离。'
    identifier = next(iter(ledger.sources))
    good = {'verdict': 'supported', 'missing_fields': [],
        'claims': [{'answer_fragment': answer, 'evidence': [{'source_id': identifier, 'quote': '当前入口按职责分离'}]}],
        'coverage': [{'field': '优化方向', 'supported': True, 'claim_indices': [0]}]}
    bad = copy.deepcopy(good)
    bad['claims'][0]['evidence'][0]['quote'] = '凭空编造的引用'
    class Runtime:
        def __init__(self): self.calls = []
        def call_chat_json(self, **kwargs):
            self.calls.append(kwargs)
            return bad if len(self.calls) == 1 else good
    runtime = Runtime()
    state = AgentState(context=context, local_answer=ledger)
    planner = LLMPlanner(runtime=runtime, system_prompt='fixture')
    decision = AgentDecision.final(answer, outcome='completed')
    planner._review_local_answer(state, decision)
    assert state.progress['local_answer_summary']['binding_error'] == 'invalid_quote'
    planner._review_local_answer(state, decision)
    planner._review_local_answer(state, decision)
    assert ledger.answer_check['valid'] and len(runtime.calls) == ledger.answer_review_calls == 2
    assert json.loads(runtime.calls[-1]['user_prompt'])['prior_binding_error'] == 'invalid_quote'


def test_analysis_report_is_retained_as_text_without_granting_verified_status(setup):
    from shinku.agent_runtime.results import export_task_result, result_data
    _reader, _paths, context = setup
    task = new_task(task_id='fixture-task', requester='fixture', session='local:fixture',
                    request='分析项目', message_id='fixture-input', contract=context['agent_completion_contract'])
    task.update(execution_attempt=1, completion_attempt=1, revision=1, updated=1000,
                completion={'accepted': False, 'reason': 'synthetic rejection'})
    report = '有依据的分析正文。' * 800 + '\n尾部：具体优化方向及其限制。'
    result = export_task_result(task, status='failed', reason='completion_no_progress', report=report)
    assert result['execution_report']['text'] == report
    assert result_data(result)['execution_report']['text'] == report
    assert result['verification']['accepted'] is False and 'verified_answer' not in result
    assert result['execution_report']['is_fact_evidence'] is False


@pytest.mark.parametrize('label', ['1) ', '2、', '3）', '## 4、'])
def test_list_labels_are_presentation_but_inline_quantities_still_need_proof(setup, label):
    reader, paths, context = setup
    ledger = restore(reader, context, [record(reader, paths[0], context)])
    identifier = next(iter(ledger.sources))
    def review(answer):
        return {'verdict': 'supported', 'missing_fields': [],
            'claims': [{'answer_fragment': line, 'evidence': [{'source_id': identifier, 'quote': '当前入口按职责分离'}]}
                       for line in answer.splitlines()],
            'coverage': [{'field': '优化方向', 'supported': True, 'claim_indices': [0]}]}
    answer = label + '当前入口按职责分离。\n' + label + '当前入口按职责分离。'
    ledger.apply_answer_review(answer, review(answer), context)
    assert ledger.answer_check['valid']
    fabricated = answer + '\n' + label + '性能提升 99%。'
    ledger.apply_answer_review(fabricated, review(fabricated), context)
    assert not ledger.answer_check['valid']
    assert ledger.answer_check['binding_error'] == 'derived_number_without_calculation'


def test_decimal_at_line_start_is_never_mistaken_for_list_number():
    from shinku.agent.local_answer import _without_list_labels
    assert _without_list_labels('3.11 是版本号\n99.5% 是一个未核实事实') == '3.11 是版本号\n99.5% 是一个未核实事实'


@pytest.mark.parametrize('change', ['none', 'unknown_id', 'wrong_text', 'wrong_quote', 'wrong_answer'])
def test_host_answer_selectors_bind_only_current_literal_text_and_still_verify_evidence(setup, change):
    reader, paths, context = setup
    ledger = restore(reader, context, [record(reader, paths[0], context)])
    answer = '当前入口按职责分离。'
    payload = ledger.review_payload('分析', context, answer=answer)
    identifier, fragment = next(iter(payload['answer_fragments'].items()))
    proof = {'verdict': 'supported', 'missing_fields': [],
        'claims': [{'answer_fragment_id': identifier, 'evidence': [
            {'source_id': next(iter(ledger.sources)), 'quote': '当前入口按职责分离'}]}],
        'coverage': [{'field': '优化方向', 'supported': True, 'claim_indices': [0]}]}
    if change == 'unknown_id': proof['claims'][0]['answer_fragment_id'] = 'model_invented'
    if change == 'wrong_text': proof['claims'][0]['answer_fragment'] = '另一份答案'
    if change == 'wrong_quote': proof['claims'][0]['evidence'][0]['quote'] = '不存在的源码'
    if change == 'wrong_answer': answer += '完全相反的新结论。'
    before = copy.deepcopy(proof)
    ledger.apply_answer_review(answer, proof, context)
    assert ledger.answer_check['valid'] is (change == 'none')
    assert proof == before and fragment == '当前入口按职责分离。'


def test_switching_to_commands_cannot_evade_readonly_exploration_limit(setup):
    reader, paths, context = setup
    class Locator:
        tool_type = 'fixture_locator'
        agent_permission_class = 'command'
        agent_completion_effects = ('command_run',)
        agent_progress_class = 'observe'
        def normalize_call(self, call): return dict(call)
        def execute(self, *, call, context):
            return {'tool_type': self.tool_type, 'followup_context': str(call),
                    'state_updates': {'verified_effects': ['command_run']}}
    locator = Locator()
    def planner(*, state):
        if state.finalization_only:
            assert len(state.local_answer.sources) == 2
            return AgentDecision.final(bind(state.local_answer, context), outcome='completed')
        if len(state.tool_calls) < 2:
            return AgentDecision.tool(ToolInvocation(reader.tool_type, {'path': str(paths[len(state.tool_calls)])}))
        return AgentDecision.tool(ToolInvocation(locator.tool_type, {'locator': len(state.tool_calls)}))
    loop = AgentLoop(planner=planner, handlers={reader.tool_type: reader, locator.tool_type: locator},
        sliced=True, max_rounds=2, policy=ExecutionPolicy(default_risk='low'), completion_gate=gate(reader))
    result = loop.run(context=context)
    for _ in range(20):
        if result.status != 'yielded': break
        result = loop.run(checkpoint=result.checkpoint)
    assert result.status == 'completed' and reader.calls == 2
    assert result.progress['local_analysis']['exploration_calls'] == 24
    assert len(result.transcript) > 24


@pytest.mark.parametrize('change', ['none', 'unknown_path', 'foreign_prefix', 'false_quantity', 'inline_quantity'])
def test_verified_source_identifiers_do_not_become_derived_date_statistics(setup, change):
    reader, paths, context = setup
    dated = paths[0].with_name('guide_2026-10-17.md')
    dated.write_text('当前入口按职责分离。')
    ledger = restore(reader, context, [record(reader, dated, context)])
    answer = '依据 `guide_2026-10-17.md`，当前入口按职责分离。'
    if change == 'unknown_path': answer = answer.replace('guide_', 'unknown_')
    if change == 'foreign_prefix': answer = answer.replace('guide_', 'foreign/guide_')
    if change == 'false_quantity': answer += '性能提升 99%。'
    if change == 'inline_quantity': answer += '性能提升 `99`。'
    proof = {'verdict': 'supported', 'missing_fields': [],
        'claims': [{'answer_fragment': answer, 'evidence': [
            {'source_id': next(iter(ledger.sources)), 'quote': '当前入口按职责分离。'}]}],
        'coverage': [{'field': '优化方向', 'supported': True, 'claim_indices': [0]}]}
    ledger.apply_answer_review(answer, proof, context)
    assert ledger.answer_check['valid'] is (change == 'none')


def test_section_ordinal_in_quote_and_sentence_final_quantity_are_parsed(setup):
    from shinku.agent.local_answer import _numbers
    reader, paths, context = setup
    paths[0].write_text('## 6. 状态归属\n当前入口按职责分离。')
    ledger = restore(reader, context, [record(reader, paths[0], context)])
    answer = '第 6 节规定状态归属。'
    review = {'verdict': 'supported', 'missing_fields': [],
        'claims': [{'answer_fragment': answer, 'evidence': [
            {'source_id': next(iter(ledger.sources)), 'quote': '## 6. 状态归属'}]}],
        'coverage': [{'field': '优化方向', 'supported': True, 'claim_indices': [0]}]}
    ledger.apply_answer_review(answer, review, context)
    assert ledger.answer_check['valid']
    assert [str(n) for n in _numbers('Value is 99. Version 0.1.0.')] == ['99']
    answer = 'Value is 99.'
    review['claims'][0]['answer_fragment'] = answer
    ledger.apply_answer_review(answer, review, context)
    assert ledger.answer_check['unsupported_numbers'] == ['99']


@pytest.mark.parametrize('known', [True, False])
def test_literal_document_references_are_identifiers_without_minting_target_contents(setup, known):
    reader, paths, context = setup
    paths[0].write_text('文档入口：docs/guide_2026-10-17.md。当前入口按职责分离。')
    ledger = restore(reader, context, [record(reader, paths[0], context)])
    answer = '当前入口按职责分离；文档索引包含 docs/' + ('guide' if known else 'forged') + '_2026-10-17.md。'
    review = {'verdict': 'supported', 'missing_fields': [],
        'claims': [{'answer_fragment': answer, 'evidence': [
            {'source_id': next(iter(ledger.sources)), 'quote': '当前入口按职责分离。'}]}],
        'coverage': [{'field': '优化方向', 'supported': True, 'claim_indices': [0]}]}
    ledger.apply_answer_review(answer, review, context)
    assert ledger.answer_check['valid'] is known
    assert len(ledger.sources) == 1  # 没有打开或注册索引目标文件。


@pytest.mark.parametrize('change', ['none', 'unknown_quote', 'conflicting_quote', 'unknown_field',
    'conflicting_field', 'unknown_claim', 'conflicting_indices', 'changed_source'])
def test_structured_field_and_numeric_quote_selectors_preserve_all_original_checks(setup, change):
    reader, paths, context = setup
    paths[0].write_text('要求 Python >=3.11。许可证为 Apache-2.0。')
    ledger = restore(reader, context, [record(reader, paths[0], context)])
    answer = '要求 Python >=3.11，许可证为 Apache-2.0。'
    payload = ledger.review_payload('分析', context, answer=answer)
    fragment = next(iter(payload['answer_fragments']))
    field = next(iter(payload['required_field_selectors']))
    quotes = list(payload['numeric_evidence_candidates'])
    assert quotes and {q['value'] for q in payload['numeric_evidence_candidates'].values()} == {'3.11', '2.0'}
    proof = {'verdict': 'supported', 'missing_fields': [],
        'claims': [{'answer_fragment_id': fragment, 'evidence': [{'quote_id': q} for q in quotes]}],
        'coverage': [{'field_id': field, 'supported': True, 'answer_fragment_ids': [fragment]}]}
    if change == 'unknown_quote': proof['claims'][0]['evidence'][0]['quote_id'] = 'invented'
    if change == 'conflicting_quote': proof['claims'][0]['evidence'][0]['quote'] = 'a fabricated quote'
    if change == 'unknown_field': proof['coverage'][0]['field_id'] = 'invented'
    if change == 'conflicting_field': proof['coverage'][0]['field'] = 'another requirement'
    if change == 'unknown_claim': proof['coverage'][0]['answer_fragment_ids'] = ['invented']
    if change == 'conflicting_indices': proof['coverage'][0]['claim_indices'] = [1]
    if change == 'changed_source':
        paths[0].write_text('修改后的 Python >=3.11。许可证为 Apache-2.0。')
        ledger.restore_reads([record(reader, paths[0], context)], {reader.tool_type: reader}, context)
    before = copy.deepcopy(proof)
    ledger.apply_answer_review(answer, proof, context)
    assert ledger.answer_check['valid'] is (change == 'none')
    assert proof == before


def test_new_host_input_can_read_added_required_sources_without_resetting_task_usage(setup):
    reader, paths, context = setup
    context['agent_task_state']['requests'] = [{'message_id': 'first', 'text': '分析已有资料'}]
    drafts = []
    def planner(*, state):
        if not state.finalization_only:
            assert state.progress['local_analysis']['phase'] == 'gathering'
        if state.finalization_only:
            drafts.append(len(state.tool_calls))
            if len(drafts) == 1:
                return AgentDecision.final('尚未通过核验的草稿', outcome='completed')
            return AgentDecision.final(bind(state.local_answer, context), outcome='completed')
        required = context['agent_completion_contract']['requirements'][0]['targets']
        read = {c['arguments']['path'] for c in state.tool_calls}
        path = next((p for p in required if p not in read), str(paths[len(state.tool_calls)]))
        return AgentDecision.tool(ToolInvocation(reader.tool_type, {'path': path}))
    loop = AgentLoop(planner=planner, handlers={reader.tool_type: reader}, sliced=True, max_rounds=1,
        policy=ExecutionPolicy(default_risk='low'), completion_gate=gate(reader))
    result = loop.run(context=context)
    while not drafts:
        result = loop.run(checkpoint=result.checkpoint)
    assert result.status == 'yielded' and reader.calls == 12
    context['agent_task_state']['requests'].append({'message_id': 'second', 'text': '再核实新增输入后分析'})
    context['agent_completion_contract']['requirements'][0]['targets'].append(str(paths[-1]))
    result = loop.run(checkpoint=result.checkpoint, context=context)
    assert result.status == 'yielded' and reader.calls == 13
    result = loop.run(checkpoint=result.checkpoint, context=context)
    assert result.status == 'completed' and drafts == [12, 13]
    assert result.progress['local_analysis']['base_tool_count'] == 12
    assert result.progress['local_analysis']['exploration_calls'] == 1
    assert len(result.transcript) > 13


@pytest.mark.parametrize('new_user_input', [True, False])
def test_only_new_host_input_releases_drafting_for_additional_effects(setup, new_user_input):
    from types import SimpleNamespace
    from shinku.agent.local_analysis import prepare_local_analysis
    reader, paths, context = setup
    context['agent_task_state']['requests'] = [{'message_id': 'first', 'text': '只读分析'}]
    records = [record(reader, p, context) for p in paths[:24]]
    ledger = restore(reader, context, records)
    state = AgentState(context=context, local_answer=ledger, tool_calls=records)
    run = SimpleNamespace(state=state, context=context)
    deps = SimpleNamespace(handlers=lambda: {reader.tool_type: reader}, completion_gate=lambda: gate(reader),
                           observe_state=lambda *_: None)
    prepare_local_analysis(run, deps)
    assert state.finalization_only
    ledger.answer_review_calls = 2
    if new_user_input:
        context['agent_task_state']['requests'].append({'message_id': 'second', 'text': '再处理新的输出要求'})
    context['agent_completion_contract']['requirements'].append(
        {'id': 'write', 'effect': 'write', 'targets': [str(paths[-1])]})
    prepare_local_analysis(run, deps)
    assert state.finalization_only is (not new_user_input)
    assert ledger.answer_review_calls == 2 and reader.calls == 24


def test_required_source_tail_remains_in_native_context_when_analysis_closes(setup):
    from types import SimpleNamespace
    from shinku.agent.planner import AgentPromptParts
    reader, paths, context = setup
    ledger = restore(reader, context, [record(reader, paths[0], context)])
    class Runtime:
        def __init__(self): self.calls = []
        def native_message_scope(self): return ('fixture',)
        def call_chat_json(self, **kwargs):
            self.calls.append(kwargs)
            return {'kind': 'wait', 'text': 'fixture'}
    context['agent_material_session'] = SimpleNamespace(
        prompt_package=lambda: {'items': [{'required': True, 'content': 'prefix', 'omitted_chars': 9000}]},
        mark_model_input=lambda *_: None)
    runtime = Runtime()
    planner = LLMPlanner(runtime=runtime, system_prompt='fixture', native_tool_history=True,
        build_user_prompt=lambda state: AgentPromptParts('original required data: ATTACHMENT_TAIL_MUST_REMAIN'))
    state = AgentState(task_id='fixture-task', context=context, local_answer=ledger)
    planner(state=state)
    original = state.native_tool_history.user_prompt
    state.finalization_only = True
    state.progress['local_analysis'] = {'phase': 'drafting', 'base_tool_count': 0}
    planner(state=state)
    call = runtime.calls[-1]
    assert call['user_prompt'] == original and 'ATTACHMENT_TAIL_MUST_REMAIN' in call['user_prompt']
    assert '当前入口按职责分离' in call['native_history'][-1]['content']
    assert call['native_tools'] is None
    assert context['agent_material_session'].prompt_package()['items'][0]['omitted_chars'] == 9000


def test_medium_complete_sources_share_existing_total_budget_without_per_file_loss(setup):
    reader, paths, context = setup
    for path in paths[:8]: path.write_text('完整源码\n' * 3600)  # 每份超过旧单文件限制，仍可在总量内逐字保存。
    records = [record(reader, path, context) for path in paths[:8]]
    ledger = restore(reader, context, records)
    assert len(ledger.sources) == 6
    assert all(row['text'] == '完整源码\n' * 3600 for row in ledger.sources.values())
    assert sum(len(row['text']) for row in ledger.sources.values() if row['valid']) <= 120000
    assert reader.calls == 8 and all(row['data']['state_updates']['read_complete'] for row in records)
