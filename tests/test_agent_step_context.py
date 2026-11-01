"""当前步骤投影、硬条件续接及原稿回读；只使用合成材料和隔离执行器。"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from shinku.agent.native_history import NativeToolHistory
from shinku.agent_runtime.work_context import focus_payload, navigation
from shinku.agent_runtime.materials import MaterialSource
from shinku.agent_runtime.context import build_execution_prompt
from shinku.agent_runtime import AgentTurnRequest
from shinku.llm.native_messages import NativeProtocolError
from test_execution_context_budget import material_context, state_for
from test_agent_conversation_loop import turn_env
from test_agent_work_planning import update, saved_task
from test_agent_runtime_boundary import settled
from work_state_fixture import plan


def fixture_focus(material_context):
    _, task, store = material_context
    sources = [MaterialSource('required', 'caller_data', '硬输入：保留原文，不允许外发。')]
    sources += [MaterialSource('tool:' + str(i), 'executor_result', '工具原文' * 3000 + str(i),
        required=False, provenance={'invocation_id': str(i)}) for i in range(10)]
    state, session = state_for(task, store, sources)
    package = session.prompt_package(); target = package['items'][1]
    task['tool_calls'] = [{'id': str(i), 'material_reference': {'reference': row['reference']}}
        for i, row in enumerate(package['items'][1:])]
    state.transcript = [{'role': 'tool', 'content': '已保留工具原文' * 1000,
        'material_reference': {'reference': row['reference']}} for row in package['items'][1:]]
    value = plan(references=[{'kind': 'material', 'id': target['reference']}])
    work = {'revision': 1, 'status': 'active', 'plan': value, 'references': [
        {'step_id': 'inspect', 'kind': 'material', 'id': target['reference'],
         'version': target['version'], 'content_sha256': target['content_sha256']}]}
    data = {'requests': task['requests'], 'contract': task['contract'], 'task_materials': package,
        'information': session.information(), 'answer_context': {'prior_user_requests': ['无关旧背景']}}
    return state, session, work, data


def test_step_projection_reduces_previews_and_preserves_original_sources_and_reads(material_context):
    state, active, work, data = fixture_focus(material_context)
    task = state.context['agent_task_state']; store = material_context[2]
    original = deepcopy((task, state.transcript, data, work))
    reads = store.read_receipts(task); usage = store.limits(task)
    projected, history = focus_payload(data, state.transcript, work, task)
    assert projected['requests'] == data['requests'] and projected['contract'] == data['contract']
    assert projected['information'] == data['information']
    items = projected['task_materials']['items']; prior = data['task_materials']['items']
    assert items[0] == prior[0] and items[1] == prior[1] and items[-2:] == prior[-2:]
    assert all(row['content'] == '' and row['omitted_chars'] == row['content_chars'] for row in items[2:-2])
    assert history[-2:] == state.transcript[-2:]
    assert len(json.dumps(projected, ensure_ascii=False)) < len(json.dumps(data, ensure_ascii=False)) // 2
    assert (task, state.transcript, data, work) == original
    assert store.read_receipts(task) == reads and store.limits(task) == usage
    for row in prior:
        assert store.resolve(row['reference'], task=task, still_authorized=lambda: True)
    assert projected['step_focus']['is_execution_evidence'] is False


@pytest.mark.parametrize('reason', ['requires_review', 'missing', 'changed_version', 'no_refs', 'expired'])
def test_unusable_plan_never_suppresses_sources(material_context, reason):
    state, active, work, data = fixture_focus(material_context)
    if reason == 'requires_review': work['status'] = 'requires_review'
    if reason == 'missing': work['references'][0]['id'] = 'tm:foreign'
    if reason == 'changed_version': work['references'][0]['version'] += '-changed'
    if reason == 'no_refs': work['references'] = []
    if reason == 'expired':
        data['information']['gaps'].append({'reference': work['references'][0]['id'],
            'reason': 'material_source_expired_or_revoked'})
    projected, history = focus_payload(data, state.transcript, work, state.context['agent_task_state'])
    assert projected == data and history == state.transcript


def test_tool_reference_checks_original_material_version_before_suppressing_previews(material_context):
    from shinku.agent.work_state import digest
    state, active, work, data = fixture_focus(material_context)
    task = state.context['agent_task_state']; row = data['task_materials']['items'][1]
    call = task['tool_calls'][0]
    call['material_reference'].update(version=row['version'], content_sha256=row['content_sha256'])
    work['references'] = [{'step_id': 'inspect', 'kind': 'tool_call', 'id': call['id'], 'receipt_sha256': digest(call)}]
    assert focus_payload(data, state.transcript, work, task)[0]['step_focus']['omitted_previews']
    row['version'] += '-changed'
    assert focus_payload(data, state.transcript, work, task) == (data, state.transcript)


def test_only_retained_tool_feedback_can_be_shortened_and_controls_survive(material_context):
    state, active, work, data = fixture_focus(material_context)
    control = {'role': 'system', 'content': '最新补充：不能删除文件，也不能发送消息。'}
    no_reference = {'role': 'tool', 'content': '没有原稿回读引用，必须完整保留。'}
    state.transcript[:0] = [control, no_reference]
    projected, history = focus_payload(data, state.transcript, work, state.context['agent_task_state'])
    assert history[:2] == [control, no_reference]
    assert projected['task_materials']['items'][0] == data['task_materials']['items'][0]


def test_navigation_has_full_plan_retrieval_and_labels_model_claims():
    work = {'revision': 7, 'status': 'active', 'plan': plan(), 'references': []}
    work['plan']['findings'] = ['待核实' * 200] * 12
    work['plan']['hypotheses'] = ['假设' * 300] * 12
    work['plan']['steps'].append({'id': 'later', 'goal': '待办' * 300, 'depends': ['inspect'],
        'requirements': [], 'status': 'pending', 'references': []})
    original = deepcopy(work); view = navigation(work)
    assert view['full_plan_retrieval'] == {'tool': 'read_work_context', 'view': 'plan', 'revision': 7}
    assert view['findings_omitted_items'] == 8 and view['hypotheses_omitted_chars'] > 0
    assert view['plan']['steps'][0] == work['plan']['steps'][0]
    assert view['plan']['steps'][1]['goal_omitted_chars'] > 0 and work == original


def test_full_plan_read_is_optional_metered_and_not_execution_evidence(tmp_path):
    value = plan(); value['findings'] = ['合成判断' * 90] * 8
    def respond(kw, count, service, tool):
        if count == 1: return update(value)
        if count == 2:
            assert 'full_plan_retrieval' in kw['user_prompt']
            return {'tool_call': {'type': 'read_work_context', 'view': 'plan'}}
        assert value['findings'][0] in kw['user_prompt']
        return {'kind': 'final', 'text': '原计划已回读，其中的判断仍待核实。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest('read-plan', '讨论持续工作方案'))
        view = settled(port, h.task_id); task = saved_task(service, h.task_id)
        assert view['status'] == 'completed' and not view['verification']['verified']
        assert len(calls) == task['budget']['requests'] == 3 and task['budget']['tools'] == 2
        assert task['progress']['stalled'] == 2 and task['work_state']['revision'] == 1
        assert 'plan' not in task['tool_calls'][-1]['data']['state_updates'] and tool.calls == 0


def test_native_hard_conditions_deduplicate_independently_from_plan_updates():
    scope = ('synthetic', 'provider'); state = SimpleNamespace(last_envelope=None, tool_calls=[])
    history = NativeToolHistory(scope, 'root\nHARD-ONE\nplan-1', None, None, 'plan-1', 'HARD-ONE')
    assert not history.request_kwargs(state, scope, 'HARD-ONE\nplan-1', 'plan-1', 'HARD-ONE')['native_history']
    rows = history.request_kwargs(state, scope, 'HARD-ONE\nplan-2', 'plan-2', 'HARD-ONE')['native_history']
    assert rows == [{'role': 'user', 'content': 'plan-2'}]
    rows = history.request_kwargs(state, scope, 'HARD-TWO\nplan-2', 'plan-2', 'HARD-TWO')['native_history']
    assert rows[-1] == {'role': 'user', 'content': 'HARD-TWO'}
    assert history.request_kwargs(state, scope, 'HARD-TWO\nplan-2', 'plan-2', 'HARD-TWO')['native_history'] == rows
    with pytest.raises(NativeProtocolError, match='hard_context_removed'):
        history.request_kwargs(state, scope, 'plan-2', 'plan-2', '')


def test_ninety_percent_summary_preserves_latest_full_contract_and_step(tmp_path, monkeypatch):
    from test_context_compaction import Summarizer, bundle, payload, window
    from shinku.llm.context_compaction import ContextCompactor
    window(monkeypatch, 2500)
    runtime = Summarizer(); manager = ContextCompactor(runtime, tmp_path / 'compactions.sqlite3')
    source = payload(15000)
    hard = {'role': 'user', 'content': '完整当前条件：读取原文件，不得修改、发送；验收包含原字节哈希。'}
    step = {'role': 'user', 'content': '当前步骤 inspect，下一步核实已有读取回执，计划不是执行证据。'}
    source['messages'].extend([hard, step]); original = deepcopy(source)
    projected = manager.project(source, bundle(), 2, 'bound-task')
    assert projected['messages'][-2:] == [hard, step] and source == original
    saved = json.loads(manager.db.execute('SELECT payload FROM context_compactions').fetchone()[0])
    assert saved['original_payload'] == original and saved['is_fact_evidence'] is False


def test_model_dictionary_cannot_opt_into_host_step_projection():
    task = {'requests': [{'text': '原要求'}], 'work_state': {'revision': 1}}
    context = {'agent_task_state': task, '_agent_conversation_turn': {'plan': plan()}}
    parts = build_execution_prompt(SimpleNamespace(context=context, transcript=[]))
    assert not parts.native_hard_context and 'step_focus' not in parts.user_prompt


def test_updated_plan_can_be_read_again_by_revision_without_resetting_progress(tmp_path):
    from test_angelos_initiative import seed
    from test_agent_conversation_loop import observed_contract
    def respond(kw, count, service, tool):
        if count == 1: return update()
        if count in {2, 5}:
            return {'tool_call': {'type': 'read_work_context', 'view': 'plan', 'revision': 1 if count == 2 else 2}}
        if count == 3: return {'tool_call': {'type': 'read_task_records', 'mode': 'detail', 'task_id': 'old'}}
        if count == 4:
            value = plan(); value['reason'] = '依据真实记录推进当前步骤'
            return update(value, 1)
        return {'kind': 'final', 'text': '旧任务具体错误为 synthetic_missing_input，原失败记录保留。',
            'task_outcome': 'completed', 'task_contract': observed_contract('read_task_records')}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        seed(service, 'old')
        h = port.submit(AgentTurnRequest('repeated-plan-read', '核实会话原错误并记录排查计划'))
        assert settled(port, h.task_id)['status'] == 'completed', port.result(h.task_id)
        task = saved_task(service, h.task_id)
        readers = [c for c in task['tool_calls'] if c['name'] == 'read_work_context']
        assert [c['status'] for c in readers] == ['ok', 'ok']
        assert [c['data']['state_updates']['revision'] for c in readers] == [1, 2]
        assert len(calls) == 6 and task['budget']['tools'] == 5 and tool.calls == 0


def test_old_or_noninteger_plan_revision_is_rejected(tmp_path):
    captured = []
    def respond(kw, count, service, tool):
        if count == 1: return update()
        context = next(iter(service._sessions.values())).context
        handler = context['_agent_conversation_turn'].work_context
        captured.append(handler.execute(call={'view': 'plan', 'revision': 2}, context=context))
        assert handler.normalize_call({'view': 'plan', 'revision': True}) is None
        return {'kind': 'final', 'text': '只讨论当前方案。', 'task_outcome': 'completed'}
    with turn_env(tmp_path, respond) as (service, tool, host, port, calls):
        h = port.submit(AgentTurnRequest('wrong-revision', '讨论方案'))
        assert settled(port, h.task_id)['status'] == 'completed'
        assert captured[0]['state_updates']['error_code'] == 'work_context_revision_conflict'
