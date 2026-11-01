"""合成命令回执经真实材料库、原生历史和补读工具；不调用外部模型。"""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shinku.agent.loop import AgentState
from shinku.agent.native_history import NativeToolHistory
from shinku.agent.task_state import AgentTaskStore, new_task
from shinku.agent.tool_context import model_tool_content, PREVIEW_CHARS
from shinku.agent_runtime.materials import MaterialPolicy, TaskMaterialStore
from shinku.agent_runtime.material_session import TaskMaterialSession, MaterialReadTool
from shinku.tools.invocation import ToolInvocation, ToolResultEnvelope


@pytest.fixture
def materials(tmp_path):
    clock, allowed = [1000.0], [True]
    ledger = AgentTaskStore(tmp_path / 'tasks.sqlite', clock=lambda: clock[0])
    task = new_task(task_id='shell-fixture', requester='fixture', session='fixture-session',
                    request='定位测试错误，不重跑副作用', message_id='m1', contract=None)
    task.update(status='running', execution_attempt=1)
    task = ledger.save(task)
    store = TaskMaterialStore(ledger, policy=MaterialPolicy(inline_chars=64))
    active = TaskMaterialSession(store, task, still_authorized=lambda: allowed[0])
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    from shinku.tools.execution import ExecutionPolicy
    active.tool_host(ToolHost(ToolRegistry({})), set(), ExecutionPolicy())
    yield active, ledger, clock, allowed
    ledger.close()


def envelope(stdout='only-output-marker', stderr='', status='ok', **updates):
    data = {'execution_session': {'session_id': 'session-1', 'command_sha256': 'a'*64,
            'status': 'completed' if status == 'ok' else 'failed', 'returncode': 0 if status == 'ok' else 7,
            'cursor': 9}, 'returncode': 0 if status == 'ok' else 7,
            'stdout': stdout, 'stderr': stderr, 'has_more': False, 'output_truncated': False,
            'verified_effects': ['command_run'] if status == 'ok' else [], **updates}
    return ToolResultEnvelope('c1', status, '执行状态仅代表命令。\nstdout:\n'+stdout+'\nstderr:\n'+stderr,
                             {'tool_type': 'fixture_shell', 'state_updates': data}, executed=True)


def retain(active, ledger, result):
    call = {'id': result.invocation_id, 'name': 'fixture_shell', 'status': result.status}
    call['material_reference'] = active.capture_tool_output(ToolInvocation('fixture_shell', id=result.invocation_id), result)
    saved = ledger.save(active.task)
    active.task.clear()
    active.task.update(saved)
    return call


def receipt(text):
    return json.loads(text.split('\n\n[host receipt]\n', 1)[1])


def long_output():
    return 'start\n' + '日志行\t"quoted"\\value\n' * 320 + 'MIDDLE_EVIDENCE' + 'later\n'*1600 + 'end'


def test_short_shell_deduplicates_without_archive_and_keeps_authoritative_fields():
    result = envelope(stderr='error-marker', status='error', artifact_receipts=[{'path': '/fixture/a', 'sha256': 'b'*64}])
    original = copy.deepcopy(result)
    text = model_tool_content(result, {})
    assert text.count('only-output-marker') == text.count('error-marker') == 1
    view = receipt(text)
    assert view['status'] == 'error' and view['data']['state_updates']['returncode'] == 7
    assert view['data']['state_updates']['artifact_receipts'] == result.data['state_updates']['artifact_receipts']
    assert view['data']['state_updates']['execution_session'] == result.data['state_updates']['execution_session']
    assert result == original


def test_long_result_retains_raw_and_omitted_middle_is_retrievable(materials):
    active, ledger, _, _ = materials
    result = envelope(long_output(), 'critical stderr', status='error')
    call = retain(active, ledger, result)
    original = copy.deepcopy(result)
    text = model_tool_content(result, call, active)
    view = receipt(text)['output_preview']
    assert 'MIDDLE_EVIDENCE' not in text and 'start' in text and 'end' in text and 'critical stderr' in text
    assert view['reader'] == 'task_material_read'
    ref = view['material_reference']['reference']
    raw = active.store.resolve(ref, task=active.task, still_authorized=active.still_authorized)
    assert raw['feedback'] == result.model_feedback and raw['state_updates'] == result.data['state_updates']
    span = view['omitted_ranges']['stdout']
    reader = MaterialReadTool(active)
    fetched, offset, left = '', span['offset'], span['length']
    while left:
        response = reader.execute(call={'reference': ref, 'offset': offset, 'length': min(left, view['page_chars'])}, context={})
        page = json.loads(response['followup_context'])
        assert page['provided'] is False
        # 实际输入前不冒记为已读；补读协议正文逐字保留，不应用shell视图。
        env = ToolResultEnvelope('page', 'ok', response['followup_context'], {'state_updates': response['state_updates']})
        wire = model_tool_content(env, {})
        active.mark_model_input(wire)
        fetched += page['content']; offset += page['returned_chars']; left -= page['returned_chars']
    assert 'MIDDLE_EVIDENCE' in fetched
    assert all(r['provided'] for r in active.store.read_receipts(active.task))
    assert result == original
    assert len(text) < len(result.model_feedback) + 1000


@pytest.mark.parametrize('failure', ['missing', 'forged', 'version', 'hash', 'wrong_call', 'expired',
                                    'permission', 'attempt', 'deadline', 'reads', 'mismatch', 'storage',
                                    'insufficient_reads', 'insufficient_chars', 'revoked', 'foreign', 'reader_blocked'])
def test_unreadable_original_falls_back_without_discarding_output(materials, failure):
    active, ledger, clock, allowed = materials
    result = envelope(long_output())
    if failure == 'storage':
        active.store.policy = replace(active.store.policy, max_source_bytes=100)
    call = retain(active, ledger, result)
    if failure == 'missing': call.pop('material_reference')
    elif failure == 'forged': call['material_reference']['reference'] = 'tm:' + 'f'*32
    elif failure == 'version': call['material_reference']['version'] += '-changed'
    elif failure == 'hash': call['material_reference']['content_sha256'] = 'f'*64
    elif failure == 'wrong_call': result.invocation_id = 'other'
    elif failure == 'expired': clock[0] += 4000
    elif failure == 'permission': allowed[0] = False
    elif failure == 'attempt': ledger.save({**active.task, 'execution_attempt': 2})
    elif failure == 'deadline': clock[0] += 301
    elif failure == 'reads':
        active.store.policy = replace(active.store.policy, max_reads=1)
        active.read(call['material_reference']['reference'], length=1)
    elif failure == 'mismatch': result.data['state_updates']['returncode'] = 9
    elif failure == 'insufficient_reads':
        active.store.policy = replace(active.store.policy, max_reads=1, page_chars=100)
    elif failure == 'insufficient_chars':
        active.store.policy = replace(active.store.policy, max_read_chars=100)
    elif failure == 'revoked': active.store.revoke(call['material_reference']['reference'], task=active.task)
    elif failure == 'foreign': active.task['session'] = 'other-session'
    elif failure == 'reader_blocked':
        from shinku.hosts.tool_host import ToolHost
        from shinku.tools.registry import ToolRegistry
        from shinku.tools.execution import ExecutionPolicy
        active.tool_host(ToolHost(ToolRegistry({})), set(), ExecutionPolicy(blocked=frozenset({'task_material_read'})))
    text = model_tool_content(result, call, active)
    assert result.model_feedback in text and text.count('MIDDLE_EVIDENCE') == 1
    assert 'output_preview' not in receipt(text)


@pytest.mark.parametrize('key', ['output_truncated', 'has_more'])
def test_upstream_partial_output_is_not_marked_complete(materials, key):
    active, ledger, _, _ = materials
    result = envelope(long_output(), **{key: True})
    call = retain(active, ledger, result)
    assert call['material_reference']['complete'] is False
    text = model_tool_content(result, call, active)
    assert receipt(text)['data']['state_updates'][key] is True
    assert not active.information()['sources'][0]['retained_complete']


@pytest.mark.parametrize('case', ['non_shell', 'different_feedback', 'unexecuted', 'material_read'])
def test_other_receipts_are_byte_identical(case):
    result = envelope()
    if case == 'non_shell': result.data['state_updates'].pop('execution_session')
    elif case == 'different_feedback': result.model_feedback = 'context does not match stdout block'
    elif case == 'unexecuted': result.executed = False
    elif case == 'material_read':
        result.model_feedback = json.dumps({'content': 'source text', 'provided': False})
        result.data = {'state_updates': {'material_read_receipt': {'reference': 'tm:fixture'}}}
    assert model_tool_content(result, {}) == result.model_feedback + '\n\n[host receipt]\n' + json.dumps(
        {'status': result.status, 'data': result.data}, ensure_ascii=False, default=str)


def native_call(ident='c1'):
    return {'role': 'assistant', 'content': '', 'reasoning_content': 'fixture reasoning', 'tool_calls': [
        {'id': ident, 'type': 'function', 'function': {'name': 'fixture_shell', 'arguments': '{}'}}]}


def test_native_history_appends_once_and_does_not_change_prior_prefix(materials):
    active, ledger, clock, _ = materials
    result = envelope(long_output())
    call = retain(active, ledger, result)
    state = AgentState(context={'agent_material_session': active})
    state.last_envelope, state.tool_calls = result, [call]
    history = NativeToolHistory('scope', 'goal', None, [])
    history.capture(native_call())
    first = history.request_kwargs(state, 'scope', 'goal')['native_history']
    assert 'output_preview' in first[1]['content']
    clock[0] += 4000
    history.capture(native_call('c2'))
    state.last_envelope = envelope('second')
    state.last_envelope.invocation_id = 'c2'
    state.tool_calls.append({'id': 'c2', 'name': 'fixture_shell', 'status': 'ok'})
    second = history.request_kwargs(state, 'scope', 'goal')['native_history']
    assert second[:2] == first and len(second) == 4
    assert history.request_kwargs(state, 'scope', 'goal')['native_history'] == second
    assert result.data['state_updates']['stdout'] == long_output()


def test_native_id_status_mismatch_still_rejected(materials):
    active, ledger, _, _ = materials
    result = envelope(long_output()); call = retain(active, ledger, result)
    state = AgentState(context={'agent_material_session': active}, tool_calls=[{**call, 'status': 'error'}])
    state.last_envelope = result
    history = NativeToolHistory('scope', 'goal', None, [])
    history.capture(native_call())
    with pytest.raises(ValueError, match='native_host_receipt_mismatch'):
        history.request_kwargs(state, 'scope', 'goal')


def test_many_results_keep_prefix_and_raw_sources_through_compaction(materials, tmp_path, monkeypatch, record_property):
    from shinku.llm.context_compaction import ContextCompactor
    from shinku.llm.native_messages import validate_native_messages
    from test_context_compaction import Summarizer, bundle, window
    active, ledger, _, _ = materials
    state = AgentState(context={'agent_material_session': active})
    history = NativeToolHistory('scope', '只诊断，不重跑命令', None, [])
    old_chars = 0; refs = []; previous = []
    for i in range(8):
        result = envelope(long_output(), 'stderr-' + str(i)); result.invocation_id = f'c{i}'
        call = retain(active, ledger, result); refs.append(call['material_reference']['reference'])
        state.last_envelope = result; state.tool_calls.append(call)
        history.capture(native_call(result.invocation_id))
        current = history.request_kwargs(state, 'scope', '只诊断，不重跑命令')['native_history']
        assert current[:len(previous)] == previous
        previous = copy.deepcopy(current)
        old_chars += len(result.model_feedback + '\n\n[host receipt]\n' + json.dumps(
            {'status': result.status, 'data': result.data}, ensure_ascii=False))
    new_chars = sum(len(m['content']) for m in current if m['role'] == 'tool')
    assert new_chars < old_chars // 2
    record_property('old_tool_chars', old_chars); record_property('new_tool_chars', new_chars)
    window(monkeypatch, 1000000)
    manager = ContextCompactor(Summarizer(), tmp_path/'compaction.sqlite', aware=True)
    payload = {'model': 'synthetic', 'messages': [{'role': 'system', 'content': '保持真实回执验收'},
        {'role': 'user', 'content': '只诊断，不重跑命令'}, *current]}
    original = copy.deepcopy(payload)
    manager.request('scope', '阶段结束，原日志按引用保留')
    projected = manager.project(payload, bundle(), 1, 'scope')
    assert projected != original and payload == original
    validate_native_messages(projected['messages'][1:])
    # 摘要会插在当前用户要求前；要求逐字保留，但不固定它的新索引。
    assert original['messages'][1] in projected['messages']
    assert projected['messages'][-2:] == original['messages'][-2:]
    for ref in refs:
        source = active.store.resolve(ref, task=active.task, still_authorized=lambda: True)
        assert source['state_updates']['stdout'] == long_output()
    saved = json.loads(manager.db.execute('SELECT payload FROM context_compactions').fetchone()[0])
    assert saved['original_payload'] == original
    manager.db.close()


@pytest.mark.parametrize('status', ['ok', 'error'])
def test_trimmed_feedback_and_large_stderr_keep_failure_state(materials, status):
    from shinku.tools.execution import result_to_envelope
    active, ledger, _, _ = materials
    result = envelope('standard output', long_output() + '\n', status=status)
    result = result_to_envelope(invocation=ToolInvocation('fixture_shell', id='c1'), result={
        'status': status, 'followup_context': result.model_feedback, 'state_updates': result.data['state_updates']})
    call = retain(active, ledger, result)
    view = receipt(model_tool_content(result, call, active))
    assert view['status'] == status
    assert 'stderr' in view['output_preview']['omitted_ranges']
    assert view['data']['state_updates']['returncode'] == (0 if status == 'ok' else 7)


def test_sdk_loop_receives_preview_then_reads_middle_without_reexecution(tmp_path, monkeypatch):
    """真实服务/SDK协议和材料读取；命令适配器是合成夹具，不证明模型决策质量。"""
    from shinku.agent_runtime.demo import demo_service
    from shinku.agent_runtime import AgentTurnRequest
    from shinku.angelos.host import ServiceHost
    from shinku.hosts.tool_host import ToolHost
    from shinku.tools.registry import ToolRegistry
    from shinku.tools.execution import ExecutionPolicy
    from test_agent_conversation_loop import native_sdk
    from test_agent_runtime_boundary import settled

    class Shell:
        tool_type = 'fixture_shell'
        agent_progress_class = 'change'
        agent_completion_effects = ('command_run',)
        agent_observation_receipt_fields = ('execution_session',)
        calls = 0
        def tool_metadata(self):
            return SimpleNamespace(description='Synthetic command result', input_schema={'type': 'object', 'properties': {}})
        def normalize_call(self, call): return call
        def execute(self, *, call, context):
            self.calls += 1
            value = envelope(long_output())
            return {'followup_context': value.model_feedback, 'state_updates': value.data['state_updates']}

    diagnostics = []
    original_lookup = TaskMaterialSession.retained_tool_output
    def traced_lookup(active, result, ref):
        try:
            value = original_lookup(active, result, ref)
            diagnostics.append({'projectable': value is not None, 'reference': ref,
                'local_items': len(active.task.get('material_package', {}).get('items', []))})
            return value
        except Exception as exc:
            diagnostics.append({'lookup_error': repr(exc)})
            raise
    monkeypatch.setattr(TaskMaterialSession, 'retained_tool_output', traced_lookup)
    service, _, _ = demo_service(tmp_path); shell = Shell()
    service.executor.tool_host = ToolHost(ToolRegistry({shell.tool_type: shell}))
    service.materials = TaskMaterialStore(service.task_store)
    old_authorize = service.authorize
    service.authorize = lambda *args: replace(old_authorize(*args), allowed_tools=frozenset({shell.tool_type}),
        policy=ExecutionPolicy(risk_by_tool={shell.tool_type: 'low'}), permission_mode='ultra')
    def respond(payload, count):
        if count == 1:
            name, args = shell.tool_type, {}
        elif count == 2:
            text = next(m['content'] for m in payload['messages'] if m['role'] == 'tool')
            assert 'MIDDLE_EVIDENCE' not in text, 'SDK received unprojected output'
            view = receipt(text)['output_preview']; span = view['omitted_ranges']['stdout']
            name, args = 'task_material_read', {'reference': view['material_reference']['reference'],
                'offset': span['offset'], 'length': min(span['length'], view['page_chars'])}
        else:
            assert count == 3 and shell.calls == 1, ('unexpected requests', count, shell.calls)
            latest = [m for m in payload['messages'] if m['role'] == 'tool'][-1]
            assert 'MIDDLE_EVIDENCE' in latest['content'], 'middle absent from material read'
            return {'role': 'assistant', 'content': json.dumps({'kind': 'final', 'task_outcome': 'completed',
                'text': '补读确认中段证据 MIDDLE_EVIDENCE；未重跑命令。'})}, 'stop'
        return {'role': 'assistant', 'content': '', 'reasoning_content': 'fixture reasoning', 'tool_calls': [
            {'id': 'sdk-' + str(count), 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}, 'tool_calls'
    errors = []
    def checked(payload, count):
        try:
            return respond(payload, count)
        except Exception as exc:
            errors.append((count, repr(exc)))
            raise
    try:
        with native_sdk(checked) as (runtime, calls):
            service.executor.profile = replace(service.executor.profile, runtime=runtime,
                runtime_meters_budget=True, native_tool_history=True)
            port = ServiceHost(service, requester='owner').port('one')
            handle = port.submit(AgentTurnRequest('shell-middle', '调查命令输出中的中段证据'))
            assert settled(port, handle.task_id)['status'] == 'completed', (diagnostics, errors)
            assert len(calls) == 3 and shell.calls == 1
            rows = [m for m in calls[1]['messages'] if m['role'] in {'assistant', 'tool'}]
            assert [m for m in calls[2]['messages'] if m['role'] in {'assistant', 'tool'}][:2] == rows
            task = service.task_store.get(handle.task_id, requester='owner', session='angelos:one')
            assert task['budget']['tools'] == 2
            records = service.materials.read_receipts(task)
            assert len(records) == 1 and records[0]['provided'] is True
            ref = task['tool_calls'][0]['material_reference']['reference']
            source = service.materials.resolve(ref, task=task, still_authorized=lambda: True)
            assert source['state_updates']['stdout'] == long_output()
    finally:
        service.coordinator.shutdown(wait=True); service.task_store.close()
