"""Actual SDK, host and local IO acceptance; no paid calls or private imports."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import httpx
import pytest


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    path = Path(__file__).resolve().parents[1] / 'scripts/verify_agent_chat_handoff.py'
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location('joint_sdk_probe', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    # 保留源码根及语料来源，只把本模块离线输出交给 pytest/批次临时目录管理。
    # 不在 teardown 删除，以便失败后从本轮批次目录检查真实 SDK/宿主回执。
    module.OUTPUT_ROOT = tmp_path_factory.mktemp('joint-artifacts')
    return module


@pytest.fixture(scope='module')
def batch(probe):
    parent = probe.OUTPUT_ROOT / ('joint-sdk-tests-' + uuid4().hex)
    return dict(workspace=parent / 'run', ledger=parent / 'budget.json',
                report=probe.run_suite(parent / 'run', parent / 'budget.json'))


def row(batch, slot):
    return next(item for item in batch['report']['rows'] if item['slot'] == slot)


def test_sdk_artifacts_use_fixture_root_without_relocating_source(probe, batch):
    """实际 SDK/文件链路输出离开仓库 tmp，源码身份和语料仍从真实检出读取。"""
    assert batch['workspace'].is_relative_to(probe.OUTPUT_ROOT)
    assert not probe.OUTPUT_ROOT.is_relative_to(probe.ROOT / 'tmp')
    assert probe.ROOT == Path(__file__).resolve().parents[1]
    assert probe.CORPUS == probe.ROOT / 'tests/fixtures/agent_chat_handoff_v1.json'


def test_fresh_driver_keeps_cli_output_boundary():
    """测试实例的覆盖不外溢到新的 CLI 模块，原路径限制仍然有效。"""
    path = Path(__file__).resolve().parents[1] / 'scripts/verify_agent_chat_handoff.py'
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location('joint_default_boundary', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    assert module.OUTPUT_ROOT == module.ROOT / 'tmp'


@pytest.mark.parametrize('slot', [
    'file_sync:task', 'attachment_async:task', 'progress_unfinished:off', 'progress_unfinished:on',
    'foreground_complete:off', 'foreground_complete:on', 'access_denied:off', 'access_denied:on',
    'ordinary_chat:off', 'ordinary_chat:on'])
def test_actual_sdk_host_chain_keeps_original_scope_and_replay_inert(batch, slot):
    item = row(batch, slot)
    assert item['passed'], item
    assert item['hidden_tools'] and item['expression_tools_absent']
    assert item['inputs_preserved'] and item['result_event_replay_inert'] and item['repeated_handoff_inert']
    assert item['platform_sends'] == item['native_history_mismatches'] == 0
    assert item['evidence_kind'].startswith('sdk_response_fixtures')
    assert item['publication_unit'] == 'transport_segment_attempt'


def test_verified_files_include_prerequisite_read_and_real_omitted_attachment_tail(batch):
    first, asynchronous = row(batch, 'file_sync:task'), row(batch, 'attachment_async:task')
    assert first['complete'] and asynchronous['complete']
    assert [read['path'] for read in first['file_reads']] == ['spec.txt']
    for name, expected in [('file_sync', b'ALPHA-47\n'), ('attachment_async', b'TAIL-73\n')]:
        root = batch['workspace'] / (name + '-task')
        assert (root / 'output.txt').read_bytes() == expected
        item = row(batch, name + ':task')
        assert len(item['file_writes']) == 1 and all(a['valid'] for a in item['artifacts'])
        result = json.loads((root / 'canonical-result.json').read_text(encoding='utf-8'))
        assert result['verification']['accepted'] is True
    assert set(json.loads((batch['workspace'] / 'file_sync-task/canonical-result.json').read_text(
        encoding='utf-8'))['verification']['satisfied']) == {'file', 'input'}
    assert [item['kind'] for item in asynchronous['batch_io']] == ['task_material_read', 'file_adapter']
    assert asynchronous['reply_status'] == 'waiting_background' and asynchronous['work_status'] == 'completed'
    import sqlite3
    with sqlite3.connect(batch['workspace'] / 'attachment_async-task/tasks.sqlite') as db:
        receipts = [json.loads(record[0]) for record in db.execute('SELECT payload FROM agent_task_material_reads')]
    assert len(receipts) == 1 and receipts[0]['offset'] > 0
    assert 'TAIL-73' in receipts[0]['content'] and receipts[0]['remaining_chars'] == 0


def test_foreground_receipts_survive_handoff_without_research_restart_or_budget_reset(batch):
    disabled, enabled = row(batch, 'progress_unfinished:off'), row(batch, 'progress_unfinished:on')
    assert not disabled['complete'] and not disabled['background_work_id'] and disabled['task_count'] == 0
    assert enabled['complete'] and enabled['task_count'] == 1 and enabled['background_work_id']
    assert enabled['source_searches'] == 1
    assert enabled['source_reads'] == ['https://query.example/a', 'https://query.example/b', 'https://query.example/c']
    task = enabled['tasks'][0]
    before = task['foreground_query']['budget_at_transfer']
    assert before['tools'] == 2 and task['budget']['tools'] == 4
    assert task['budget']['requests'] > before['requests'] > 0
    for key in ('max_requests', 'max_tokens', 'max_tools', 'max_active_seconds'):
        assert before[key] == task['budget'][key] == batch['report']['selection']['task_limits'][key]
    assert task['foreground_query']['remaining'] and len(task['foreground_query']['receipt_ids']) == 2
    assert len(enabled['source_reads']) == len(set(enabled['source_reads']))


@pytest.mark.parametrize('answer, accepted', [
    ('甲店：10 元 https://query.example/a；乙店：20 元 https://query.example/b；丙店：30 元 https://query.example/c', True),
    ('丙店价格是30元（https://query.example/c）。乙店价格是20元（https://query.example/b）。甲店10元最低（https://query.example/a）。', True),
    ('甲店20元 https://query.example/a；乙店10元 https://query.example/b；丙店30元 https://query.example/c', False),
    ('甲店100元 https://query.example/a；乙店20元 https://query.example/b；丙店30元 https://query.example/c', False),
    ('甲店10美元 https://query.example/a；乙店20元 https://query.example/b；丙店30元 https://query.example/c', False),
    ('甲店10元；乙店20元 https://query.example/b；丙店30元 https://query.example/c', False)])
def test_price_oracle_preserves_numeric_meaning_and_source_without_requiring_exact_wording(probe, answer, accepted):
    case = next(item for item in probe.corpus()['cases'] if item['id'] == 'progress_unfinished')
    assert probe.original_prices_preserved(case, answer) is accepted


@pytest.mark.parametrize('case', ['foreground_complete', 'access_denied', 'ordinary_chat'])
def test_completed_blocked_and_chat_requests_never_spawn_continuation(batch, case):
    for arm in ('off', 'on'):
        item = row(batch, case + ':' + arm)
        assert item['task_count'] == 0 and item['background_work_id'] == ''
        assert item['complete'] is (case != 'access_denied')
        if case == 'ordinary_chat':
            assert not item['batch_io'] and item['expression_preparations'] == 0
        if case == 'access_denied':
            assert item['source_searches'] == 1 and item['source_reads'] == ['https://query.example/a']


@pytest.mark.parametrize('case', [
    'completed', 'partial', 'failed', 'canceled', 'unverified', 'waiting_user', 'long_material', 'untrusted_source'])
def test_separate_expression_and_neutral_check_preserve_all_bound_facts(probe, batch, case):
    item = row(batch, case + ':expression')
    assert item['passed'] and item['expression_status'] == 'ready'
    assert item['required_facts_present'] and item['canonical_unchanged'] and item['replay_inert']
    assert item['platform_sends'] == item['task_sdk_calls'] == 0
    assert item['quality_review'] == 'fixture_not_provider_quality'
    assert item['evidence_kind'] == 'sdk_fixture_expression_of_controlled_canonical_fixture'
    root = batch['workspace'] / (case + '-expression')
    result = json.loads((root / 'canonical-result.json').read_text(encoding='utf-8'))
    assert result['execution']['status'] == next(r['status'] for r in probe.corpus()['expression_cases'] if r['id'] == case)
    assert result['delivery']['sent'] is False and result['deployment']['status'] == 'not_deployed'
    assert result['execution_report']['is_fact_evidence'] is False
    assert '12.5' in item['text'] and 'ms' in item['text']


def test_independent_journal_matches_actual_sdk_and_all_adapter_dispatches(probe, batch):
    saved = probe.inspect_ledger(batch['ledger'])
    report = batch['report']
    assert report['all_selected_passed'] and report['selected_slots'] == 18 and report['unrun_slots'] == 0
    assert report['provider_requests'] == report['platform_sends'] == 0
    assert report['provider_cost'] is None and report['reported_tokens_evidence'] == 'fixture'
    assert report['role_attempts'] == {'chat': 27, 'execution': 15, 'expression': 16, 'check': 16}
    assert len(saved['attempts']) == saved['budget']['requests'] == 74
    assert len(saved['io']) == saved['budget']['tools'] == 19
    assert sum(item['kind'] == 'task_material_read' for item in saved['io']) == 1
    assert len(saved['slots']) == 18 and set(saved['slots'].values()) == {'finished'}
    assert all(not item.get('native_tools') for item in saved['attempts'] if item['role'] in {'expression', 'check'})
    assert saved['identity']['code_identity'] == report['selection']['code_identity']
    for path in batch['workspace'].rglob('*'):
        if path.is_file():
            assert probe.INTERNAL_SENTINEL.encode() not in path.read_bytes(), path
    assert probe.INTERNAL_SENTINEL not in batch['ledger'].read_text(encoding='utf-8')


def new_ledger(probe, tmp_path, **kwargs):
    saved = probe.JointLedger(tmp_path / 'journal.json', {'mode': 'offline', 'slots': ['fixed:on']}, **kwargs)
    saved.claim('fixed:on')
    return saved


def sdk_call(probe, runtime, role='chat'):
    with runtime.ledger.role(role):
        return runtime.sdk.chat.completions.create(model=probe.CHAT_MODEL,
            messages=[dict(role='user', content='one isolated fixture')], max_tokens=16)


def observing_runtime(probe, ledger, observed):
    def provider(request):
        # Assert the reservation was durable before HTTP or any provider work.
        saved = probe.inspect_ledger(ledger.path)
        assert saved['budget']['requests'] == len(saved['attempts'])
        assert saved['attempts'][-1]['state'] == 'reserved'
        observed.append(json.loads(request.content))
        return probe.json_response(observed[-1], {'ok': True})
    return probe.ChatSDKRuntime(ledger, 'offline-fixture-key', live=False,
        forward_transport=httpx.MockTransport(provider))


def test_sdk_reservation_precedes_forwarding_and_one_role_cannot_borrow_another(probe, tmp_path):
    ledger = new_ledger(probe, tmp_path, role_limits=dict(chat=1, execution=1, expression=1, check=1))
    observed = []
    runtime = observing_runtime(probe, ledger, observed)
    try:
        sdk_call(probe, runtime)
        with pytest.raises(probe.TaskBudgetExceeded, match='role_budget'):
            sdk_call(probe, runtime)
        sdk_call(probe, runtime, 'expression')
        assert len(observed) == ledger.budget.requests == 2
        assert [item['role'] for item in ledger.attempts] == ['chat', 'expression']
    finally:
        runtime.close()


@pytest.mark.parametrize('limit, expected', [
    ({'max_requests': 1}, 'model_budget'), ({'max_tokens': 10}, 'token_budget'),
    ({'max_active_seconds': .001}, 'time_budget')])
def test_batch_budget_stops_before_next_actual_sdk_request(probe, tmp_path, limit, expected):
    ledger = new_ledger(probe, tmp_path, limits={**probe.LIMITS, **limit})
    observed = []
    runtime = observing_runtime(probe, ledger, observed)
    try:
        if 'max_requests' in limit:
            sdk_call(probe, runtime)
        if 'max_active_seconds' in limit:
            ledger.budget.active_seconds = .002
        with pytest.raises(probe.TaskBudgetExceeded, match=expected):
            sdk_call(probe, runtime)
        assert len(observed) == (1 if 'max_requests' in limit else 0)
    finally:
        runtime.close()


@pytest.mark.parametrize('failure', ['reservation_save', 'tool_save'])
def test_failed_durable_reservation_prevents_provider_or_file_effect(probe, tmp_path, monkeypatch, failure):
    ledger = new_ledger(probe, tmp_path)
    observed = []
    runtime = observing_runtime(probe, ledger, observed)
    def denied():
        raise PermissionError('isolated journal failure')
    monkeypatch.setattr(ledger, 'save', denied)
    try:
        if failure == 'reservation_save':
            with pytest.raises(PermissionError):
                sdk_call(probe, runtime)
            assert ledger.budget.requests == 1
        else:
            writer = probe.benchmark.FileAdapter(tmp_path.resolve(), dict(initial={}, expected={'output.txt': 'ok'}), 'write', ledger)
            with pytest.raises(PermissionError):
                writer.execute(call=dict(path=str(tmp_path / 'output.txt'), content='ok'), context={})
            assert not (tmp_path / 'output.txt').exists() and not writer.calls and ledger.budget.tools == 1
        assert observed == []
    finally:
        runtime.close()


def test_material_dispatch_cannot_read_after_shared_io_budget_exhaustion(probe, tmp_path, monkeypatch):
    ledger = new_ledger(probe, tmp_path, limits={**probe.LIMITS, 'max_tools': 1})
    ledger.use_tool('file_adapter')
    store = probe.AgentTaskStore(tmp_path / 'task.sqlite')
    material = probe.MeteredMaterialStore(store, approved_roots=(tmp_path,), ledger=ledger)
    touched = []
    monkeypatch.setattr(probe.TaskMaterialStore, 'read', lambda *a, **k: touched.append(True))
    try:
        with pytest.raises(probe.TaskBudgetExceeded, match='tool_budget'):
            material.read('fixture', task={}, still_authorized=lambda: True)
        assert touched == [] and ledger.budget.tools == len(ledger.io) == 1
    finally:
        store.close()


@pytest.mark.parametrize('failures, winerror', [(2, 5), (5, 5), (1, 32)])
def test_atomic_journal_replace_retry_is_bounded_and_only_for_windows_access_denied(
        probe, tmp_path, monkeypatch, failures, winerror):
    import agent_chat_handoff_support as support
    ledger = new_ledger(probe, tmp_path)
    observed, calls, delays = [], [], []
    runtime = observing_runtime(probe, ledger, observed)
    original = support.os.replace
    def replace(source, destination):
        calls.append(True)
        if len(calls) <= failures:
            error = PermissionError('isolated replace failure')
            error.winerror = winerror
            raise error
        return original(source, destination)
    monkeypatch.setattr(support.os, 'replace', replace)
    monkeypatch.setattr(support.time, 'sleep', delays.append)
    try:
        if failures < 5 and winerror == 5:
            sdk_call(probe, runtime)
            assert len(observed) == ledger.budget.requests == 1
            saved = probe.inspect_ledger(ledger.path)
            assert len(saved['journal_write_retries']) == 2 and len(delays) == 2
        else:
            with pytest.raises(PermissionError):
                sdk_call(probe, runtime)
            assert observed == [] and len(calls) == (5 if winerror == 5 else 1)
            assert len(delays) == (4 if winerror == 5 else 0)
        assert delays == [.02 * 2 ** n for n in range(len(delays))]
    finally:
        runtime.close()


def test_finished_slot_interrupted_journal_and_exclusive_lock_never_start_a_repeated_batch(probe, tmp_path):
    ledger = new_ledger(probe, tmp_path)
    with pytest.raises(ValueError, match='already_exists'):
        probe.JointLedger(ledger.path, ledger.identity)
    with probe.acceptance.exclusive_ledger(ledger.path):
        with pytest.raises(FileExistsError):
            with probe.acceptance.exclusive_ledger(ledger.path):
                pytest.fail('duplicate lock admitted')
    ledger.finish('fixed:on')
    with pytest.raises(ValueError, match='claimed'):
        ledger.claim('fixed:on')
    with pytest.raises(ValueError, match='active_slot'):
        ledger.use_tool()
    with ledger.role('chat'), pytest.raises(ValueError, match='active_slot'):
        ledger.budget.reserve_request(dict(messages=[], max_tokens=1))


@pytest.mark.parametrize('mutation', [
    lambda v: v['budget'].update(requests=True), lambda v: v['budget'].update(max_requests=177),
    lambda v: v['slots'].update({'unapproved': 'finished'}),
    lambda v: v.update(io=[{'index': 1, 'slot': 'fixed:on', 'kind': 'task_material_read'}]),
    lambda v: v.update(journal_write_retries=[{'slot': 'fixed:on', 'winerror': 5, 'delay_seconds': .32}]),
    lambda v: v['identity'].update(slots=['fixed:on', 'fixed:on'])])
def test_inspection_rejects_corrupt_counts_expanded_limits_and_unapproved_metadata(probe, tmp_path, mutation):
    ledger = new_ledger(probe, tmp_path)
    saved = json.loads(ledger.path.read_text(encoding='utf-8'))
    mutation(saved)
    ledger.path.write_text(json.dumps(saved), encoding='utf-8')
    with pytest.raises(ValueError):
        probe.inspect_ledger(ledger.path)
    with pytest.raises(ValueError, match='another_writer'):
        ledger.save()


@pytest.mark.parametrize('fault', ['neutral_rejection', 'expression_timeout', 'native_tool_response', 'false_binding'])
def test_expression_failure_falls_back_without_task_calls_or_repeated_model_attempts(probe, tmp_path, fault):
    ledger = new_ledger(probe, tmp_path)
    case = {'id': 'isolated-expression-fault', 'status': 'failed', 'accepted': False}
    base = probe.FixtureProvider(case, 'expression', tmp_path / 'expression')
    observed = []
    def provider(request):
        payload = json.loads(request.content)
        observed.append(payload)
        if fault == 'expression_timeout':
            raise httpx.ReadTimeout('controlled timeout', request=request)
        if fault == 'native_tool_response':
            return probe.json_response(payload, tool='benchmark_write', arguments={'path': 'outside.txt', 'content': 'bad'})
        if fault == 'false_binding':
            return probe.json_response(payload, {'binding_hash': 'wrong', 'layout': [{'text': '已上线并发送'}]})
        user = payload['messages'][-1]['content']
        data = json.loads(user)
        if 'candidate_text' in data:
            return probe.json_response(payload, dict(binding_hash=data['binding_hash'], supported=False,
                covered_fact_ids=[], additional_claims=['已上线'], contradictions=['fixture'], omitted_limitations=[]))
        return base(request)
    root = tmp_path / 'expression'
    root.mkdir()
    store = probe.AgentTaskStore(root / 'tasks.sqlite')
    runtime = probe.ChatSDKRuntime(ledger, 'offline-fixture-key', live=False, forward_transport=httpx.MockTransport(provider))
    service = probe.ExpressionService(probe.ExpressionArchive(store), probe.profile(runtime))
    source = probe.canonical_expression(case, root)
    try:
        record = service.prepare(source, scope=probe.SCOPE, check_current=lambda: '')
        before = len(observed)
        repeated = service.prepare(source, scope=probe.SCOPE, check_current=lambda: '')
        assert record['status'] == 'fallback' and record['fallback_reason']
        assert repeated['expression_id'] == record['expression_id'] and len(observed) == before
        assert all(item['role'] in {'expression', 'check'} for item in ledger.attempts)
        assert ledger.io == [] and not any(p.get('tools') for p in observed)
        assert not (root / 'outside.txt').exists()
        assert all(b['text'] in record['text'] for b in probe.fact_blocks(source) if b.get('required', True))
        assert record['delivery'].get('sent') is not True
    finally:
        runtime.close()
        store.close()


def test_sdk_http_failures_reserve_each_attempt_and_keep_unknown_usage(probe, tmp_path):
    ledger = new_ledger(probe, tmp_path)
    observed = []
    def unavailable(request):
        observed.append(True)
        return httpx.Response(429, json={'error': {'message': 'controlled quota', 'type': 'rate_limit'}})
    runtime = probe.ChatSDKRuntime(ledger, 'offline-fixture-key', live=False, forward_transport=httpx.MockTransport(unavailable))
    try:
        fallback = {'kind': 'final', 'text': 'controlled HTTP failure'}
        response = runtime.call_chat_json(system_prompt='Return a JSON value.', user_prompt='isolated', fallback=fallback)
        assert response == fallback and len(observed) == ledger.budget.requests == 2
        assert ledger.budget.estimated_requests == 2 and ledger.budget.tokens > 0
        assert all(item['state'] == 'error' and item['sdk_usage'] is None for item in ledger.attempts)
        assert ledger.budget.reported_tokens == 0 and not ledger.io
    finally:
        runtime.close()


def test_joint_oracle_does_not_pass_when_chat_provider_never_responds(probe, tmp_path):
    ledger = new_ledger(probe, tmp_path)
    case = next(item for item in probe.corpus()['cases'] if item['kind'] == 'chat')
    def unavailable(request):
        return httpx.Response(429, json={'error': {'message': 'controlled HTTP failure', 'type': 'rate_limit'}})
    item = probe.run_joint(case, 'off', tmp_path / 'ordinary-chat', ledger,
        ('offline-fixture-key', 'offline-fixture-key'), live=False, fixture=unavailable)
    assert not item['passed'] and not item['complete'] and item['failures']
    assert item['task_count'] == 0 and item['source_reads'] == [] and item['file_writes'] == []
    assert ledger.budget.requests == len(ledger.attempts) == 4
    assert all(a['state'] == 'error' and a['sdk_usage'] is None for a in ledger.attempts)
    assert not ledger.io and item['platform_sends'] == 0


@pytest.mark.parametrize('case_id', ['file_sync', 'attachment_async'])
def test_generated_output_can_be_read_back_before_original_completion(probe, tmp_path, case_id):
    case = next(item for item in probe.corpus()['cases'] if item['id'] == case_id)
    root = tmp_path / case_id
    ledger = new_ledger(probe, tmp_path)

    class ReadbackProvider(probe.FixtureProvider):
        def __call__(self, request):
            payload = json.loads(request.content)
            names = [row['function']['name'] for row in payload.get('tools', [])]
            if (payload['model'] == probe.EXECUTION_MODEL and 'benchmark_read' in names
                    and self.host.writer.calls and not any(row['path'] == 'output.txt' for row in self.host.reader.calls)):
                return probe.json_response(payload, tool='benchmark_read',
                    arguments={'path': str(root / 'output.txt')}, reasoning=True)
            return super().__call__(request)

    item = probe.run_joint(case, 'task', root, ledger, ('offline-fixture-key', 'offline-fixture-key'),
                           live=False, fixture=ReadbackProvider(case, 'task', root))
    assert item['passed'] and item['complete'], item['failures']
    assert len(item['file_writes']) == 1 and item['artifacts'][0]['valid']
    assert any(read['path'] == 'output.txt' for read in item['file_reads'])
    assert (root / 'output.txt').read_bytes() == case['expected']['output.txt'].encode()
    assert all((root / name).read_bytes() == text.encode() for name, text in case['initial'].items())
    reader = probe.CompleteFileReader(root, case, 'read')
    assert reader.normalize_call({'path': str(root / 'unapproved.txt')}) is None
    assert reader.normalize_call({'path': str(tmp_path / 'outside.txt')}) is None
    assert item['result_event_replay_inert'] and item['platform_sends'] == 0


@pytest.mark.parametrize('fault', ['http_failure', 'invalid_schema'])
def test_unfinished_query_requires_actual_source_chain_even_when_incomplete_is_expected(probe, tmp_path, fault):
    case = next(item for item in probe.corpus()['cases'] if item['id'] == 'progress_unfinished')
    ledger = new_ledger(probe, tmp_path)

    def unavailable(request):
        if fault == 'http_failure':
            return httpx.Response(429, json={'error': {'message': 'controlled HTTP failure', 'type': 'rate_limit'}})
        return probe.json_response(json.loads(request.content), {'unexpected_fixture_schema': True})

    item = probe.run_joint(case, 'off', tmp_path / 'query', ledger,
        ('offline-fixture-key', 'offline-fixture-key'), live=False, fixture=unavailable)
    assert not item['passed'], 'An unexercised negative case cannot be accepted.'
    assert 'intended_query_chain_not_observed' in item['failures']
    assert not item['query_chain_observed'] and item['source_reads'] == []
    assert item['task_count'] == 0 and not ledger.io and item['platform_sends'] == 0


def test_empty_source_slices_do_not_exercise_the_read_price_comparison_case(probe, tmp_path):
    case = next(item for item in probe.corpus()['cases'] if item['id'] == 'progress_unfinished')
    root = tmp_path / 'empty-query'
    ledger = new_ledger(probe, tmp_path)

    class EmptyReadProvider(probe.FixtureProvider):
        def __call__(self, request):
            response = super().__call__(request)
            body = response.json()
            calls = body['choices'][0]['message'].get('tool_calls', [])
            for call in calls:
                if call['function']['name'] == 'query_read':
                    args = json.loads(call['function']['arguments'])
                    call['function']['arguments'] = json.dumps({**args, 'offset': 999999})
            return httpx.Response(200, json=body)

    item = probe.run_joint(case, 'off', root, ledger, ('offline-fixture-key', 'offline-fixture-key'),
                           live=False, fixture=EmptyReadProvider(case, 'off', root))
    assert not item['passed'] and 'intended_query_chain_not_observed' in item['failures']
    assert item['source_reads'] and not item['meaningful_source_reads'] and not item['query_chain_observed']
    assert all(read['content_chars'] == 0 for read in item['source_read_observations'])
    assert item['platform_sends'] == 0


@pytest.mark.parametrize('shape', ['markdown_fence', 'length', 'unexpected_keys', 'native_tool'])
def test_response_diagnostics_preserve_sdk_shape_without_provider_content_or_reasoning(probe, tmp_path, shape):
    secret = 'PRIVATE_PROVIDER_RESPONSE_DO_NOT_STORE'
    ledger = new_ledger(probe, tmp_path)

    def provider(request):
        payload = json.loads(request.content)
        content = ('```json\n{"free_segments": []}\n```' if shape == 'markdown_fence' else
                   '{"free_segments": [' if shape == 'length' else
                   json.dumps({'unexpected_private_key': secret}) if shape == 'unexpected_keys' else None)
        message = {'role': 'assistant', 'content': content, 'reasoning_content': probe.INTERNAL_SENTINEL}
        if shape == 'native_tool':
            message['tool_calls'] = [{'id': 'untrusted-fixture', 'type': 'function',
                'function': {'name': secret, 'arguments': json.dumps({'body': secret})}}]
        return httpx.Response(200, json={'id': 'fixture-shape', 'object': 'chat.completion', 'created': 1,
            'model': payload['model'], 'choices': [{'index': 0, 'message': message,
                'finish_reason': 'length' if shape == 'length' else 'tool_calls' if shape == 'native_tool' else 'stop'}],
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})

    runtime = probe.ChatSDKRuntime(ledger, 'offline-fixture-key', live=False, forward_transport=httpx.MockTransport(provider))
    try:
        response = sdk_call(probe, runtime)
        assert response.choices and not ledger.io
        item = probe.inspect_ledger(ledger.path)['attempts'][0]['response_metadata']['choices'][0]
        assert item['json_wrapping'] == ('markdown_fence' if shape == 'markdown_fence' else 'none' if shape == 'native_tool' else 'plain')
        assert item['json_shape'] == ('invalid' if shape == 'length' else 'empty' if shape == 'native_tool' else 'object')
        assert item['native_tool_calls'] == (1 if shape == 'native_tool' else 0)
        if shape == 'length':
            assert item['finish_reason'] == 'length'
        if shape == 'unexpected_keys':
            assert item['recognised_keys'] == []
        text = ledger.path.read_text(encoding='utf-8')
        assert secret not in text and 'unexpected_private_key' not in text and probe.INTERNAL_SENTINEL not in text
        assert ledger.budget.requests == 1 and ledger.budget.reported_tokens == 2
    finally:
        runtime.close()


def test_connection_failure_cause_types_do_not_store_exception_messages(probe, tmp_path):
    ledger = new_ledger(probe, tmp_path)
    secret = 'PRIVATE_EXCEPTION_BODY_DO_NOT_STORE'

    def disconnected(request):
        try:
            raise RuntimeError(secret)
        except RuntimeError as cause:
            raise httpx.ConnectError(secret, request=request) from cause

    runtime = probe.ChatSDKRuntime(ledger, 'offline-fixture-key', live=False,
        forward_transport=httpx.MockTransport(disconnected))
    try:
        with pytest.raises(Exception) as failure:
            sdk_call(probe, runtime)
        assert type(failure.value).__name__ == 'APIConnectionError'
        item = probe.inspect_ledger(ledger.path)['attempts'][0]
        assert item['error_cause_types'] == ['APIConnectionError', 'ConnectError', 'RuntimeError']
        assert item['state'] == 'error' and item['sdk_usage'] is None and secret not in ledger.path.read_text()
        assert ledger.budget.requests == ledger.budget.estimated_requests == 1 and not ledger.io
    finally:
        runtime.close()


@pytest.mark.parametrize('corruption', ['raw_body', 'unexpected_key', 'invalid_finish_type'])
def test_journal_inspection_rejects_unsafe_response_diagnostic_fields(probe, tmp_path, corruption):
    ledger = new_ledger(probe, tmp_path)
    runtime = observing_runtime(probe, ledger, [])
    try:
        sdk_call(probe, runtime)
    finally:
        runtime.close()
    value = json.loads(ledger.path.read_text())
    choice = value['attempts'][0]['response_metadata']['choices'][0]
    if corruption == 'raw_body':
        choice['raw_content'] = 'A provider body must not enter response metadata.'
    elif corruption == 'unexpected_key':
        choice['recognised_keys'] = ['unapproved_private_field']
    else:
        choice['finish_reason'] = {}
    ledger.path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='attempt_metadata_invalid'):
        probe.inspect_ledger(ledger.path)


def test_expression_diagnostic_preview_limits_model_roles_and_frozen_cases(probe):
    value = probe.preview(expression_diagnostic=True)
    assert set(value['models']) == {'chat_expression_check'}
    assert value['slots'] == ['completed:expression', 'failed:expression', 'untrusted_source:expression']
    assert value['batch_limits']['max_requests'] == 6 and value['batch_limits']['max_tokens'] == 150000
    assert value['batch_limits']['max_active_seconds'] == 210 and value['effective_adapter_limit'] == 0
    assert value['permitted_roles'] == ['expression', 'check'] and value['adapter_dispatch_allowed'] is False
    assert value['expression_output_limit'] == 2048 and value['selection_hash'] != probe.preview()['selection_hash']
    with pytest.raises(ValueError, match='diagnostic_limits_expanded'):
        probe.preview({**probe.DIAGNOSTIC_LIMITS, 'max_requests': 7}, expression_diagnostic=True)


def test_expression_diagnostic_runs_three_original_sdk_samples_without_task_or_adapter_dispatch(probe):
    parent = probe.OUTPUT_ROOT / ('expression-diagnostic-tests-' + uuid4().hex)
    report = probe.run_suite(parent / 'run', parent / 'budget.json', expression_diagnostic=True)
    assert report['all_selected_passed'] and report['selected_slots'] == 3 and report['unrun_slots'] == 0
    assert report['role_attempts'] == {'chat': 0, 'execution': 0, 'expression': 3, 'check': 3}
    saved = probe.inspect_ledger(parent / 'budget.json')
    assert saved['budget']['requests'] == 6 and saved['budget']['tools'] == 0 and not saved['io']
    assert report['provider_requests'] == report['platform_sends'] == 0
    for item in report['rows']:
        assert item['task_sdk_calls'] == 0 and item['required_facts_present'] and item['replay_inert']
        assert item['evidence_kind'] == 'sdk_fixture_expression_of_controlled_canonical_fixture'
    assert all('response_metadata' in attempt for attempt in saved['attempts'])


@pytest.mark.parametrize('role', ['chat', 'execution'])
def test_expression_diagnostic_denies_unused_sdk_roles_before_reservation(probe, tmp_path, role):
    selection = probe.preview(expression_diagnostic=True)
    ledger = probe.JointLedger(tmp_path / 'journal.json', probe.ledger_identity(selection, 'offline'),
        selection['batch_limits'], selection['role_limits'])
    ledger.claim(selection['slots'][0])
    observed = []
    runtime = observing_runtime(probe, ledger, observed)
    try:
        with pytest.raises(ValueError, match='sdk_role_unbound'):
            sdk_call(probe, runtime, role=role)
        with pytest.raises(ValueError, match='adapter_dispatch_forbidden'):
            ledger.use_tool()
        assert not observed and ledger.budget.requests == ledger.budget.tools == 0
        assert probe.inspect_ledger(ledger.path)['attempts'] == []
    finally:
        runtime.close()


def test_expression_diagnostic_credential_selection_does_not_require_execution_key(probe, tmp_path):
    path = tmp_path / 'missing-execution-key.env'
    path.write_text('SHINKU_CHAT_MODEL_NAME=' + probe.CHAT_MODEL + '\nSHINKU_CHAT_BASE_URL=' + probe.CHAT_ENDPOINT
        + '\nSHINKU_CHAT_API_PROTOCOL=openai\nSHINKU_CHAT_API_KEY=offline-fixture-key\n', encoding='utf-8')
    assert probe.read_keys(path, execution_required=False) == ('offline-fixture-key', '')
    with pytest.raises(ValueError, match='credentials_required'):
        probe.read_keys(path)


@pytest.mark.parametrize('corruption', ['role_policy', 'adapter_policy', 'attempt_role'])
def test_expression_diagnostic_journal_keeps_declared_dispatch_denials(probe, tmp_path, corruption):
    selection = probe.preview(expression_diagnostic=True)
    ledger = probe.JointLedger(tmp_path / 'journal.json', probe.ledger_identity(selection, 'offline'),
        selection['batch_limits'], selection['role_limits'])
    ledger.claim(selection['slots'][0])
    runtime = observing_runtime(probe, ledger, [])
    try:
        sdk_call(probe, runtime, role='expression')
    finally:
        runtime.close()
    value = json.loads(ledger.path.read_text())
    if corruption == 'role_policy':
        value['identity']['permitted_roles'] = 'expression'
    elif corruption == 'adapter_policy':
        value['identity']['adapter_dispatch_allowed'] = 'false'
    else:
        value['attempts'][0]['role'] = 'chat'
    ledger.path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='dispatch_policy_invalid|slots_or_roles_invalid'):
        probe.inspect_ledger(ledger.path)


@pytest.mark.parametrize('role, changes', [
    ('expression', {'tools': [{'type': 'function', 'function': {'name': 'benchmark_write', 'parameters': {'type': 'object'}}}]}),
    ('check', {'tool_choice': 'auto'}), ('chat', {'model': 'unapproved-model'}),
    ('chat', {'stream': True}), ('execution', {})])
def test_chat_transport_rejects_wrong_model_tools_stream_and_role_before_forwarding(probe, tmp_path, role, changes):
    ledger = new_ledger(probe, tmp_path)
    observed = []
    runtime = observing_runtime(probe, ledger, observed)
    try:
        with ledger.role(role), pytest.raises(ValueError, match='payload_or_role'):
            runtime.sdk.chat.completions.create(**{**dict(model=probe.CHAT_MODEL,
                messages=[dict(role='user', content='isolated')], max_tokens=16), **changes})
        assert not observed and ledger.budget.requests == 1
        assert ledger.attempts[0]['state'] == 'error' and ledger.attempts[0]['forwarded'] is False
    finally:
        runtime.close()


def test_live_candidate_configuration_requires_both_keys_and_exact_selected_endpoint(probe, tmp_path):
    path = tmp_path / 'candidate.env'
    settings = dict(SHINKU_CHAT_MODEL_NAME=probe.CHAT_MODEL, SHINKU_CHAT_BASE_URL=probe.CHAT_ENDPOINT,
        SHINKU_CHAT_API_PROTOCOL='openai', SHINKU_CHAT_API_KEY='isolated-fixture-chat-secret',
        SHINKU_AGENT_API_KEY='isolated-fixture-execution-secret')
    def write(values):
        path.write_text('\n'.join(key + '=' + value for key, value in values.items()), encoding='utf-8')
    write(settings)
    assert probe.read_keys(path) == ('isolated-fixture-chat-secret', 'isolated-fixture-execution-secret')
    write({**settings, 'SHINKU_CHAT_MODEL_NAME': 'different'})
    with pytest.raises(ValueError, match='candidate_changed'):
        probe.read_keys(path)
    write({**settings, 'SHINKU_CHAT_BASE_URL': 'https://different.example/v1'})
    with pytest.raises(ValueError, match='candidate_changed'):
        probe.read_keys(path)
    write({**settings, 'SHINKU_AGENT_API_KEY': ''})
    with pytest.raises(ValueError, match='credentials_required'):
        probe.read_keys(path)


def test_frozen_corpus_code_selection_and_paths_require_new_preview(probe, tmp_path, monkeypatch):
    selection = probe.preview()
    assert selection['selection_hash'] == probe.canonical_hash({k: v for k, v in selection.items() if k != 'selection_hash'})
    assert selection['live_authorized'] is False and selection['platform_sends'] == 0
    assert selection['code_identity']['source_files'] > 100 and len(selection['code_identity']['sha256']) == 64
    assert probe.preview()['code_identity'] == selection['code_identity']
    original = probe.corpus()
    changed = copy.deepcopy(original)
    changed['cases'][0]['request'] += ' changed'
    candidate = tmp_path / 'corpus.json'
    candidate.write_text(json.dumps(changed), encoding='utf-8')
    monkeypatch.setattr(probe, 'CORPUS', candidate)
    with pytest.raises(ValueError, match='corpus_changed'):
        probe.preview()
    for limits in ({}, {**probe.LIMITS, 'max_tools': 81}, {**probe.LIMITS, 'max_tokens': True}):
        with pytest.raises(ValueError):
            probe.validate_limits(limits)
    with pytest.raises(ValueError, match='fresh_workspace'):
        probe.validate_paths(tmp_path / 'run', tmp_path / 'budget.json')


@pytest.mark.parametrize('hash_value', [None, 'wrong'])
def test_live_missing_or_stale_review_fails_before_credentials_or_state(probe, tmp_path, monkeypatch, hash_value):
    touched = []
    monkeypatch.setattr(probe, 'read_keys', lambda path: touched.append(path))
    with pytest.raises(ValueError, match='selection_hash'):
        probe.run_suite(tmp_path / 'run', tmp_path / 'budget.json', live=True,
                        env_file=tmp_path / 'unused.env', expected_selection_hash=hash_value)
    assert not touched and not (tmp_path / 'run').exists() and not (tmp_path / 'budget.json').exists()


def test_live_code_drift_rejects_previously_reviewed_selection(probe, tmp_path, monkeypatch):
    reviewed = probe.preview()['selection_hash']
    monkeypatch.setattr(probe, 'code_identity', lambda: dict(sha256='changed', source_files=0))
    with pytest.raises(ValueError, match='selection_hash'):
        probe.run_suite(tmp_path / 'run', tmp_path / 'budget.json', live=True,
                        expected_selection_hash=reviewed, env_file=tmp_path / 'unused.env')
    assert not (tmp_path / 'budget.json').exists()


@pytest.mark.parametrize('preview_args', [[], ['--expression-diagnostic', '--expression-output-tokens', '4096'],
    ['--expression-diagnostic', '--expression-output-tokens', '4096', '--expression-timeout-seconds', '60',
     '--expression-http-timeout-seconds', '55']], ids=['joint_default', 'diagnostic_4096', 'diagnostic_slow'])
def test_default_preview_reads_no_credentials_opens_no_state_and_cannot_connect(probe, preview_args):
    code = r'''
import json, pathlib, runpy, socket, sys
script = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(script.parent))
def no_network(*a, **k): raise AssertionError('preview network')
socket.socket.connect = socket.socket.connect_ex = no_network
def guard(event, args):
    if event == 'open':
        path, mode, flags = args
        if str(path).endswith(('.env', '.env.private')) or isinstance(mode, str) and any(c in mode for c in 'wax+'):
            raise AssertionError('preview credential or write')
    if event in ('sqlite3.connect', 'os.mkdir'): raise AssertionError('preview state creation')
sys.addaudithook(guard)
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name='__main__')
'''
    process = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', code, str(Path(probe.__file__)), *preview_args],
        cwd=probe.ROOT, env={**os.environ, 'SHINKU_ENV_FILE': 'tmp/joint-preview-unused.env',
                            'SHINKU_CONTEXT_BUDGET_MODE': 'off'}, capture_output=True, text=True, encoding='utf-8')
    assert process.returncode == 0, process.stderr
    selection = json.loads(process.stdout)
    assert selection['models']['chat_expression_check']['model'] == 'gemini-2.5-pro' and not selection['production_enabled']
    if preview_args:
        assert len(selection['slots']) == 3 and set(selection['models']) == {'chat_expression_check'}
        assert selection['expression_output_limit'] == 4096 and selection['effective_adapter_limit'] == 0
    else:
        assert len(selection['slots']) == 18 and selection['models']['execution']['mode'] == 'thinking'


@pytest.mark.parametrize('mutation', ['valid', 'wrong_binding', 'omitted_fact', 'forged_tool', 'truncated'])
def test_wrapped_sdk_expression_still_obeys_original_fact_and_tool_gates(probe, tmp_path, mutation):
    case = next(item for item in probe.corpus()['expression_cases'] if item['id'] == 'completed')
    root = tmp_path / 'wrapped-expression'
    ledger = new_ledger(probe, tmp_path)

    class WrappedProvider(probe.FixtureProvider):
        def __call__(self, request):
            reply = super().__call__(request).json()
            choice = reply['choices'][0]
            candidate = json.loads(choice['message']['content'])
            if 'layout' in candidate:
                if mutation == 'wrong_binding':
                    candidate['binding_hash'] = 'a' * 64
                elif mutation == 'omitted_fact':
                    candidate['layout'] = candidate['layout'][:1]
                elif mutation == 'forged_tool':
                    candidate['tool_call'] = {'type': 'shell', 'command': 'untrusted tool data'}
                if mutation == 'truncated':
                    choice['finish_reason'] = 'length'
            choice['message']['content'] = '```json\n' + json.dumps(candidate, ensure_ascii=False) + '\n```'
            return httpx.Response(200, json=reply)

    result = probe.run_expression(case, root, ledger, 'offline-fixture-key', live=False,
        fixture=WrappedProvider(case, 'expression', root))
    assert result['required_facts_present'] and result['canonical_unchanged'] and result['replay_inert']
    assert result['platform_sends'] == result['task_sdk_calls'] == ledger.budget.tools == 0
    assert result['passed'] is (mutation == 'valid')
    assert result['expression_status'] == ('ready' if mutation == 'valid' else 'fallback')
    saved = probe.inspect_ledger(ledger.path)
    assert len(saved['attempts']) == (2 if mutation == 'valid' else 1)
    assert all(a['role'] in ('expression', 'check') for a in saved['attempts'])
    assert saved['attempts'][0]['response_metadata']['choices'][0]['json_wrapping'] == 'markdown_fence'
    assert saved['attempts'][0]['response_metadata']['choices'][0]['recognised_keys'] == ['binding_hash', 'layout']


def test_larger_expression_diagnostic_limit_changes_review_before_key_or_state(probe, tmp_path, monkeypatch):
    old = probe.preview(expression_diagnostic=True)
    proposed = probe.preview(expression_diagnostic=True, expression_output_tokens=4096)
    assert old['expression_output_limit'] == probe.preview()['expression_output_limit'] == 2048
    assert proposed['expression_output_limit'] == 4096 and proposed['selection_hash'] != old['selection_hash']
    assert proposed['batch_limits'] == old['batch_limits'] and proposed['slots'] == old['slots']
    assert proposed['effective_adapter_limit'] == 0 and proposed['permitted_roles'] == ['expression', 'check']
    touched = []
    monkeypatch.setattr(probe, 'read_keys', lambda *a, **k: touched.append(a))
    with pytest.raises(ValueError, match='selection_hash'):
        probe.run_suite(tmp_path / 'unused', tmp_path / 'budget.json', live=True, env_file=tmp_path / 'unused.env',
            expression_diagnostic=True, expression_output_tokens=4096, expected_selection_hash=old['selection_hash'])
    assert not touched and not (tmp_path / 'budget.json').exists() and not (tmp_path / 'unused').exists()


@pytest.mark.parametrize('limit', [127, 4097, True, '2048'])
def test_expression_diagnostic_output_limits_cannot_expand_beyond_original_profile(probe, limit):
    with pytest.raises(ValueError, match='expression_output_limit'):
        probe.preview(expression_diagnostic=True, expression_output_tokens=limit)
    with pytest.raises(ValueError, match='diagnostic_only'):
        probe.preview(expression_output_tokens=4096)


def test_4096_expression_diagnostic_reaches_both_sdk_roles_without_extra_calls(probe, tmp_path):
    selection = probe.preview(expression_diagnostic=True, expression_output_tokens=4096)
    ledger = probe.JointLedger(tmp_path / 'journal.json', probe.ledger_identity(selection, 'offline'),
        selection['batch_limits'], selection['role_limits'])
    ledger.claim(selection['slots'][0])
    case = next(item for item in probe.corpus()['expression_cases'] if item['id'] == 'completed')
    root = tmp_path / 'larger-output'

    class ObservedProvider(probe.FixtureProvider):
        def __call__(self, request):
            assert json.loads(request.content)['max_tokens'] == 4096
            return super().__call__(request)

    result = probe.run_expression(case, root, ledger, 'offline-fixture-key', live=False,
        fixture=ObservedProvider(case, 'expression', root), max_output_tokens=4096)
    assert result['passed'] and result['replay_inert'] and result['canonical_unchanged']
    assert len(ledger.attempts) == 2 and [a['role'] for a in ledger.attempts] == ['expression', 'check']
    assert not ledger.io and ledger.budget.tools == result['task_sdk_calls'] == result['platform_sends'] == 0


@pytest.mark.parametrize('mutation,bad_checks', [
    ('valid', set()), ('wrong_binding', {'binding_matches'}),
    ('unsupported', {'supported_true'}), ('non_boolean_supported', {'supported_true'}),
    ('missing_fact', {'coverage_exact'}), ('duplicate_fact', {'coverage_exact'}),
    ('invalid_ids', {'covered_ids_valid', 'coverage_exact'}),
    ('additional_claims', {'additional_claims_empty'}), ('contradictions', {'contradictions_empty'}),
    ('omitted_limitations', {'omitted_limitations_empty'}), ('unexpected_field', {'schema_exact'}),
])
def test_neutral_rejection_diagnostics_match_original_checker_without_storing_prose(probe, tmp_path, mutation, bad_checks):
    from shinku.result_expression.facts import check_neutral
    secret = 'PRIVATE_NEUTRAL_REPLY_REASON_DO_NOT_STORE'
    blocks = [{'id': 'test_fact', 'text': 'The recorded result.'}]
    verdict = dict(binding_hash='fixture-binding', supported=True, covered_fact_ids=['test_fact'],
                   additional_claims=[], contradictions=[], omitted_limitations=[])
    if mutation == 'wrong_binding': verdict['binding_hash'] = secret
    elif mutation == 'unsupported': verdict['supported'] = False
    elif mutation == 'non_boolean_supported': verdict['supported'] = 1
    elif mutation == 'missing_fact': verdict['covered_fact_ids'] = []
    elif mutation == 'duplicate_fact': verdict['covered_fact_ids'] *= 2
    elif mutation == 'invalid_ids': verdict['covered_fact_ids'] = secret
    elif mutation in {'additional_claims', 'contradictions', 'omitted_limitations'}: verdict[mutation] = [secret]
    elif mutation == 'unexpected_field': verdict[secret] = secret
    ledger = new_ledger(probe, tmp_path)
    seen = []
    def provider(request):
        seen.append(request)
        return httpx.Response(200, json={'id': 'fixture-neutral', 'object': 'chat.completion', 'created': 1,
            'model': probe.CHAT_MODEL, 'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
                'role': 'assistant', 'content': '```json\n' + json.dumps(verdict) + '\n```',
                'reasoning_content': secret}}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}})
    runtime = probe.ChatSDKRuntime(ledger, 'offline-fixture-key', live=False, forward_transport=httpx.MockTransport(provider))
    try:
        with ledger.role('check'):
            runtime.sdk.chat.completions.create(model=probe.CHAT_MODEL, messages=[{'role': 'user', 'content': json.dumps(
                dict(binding_hash='fixture-binding', fact_blocks=blocks, candidate_text=secret))}])
        attempt = probe.inspect_ledger(ledger.path)['attempts'][0]
        flags = attempt['response_metadata']['choices'][0]['neutral_checks']
        assert {k for k, v in flags.items() if not v} == bad_checks
        assert all(flags.values()) == (check_neutral(verdict, blocks, 'fixture-binding') == '')
        assert all(type(v) is bool for v in flags.values())
        assert len(seen) == ledger.budget.requests == 1 and not ledger.io
        assert secret not in ledger.path.read_text(encoding='utf-8')
    finally:
        runtime.close()


@pytest.mark.parametrize('corruption', ['neutral_text', 'neutral_extra_key', 'timeout_nan', 'timeout_boolean'])
def test_new_diagnostic_fields_reject_content_or_invalid_timeout_in_saved_journal(probe, tmp_path, corruption):
    ledger = new_ledger(probe, tmp_path)
    runtime = observing_runtime(probe, ledger, [])
    try:
        sdk_call(probe, runtime)
    finally:
        runtime.close()
    value = json.loads(ledger.path.read_text())
    attempt = value['attempts'][0]
    if corruption.startswith('neutral_'):
        flags = dict.fromkeys(['schema_exact', 'binding_matches', 'supported_true', 'coverage_exact',
            'covered_ids_valid', 'additional_claims_empty', 'contradictions_empty', 'omitted_limitations_empty'], True)
        flags['schema_exact' if corruption == 'neutral_text' else 'raw_reply'] = 'private text'
        attempt['response_metadata']['choices'][0]['neutral_checks'] = flags
    else:
        attempt['http_timeout_seconds'] = float('nan') if corruption == 'timeout_nan' else True
    ledger.path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='attempt_metadata_invalid'):
        probe.inspect_ledger(ledger.path)


def test_expression_deadline_cannot_silently_expand_declared_http_timeout(probe, tmp_path):
    ledger = new_ledger(probe, tmp_path)
    observed = []
    def provider(request):
        observed.append(request.extensions['timeout'])
        return httpx.Response(200, json={'id': 'timeout-clamp', 'object': 'chat.completion', 'created': 1,
            'model': probe.CHAT_MODEL, 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': '{}'}}]})
    runtime = probe.ChatSDKRuntime(ledger, 'offline-fixture-key', live=False, forward_transport=httpx.MockTransport(provider))
    try:
        probe.RoleExpressionAdapter(runtime, 'expression').call_expression_json(
            system_prompt='Return JSON.', user_prompt='facts', timeout_seconds=30)
        assert len(observed) == 1 and all(v == 25 for v in observed[0].values())
        assert probe.inspect_ledger(ledger.path)['attempts'][0]['http_timeout_seconds'] == 25
    finally:
        runtime.close()


@pytest.mark.parametrize('host,http', [(0, 0), (61, 25), (30, 31), (True, 1), (30, float('nan')), (30, '25')])
def test_diagnostic_timeouts_stay_within_original_profile_and_host_deadline(probe, host, http):
    with pytest.raises(ValueError, match='timeouts_invalid'):
        probe.preview(expression_diagnostic=True, expression_timeout_seconds=host, expression_http_timeout_seconds=http)


def test_slow_diagnostic_review_rejects_old_hash_before_credentials_or_state(probe, tmp_path, monkeypatch):
    old = probe.preview(expression_diagnostic=True, expression_output_tokens=4096)
    args = dict(expression_diagnostic=True, expression_output_tokens=4096,
                expression_timeout_seconds=60, expression_http_timeout_seconds=55)
    new = probe.preview(**args)
    assert new['selection_hash'] != old['selection_hash'] and new['slots'] == old['slots']
    assert new['batch_limits'] == old['batch_limits'] and new['effective_adapter_limit'] == 0
    assert probe.preview(**{**args, 'expression_timeout_seconds': 60.0}) == new
    with pytest.raises(ValueError, match='diagnostic_only'):
        probe.preview(expression_timeout_seconds=60, expression_http_timeout_seconds=55)
    touched = []
    monkeypatch.setattr(probe, 'read_keys', lambda *a, **k: touched.append(a))
    with pytest.raises(ValueError, match='selection_hash'):
        probe.run_suite(tmp_path / 'unused', tmp_path / 'unused.json', live=True, env_file=tmp_path / 'unused.env',
            expected_selection_hash=old['selection_hash'], **args)
    assert not touched and not (tmp_path / 'unused').exists() and not (tmp_path / 'unused.json').exists()


def test_reviewed_timeouts_reach_both_expression_and_check_without_extra_requests(probe, tmp_path):
    selection = probe.preview(expression_diagnostic=True, expression_output_tokens=4096,
                              expression_timeout_seconds=60, expression_http_timeout_seconds=55)
    ledger = probe.JointLedger(tmp_path / 'journal.json', probe.ledger_identity(selection, 'offline'),
        selection['batch_limits'], selection['role_limits'])
    ledger.claim(selection['slots'][0])
    case = next(c for c in probe.corpus()['expression_cases'] if c['id'] == 'completed')
    root = tmp_path / 'slow-expression'
    class ObservedProvider(probe.FixtureProvider):
        def __call__(self, request):
            assert all(v == 55 for v in request.extensions['timeout'].values())
            return super().__call__(request)
    result = probe.run_expression(case, root, ledger, 'offline-fixture-key', live=False,
        fixture=ObservedProvider(case, 'expression', root), max_output_tokens=4096, timeout_seconds=60, http_timeout_seconds=55)
    assert result['passed'] and result['replay_inert'] and result['canonical_unchanged']
    assert len(ledger.attempts) == 2 and [a['role'] for a in ledger.attempts] == ['expression', 'check']
    assert all(a['http_timeout_seconds'] == 55 for a in ledger.attempts) and not ledger.io
    assert all(ledger.attempts[-1]['response_metadata']['choices'][0]['neutral_checks'].values())


def test_finished_slot_persists_inflight_time_exhaustion_and_blocks_new_request(probe, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import shinku.agent.budget as budget_module

    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(budget_module, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    limits = {**probe.LIMITS, 'max_active_seconds': 2}
    ledger = probe.JointLedger(tmp_path / 'terminal.json',
        {'mode': 'offline', 'slots': ['first:on', 'next:on']}, limits=limits)
    ledger.claim('first:on')
    observed = []
    runtime = observing_runtime(probe, ledger, observed)
    try:
        with ledger.activity():
            sdk_call(probe, runtime)
            # The accepted in-flight request finishes after its admission cap.
            clock.now += 3
        ledger.finish('first:on')
        saved = probe.inspect_ledger(ledger.path)
        assert saved['slots'] == {'first:on': 'finished'}
        assert saved['budget']['reason'] == 'task_time_budget_exhausted'
        assert saved['budget'] == ledger.budget.snapshot()
        assert saved['budget']['active_seconds'] == 3
        assert saved['budget']['requests'] == len(saved['attempts']) == len(observed) == 1
        assert saved['attempts'][0]['state'] == 'response' and saved['attempts'][0]['forwarded']
        assert saved['budget']['reported_tokens'] == saved['budget']['tokens'] == 100
        assert saved['budget']['tools'] == len(saved['io']) == 0
        with pytest.raises(probe.TaskBudgetExceeded, match='task_time_budget_exhausted'):
            ledger.claim('next:on')
        assert probe.inspect_ledger(ledger.path) == saved and len(observed) == 1
    finally:
        runtime.close()


def test_report_retains_host_selected_interface_without_model_response_text(batch):
    for slot in ('file_sync:task', 'attachment_async:task'):
        selected = [stage for stage in row(batch, slot)['chat_protocol_diagnostics']
                    if stage['name'] == 'chat_model_result']
        assert selected and selected[0]['selected_interface'] == 'agent_task_submit'
        assert all(not set(stage).intersection({'text', 'response', 'reasoning_content', 'system_prompt'})
                   for stage in selected)
    selected = [stage for stage in row(batch, 'ordinary_chat:on')['chat_protocol_diagnostics']
                if stage['name'] == 'chat_model_result']
    assert selected[0]['selected_interface'] == ''


@pytest.mark.parametrize('native', [True, False])
def test_actual_sdk_chat_sees_attachment_descriptor_before_admission_without_its_bytes(probe, tmp_path, native):
    case = next(item for item in probe.corpus()['cases'] if item['id'] == 'attachment_async')
    root = tmp_path / 'attachment'
    ledger = new_ledger(probe, tmp_path)

    class AttachmentProvider(probe.FixtureProvider):
        observed_manifest = None

        def __call__(self, request):
            payload = json.loads(request.content)
            tools = {row['function']['name'] for row in payload.get('tools', [])}
            if 'agent_task_submit' in tools:
                user = payload['messages'][-1]['content']
                marker = '【宿主绑定的任务资料描述，仅为当前请求数据】\n'
                manifest = json.loads(user.split(marker, 1)[1])
                assert manifest['descriptor_only'] is True
                assert manifest['materials'] == [{'kind': 'attachment', 'relation': 'direct',
                    'local_reference_supplied': True}]
                assert manifest['gaps'] == [] and manifest['omitted_materials'] == 0
                assert str(root / 'attachment.txt') not in user
                assert case['attachment_tail'] not in user and 'f' * 400 not in user
                assert self.host.reader.calls == self.host.writer.calls == []
                assert not self.host.material_store.read_receipts({'task_id': 'unused',
                    'requester': probe.SCOPE.requester, 'session': probe.SCOPE.session})
                self.observed_manifest = manifest
                arguments = {'goal': '按附件处理'}
                if native:
                    return probe.json_response(payload, tool='agent_task_submit', arguments=arguments)
                return probe.json_response(payload, {'kind': 'tool', 'tool_call': {
                    'name': 'agent_task_submit', 'arguments': arguments}})
            return super().__call__(request)

    provider = AttachmentProvider(case, 'task', root)
    item = probe.run_joint(case, 'task', root, ledger, ('offline-fixture-key', 'offline-fixture-key'),
        live=False, fixture=provider)
    assert provider.observed_manifest is not None
    assert item['passed'] and item['complete'], item['failures']
    assert item['task_count'] == 1 and item['work_status'] == 'completed'
    assert item['inputs_preserved'] and len(item['file_writes']) == 1
    assert (root / 'output.txt').read_bytes() == b'TAIL-73\n'
    source = json.loads((root / 'canonical-result.json').read_text(encoding='utf-8'))
    assert source['original_requests'][0]['text'].startswith(case['request'])
    assert '只写一次' in source['original_requests'][0]['text']
    assert [stage['selected_interface'] for stage in item['chat_protocol_diagnostics']
        if stage['name'] == 'chat_model_result'] == ['agent_task_submit']


def test_actual_sdk_retained_long_fixture_recovers_exact_tail_without_canonical_upgrade(probe, tmp_path):
    case = probe.supplemental_corpus()['expression_cases'][0]
    ledger = new_ledger(probe, tmp_path)
    root = tmp_path / 'long'
    provider = probe.FixtureProvider(case, 'expression', root)
    item = probe.run_expression(case, root, ledger, 'offline-fixture-key', live=False, fixture=provider)
    assert item['passed'], item
    assert item['material_recovery'] == dict(retained_source=True, canonical_tail_present=False,
        presentation_tail_present=True, page_reads=4, projection_status='complete')
    assert item['canonical_unchanged'] and item['replay_inert'] and item['task_sdk_calls'] == 0
    assert item['platform_sends'] == 0 and len(ledger.attempts) == 2
    assert [record['role'] for record in ledger.attempts] == ['expression', 'check']
    assert [record['kind'] for record in ledger.io] == ['file_adapter', *(['task_material_read'] * 4)]
    result = json.loads((root / 'canonical-result.json').read_text(encoding='utf-8'))
    assert result['execution']['status'] == 'partial' and result['verification']['accepted'] is False
    assert result['information']['status'] == 'incomplete'
    store = probe.AgentTaskStore(root / 'tasks.sqlite')
    try:
        archive = probe.ExpressionArchive(store)
        with store._transaction() as db:
            identifiers = [row[0] for row in db.execute('SELECT expression_id FROM agent_result_expressions')]
        records = [archive.get(identifier, scope=probe.SCOPE) for identifier in identifiers]
        expected = json.loads((root / 'retained-receipt.json').read_text(encoding='utf-8'))['state_updates']['result_facts']
        assert records[0]['presentation_projection']['facts'] == expected
        assert records[0]['canonical_result'] == result
    finally:
        store.close()


@pytest.mark.parametrize('fault', ['quota', 'changed_origin'])
def test_retained_fixture_failure_uses_honest_template_without_models_or_execution(probe, tmp_path, monkeypatch, fault):
    case = probe.supplemental_corpus()['expression_cases'][0]
    ledger = new_ledger(probe, tmp_path)
    root = tmp_path / fault
    original = probe.retained_expression

    def changed(case, root, store, ledger):
        options = {'policy': probe.MaterialPolicy(inline_chars=1024, page_chars=2048, max_read_chars=16)} \
            if fault == 'quota' else {}
        source, materials = original(case, root, store, ledger, **options)
        if fault == 'changed_origin':
            (root / 'retained-receipt.json').write_text('CHANGED_UNVERIFIED_ORIGIN', encoding='utf-8')
        return source, materials

    monkeypatch.setattr(probe, 'retained_expression', changed)
    item = probe.run_expression(case, root, ledger, 'offline-fixture-key', live=False,
        fixture=probe.FixtureProvider(case, 'expression', root))
    assert not item['passed'] and item['expression_status'] == 'fallback'
    assert item['fallback_reason'] == 'expression_required_material_unavailable'
    assert item['canonical_unchanged'] and item['replay_inert']
    assert not item['material_recovery']['presentation_tail_present']
    assert not ledger.attempts and item['platform_sends'] == item['task_sdk_calls'] == 0
    assert 'CHANGED_UNVERIFIED_ORIGIN' not in item['text']


def test_request_audit_checks_system_native_and_call_history_without_rejecting_user_data(probe):
    value = dict(messages=[dict(role='user', content='Do not invoke benchmark_write.')])
    assert all(probe.chat_request_isolation(value).values())
    for channel, changes in [
        ('system_catalog_absent', dict(messages=[dict(role='system', content='benchmark_write schema')])),
        ('native_catalog_absent', dict(tools=[dict(type='function', function=dict(name='benchmark_write'))])),
        ('execution_call_history_absent', dict(messages=[dict(role='assistant', tool_calls=[
            dict(function=dict(name='benchmark_write', arguments='{}'))])])),
    ]:
        flags = probe.chat_request_isolation({**value, **changes})
        assert flags[channel] is False and sum(not item for item in flags.values()) == 1


def test_new_admission_profile_does_not_expand_original_batch_or_accept_unknown_profiles(probe, tmp_path):
    limits = dict(max_requests=192, max_tokens=3000000, max_tools=100, max_active_seconds=1800)
    roles = dict(chat=64, execution=64, expression=32, check=32)
    with pytest.raises(ValueError, match='limits_invalid_or_expanded'):
        probe.validate_limits(limits)
    with pytest.raises(ValueError, match='role_limits_invalid_or_expanded'):
        probe.validate_roles(roles)
    for profile in ('arbitrary-budget', [], True):
        with pytest.raises(ValueError, match='admission_profile_invalid'):
            probe.validate_limits(limits, profile=profile)
    with pytest.raises(ValueError, match='limits_invalid_or_expanded'):
        probe.JointLedger(tmp_path / 'forbidden.json', {'slots': ['fixed:on']}, limits)
    assert not (tmp_path / 'forbidden.json').exists()
    value = {**limits, 'max_active_seconds': limits['max_active_seconds'] + 1}
    with pytest.raises(ValueError, match='limits_invalid_or_expanded'):
        probe.validate_limits(value, profile='remaining-gates-v1')
