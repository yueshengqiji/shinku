"""阶段观测的口径、并发、持久化和真实请求边界；使用合成数据，不访问模型。"""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace

import pytest

from shinku.phase_timing import bind_observer, phase_scope, summarize_phases
from shinku.llm.request_ledger import bind_provider_calls, observed_completion
from shinku.qq.trace import TurnTraceStore
from shinku.angelos.usage_view import UsageView


def reply():
    return {'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}


def turn():
    return SimpleNamespace(conversation_type='private', conversation_id='fixture',
                           primary_message_id='m', combined_text='SECRET_USER_BODY')


def phases(result):
    return {r['phase']: r for r in result['phases']}


def test_nested_duration_and_percentiles_are_inclusive_not_additive():
    events, clock = [], [0.0]
    with bind_observer('task', events.append, clock=lambda: clock[0]), phase_scope('execution'):
        clock[0] = 1
        with phase_scope('tool'):
            clock[0] = 4
        clock[0] = 5
        with phase_scope('tool'):
            clock[0] = 6
        clock[0] = 10
    result = summarize_phases(events)
    rows = phases(result)
    assert rows['execution']['totalMs'] == 10000
    assert rows['tool']['totalMs'] == 4000
    assert rows['tool']['p50Ms'] == 1000 and rows['tool']['p95Ms'] == 3000
    assert rows['tool']['samples'] == 2
    assert events[1]['parent_id'] == events[0]['span_id']
    assert result['durationMode'] == 'inclusive_not_additive'


def test_exceptions_and_interrupted_start_do_not_become_zero_or_success():
    events = []
    with bind_observer('task', events.append):
        with pytest.raises(RuntimeError):
            with phase_scope('tool'):
                raise RuntimeError('SECRET_EXCEPTION_BODY')
    row = phases(summarize_phases(events))['tool']
    assert row['failed'] == 1 and row['measured'] == 1
    row = phases(summarize_phases(events[:1]))['tool']
    assert row['unsettled'] == 1 and row['totalMs'] is None and row['p95Ms'] is None
    assert 'SECRET_EXCEPTION_BODY' not in json.dumps(events)


def test_two_threads_have_independent_tasks_and_parent_spans():
    barrier, events, lock = threading.Barrier(2), [], threading.Lock()
    def emit(row):
        with lock:
            events.append(row)
    def worker(binding):
        with bind_observer(binding, emit), phase_scope('execution'):
            barrier.wait(timeout=2)
            with phase_scope('tool'):
                barrier.wait(timeout=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker, ['a', 'b']))
    for binding in ('a', 'b'):
        selected = [r for r in events if r['binding'] == binding]
        root = next(r for r in selected if r['phase'] == 'execution')
        assert all(r['parent_id'] == root['span_id'] for r in selected if r['phase'] == 'tool')
        assert len(selected) == 4


def test_actual_provider_requests_are_attributed_once_and_missing_usage_unknown():
    view = UsageView()
    with bind_observer('task', view.observe_phase), bind_provider_calls('task', view), phase_scope('planning'):
        with phase_scope('answer_review'):
            observed_completion(lambda **kw: reply(), {})
        with pytest.raises(TimeoutError):
            observed_completion(lambda **kw: (_ for _ in ()).throw(TimeoutError('SECRET')), {})
    calls = view.read(['task'])
    rows = phases(summarize_phases(view.read_phases(['task']), calls))
    assert rows['answer_review']['requests'] == 1 and rows['answer_review']['knownInputTokens'] == 100
    assert rows['planning']['requests'] == 1 and rows['planning']['requestFailed'] == 1
    assert rows['planning']['knownInputTokens'] is None and rows['planning']['incompleteUsage'] == 1
    assert sum(r['requests'] for r in rows.values()) == 2


def test_trace_restart_privacy_and_external_ledger_are_preserved(tmp_path):
    path, upstream = tmp_path/'trace.json', []
    store = TurnTraceStore(path=path)
    ident = store.begin(turn())
    with bind_provider_calls('external-test', upstream.append), store.observe(ident), phase_scope('routing'):
        observed_completion(lambda **kw: reply(), {'messages': ['SECRET_PROMPT']})
    store.finish(ident, status='finished')
    result = TurnTraceStore(path=path).snapshot()[0]
    rows = phases(result['performance'])
    assert rows['routing']['samples'] == 1 and rows['routing']['requests'] == 1
    assert rows['routing']['knownOutputTokens'] == 20
    assert len(upstream) == 2 and {r['binding'] for r in upstream} == {'external-test'}
    assert 'SECRET' not in path.read_text()
    assert not result['performance']['truncated']


def test_trace_capacity_reports_gap_instead_of_complete_sample(tmp_path):
    store = TurnTraceStore(path=tmp_path/'trace.json', max_stages=2)
    ident = store.begin(turn())
    with store.observe(ident), phase_scope('execution'), phase_scope('tool'):
        pass
    assert store.snapshot()[0]['performance']['truncated']
    assert TurnTraceStore(path=store.path).snapshot()[0]['performance']['truncated']


def test_phase_store_dedup_restart_and_terminal_survives_capacity(tmp_path):
    path = tmp_path/'usage.sqlite3'
    view = UsageView(path)
    start = {'binding': 'task', 'span_id': 'one', 'parent_id': '',
             'phase': 'tool', 'status': 'started', 'duration_ms': None}
    view.observe_phase(start)
    for i in range(511):
        view.observe_phase(dict(start, span_id=str(i)))
    view.observe_phase(dict(start, span_id='over-limit'))
    assert not view.phase_healthy
    view.observe_phase(dict(start, status='completed', duration_ms=12))
    view.observe_phase(start)
    fresh = UsageView(path)
    rows = fresh.read_phases(['task'])
    assert len(rows) == 512
    assert next(r for r in rows if r['span_id'] == 'one')['duration_ms'] == 12
    assert fresh.read_phases(['other']) == []


def test_observation_storage_failure_never_changes_result_or_replays():
    view, called = UsageView(), []
    view.db.close()
    with bind_observer('task', view.observe_phase), phase_scope('tool'):
        called.append('effect')
    assert called == ['effect'] and not view.phase_healthy


def test_old_usage_is_unclassified_and_empty_is_unknown():
    calls = [{'binding': 'task', 'request_id': 'old', 'status': 'completed',
              'elapsed_seconds': None, 'usage': {}}]
    row = phases(summarize_phases([], calls))['unclassified']
    assert row['requests'] == 1 and row['totalMs'] is None and row['requestMs'] is None
    assert row['knownInputTokens'] is None and row['knownOutputTokens'] is None
    assert not summarize_phases([])['available']


def test_same_request_label_in_two_bindings_does_not_drop_usage():
    view = UsageView()
    for binding in ('preparation', 'task'):
        view({'binding': binding, 'request_id': 'legacy-id', 'phase': 'planning',
              'status': 'completed', 'usage': {'input_tokens': 100}, 'elapsed_seconds': 1})
    row = phases(summarize_phases([], view.read(['preparation', 'task'])))['planning']
    assert row['requests'] == 2 and row['knownInputTokens'] == 200


def test_compaction_calls_are_separate_from_parent_planning():
    from shinku.llm.context_compaction import summary_call_scope
    view = UsageView()
    with bind_observer('task', view.observe_phase), bind_provider_calls('task', view), phase_scope('planning'):
        with summary_call_scope():
            observed_completion(lambda **kw: reply(), {})
    rows = phases(summarize_phases(view.read_phases(['task']), view.read(['task'])))
    assert rows['compaction']['requests'] == 1 and rows['planning']['requests'] == 0
