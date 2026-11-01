"""实际传输边界的离线账本回归：重试、缺用量、流结束、预算拒绝与故障隔离。"""
from types import SimpleNamespace

import pytest

from shinku.agent.budget import TaskBudget, TaskBudgetExceeded, bind_task_budget
from shinku.llm.request_ledger import JsonCallLedger, bind_provider_calls, observed_completion
from shinku.llm.runtime_transport import attempt_completion, RuntimeTransportDependencies, strip_cache_hints


def response(n=10):
    return {'model': 'fixture-v1', 'usage': {'prompt_tokens': n, 'completion_tokens': 2,
            'prompt_cache_hit_tokens': 3, 'prompt_cache_miss_tokens': n - 3}}


def test_each_attempt_is_separate_and_unknown_failure_is_not_free():
    events = []
    with bind_provider_calls('prepare:fixture', events.append):
        with pytest.raises(TimeoutError):
            observed_completion(lambda **kw: (_ for _ in ()).throw(TimeoutError()), {'model': 'fixture'})
        assert observed_completion(lambda **kw: response(), {'model': 'fixture'}) == response()
    assert [r['status'] for r in events] == ['started', 'timeout', 'started', 'completed']
    assert events[0]['request_id'] == events[1]['request_id'] != events[2]['request_id']
    assert events[1]['usage']['input_tokens'] is None and events[3]['usage']['input_tokens'] == 10
    assert events[3]['reported_model'] == 'fixture-v1' and all(r['phase'] == 'preparation' for r in events)
    assert not {'messages', 'tools', 'url', 'error'} & events[3].keys()


def test_stream_cumulative_usage_is_recorded_once_at_end_or_unknown_on_close():
    events = []
    with bind_provider_calls('fixture', events.append):
        stream = observed_completion(lambda **kw: iter([response(10), response(20)]), {'stream': True})
        assert len(list(stream)) == 2
        assert events[-1]['usage']['input_tokens'] == 20 and events[-1]['status'] == 'completed'
        stream = observed_completion(lambda **kw: iter([response(10), response(20)]), {'stream': True})
        next(stream); stream.close()
        assert events[-1]['status'] == 'unknown' and events[-1]['usage']['input_tokens'] == 10


def test_sink_failure_does_not_retry_or_change_provider_result():
    calls = []
    with bind_provider_calls('fixture', lambda r: (_ for _ in ()).throw(OSError('disk full'))):
        assert observed_completion(lambda **kw: calls.append(kw) or response(), {}) == response()
    assert len(calls) == 1


def test_disk_ledger_restart_upserts_and_scope_counts(tmp_path):
    path = tmp_path / 'calls.jsonl'
    ledger = JsonCallLedger(path)
    with bind_provider_calls('task-1', ledger):
        observed_completion(lambda **kw: response(), {'model': 'fixture'})
    with bind_provider_calls('task-2', ledger):
        observed_completion(lambda **kw: response(20), {'model': 'fixture'})
    fresh = JsonCallLedger(path)
    snap = fresh.snapshot('task-1', expected_requests=1)
    assert snap['requests'] == 1 and snap['totals']['input_tokens'] == 10 and snap['ledger_complete']
    assert not fresh.snapshot('task-1')['ledger_complete']
    assert not fresh.snapshot('task-1', expected_requests=2)['ledger_complete']
    assert path.stat().st_mode & 0o777 == 0o600


def test_real_runtime_attempt_counts_compatibility_retry_but_not_budget_rejection():
    calls, events = [], []
    def create(**kw):
        calls.append(kw)
        if 'prompt_cache_key' in kw:
            raise TypeError('unsupported cache hint')
        return response()
    bundle = SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    deps = RuntimeTransportDependencies(add_metric=lambda *a: None, attempt_completion=None, note_error=lambda *a, **k: None,
        retry_without_cache_hints=lambda e: False, strip_cache_hints=strip_cache_hints,
        runtime_policy=lambda: SimpleNamespace(http_attempts=2))
    with bind_provider_calls('fixture', events.append):
        result = attempt_completion(bundle=bundle, payload={'prompt_cache_key': 'fixture'}, attempt=1, deps=deps)
    assert result == response() and len(calls) == 2
    assert [e['status'] for e in events] == ['started', 'failed', 'started', 'completed']
    events.clear(); calls.clear()
    budget = TaskBudget(max_requests=1, max_tokens=10, max_tools=1, max_active_seconds=30, managed=True)
    with bind_provider_calls('fixture', events.append), bind_task_budget(budget), pytest.raises(TaskBudgetExceeded):
        attempt_completion(bundle=bundle, payload={'messages': [{'role': 'user', 'content': 'x' * 1000}]}, attempt=1, deps=deps)
    assert events == calls == []
