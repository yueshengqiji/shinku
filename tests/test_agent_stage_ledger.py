"""Atomic ledger replacement retries never purchase another request/effect."""
import ctypes
import errno
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest

from test_agent_stage_live_benchmark import MockProvider, probe


def denied(winerror=5):
    error = PermissionError(errno.EACCES, 'isolated replacement denial')
    if winerror is not None:
        error.winerror = winerror
    return error


def inject(monkeypatch, ledger, *, failures=2, winerror=5, phase=None):
    original = probe.os.replace
    calls, delays = [], []
    def replace(source, destination):
        if Path(destination) == ledger.path:
            value = json.loads(Path(source).read_text(encoding='utf-8'))
            eligible = phase is None or value['attempts'] and value['attempts'][-1]['state'] == phase
            if eligible:
                calls.append(Path(source).read_bytes())
                if len(calls) <= failures:
                    raise denied(winerror)
        return original(source, destination)
    monkeypatch.setattr(probe.os, 'replace', replace)
    # Isolate file retry waits; do not replace worker polling's global sleep.
    monkeypatch.setattr(probe, 'time', SimpleNamespace(sleep=delays.append, monotonic=probe.time.monotonic))
    return calls, delays


def test_replace_retries_same_flushed_snapshot_and_preserves_old_file_until_success(tmp_path, monkeypatch):
    ledger = probe.PairedLedger(tmp_path/'budget.json')
    old = ledger.path.read_bytes()
    calls, delays = inject(monkeypatch, ledger)
    synced = []
    original_sync = probe.os.fsync
    def sync(fd):
        synced.append(True)
        return original_sync(fd)
    monkeypatch.setattr(probe.os, 'fsync', sync)
    def wait(delay):
        assert ledger.path.read_bytes() == old
        assert synced == [True]
        delays.append(delay)
    monkeypatch.setattr(probe.time, 'sleep', wait)
    ledger.claim('simple_file', 'react')
    assert delays == [.02, .04] and len(calls) == 3 and len(set(calls)) == 1
    saved = json.loads(ledger.path.read_text())
    assert saved['slots'] == {'simple_file:react': 'in_progress'}
    assert set(saved) == {'identity', 'budget', 'attempts', 'slots'}
    assert saved['budget']['requests'] == ledger.budget.requests == 0


@pytest.mark.parametrize('winerror,count', [(5, 5), (32, 1), (None, 1)])
def test_permanent_or_other_denials_preserve_original_and_do_not_reset_claim(tmp_path, monkeypatch, winerror, count):
    ledger = probe.PairedLedger(tmp_path/'budget.json'); old = ledger.path.read_bytes()
    calls, delays = inject(monkeypatch, ledger, failures=99, winerror=winerror)
    with pytest.raises(PermissionError):
        ledger.claim('simple_file', 'react')
    assert len(calls) == count and delays == ([.02, .04, .08, .16] if winerror == 5 else [])
    assert ledger.path.read_bytes() == old
    assert ledger.slots == {'simple_file:react': 'in_progress'}
    assert json.loads(ledger.path.with_suffix('.json.new').read_text())['slots'] == ledger.slots
    with pytest.raises(ValueError, match='already_claimed'):
        ledger.claim('simple_file', 'react')


@pytest.mark.parametrize('operation', ['sync', 'replace'])
def test_non_permission_io_errors_are_not_retried_or_hidden(tmp_path, monkeypatch, operation):
    ledger = probe.PairedLedger(tmp_path/'budget.json'); old = ledger.path.read_bytes()
    calls, delays = [], []
    def fail(*args):
        calls.append(True)
        raise OSError(errno.ENOSPC, 'isolated full disk')
    monkeypatch.setattr(probe.os, 'fsync' if operation == 'sync' else 'replace', fail)
    monkeypatch.setattr(probe.time, 'sleep', delays.append)
    with pytest.raises(OSError, match='full disk'):
        ledger.claim('simple_file', 'react')
    assert calls == [True] and delays == [] and ledger.path.read_bytes() == old


def make_runtime(root, ledger, forwarded):
    fixture = SimpleNamespace(root=root, sample={'id':'simple_file', 'kind':'simple'},
                              reader=SimpleNamespace(calls=[]), writer=SimpleNamespace(calls=[]))
    def respond(request):
        saved = json.loads(ledger.path.read_text())
        assert saved['budget']['requests'] == 1 and saved['attempts'][0]['state'] == 'reserved'
        forwarded.append(True)
        return httpx.Response(200, json=dict(id='fixture', object='chat.completion', created=0,
            model=probe.MODEL, choices=[dict(index=0, finish_reason='stop', message=dict(role='assistant',
            content='{"kind":"final","text":"fixture"}', reasoning_content='fixture-native-history'))],
            usage=dict(prompt_tokens=12, completion_tokens=4, total_tokens=16)))
    return probe.PairedRuntime(fixture, ledger, 'offline-fixture', 'react', live=False,
                               forward_transport=httpx.MockTransport(respond))


def sdk_call(runtime):
    return runtime.sdk.chat.completions.create(model=probe.MODEL,
        messages=[dict(role='user', content='isolated request')],
        extra_body={'thinking': {'type':'enabled'}})


@pytest.mark.parametrize('phase', ['reserved', 'response'])
@pytest.mark.parametrize('failures,extra_replace_denial',
                         [(2, False), (99, False), (2, True)],
                         ids=['2', '99', '2-extra-replace-denial'])
def test_sdk_reservation_and_response_saves_never_repeat_transport(
        tmp_path, monkeypatch, phase, failures, extra_replace_denial):
    ledger = probe.PairedLedger(tmp_path/'budget.json'); ledger.claim('simple_file', 'react')
    forwarded = []; runtime = make_runtime(tmp_path, ledger, forwarded)
    try:
        if extra_replace_denial:
            original_replace = probe.os.replace
            pending = [True]
            def replace(source, destination):
                if Path(destination) == ledger.path and pending:
                    value = json.loads(Path(source).read_text(encoding='utf-8'))
                    if value['attempts'] and value['attempts'][-1]['state'] == phase:
                        pending.pop()
                        raise denied()
                return original_replace(source, destination)
            monkeypatch.setattr(probe.os, 'replace', replace)
        calls, delays = inject(monkeypatch, ledger, failures=failures, phase=phase)
        if failures > 4:
            with pytest.raises(PermissionError):
                sdk_call(runtime)
        else:
            assert sdk_call(runtime).usage.total_tokens == 16
        # Fault injection and a real Windows denial can both consume retries.
        # Check the bound and every wait without assuming injection is the only
        # source of an access denial. The snapshot and transport checks below
        # still enforce one reservation, one response and no repeated purchase.
        minimum = 5 if failures > 4 else failures + 1 + int(extra_replace_denial)
        assert minimum <= len(calls) <= 5
        assert delays == [.02 * 2 ** attempt for attempt in range(len(calls) - 1)]
        assert len(forwarded) == (0 if phase == 'reserved' and failures > 4 else 1)
        assert ledger.budget.requests == len(ledger.attempts) == len(runtime.calls) == 1
        assert len(set(calls)) == 1
        if forwarded:
            assert ledger.attempts[0]['state'] == 'response'
            assert ledger.attempts[0]['sdk_usage']['total_tokens'] == 16
            assert ledger.budget.reported_tokens == 16
        if phase == 'response' and failures > 4:
            # The persisted reservation remains spent/unknown if receipt save fails.
            saved = json.loads(ledger.path.read_text())
            assert saved['attempts'][0]['state'] == 'reserved'
            assert saved['budget']['requests'] == 1
        if phase == 'reserved' and failures > 4:
            assert ledger.budget.tokens > 0 and ledger.attempts[0]['state'] == 'reserved'
        with pytest.raises(ValueError, match='requires_review'):
            probe.run_suite(tmp_path/'forbidden-replay', ledger, 'unused', live=False, forward_factory=lambda *_:None)
        assert not (tmp_path/'forbidden-replay').exists()
    finally:
        runtime.close()


@pytest.mark.parametrize('failures', [2, 99])
def test_tool_reservation_precedes_one_real_write_and_receipt(tmp_path, monkeypatch, failures):
    ledger = probe.PairedLedger(tmp_path/'budget.json'); ledger.claim('simple_file', 'react')
    writer = probe.benchmark.FileAdapter(tmp_path, dict(initial={}, expected={'output.txt':'ok'}), 'write', ledger)
    calls, delays = inject(monkeypatch, ledger, failures=failures)
    if failures > 4:
        with pytest.raises(PermissionError):
            writer.execute(call=dict(path=str(tmp_path/'output.txt'), content='ok'), context={})
        assert not (tmp_path/'output.txt').exists() and writer.calls == []
    else:
        result = writer.execute(call=dict(path=str(tmp_path/'output.txt'), content='ok'), context={})
        assert (tmp_path/'output.txt').read_bytes() == b'ok' and len(writer.calls) == 1
        receipt = result['state_updates']['artifact_receipts'][0]
        assert writer.agent_verify_result(receipt=receipt, context={})['result_valid']
        assert json.loads(ledger.path.read_text())['budget']['tools'] == 1
    assert ledger.budget.tools == 1 and len(calls) == (5 if failures > 4 else 3)


@pytest.mark.parametrize('case_id,arm', [('early_final_repair','react'), ('cross_slice_code','staged')])
def test_original_failed_cases_complete_with_replacement_faults_and_exact_receipts(tmp_path, monkeypatch, case_id, arm):
    ledger = probe.PairedLedger(tmp_path/'budget.json'); ledger.claim(case_id, arm)
    case = next(c for c in probe.corpus()['cases'] if c['id'] == case_id)
    handler = MockProvider(case, arm, ledger)
    calls, delays = inject(monkeypatch, ledger, phase='response')
    saves = []
    original_save = probe.write_snapshot
    def save(path, value):
        before = len(delays)
        retries = original_save(path, value)
        waits = delays[before:]
        # A separate real Windows denial may occur on another save. Its
        # recovery is valid, but each snapshot still has exactly its own
        # bounded backoff and must not re-enter any SDK/tool operation.
        assert 0 <= retries <= 4
        assert waits == [.02 * 2 ** i for i in range(retries)]
        saves.append(retries)
        return retries
    monkeypatch.setattr(probe, 'write_snapshot', save)
    row = probe.run_case(case, arm, tmp_path/'run', ledger, 'offline-fixture', live=False,
                         forward_transport=httpx.MockTransport(handler))
    assert any(retries >= 2 for retries in saves) and len(delays) == sum(saves)
    assert row['fixture_pass'] and row['task_success'] and row['contract_preserved'] and row['native_history_exact']
    assert all(a['valid'] for a in row['artifact_checks'])
    assert row['duplicate_effects'] == row['duplicate_publications'] == 0
    assert row['query_unchanged'] and row['publications'] == 1
    assert row['provider_requests'] == row['platform_sends'] == 0
    assert ledger.budget.requests == len(ledger.attempts) == len(handler.calls) + row['controlled_requests']
    assert len(row['writes']) == len(case['expected'])


@pytest.mark.parametrize('other_snapshot_denial', [False, True])
def test_report_replacement_retries_do_not_repeat_completed_cases(tmp_path, monkeypatch, other_snapshot_denial):
    ledger = probe.PairedLedger(tmp_path/'budget.json')
    calls, delays, handlers, saves, snapshots, injected = {}, [], {}, [], {}, []
    original = probe.os.replace
    original_save = probe.write_snapshot
    def replace(source, destination):
        if Path(destination).name == 'report.json':
            raw = Path(source).read_bytes()
            rows = len(json.loads(raw)['rows'])
            snapshots.setdefault(rows, []).append(raw)
            calls[rows] = calls.get(rows, 0) + 1
            if rows in (0, 1) and calls[rows] <= 2:
                raise denied()
        elif Path(destination) == ledger.path and other_snapshot_denial and not injected:
            if json.loads(Path(source).read_text())['budget']['requests'] > 0:
                injected.append(True)
                raise denied()
        return original(source, destination)
    def save(path, value):
        before = len(delays)
        retries = original_save(path, value)
        # Report and budget snapshots may independently encounter a real
        # Windows denial. Check every bounded retry, not a batch-wide sum.
        assert 0 <= retries <= 4
        assert delays[before:] == [.02 * 2 ** i for i in range(retries)]
        saves.append((Path(path), len(value.get('rows', [])), retries))
        return retries
    def factory(case, arm):
        key = (case['id'], arm)
        assert key not in handlers
        handler = MockProvider(case, arm, ledger); handlers[key] = handler
        return httpx.MockTransport(handler)
    monkeypatch.setattr(probe.os, 'replace', replace)
    monkeypatch.setattr(probe, 'time', SimpleNamespace(sleep=delays.append, monotonic=probe.time.monotonic))
    monkeypatch.setattr(probe, 'write_snapshot', save)
    report = probe.run_suite(tmp_path/'run', ledger, 'offline-fixture', live=False, forward_factory=factory)
    assert len(delays) == sum(retries for path, rows, retries in saves)
    assert all(any(path.name == 'report.json' and count == rows and retries >= 2
                   for path, count, retries in saves) for rows in (0, 1))
    assert all(len(snapshots[rows]) >= 3 and len(set(snapshots[rows])) == 1 for rows in (0, 1))
    assert injected == ([True] if other_snapshot_denial else [])
    if other_snapshot_denial:
        assert any(path == ledger.path and retries >= 1 for path, rows, retries in saves)
    assert len(handlers) == len(report['rows']) == 30 and all(r['fixture_pass'] for r in report['rows'])
    assert all(r['duplicate_effects'] == r['duplicate_publications'] == 0 for r in report['rows'])
    assert all(s == 'finished' for s in ledger.slots.values())
    assert ledger.budget.requests == len(ledger.attempts) == sum(
        len(handler.calls) for handler in handlers.values()) + sum(row['controlled_requests'] for row in report['rows'])
    assert report['provider_requests'] == report['platform_sends'] == 0
    assert json.loads((tmp_path/'run/report.json').read_text()) == report


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows deny-delete handle semantics')
def test_real_windows_handle_denies_replace_then_release_allows_same_snapshot(tmp_path, monkeypatch):
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    close = kernel.CloseHandle; close.argtypes = [wintypes.HANDLE]; close.restype = wintypes.BOOL
    ledger = probe.PairedLedger(tmp_path/'budget.json'); old = ledger.path.read_bytes()
    handle = create(str(ledger.path), 0x80000000, 3, None, 3, 0x80, None)
    assert handle != ctypes.c_void_p(-1).value
    opened, delays, errors = [handle], [], []
    original = probe.os.replace
    def replace(*args):
        try:
            return original(*args)
        except PermissionError as exc:
            errors.append(exc.winerror)
            raise
    def release(delay):
        assert ledger.path.read_bytes() == old
        delays.append(delay)
        assert close(opened.pop())
    monkeypatch.setattr(probe.os, 'replace', replace)
    monkeypatch.setattr(probe.time, 'sleep', release)
    try:
        ledger.claim('simple_file', 'react')
        assert errors == [5] and delays == [.02]
        assert json.loads(ledger.path.read_text())['slots'] == {'simple_file:react':'in_progress'}
    finally:
        if opened:
            close(opened.pop())
