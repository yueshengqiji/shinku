"""Real JSON queues remain atomic and FIFO through bounded replacement failures."""
import ctypes
import errno
import json
import sys
import time

import pytest

from shinku.agent import pending
from shinku.agent.approval import make_tool_approval_request
from shinku.agent.pending import JsonPendingApprovalStore, PendingApprovalRecord
from shinku.tools.invocation import ToolInvocation


def record(key):
    request = make_tool_approval_request(task_id=key, invocation=ToolInvocation('inspect_files', {'n': key}))
    return PendingApprovalRecord(request, key, {}, (), 100)


def denied(winerror=5):
    error = PermissionError(errno.EACCES, 'fixed test denial')
    error.winerror = winerror
    return error


@pytest.mark.parametrize('winerror', [5, 32, 33])
def test_transient_windows_denial_replaces_once_and_reloads_fifo(tmp_path, monkeypatch, winerror):
    store = JsonPendingApprovalStore(tmp_path / 'queue.json')
    store.put('fixture', record('first'))
    actual, calls, delays = pending.os.replace, [], []
    def replace(source, target):
        calls.append((source, target))
        if len(calls) <= 2:
            raise denied(winerror)
        return actual(source, target)
    monkeypatch.setattr(pending.os, 'replace', replace)
    monkeypatch.setattr(time, 'sleep', delays.append)
    store.put('fixture', record('second'))
    assert len(calls) == 3 and len(delays) == 2 and sum(delays) <= 1
    data = json.loads(store.path.read_text(encoding='utf-8'))
    assert [r['task_id'] for r in data['fixture']] == ['first', 'second']
    reloaded = JsonPendingApprovalStore(store.path)
    assert reloaded.get('fixture').task_id == 'first'
    reloaded.delete('fixture')
    assert reloaded.get('fixture').task_id == 'second'
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize('operation', ['put', 'delete'])
def test_persistent_failure_preserves_both_memory_and_old_disk(tmp_path, monkeypatch, operation):
    store = JsonPendingApprovalStore(tmp_path / 'queue.json')
    store.put('fixture', record('first'))
    before = store.path.read_bytes()
    calls, delays = [], []
    def fail(*args):
        calls.append(args)
        raise denied()
    monkeypatch.setattr(pending.os, 'replace', fail)
    monkeypatch.setattr(time, 'sleep', delays.append)
    with pytest.raises(PermissionError):
        store.put('fixture', record('second')) if operation == 'put' else store.delete('fixture')
    assert 1 < len(calls) <= 7 and sum(delays) <= 1
    assert store.path.read_bytes() == before and store.get('fixture').task_id == 'first'
    assert JsonPendingApprovalStore(store.path).get('fixture').task_id == 'first'
    assert len(list(tmp_path.iterdir())) == 1
    # Failure must not append twice when the caller explicitly tries again.
    monkeypatch.undo()
    if operation == 'put':
        store.put('fixture', record('second'))
        assert len(json.loads(store.path.read_text(encoding='utf-8'))['fixture']) == 2
    else:
        store.delete('fixture')
        assert store.get('fixture') is None


def test_unrelated_permission_error_is_not_retried_or_committed(tmp_path, monkeypatch):
    store = JsonPendingApprovalStore(tmp_path / 'queue.json')
    calls = []
    def fail(*args):
        calls.append(args)
        raise PermissionError(errno.EACCES, 'non-Windows denial')
    monkeypatch.setattr(pending.os, 'replace', fail)
    with pytest.raises(PermissionError):
        store.put('fixture', record('first'))
    assert len(calls) == 1 and store.get('fixture') is None
    assert not list(tmp_path.iterdir())


def test_actual_windows_handle_without_delete_sharing_is_released_during_retry(tmp_path, monkeypatch):
    if sys.platform != 'win32':
        return
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                                  wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    store = JsonPendingApprovalStore(tmp_path / 'queue.json')
    store.put('fixture', record('first'))
    handle = kernel.CreateFileW(str(store.path), 0x80000000, 3, None, 3, 0x80, None)
    assert handle != wintypes.HANDLE(-1).value
    released = []
    def release(_delay):
        if not released:
            assert kernel.CloseHandle(handle)
            released.append(True)
    monkeypatch.setattr(time, 'sleep', release)
    try:
        store.put('fixture', record('second'))
        assert released == [True]
        assert [r['task_id'] for r in json.loads(store.path.read_text(encoding='utf-8'))['fixture']] == ['first', 'second']
    finally:
        if not released:
            kernel.CloseHandle(handle)
