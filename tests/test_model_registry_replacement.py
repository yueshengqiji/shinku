"""Atomic registry retries cannot reload or select a model twice."""
import errno
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shinku.providers import service
import test_contract_c2_4 as contract_case


def error(winerror):
    value = PermissionError(errno.EACCES, "isolated registry denial")
    if winerror is not None:
        value.winerror = winerror
    return value


@pytest.mark.parametrize("winerror,failures,success", [(5, 2, True), (5, 99, False), (32, 2, False), (None, 2, False)])
def test_real_select_response_preserves_snapshot_and_reload_boundary(monkeypatch, winerror, failures, success):
    case = contract_case.SelectTests("test_the_underscore_spelling_is_accepted")
    case.setUp()
    try:
        old = case.store.path.read_bytes()
        original = service.os.replace
        attempts, delays, synced = [], [], []
        original_sync = service.os.fsync
        def sync(fd):
            synced.append(True)
            return original_sync(fd)
        def replace(source, target):
            if Path(target) == case.store.path:
                attempts.append(Path(source).read_bytes())
                if len(attempts) <= failures:
                    raise error(winerror)
            return original(source, target)
        def wait(delay):
            assert synced == [True] and case.store.path.read_bytes() == old
            delays.append(delay)
        monkeypatch.setattr(service.os, "replace", replace)
        monkeypatch.setattr(service.os, "fsync", sync)
        monkeypatch.setattr(service, "time", SimpleNamespace(sleep=wait))
        response = case.client.post(contract_case.SELECT_PATH, json=dict(provider_id="glm", model_id="glm-4.5")).json()
        assert response["ok"] is success
        assert case.engine.count == int(success)
        assert len(set(attempts)) == 1
        expected = 3 if success else 5 if winerror == 5 else 1
        assert len(attempts) == expected
        assert delays == ([.02, .04] if success else [.02, .04, .08, .16] if winerror == 5 else [])
        if success:
            assert response["activeModel"] == "glm-4.5"
            assert case.store.get_provider("glm").chat_model == "glm-4.5"
            assert case.config.CHAT_MODEL_NAME == "glm-4.5"
        else:
            assert response["status"] == "select_failed"
            assert case.store.path.read_bytes() == old
            assert case.config.CHAT_MODEL_NAME == "deepseek-chat"
    finally:
        case.doCleanups()


@pytest.mark.parametrize("operation", ["fsync", "replace"])
def test_other_io_errors_stop_without_retry_or_reload(monkeypatch, operation):
    case = contract_case.SelectTests("test_the_underscore_spelling_is_accepted"); case.setUp()
    try:
        old = case.store.path.read_bytes(); calls = []
        def fail(*args):
            calls.append(True)
            raise OSError(errno.ENOSPC, "isolated disk full")
        monkeypatch.setattr(service.os, operation, fail)
        monkeypatch.setattr(service, "time", SimpleNamespace(sleep=lambda _: pytest.fail("unexpected retry")))
        value = case.client.post(contract_case.SELECT_PATH, json=dict(provider_id="glm", model_id="glm-4.5")).json()
        assert not value["ok"] and value["status"] == "select_failed"
        assert calls == [True] and case.engine.count == 0 and case.store.path.read_bytes() == old
    finally:
        case.doCleanups()
