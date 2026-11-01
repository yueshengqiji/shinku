"""POSIX read receipts must survive changes to Windows path classification."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from shinku.agent.local_answer import LocalAnswerLedger


@pytest.mark.parametrize("path,accepted", [
    ("/tmp/input.txt", True), ("/tmp/input", True), ("/tmp/输入.txt", True),
    ("C:/fixture/input.txt", True), ("input.txt", False), ("../input.txt", False),
])
def test_absolute_read_receipt_is_required_for_each_path_syntax(path, accepted):
    content = "VALUE=13"
    sha = hashlib.sha256(content.encode()).hexdigest()
    verified = []

    class Reader:
        agent_local_answer_source = True
        agent_permission_class = "read"
        agent_completion_effects = ("read",)
        agent_completion_target_arguments = ("path",)

        def agent_verify_result(self, *, receipt, context):
            verified.append(receipt["path"])
            return {"result_valid": True, "sha256": sha, "file_text": content}

    updates = {"path": path, "sha256": sha, "read_complete": True,
               "verified_effects": ["read"], "completion_targets": [path]}
    ledger = LocalAnswerLedger(enabled=True)
    ledger.observe(SimpleNamespace(name="reader", arguments={"path": path}),
                   SimpleNamespace(status="ok", data={"state_updates": updates}), Reader(), {})
    assert bool(ledger.sources) is accepted
    assert verified == ([path] if accepted else [])
    if accepted:
        assert next(iter(ledger.sources.values()))["text"] == content


def test_posix_receipt_still_requires_fresh_matching_bytes():
    class ChangedReader:
        agent_local_answer_source = True
        agent_permission_class = "read"
        agent_completion_effects = ("read",)
        agent_completion_target_arguments = ("path",)

        def agent_verify_result(self, **kwargs):
            return {"result_valid": False, "sha256": "changed", "file_text": "changed"}

    path = "/tmp/input.txt"
    updates = {"path": path, "sha256": "original", "read_complete": True,
               "verified_effects": ["read"], "completion_targets": [path]}
    ledger = LocalAnswerLedger(enabled=True)
    ledger.observe(SimpleNamespace(name="reader", arguments={"path": path}),
                   SimpleNamespace(status="ok", data={"state_updates": updates}), ChangedReader(), {})
    assert len(ledger.sources) == 1
    assert next(iter(ledger.sources.values()))["valid"] is False
