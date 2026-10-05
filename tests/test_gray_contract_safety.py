"""Even Ultra must not escape the live test's exact disposable-file guard."""
import importlib
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.fixture
def guarded(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    gray = importlib.import_module("gray_contract_repair")
    root = tmp_path / "fixture"
    root.mkdir()
    target = root / "target.zip"
    config = replace(gray.gray.PrivateConfig.from_environment(), root=root, data_root=root,
        read_roots=(root,), write_roots=(root,), delete_roots=(root,), audit_path=root / "audit.jsonl")
    return gray.StrictList(config, root), gray.StrictDelete(config, target), root, target


def test_list_guard_rejects_parent_other_drive_or_nonfixture_directory(guarded):
    listing, _, root, _ = guarded
    for path in [root.parent, root / "other", root / "..", "C:/Windows"]:
        assert listing.normalize_call({"path": str(path)}) is None
        result = listing.execute(call={"path": str(path)}, context={"agent_permission_mode": "ultra"})
        assert result.get("error") or not result.get("state_updates", {}).get("entries")
    assert listing.events == []
    assert listing.normalize_call({"path": str(root)}) is not None


def test_delete_guard_rejects_recursive_directory_keepfile_and_escape(guarded):
    _, deletion, root, target = guarded
    for path, recursive in [(root, False), (target, True), (root / "keep.zip", False),
                            (root.parent / "target.zip", False), ("C:/Windows/test.zip", False)]:
        call = {"path": str(path), "confirm": True, "recursive": recursive}
        assert deletion.normalize_call(call) is None
        deletion.execute(call=call, context={"agent_permission_mode": "ultra"})
    assert deletion.events == []
    assert deletion.normalize_call({"path": str(target), "confirm": True}) is not None
