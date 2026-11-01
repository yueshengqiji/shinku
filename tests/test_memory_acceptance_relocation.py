"""Historical batches can move hosts without weakening input provenance."""
import json
from pathlib import PurePosixPath, PureWindowsPath
from types import SimpleNamespace

import pytest

from scripts import run_memory_acceptance as runner


@pytest.mark.parametrize("old_root,path_type", [
    ("C:/Users/fixture/shinku-standalone", PureWindowsPath),
    ("//fixture-host/share/shinku-standalone", PureWindowsPath),
    ("/old-workspace/shinku-standalone", PurePosixPath),
])
def test_prepare_relocated_inputs_keeps_hashes_originals_and_budget_history(tmp_path, monkeypatch, old_root, path_type):
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    fixture = root / "tests/fixtures/demo.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(json.dumps({"source_kind": "fictional_only", "cases": [{"id": "demo"}]}))
    driver = root / "scripts/driver.py"
    driver.parent.mkdir()
    driver.write_text("# preserved historical driver\n")
    source = root / "tmp/previous/manifest.json"
    source.parent.mkdir(parents=True)
    history = {"earlier_unknown_requests": 2, "reserved_tokens": 170626}
    runner.dump(source, {
        "fixture": str(path_type(old_root) / "tests/fixtures/demo.json"),
        "fixture_sha256": runner.sha(fixture),
        "driver": str(path_type(old_root) / "scripts/driver.py"),
        "driver_sha256": runner.sha(driver),
        "schedule": [{"id": "demo", "arm": "model"}],
        "prior_usage_history": history,
        "authorization": "fictional test authorization",
    })
    runner.dump(source.parent / "summary.json", {"budget": {"estimated_requests": 1, "tokens": 85313}})
    original = {p: p.read_bytes() for p in (source, fixture, driver, source.parent / "summary.json")}
    args = SimpleNamespace(source_manifest=str(source), source_root=old_root,
        retry=[], synthetic_fixtures=True, max_requests=66, max_tokens=1400000, max_seconds=1800, per_case=6)
    output = root / "tmp/new-batch"

    runner.prepare(args, output)

    manifest = runner.read(output / "manifest.json")
    assert manifest["source_manifest_sha256"] == runner.sha(source)
    assert manifest["driver"] == str(driver)
    assert manifest["driver_sha256"] == runner.sha(driver)
    assert manifest["source_repository_root"] == old_root
    assert manifest["resolved_source_paths"] == {"fixture": str(fixture), "driver": str(driver)}
    assert manifest["prior_usage_history"] == history
    assert manifest["prior_budget"] == {"estimated_requests": 1, "tokens": 85313}
    assert manifest["prior_consecutive_technical_failures"] == 1
    assert all(p.read_bytes() == data for p, data in original.items())
    runner.frozen(output)
    fixture.write_text("changed historical fixture")
    with pytest.raises(ValueError, match="source_manifest_hash_mismatch:fixture"):
        runner.prepare(args, root / "tmp/changed-input")
    assert not (root / "tmp/changed-input").exists()


@pytest.mark.parametrize("value,previous_root,error", [
    ("C:/old/repo/scripts/driver.py", None, "outside_repository"),
    ("C:/other/repo/scripts/driver.py", "C:/old/repo", "outside_declared_source_root"),
    ("C:/old/repo-sibling/scripts/driver.py", "C:/old/repo", "outside_declared_source_root"),
    ("C:/old/repo/../outside.py", "C:/old/repo", "parent_traversal"),
    ("C:relative.py", "C:/old/repo", "requires_absolute_paths"),
    ("/old/repo/../outside.py", "/old/repo", "parent_traversal"),
])
def test_relocation_rejects_undeclared_roots_and_escaping_paths(tmp_path, monkeypatch, value, previous_root, error):
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    with pytest.raises(ValueError, match=error):
        runner.source_path(value, previous_root)


def test_relocation_rejects_symlinks_that_escape_current_repository(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "scripts").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner, "ROOT", root)
    with pytest.raises(ValueError):
        runner.source_path("C:/old/repo/scripts/driver.py", "C:/old/repo")


def test_current_relative_and_absolute_inputs_need_no_relocation(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    monkeypatch.setattr(runner, "ROOT", root)
    expected = root / "scripts/driver.py"
    assert runner.source_path("scripts/driver.py") == expected
    assert runner.source_path(str(expected)) == expected
