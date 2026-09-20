from __future__ import annotations

import zipfile
from pathlib import Path

from scripts.audit_shinku_artifact import audit_artifact
from scripts.build_shinku_release_bundle import build_bundle
from scripts.build_shinku_release_manifest import build_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_d4_lock_files_are_pinned_and_non_editable() -> None:
    cleanenv = (ROOT / "requirements-cleanenv.lock").read_text(encoding="utf-8")
    build = (ROOT / "requirements-build.lock").read_text(encoding="utf-8")
    assert "-e " not in cleanenv
    assert "setuptools==84.0.0" in build
    assert "wheel==0.48.0" in build
    assert "fastapi==0.141.1" in cleanenv


def test_d4_manifest_includes_license_tests_and_locks() -> None:
    result = build_manifest(ROOT)
    assert result["ok"] is True
    files = {item["path"] for item in result["files"]}
    assert "LICENSE" in files
    assert "tests/test_contract_d4.py" in files
    assert "requirements-cleanenv.lock" in files
    assert "requirements-build.lock" in files


def test_d4_source_bundle_is_clean(tmp_path: Path) -> None:
    artifact = tmp_path / "shinku-source.zip"
    result = build_bundle(ROOT, artifact)
    assert result["ok"] is True
    audited = audit_artifact(artifact)
    assert audited["ok"] is True
    with zipfile.ZipFile(artifact) as archive:
        assert ".env" not in archive.namelist()
        assert "LICENSE" in archive.namelist()
