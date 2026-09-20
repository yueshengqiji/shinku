"""D1 release-boundary audit for the independent Shinku repository."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许直接执行 `python scripts/audit_shinku_release.py`，而不是只能作为测试模块导入。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.check_shinku_admission import audit as admission_audit


MEDIA_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".mp3", ".wav",
    ".ogg", ".mp4", ".avi", ".mkv", ".flac", ".ttf", ".otf", ".woff", ".woff2",
})
IGNORED_DIRS = frozenset({
    ".git", ".pytest_cache", "__pycache__", ".venv", "venv", "build", "dist", "data",
})


def _iter_release_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.name in {".env", ".env.local"}:
            continue
        yield path


def audit_release(root: str | Path) -> dict[str, object]:
    repo = Path(root).resolve()
    findings: list[str] = []
    required = (
        "README.md",
        "LICENSE",
        "NOTICE",
        "pyproject.toml",
        "docs/ADMISSION.md",
        "docs/SOURCE_RECORD.md",
        "docs/THIRD_PARTY_LICENSES.md",
    )
    for relative in required:
        if not (repo / relative).is_file():
            findings.append(f"missing_required:{relative}")

    media = [
        path.relative_to(repo).as_posix()
        for path in _iter_release_files(repo)
        if path.suffix.lower() in MEDIA_SUFFIXES
    ]
    findings.extend(f"media_in_tree:{path}" for path in media)

    license_path = repo / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8") if license_path.exists() else ""
    pyproject_path = repo / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8") if pyproject_path.exists() else ""
    license_ok = (
        "Apache License" in license_text
        and "Version 2.0" in license_text
        and 'license = { file = "LICENSE" }' in pyproject
    )
    if not license_ok:
        findings.append("apache_license_metadata_mismatch")

    notice_path = repo / "NOTICE"
    notice = notice_path.read_text(encoding="utf-8") if notice_path.exists() else ""
    if "当前声明（D1 发布边界审计" not in notice:
        findings.append("notice_is_stale")
    if "第三方 Python 依赖" not in notice:
        findings.append("notice_missing_dependency_reference")

    admission = admission_audit(repo)
    return {
        "ok": not findings and bool(admission["ok"]),
        "release_ready": not findings and bool(admission["ok"]),
        "release_gate": "" if not findings and admission["ok"] else "automated release checks are not green",
        "admission": admission,
        "media_files": media,
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = audit_release(args.root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"AUTOMATED_AUDIT={'PASS' if result['ok'] else 'FAIL'}")
        print(f"RELEASE_READY={'YES' if result['release_ready'] else 'NO'}")
        print(f"RELEASE_GATE={result['release_gate']}")
        for item in result["findings"]:
            print(f"  {item}")
        for item in result["admission"]["findings"]:
            print(f"  admission:{item['check']}:{item['detail']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
