"""Build a deterministic source release bundle from the clean manifest roots."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# 允许直接执行 `python scripts/build_shinku_release_bundle.py`。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_shinku_release_manifest import (
    FORBIDDEN_NAMES,
    MEDIA_SUFFIXES,
    _files_under,
)


def build_bundle(root: str | Path, output: str | Path) -> dict[str, object]:
    repo = Path(root).resolve()
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    findings: list[str] = []

    roots = (
        "src",
        "scripts",
        "tests",
        "docs",
        "README.md",
        "LICENSE",
        "NOTICE",
        "pyproject.toml",
        "requirements-cleanenv.lock",
        "requirements-build.lock",
        ".env.example",
        ".gitignore",
    )
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative_root in roots:
            for path in _files_under(repo, relative_root):
                relative = path.relative_to(repo).as_posix()
                if path.name in FORBIDDEN_NAMES:
                    findings.append(f"secret_file:{relative}")
                    continue
                if path.suffix.lower() in MEDIA_SUFFIXES:
                    findings.append(f"media_file:{relative}")
                    continue
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes())
                entries.append(relative)

    return {
        "ok": not findings,
        "artifact": str(target),
        "file_count": len(entries),
        "files": entries,
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or root / "dist" / "shinku-0.1.0-source.zip"
    result = build_bundle(root, output)
    print(f"BUNDLE={'PASS' if result['ok'] else 'FAIL'}")
    print(f"ARTIFACT={result['artifact']}")
    print(f"FILES={result['file_count']}")
    for item in result["findings"]:
        print(f"  {item}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
