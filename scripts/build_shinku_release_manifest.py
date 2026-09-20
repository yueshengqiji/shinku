"""Build a deterministic clean-tree release manifest without secrets or runtime data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOTS = (
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
EXCLUDED_PARTS = frozenset({
    ".git", ".pytest_cache", "__pycache__", ".venv", "venv", "build", "dist", "data",
})
FORBIDDEN_NAMES = frozenset({".env", ".env.local"})
MEDIA_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".mp3", ".wav",
    ".ogg", ".mp4", ".avi", ".mkv", ".flac", ".ttf", ".otf", ".woff", ".woff2",
})


def _files_under(root: Path, relative: str):
    path = root / relative
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    for item in sorted(path.rglob("*")):
        parts = item.relative_to(root).parts
        if item.is_file() and not any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in parts):
            yield item


def build_manifest(root: str | Path) -> dict[str, object]:
    repo = Path(root).resolve()
    files: list[dict[str, object]] = []
    findings: list[str] = []
    seen: set[Path] = set()
    for relative_root in ROOTS:
        for path in _files_under(repo, relative_root):
            if path in seen:
                continue
            seen.add(path)
            relative = path.relative_to(repo).as_posix()
            if path.name in FORBIDDEN_NAMES:
                findings.append(f"secret_file:{relative}")
                continue
            if path.suffix.lower() in MEDIA_SUFFIXES:
                findings.append(f"media_file:{relative}")
                continue
            payload = path.read_bytes()
            files.append({
                "path": relative,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })
    return {
        "ok": not findings,
        "file_count": len(files),
        "files": files,
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = build_manifest(args.root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"MANIFEST={'PASS' if result['ok'] else 'FAIL'}")
        print(f"FILES={result['file_count']}")
        for item in result["findings"]:
            print(f"  {item}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
