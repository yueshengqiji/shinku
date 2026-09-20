"""Audit a Shinku source or wheel artifact for release-boundary violations."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


FORBIDDEN_NAMES = {".env", ".env.local"}
FORBIDDEN_PARTS = {".git", "data", "build", "dist", "__pycache__", ".pytest_cache"}
MEDIA_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg", ".mp3", ".wav",
    ".ogg", ".mp4", ".avi", ".mkv", ".flac", ".ttf", ".otf", ".woff", ".woff2",
}


def audit_artifact(path: str | Path) -> dict[str, object]:
    artifact = Path(path).resolve()
    findings: list[str] = []
    if not artifact.is_file():
        return {"ok": False, "artifact": str(artifact), "file_count": 0, "findings": ["missing_artifact"]}
    with zipfile.ZipFile(artifact) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        for name in names:
            parts = set(Path(name).parts)
            path_obj = Path(name)
            if path_obj.name in FORBIDDEN_NAMES:
                findings.append(f"secret_file:{name}")
            if parts & FORBIDDEN_PARTS:
                findings.append(f"runtime_or_build_path:{name}")
            if path_obj.suffix.lower() in MEDIA_SUFFIXES:
                findings.append(f"media_file:{name}")
        if artifact.suffix.lower() == ".zip":
            required = {
                "README.md",
                "LICENSE",
                "NOTICE",
                "pyproject.toml",
                "docs/MCP_SCOPE.md",
                "requirements-cleanenv.lock",
                "requirements-build.lock",
            }
            findings.extend(f"missing_required:{name}" for name in sorted(required - set(names)))
        else:
            if not any(name.startswith("shinku/") for name in names):
                findings.append("wheel_missing_shinku_package")
            if not any(name.endswith(".dist-info/METADATA") for name in names):
                findings.append("wheel_missing_metadata")
    return {
        "ok": not findings,
        "artifact": str(artifact),
        "file_count": len(names),
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    args = parser.parse_args(argv)
    result = audit_artifact(args.artifact)
    print(f"ARTIFACT_AUDIT={'PASS' if result['ok'] else 'FAIL'}")
    print(f"FILES={result['file_count']}")
    for item in result["findings"]:
        print(f"  {item}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
