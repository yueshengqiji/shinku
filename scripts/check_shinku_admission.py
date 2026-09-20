"""Automated release-boundary checks for the independent Shinku tree.

The audit deliberately distinguishes executable boundaries from provenance docs:
SOURCE_RECORD.md may mention the old project for evidence, while runtime imports,
active environment assignments and dependency declarations must not depend on it.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


OLD_IMPORT_ROOTS = frozenset({
    "akane",
    "companion_v01",
    "companion_shared",
    "code_shared",
})
OLD_ENV_PREFIXES = ("COMPANION_",)
_LEDGER_PATH_RE = re.compile(r"`([^`]+)`")
_DEPENDENCY_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class Finding:
    check: str
    detail: str


def _dependency_name(spec: str) -> str:
    match = _DEPENDENCY_NAME_RE.match(str(spec).strip())
    return match.group(0).lower().replace("_", "-") if match else ""


def _declared_dependencies(root: Path) -> set[str]:
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = document.get("project") or {}
    values = list(project.get("dependencies") or [])
    optional = project.get("optional-dependencies") or {}
    for group in optional.values():
        values.extend(group or [])
    return {_dependency_name(value) for value in values if _dependency_name(value)}


def _ledger_paths(root: Path) -> tuple[str, ...]:
    text = (root / "docs" / "SOURCE_RECORD.md").read_text(encoding="utf-8")
    return tuple(value.replace("\\", "/").strip() for value in _LEDGER_PATH_RE.findall(text))


def _ledger_covers(path: str, entries: tuple[str, ...]) -> bool:
    normalized = path.replace("\\", "/")
    for entry in entries:
        clean = entry.rstrip("/")
        if not clean or clean.startswith("http"):
            continue
        if normalized == clean or normalized.startswith(clean + "/"):
            return True
    return False


def _import_findings(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    source_root = root / "src"
    for path in sorted(source_root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except SyntaxError as exc:
            findings.append(Finding("import_graph", f"{relative}: syntax error: {exc}"))
            continue
        for node in ast.walk(tree):
            imported = node.module if isinstance(node, ast.ImportFrom) else None
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else []
            candidates = ([imported] if imported else []) + names
            for value in candidates:
                root_name = str(value or "").split(".", 1)[0].lower()
                if root_name in OLD_IMPORT_ROOTS:
                    findings.append(Finding("import_graph", f"{relative}: {value}"))
    return findings


def _active_env_findings(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for name in (".env", ".env.example"):
        path = root / name
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key = stripped.split("=", 1)[0].strip()
            if key.startswith(OLD_ENV_PREFIXES):
                findings.append(Finding("active_env", f"{name}:{number}: {key}"))
    return findings


def audit(root: str | Path) -> dict[str, object]:
    repo = Path(root).resolve()
    findings = _import_findings(repo) + _active_env_findings(repo)
    ledger = _ledger_paths(repo)
    source_files = sorted((repo / "src" / "shinku").rglob("*.py"))
    uncovered = [path.relative_to(repo).as_posix() for path in source_files if not _ledger_covers(path.relative_to(repo).as_posix(), ledger)]
    findings.extend(Finding("source_record", path) for path in uncovered)

    declared = _declared_dependencies(repo)
    license_path = repo / "docs" / "THIRD_PARTY_LICENSES.md"
    license_text = license_path.read_text(encoding="utf-8").lower() if license_path.exists() else ""
    missing_licenses = sorted(name for name in declared if name not in license_text)
    findings.extend(Finding("license_inventory", name) for name in missing_licenses)
    return {
        "ok": not findings,
        "checks": {
            "import_graph": not any(item.check == "import_graph" for item in findings),
            "active_env": not any(item.check == "active_env" for item in findings),
            "source_record": not any(item.check == "source_record" for item in findings),
            "license_inventory": not any(item.check == "license_inventory" for item in findings),
        },
        "source_files": len(source_files),
        "dependencies": sorted(declared),
        "findings": [{"check": item.check, "detail": item.detail} for item in findings],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    result = audit(args.root)
    for name, passed in result["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if result["findings"]:
        for finding in result["findings"]:
            print(f"  {finding['check']}: {finding['detail']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
