"""Read-only preflight for a future Shinku cutover.

This command never stops a process, edits NapCat, changes ports, or enables QQ
egress. It only checks both health surfaces and the local release gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.audit_shinku_release import audit_release


Fetcher = Callable[[str], dict[str, Any]]


def _fetch_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=3) as response:  # noqa: S310 - loopback preflight URL
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {"value": payload}


def _probe(base_url: str, fetcher: Fetcher = _fetch_json) -> dict[str, Any]:
    base = base_url.rstrip("/")
    result: dict[str, Any] = {"base_url": base, "ok": False}
    try:
        health = fetcher(f"{base}/health")
        ready = fetcher(f"{base}/health/ready")
    except Exception as exc:  # pragma: no cover - exact network errors vary by host
        result["error"] = type(exc).__name__
        return result
    result.update({"ok": health.get("status") == "ok" and ready.get("status") in {"ok", "ready"}})
    result["health"] = {key: health.get(key) for key in ("status", "project", "version", "port") if key in health}
    result["ready"] = {key: ready.get(key) for key in ("status", "service", "detail") if key in ready}
    return result


def assess(root: str | Path, *, old_url: str, new_url: str, fetcher: Fetcher = _fetch_json) -> dict[str, Any]:
    release = audit_release(root)
    old = _probe(old_url, fetcher)
    new = _probe(new_url, fetcher)
    checks = {
        "release_ready": bool(release["release_ready"]),
        "old_ready": bool(old["ok"]),
        "new_ready": bool(new["ok"]),
    }
    return {
        "read_only": True,
        "manual_confirmation_required": True,
        "checks": checks,
        "ready_for_manual_cutover": all(checks.values()),
        "release": {"release_ready": release["release_ready"], "findings": release["findings"]},
        "old": old,
        "new": new,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_REPO_ROOT)
    parser.add_argument("--old-url", default="http://127.0.0.1:9998")
    parser.add_argument("--new-url", default="http://127.0.0.1:19998")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = assess(args.root, old_url=args.old_url, new_url=args.new_url)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"READ_ONLY={'YES' if result['read_only'] else 'NO'}")
        print(f"OLD_READY={'YES' if result['checks']['old_ready'] else 'NO'}")
        print(f"NEW_READY={'YES' if result['checks']['new_ready'] else 'NO'}")
        print(f"RELEASE_READY={'YES' if result['checks']['release_ready'] else 'NO'}")
        print(f"READY_FOR_MANUAL_CUTOVER={'YES' if result['ready_for_manual_cutover'] else 'NO'}")
    return 0 if result["ready_for_manual_cutover"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
