"""Local-only Stage B cost diagnostic, fictional isolated sources, no network.

Reports warmed end-to-end inspect cost separately from cold index preparation.
It measures engineering overhead, NOT model answer quality or BGE retrieval gain.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore


def snapshot():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT / "src/shinku/memory/search_index.py",
                         ROOT / "src/shinku/memory/store.py", ROOT / "src/shinku/memory/service.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    report = {"label": args.label, "source_snapshot_before": snapshot(), "runs": [],
              "isolated": True, "network_calls": 0, "embeddings": False, "model_qa": False}
    now = 1790860000
    for count in (10, 100, 500):
        with tempfile.TemporaryDirectory(prefix="shinku-b-cost-") as tmp:
            store = MemoryStore(Path(tmp) / "memory.sqlite3", event_scan_limit=800)
            policy = MemoryPolicy(search_keys_enabled=False, master_user_ids=("owner",))
            service = MemoryService(store=store, router=MemoryRouter(), policy=policy, clock=lambda: now)
            store.record_event(conversation_id="owner", scope="private:owner", text="实际设置强度0.65，不是建议的0.8。",
                source_id="turn:owner:target", created_at=now, metadata={"role": "user", "sender_id": "owner"})
            store.record_event(conversation_id="owner", scope="private:owner", text="下次可以尝试0.8；目前仍然是0.65。",
                source_id="turn:owner:target:assistant", created_at=now, metadata={"role": "assistant", "sender_id": "bot"})
            for index in range(count - 2):
                store.record_event(conversation_id="owner", scope="private:owner", text=f"整理无关文件和草稿，记录编号{index}。",
                    source_id=f"noise:{index}", created_at=now-100-index, metadata={"role": "user", "sender_id": "owner"})
            for branch in ("raw", "enhanced"):
                started = time.perf_counter()
                service.policy = replace(policy, search_keys_enabled=branch == "enhanced")
                store.configure_search_keys(enabled=branch == "enhanced")
                service.inspect(conversation_id="owner", conversation_type="private", requester_id="owner",
                                query="实际设置强度", mode="retrieve")
                report["runs"].append({"candidate_count": count, "branch": branch, "kind": "warmup",
                                       "seconds": time.perf_counter() - started})
            for repeat in range(1, 4):
                for branch in (("raw", "enhanced") if repeat % 2 else ("enhanced", "raw")):
                    service.policy = replace(policy, search_keys_enabled=branch == "enhanced")
                    store.configure_search_keys(enabled=branch == "enhanced")
                    with patch.object(store, "_connect", wraps=store._connect) as connects, \
                         patch.object(store, "expire", wraps=store.expire) as expires:
                        started = time.perf_counter()
                        result = service.inspect(conversation_id="owner", conversation_type="private", requester_id="owner",
                                                 query="实际设置强度", mode="retrieve")
                        elapsed = time.perf_counter() - started
                    row = {"candidate_count": count, "branch": branch, "repeat": repeat, "kind": "measured",
                           "seconds": elapsed, "connections": connects.call_count, "expiry_scans": expires.call_count,
                           "source_ids": [r["source_id"] for r in result["records"]],
                           "rendered_hash": hashlib.sha256(result["rendered"].encode()).hexdigest(),
                           "source_present": "0.65" in result["rendered"]}
                    report["runs"].append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
    report["source_snapshot_after"] = snapshot()
    report["source_unchanged_during_run"] = report["source_snapshot_after"] == report["source_snapshot_before"]
    report["summary"] = {str(count): {branch: {
        "median_seconds": statistics.median(r["seconds"] for r in report["runs"] if
                           r["candidate_count"] == count and r["branch"] == branch and r["kind"] == "measured"),
        "all_sources_present": all(r["source_present"] for r in report["runs"] if
                           r["candidate_count"] == count and r["branch"] == branch and r["kind"] == "measured")}
        for branch in ("raw", "enhanced")} for count in (10, 100, 500)}
    path = ROOT / "docs" / ("MEMORY_STAGE_B_COST_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("REPORT " + str(path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
