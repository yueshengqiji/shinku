"""Read-only storage baseline and isolated time/restore replay for memory.

No QQ sender, configured chat model, production service restart or implicit
VACUUM. The replay uses SQLite + the actual lifecycle engine with a fake clock;
it is not a public-model semantic evaluation.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shinku.memory import LifecyclePolicy, MemoryStore  # noqa: E402


def readonly_baseline(path: Path) -> dict:
    database = path.resolve(strict=True)
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("BEGIN")
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        counts, logical = {}, {}
        for table, text_column in (("memory_turns", "content"), ("memory_events", "text"), ("memory_facts", "text"), ("memory_embeddings", "vector_blob")):
            if table not in tables:
                continue
            counts[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            logical[table] = conn.execute(f"SELECT COALESCE(sum(length(CAST({text_column} AS BLOB))),0) FROM {table}").fetchone()[0]
        pages = conn.execute("PRAGMA page_count").fetchone()[0]
        event_layers = {f"{row[0]}:{row[1]}": row[2] for row in conn.execute("SELECT layer,status,count(*) FROM memory_events GROUP BY layer,status")} if "memory_events" in tables else {}
        free_pages = conn.execute("PRAGMA freelist_count").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        try:
            allocation = {row[0]: row[1] for row in conn.execute("SELECT name,sum(pgsize) FROM dbstat GROUP BY name")}
            allocation_status = "measured_with_dbstat"
        except sqlite3.OperationalError:
            allocation, allocation_status = {}, "dbstat_unavailable"
    return {"database": str(database), "observed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "read_only_sql": True,
            "counts": counts, "logical_payload_bytes": logical, "allocated_bytes_by_object": allocation,
            "event_layer_status_counts": event_layers,
            "allocation_status": allocation_status, "page_count": pages, "page_size": page_size,
            "reusable_free_bytes": free_pages * page_size,
            "file_bytes": {name: candidate.stat().st_size if candidate.exists() else 0 for name, candidate in (("main", database), ("wal", Path(str(database) + "-wal")), ("shm", Path(str(database) + "-shm")))},
            "backup_bytes": "not_scanned", "physical_cleanup_executed": False}


def replay() -> dict:
    rows = []
    with tempfile.TemporaryDirectory(prefix="shinku-memory-lifecycle-replay-") as scratch:
        def fixture(name, **changes):
            policy = replace(LifecyclePolicy(mode="active"), **changes)
            clock = [1_800_000_000]
            store = MemoryStore(Path(scratch) / f"{name}.sqlite3", lifecycle_policy=policy)
            store.lifecycle.wall_time = lambda: clock[0]
            store.record_event(conversation_id="fictional-owner", scope="private:fictional-owner", text="我喜欢安静的环境和自然的红茶。", source_id="fixture-source", created_at=clock[0], expires_at=clock[0] + 60, metadata={"sender_id": "fictional-owner", "role": "user"})
            record = store.event_by_source_id(source_id="fixture-source", scope="private:fictional-owner")
            return store, clock, record

        def strengthen(store, record, clock, key):
            return store.lifecycle.reinforce_remention(record=record, user_text=record.text, actor_id="fictional-owner", evidence_id=key, now=clock[0], session_id="fixture-session")

        def run(name, check):
            started = time.monotonic()
            try:
                detail = check()
                rows.append({"name": name, "passed": True, "detail": detail, "elapsed_seconds": round(time.monotonic() - started, 4)})
            except Exception as exc:
                rows.append({"name": name, "passed": False, "error": type(exc).__name__, "message": str(exc)[:300]})

        def tier_path():
            store, clock, record = fixture("tiers")
            assert store.lifecycle.state(record.record_id)["tier"] == 1
            path = [1]
            for index, day in enumerate((0, 1, 2, 7, 10, 15)):
                clock[0] = 1_800_000_000 + day * 86400
                assert strengthen(store, record, clock, f"request-{index}")["accepted"]
                path.append(store.lifecycle.state(record.record_id)["tier"])
            assert path[-1] == 3
            assert store.event_by_source_id(source_id=record.source_id, scope=record.scope).confidence == record.confidence
            return {"tier_path": path, "truth_confidence_unchanged": True}
        run("L1_L2_spaced_L3", tier_path)

        def burst():
            store, clock, record = fixture("burst")
            assert strengthen(store, record, clock, "first")["accepted"]
            accepted = sum(bool(strengthen(store, record, clock, f"burst-{i}")["accepted"]) for i in range(30))
            assert accepted == 0
            assert store.lifecycle.state(record.record_id)["tier"] == 2
            return {"additional_burst_accepts": accepted}
        run("burst_does_not_permanent", burst)

        def duplicate_restart():
            store, clock, record = fixture("restart")
            assert strengthen(store, record, clock, "stable-request")["accepted"]
            reopened = MemoryStore(store.path, lifecycle_policy=store.lifecycle.policy)
            assert not strengthen(reopened, record, clock, "stable-request")["accepted"]
            assert reopened.lifecycle.state(record.record_id)["reinforce_count"] == 1
            return {"count_after_restart_replay": 1}
        run("idempotence_across_restart", duplicate_restart)

        def retrieval_only():
            store, clock, record = fixture("retrieval")
            for _ in range(4):
                store.search(query="红茶", scope=record.scope, now=clock[0])
            assert store.lifecycle.state(record.record_id)["reinforce_count"] == 0
            return {"retrievals": 4, "reinforcements": 0}
        run("retrieval_not_rehearsal", retrieval_only)

        def vacation():
            store, clock, record = fixture("vacation")
            clock[0] += 365 * 86400
            assert store.lifecycle.scan(now=clock[0], scope=record.scope)["actions"] == []
            assert store.event_by_source_id(source_id=record.source_id, scope=record.scope)
            return {"days_silent": 365, "original_retained": True}
        run("vacation_requires_idle_turns", vacation)

        def shadow():
            store, clock, record = fixture("shadow", mode="shadow")
            proposal = strengthen(store, record, clock, "r")
            assert not proposal["applied"] and proposal["proposed_tier"] == 2
            assert store.lifecycle.state(record.record_id)["tier"] == 1
            assert store.event_by_source_id(source_id=record.source_id, scope=record.scope).text == record.text
            return {"proposed_tier": 2, "real_tier": 1, "original_retained": True}
        run("shadow_no_tier_or_text_mutation", shadow)

        def scope_isolation():
            store, clock, record = fixture("scope")
            for i in range(20):
                clock[0] += 86400
                store.lifecycle.note_interaction(scope="group:noise", evidence_id=f"noise-{i}", now=clock[0])
            assert store.lifecycle.scan(now=clock[0], scope=record.scope)["actions"] == []
            assert not store.search(query="红茶", scope="group:noise", now=clock[0])
            return {"foreign_scope_cannot_read_or_age_owner": True}
        run("scope_privacy_and_clock", scope_isolation)

        def source_guard():
            store, clock, record = fixture("source-guard")
            assert not store.lifecycle.reinforce_used(record=record, quote=record.record_id, final_text=record.record_id, actor_id="fictional-owner", evidence_id="forged", now=clock[0])["accepted"]
            store.forget(query="红茶", scope=record.scope, now=clock[0])
            assert not strengthen(store, record, clock, "old-snapshot")["accepted"]
            return {"supplied_ids_and_stale_sources_rejected": True}
        run("bound_evidence_required", source_guard)

        def backup_restore():
            store, clock, record = fixture("restore-source")
            backup_path = Path(scratch) / "restore-target.sqlite3"
            with closing(sqlite3.connect(store.path)) as source, closing(sqlite3.connect(backup_path)) as destination:
                source.backup(destination)
            store.forget(query="红茶", scope=record.scope, purge=True, now=clock[0])
            revocations = store.lifecycle.export_revocations(scope=record.scope)
            restored = MemoryStore(backup_path)
            result = restored.lifecycle.import_revocations(revocations, approved_scope=record.scope, confirmed=True)
            assert not restored.search(query="红茶", scope=record.scope, now=clock[0])
            assert result["records_disabled"] == 1
            return {"post_backup_revocations_reapplied": True, **result}
        run("forget_then_restore_no_resurrection", backup_restore)

        def budget():
            store, clock, record = fixture("budget", auto_raw_budget_bytes=1)
            for index in range(7):
                clock[0] += 3 * 86400
                strengthen(store, record, clock, f"r-{index}")
            assert store.lifecycle.state(record.record_id)["tier"] == 2
            return {"budget_blocks_new_auto_permanent": True}
        run("bounded_auto_permanent_growth", budget)

        def recall_after_reclaim():
            store, clock, _ = fixture("summary", l1_seconds=1, l2_seconds=2, migration_grace_seconds=1, summary_grace_seconds=2, minimum_idle_turns=1, summary_max_chars=25)
            raw = "我喜欢安静的环境和自然的红茶。" + "这是一段普通天气闲聊。" * 40
            store.record_event(conversation_id="fictional-owner", scope="private:fictional-owner", text=raw, source_id="long-fixture", created_at=clock[0], metadata={"sender_id": "fictional-owner", "role": "user"})
            for index in range(3):
                clock[0] += 86400
                store.lifecycle.note_interaction(scope="private:fictional-owner", evidence_id=f"activity-{index}", now=clock[0])
            first = store.lifecycle.scan(now=clock[0], scope="private:fictional-owner")
            assert any(item["action"] == "summarize" for item in first["actions"])
            result = store.lifecycle.scan(now=clock[0] + 3, scope="private:fictional-owner", allow_reclaim=True, approved_scope="private:fictional-owner")
            current = store.event_by_source_id(source_id="long-fixture", scope="private:fictional-owner")
            assert current and current.metadata["raw_detail_unavailable"] and len(current.text) < len(raw)
            assert any(item.source_id == "long-fixture" for item in store.search(query="红茶", scope="private:fictional-owner", now=clock[0] + 3))
            assert result["logical_bytes_reclaimed"] > 0
            return {"summary_recall_retained": True, "logical_bytes_reclaimed": result["logical_bytes_reclaimed"], "raw_detail_unavailable": True, "file_compacted": False}
        run("summary_grace_reclaim_recall", recall_after_reclaim)

    return {"kind": "isolated_actual_sqlite_time_replay", "cases": rows, "passed": sum(row["passed"] for row in rows), "total": len(rows),
            "all_passed": all(row["passed"] for row in rows), "model_calls": 0, "qq_messages_sent": 0,
            "production_mutations": 0, "scope": "disposable_fictional_fixture_only"}


def main() -> int:
    parser = argparse.ArgumentParser(description="只读记忆基线 / 隔离生命周期回放 / 恢复撤销记录")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--baseline", type=Path)
    modes.add_argument("--replay", action="store_true")
    modes.add_argument("--export-revocations", type=Path)
    modes.add_argument("--restore-revocations", type=Path)
    parser.add_argument("--scope", default="")
    parser.add_argument("--revocations", type=Path)
    parser.add_argument("--confirm", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.baseline:
        result = readonly_baseline(args.baseline)
    elif args.replay:
        result = replay()
    elif args.export_revocations:
        if not args.scope or not args.output:
            parser.error("撤销导出需要 --scope 与显式 --output")
        database = args.export_revocations.resolve(strict=True)
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM memory_revocations WHERE scope=? ORDER BY revoked_at", (args.scope,)).fetchall()
        result = {"version": 1, "scope": args.scope, "revocations": [dict(row) for row in rows]}
    else:
        if not args.scope or not args.revocations or args.confirm != "REAPPLY_REVOCATIONS":
            parser.error("恢复需要 --scope、--revocations 和 --confirm REAPPLY_REVOCATIONS；在恢复库上线前执行")
        database = args.restore_revocations.resolve(strict=True)
        payload = json.loads(args.revocations.read_text(encoding="utf-8"))
        result = MemoryStore(database).lifecycle.import_revocations(payload, approved_scope=args.scope, confirmed=True)
    output = args.output or ROOT / "docs" / ("MEMORY_LIFECYCLE_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Evidence files are append-only artifacts; never overwrite a prior run.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"report": str(output.resolve()), "kind": result.get("kind", "read_only_baseline"), "passed": result.get("passed"), "total": result.get("total")}, ensure_ascii=False))
    return 0 if result.get("all_passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
