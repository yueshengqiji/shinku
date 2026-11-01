"""F local maintenance demo: preview, fault, repair, backup reconciliation, rollback.

All destructive fault simulation is confined to a newly created fictional DB.
There is no saved profile, model call, embedding, QQ transport or production DB.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from contextlib import closing
import time
from unittest.mock import patch

import gray_memory_stage_a as a
import gray_memory_sequence as sequence
from shinku.memory.source_reader import MemorySourceReader
from shinku.memory.models import MemoryRecord

ROOT = a.ROOT


def source_fingerprint(store):
    with store._connect() as conn:
        events = [list(row) for row in conn.execute("SELECT event_id,scope,source_id,text,layer,created_at,status,metadata_json "
                                                  "FROM memory_events ORDER BY event_id")]
        facts = [list(row) for row in conn.execute("SELECT fact_id,scope,source_id,text,created_at,updated_at,status,metadata_json "
                                                 "FROM memory_facts ORDER BY fact_id")]
    return hashlib.sha256(json.dumps([events, facts], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    output = ROOT / "docs" / ("MEMORY_STAGE_F_MAINTENANCE_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(exist_ok=False)
    report = {"status": "running", "source_snapshot_before": {**sequence.snapshot(),
        "scripts/gray_memory_maintenance.py": a.sha(Path(__file__))}, "phases": [],
        "isolation": {"fictional_only": True, "new_database": str(output / "memory.sqlite3"),
            "production_memory_read": False, "production_memory_write": False, "model_calls": 0,
            "embedding_calls": 0, "qq_transport": False}, "fault_tables": ["memory_search_keys", "memory_current_keys"]}
    now = a.timestamp("2026-10-02T09:00:00+08:00")
    policy = a.MemoryPolicy(search_keys_enabled=True, current_understanding_enabled=True,
        evidence_package_enabled=True, history_needs_enabled=True, master_user_ids=(a.OWNER,))
    store = a.MemoryStore(output / "memory.sqlite3")
    memory = a.MemoryService(store=store, router=a.MemoryRouter(), clock=lambda: now, policy=policy)
    scope = memory.scope_for(a.OWNER, "private")
    counters = {"sqlite_connections": 0, "source_handle_requests": 0, "matched_handles": 0}
    connect = store._connect
    read_sources = MemorySourceReader.read_sources
    def counted_connect():
        counters["sqlite_connections"] += 1
        return connect()
    def counted_reads(reader, **kwargs):
        result = read_sources(reader, **kwargs)
        if reader.store is store:
            counters["source_handle_requests"] += len(kwargs["requests"])
            counters["matched_handles"] += sum(row is not None for row in result)
        return result
    store._connect = counted_connect
    def phase(name, operation):
        before = dict(counters)
        started = time.monotonic()
        value = operation()
        report["phases"].append({"name": name, "result": value, "seconds": round(time.monotonic() - started, 6),
            "cost": {key: counters[key] - before[key] for key in counters}})
        return value
    def ask(question="你记得星河项目方案A实际强度吗？"):
        return memory.prepare_turn(conversation_id=a.OWNER, conversation_type="private", user_message=question,
            message_meta={"sender_id": a.OWNER}).as_dict()
    try:
        with patch.object(MemorySourceReader, "read_sources", counted_reads):
            memory.record_turn(conversation_id=a.OWNER, conversation_type="private", message_id="param",
                text="星河项目方案A实际强度0.65。", metadata={"sender_id": a.OWNER})
            memory.record_turn(conversation_id=a.OWNER, conversation_type="private", message_id="preference",
                text="记住我平时喜欢清淡的面食。", metadata={"sender_id": a.OWNER})
            before = phase("working", ask)
            phase("natural_preference_and_workbench", lambda: ask("帮我推荐晚饭。"))
            original_source_hash = source_fingerprint(store)
            with store._connect() as conn:
                # These tables were created in this run's new fictional DB.
                conn.execute("DROP TABLE memory_search_keys")
                conn.execute("DROP TABLE memory_current_keys")
            fallback = phase("missing_derived_tables_fallback", ask)
            if "0.65" not in fallback["rendered"]:
                raise AssertionError("fault_fallback_lost_source")
            phase("scope_bounded_preview", lambda: {
                "search": store.search_index.rebuild(scope=scope, now=now, dry_run=True, force=True),
                "current": memory.current_understanding.rebuild(scope=scope, now=now, dry_run=True, force=True)})
            with store._connect() as conn:
                if conn.execute("SELECT count(*) FROM sqlite_master WHERE name IN ('memory_search_keys','memory_current_keys')").fetchone()[0]:
                    raise AssertionError("preview_created_schema")
            phase("explicit_scope_repair", lambda: {
                "search": store.search_index.rebuild(scope=scope, now=now, dry_run=False, force=True),
                "current": memory.current_understanding.rebuild(scope=scope, now=now, dry_run=False, force=True)})
            phase("idempotent_retry", lambda: {
                "search": store.search_index.rebuild(scope=scope, now=now, dry_run=False),
                "current": memory.current_understanding.rebuild(scope=scope, now=now, dry_run=False)})
            repaired = phase("repaired_same_question", ask)
            report["repair_source_tables_unchanged"] = original_source_hash == source_fingerprint(store)
            report["repair_render_identical"] = before["rendered"] == repaired["rendered"]
            phase("metadata_only_inspection", lambda: {
                "search": store.search_index.inspect(scope=scope),
                "current": memory.current_understanding.inspect(scope=scope, now=now),
                "workbench": memory.history_workbench.inspect(scope=scope, requester=a.OWNER, now=now)})
            backup = output / "isolated-old-backup.sqlite3"
            with closing(sqlite3.connect(store.path)) as source, closing(sqlite3.connect(backup)) as destination:
                source.backup(destination)
            old = tuple(MemoryRecord(**{key: row[key] for key in
                ("record_id", "layer", "scope", "text", "created_at", "score", "importance", "confidence", "source_id", "metadata")})
                for row in before["records"])
            forgotten = memory.record_turn(conversation_id=a.OWNER, conversation_type="private", message_id="forget",
                text="忘记星河项目", metadata={"sender_id": a.OWNER})
            ledgers = [store.lifecycle.export_revocations(scope=item) for item in (scope, memory.owner_scope)]
            phase("after_forget", ask)
            # Restore only this run's explicit fictional path, then reconcile
            # revocations BEFORE any retrieval or derived rebuilding.
            with store._lock, closing(sqlite3.connect(backup)) as source, closing(sqlite3.connect(store.path)) as destination:
                source.backup(destination)
            phase("restored_backup_reconciled_first", lambda: [store.lifecycle.import_revocations(
                ledger, approved_scope=ledger["scope"], confirmed=True) for ledger in ledgers])
            after_restore = phase("restored_then_reindexed", lambda: {
                "search": store.search_index.rebuild(scope=scope, now=now, dry_run=False, force=True),
                "current": memory.current_understanding.rebuild(scope=scope, now=now, dry_run=False, force=True),
                "answer_sources": ask()})
            report["revoked_text_not_restored_to_delivery"] = "0.65" not in after_restore["answer_sources"]["rendered"]
            memory.policy = replace(policy, search_keys_enabled=False, current_understanding_enabled=False,
                evidence_package_enabled=False, history_needs_enabled=False)
            store.configure_search_keys(enabled=False)
            rollback = phase("feature_rollback_with_old_selected_handles", lambda: {"rendered": memory._render(old),
                "fresh_question": ask(), "revocations": sum(len(ledger["revocations"]) for ledger in ledgers)})
            report["rollback_keeps_revocation"] = "0.65" not in rollback["rendered"] and "0.65" not in rollback["fresh_question"]["rendered"]
            report["forgotten"] = forgotten.as_dict() if forgotten else {}
            if not all(report[key] for key in ("repair_source_tables_unchanged", "repair_render_identical",
                                               "revoked_text_not_restored_to_delivery", "rollback_keeps_revocation")):
                raise AssertionError("maintenance_acceptance_failed")
            report["status"] = "completed"
            return 0
    except Exception as exc:
        report["status"], report["error_type"] = "failed", type(exc).__name__
        raise
    finally:
        report["source_snapshot_after"] = {**sequence.snapshot(), "scripts/gray_memory_maintenance.py": a.sha(Path(__file__))}
        report["source_unchanged_during_run"] = report["source_snapshot_before"] == report["source_snapshot_after"]
        report["cost_total"] = counters
        memory.lifecycle_review.stop()
        store.lifecycle.stop_worker()
        store.stop_embedding_worker()
        (output / "maintenance.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# F 隔离维护演示", "", "状态：" + report["status"], "",
            "所有故障及备份恢复仅发生在本次新建虚构 SQLite 库，不是生产操作；没有模型回复或线上证明。", ""]
        for row in report["phases"]:
            lines += ["## " + row["name"], "", f"耗时{row['seconds']}秒；成本{row['cost']}。", "",
                "```json", json.dumps(row["result"], ensure_ascii=False, indent=2), "```", ""]
        (output / "MAINTENANCE.md").write_text("\n".join(lines), encoding="utf-8")
        print("REPORT " + str(output), flush=True)
        print("STATUS " + report["status"], flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
