"""F combined failure/rollback guards on an isolated DB; no production actions."""
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.source_reader import MemorySourceReader
from shinku.memory.workbench import HistoryWorkbench


class UpgradeMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.now = 1790900000
        self.policy = MemoryPolicy(search_keys_enabled=True, current_understanding_enabled=True,
                                   evidence_package_enabled=True)
        self.store = MemoryStore(Path(self.temp.name) / "memory.sqlite3")
        self.service = MemoryService(store=self.store, router=MemoryRouter(), policy=self.policy, clock=lambda: self.now)
        self.reader = MemorySourceReader(self.store)

    def tearDown(self):
        self.temp.cleanup()

    def event(self, text="星河项目方案A费用260元。", source="one", metadata=None):
        self.store.record_event(conversation_id="owner", scope="private:owner", text=text, source_id=source,
            created_at=self.now, metadata=metadata or {"role": "user", "sender_id": "owner"})
        row = self.store.event_by_source_id(source_id=source, scope="private:owner")
        return self.reader.read_source(record_id=row.record_id, scope=row.scope, now=self.now)

    def summary(self, parent, *, source="summary:one"):
        return self.event(parent.text, source=source, metadata={"record_type": "summary",
            "raw_detail_unavailable": False, "source_record_ids": [parent.record_id],
            "source_content_hashes": {parent.record_id: hashlib.sha256(parent.text.encode()).hexdigest()}})

    def test_search_missing_table_has_noncreating_preview_and_idempotent_repair(self):
        self.event()
        with self.store._connect() as conn:
            conn.execute("DROP TABLE memory_search_keys")
        preview = self.store.search_index.rebuild(scope="private:owner", now=self.now, force=True, dry_run=True)
        self.assertTrue(preview["schema_missing"])
        self.assertEqual(preview["written"], 0)
        with self.store._connect() as conn:
            self.assertIsNone(conn.execute("SELECT name FROM sqlite_master WHERE name='memory_search_keys'").fetchone())
        self.assertEqual(self.store.search_index.rebuild(scope="private:owner", now=self.now, force=True, dry_run=False)["written"], 1)
        self.assertEqual(self.store.search_index.rebuild(scope="private:owner", now=self.now, dry_run=False)["written"], 0)

    def test_current_missing_table_has_noncreating_preview_and_idempotent_repair(self):
        self.service.record_turn(conversation_id="owner", conversation_type="private", message_id="pref",
            text="记住我不吃香菜。", metadata={"sender_id": "owner"})
        view = self.service.current_understanding
        with self.store._connect() as conn:
            conn.execute("DROP TABLE memory_current_keys")
        preview = view.rebuild(scope="private:owner", now=self.now, force=True, dry_run=True)
        self.assertTrue(preview["schema_missing"])
        self.assertEqual(preview["written"], 0)
        self.assertGreater(view.rebuild(scope="private:owner", now=self.now, force=True, dry_run=False)["written"], 0)
        self.assertEqual(view.rebuild(scope="private:owner", now=self.now, dry_run=False)["written"], 0)

    def test_rollback_keeps_source_revocation_even_with_old_selected_snapshot(self):
        row = self.event()
        self.store.forget(query="星河", record_id=row.record_id, scope=row.scope, now=self.now)
        self.service.policy = replace(self.policy, search_keys_enabled=False, current_understanding_enabled=False,
                                      evidence_package_enabled=False)
        self.store.configure_search_keys(enabled=False)
        self.assertNotIn("260元", self.service._render((row,)))

    def test_recent_render_keeps_revocation_even_with_old_selected_snapshot(self):
        row = self.event()
        self.store.forget(query="星河", record_id=row.record_id, scope=row.scope, now=self.now)
        self.assertNotIn("260元", self.service._render((row,), mode="recent"))

    def test_explicit_summary_type_is_not_labelled_original_source(self):
        parent = self.event()
        derived = self.summary(parent)
        self.assertEqual(derived.metadata["reading_representation"], "summary")

    def test_changed_parent_cannot_supply_old_summary_text(self):
        parent = self.event()
        derived = self.summary(parent)
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_events SET text=? WHERE source_id='one'", ("星河项目方案A费用300元。",))
        self.assertIsNone(self.reader.read_source(record_id=derived.record_id, scope=derived.scope, now=self.now))

    def test_changed_grandparent_invalidates_transitive_summary(self):
        parent = self.event()
        mid = self.summary(parent)
        derived = self.summary(mid, source="summary:two")
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_events SET text=? WHERE source_id='one'", ("星河项目方案A费用300元。",))
        self.assertIsNone(self.reader.read_source(record_id=derived.record_id, scope=derived.scope, now=self.now))

    def test_dependency_cycle_fails_closed_instead_of_recursing_forever(self):
        import json
        row = self.event()
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_events SET metadata_json=? WHERE source_id='one'", (json.dumps({
                "source_record_ids": [row.record_id], "source_content_hashes": {row.record_id: hashlib.sha256(row.text.encode()).hexdigest()}}),))
        self.assertIsNone(self.reader.read_source(record_id=row.record_id, scope=row.scope, now=self.now))

    def test_global_reset_clears_workbench_handles(self):
        row = self.event("我喜欢清淡面食。")
        workbench = HistoryWorkbench(self.store)
        workbench.put(scope=row.scope, requester="owner", topic="diet", records=(row,), allowed_scopes=(row.scope,), now=self.now)
        self.store.reset_all()
        self.assertFalse(workbench.inspect(scope=row.scope, requester="owner", now=self.now)["source_handles"])


if __name__ == "__main__":
    unittest.main()
