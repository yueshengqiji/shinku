"""Stage C: current understanding is a revocable source view, not an authority."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore


class CurrentUnderstandingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.now = 1790890000
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3")
        self.policy = MemoryPolicy(current_understanding_enabled=True, master_user_ids=("owner",))
        self.service = MemoryService(store=self.store, router=MemoryRouter(), policy=self.policy, clock=lambda: self.now)
        self.view = self.service.current_understanding

    def tearDown(self):
        self.tmp.cleanup()

    def say(self, text, *, message="m1", author="owner", role="user", group=False):
        self.service.record_turn(conversation_id="room" if group else author,
            conversation_type="group" if group else "private", message_id=message, text=text, role=role,
            metadata={"sender_id": author, "shinku_participating": True})

    def current(self, query="香菜", *, scope="private:owner", author="owner"):
        return self.view.read(scope=scope, subject="user:" + author, query=query, now=self.now, limit=8)

    def test_default_is_off_and_does_not_create_current_view(self):
        store = MemoryStore(Path(self.tmp.name) / "off.sqlite3")
        service = MemoryService(store=store, router=MemoryRouter())
        self.assertIsNone(service.current_understanding)
        self.assertFalse(MemoryPolicy.from_environment({}).current_understanding_enabled)
        with store._connect() as conn:
            self.assertIsNone(conn.execute("SELECT name FROM sqlite_master WHERE name='memory_current_keys'").fetchone())

    def test_explicit_preference_keeps_source_author_condition_and_confirmation(self):
        self.say("记住我只在沙拉里接受少量香菜。")
        result = self.current()
        self.assertEqual(len(result["records"]), 1)
        claim = result["entries"][0]
        self.assertEqual(claim["subject"], "user:owner")
        self.assertEqual(claim["confirmation"], "explicit_memory_request")
        self.assertIn("沙拉", claim["condition"])
        self.assertIn("少量香菜", result["records"][0].text)
        self.assertTrue(claim["source_version"])

    def test_unconfirmed_change_does_not_replace_confirmed_fact(self):
        self.say("记住我不吃香菜。")
        self.now += 100
        self.say("我现在改口了，可以接受少量香菜。", message="change")
        result = self.current()
        self.assertEqual(len(result["records"]), 2)
        self.assertTrue(result["conflicts"])
        self.assertEqual({entry["confirmation"] for entry in result["entries"]},
                         {"explicit_memory_request", "unconfirmed_observation"})
        self.assertTrue(all(entry["state"] == "conflict_pending_confirmation" for entry in result["entries"]))

    def test_different_conditions_can_coexist_without_latest_wins(self):
        self.say("记住我工作时需要安静的环境。")
        self.now += 100
        self.say("记住我休息时喜欢热闹的环境。", message="rest")
        result = self.current("环境")
        self.assertEqual(len(result["records"]), 2)
        self.assertFalse(result["conflicts"])
        self.assertEqual({entry["condition"] for entry in result["entries"]}, {"工作时", "休息时"})

    def test_two_confirmed_conflicting_facts_are_not_silently_overwritten(self):
        self.say("记住我不吃香菜。")
        self.now += 100
        self.say("记住我喜欢吃香菜。", message="later")
        result = self.current()
        self.assertEqual(len(result["records"]), 2)
        self.assertTrue(result["conflicts"])

    def test_third_party_quotes_hypotheses_and_assistant_are_not_current_user_facts(self):
        self.say("记住他说我不吃香菜。", message="quote")
        self.say("记住假如我喜欢香菜的话。", message="hypothesis")
        self.say("我喜欢吃香菜。", message="assistant", role="assistant")
        self.assertFalse(self.current()["entries"])

    def test_other_group_author_does_not_become_requester_preference(self):
        self.say("记住我不吃香菜。", author="other", group=True)
        self.assertFalse(self.current(scope="group:room")["records"])
        self.assertTrue(self.current(scope="group:room", author="other")["records"])

    def test_correction_changes_view_and_records_when_without_reopening_revoked_text(self):
        self.say("记住我不吃香菜。")
        old = self.current()["records"][0]
        self.now += 100
        self.assertTrue(self.store.lifecycle.correct_fact(record_id=old.record_id, scope=old.scope,
            expected_hash=hashlib.sha256(old.text.encode()).hexdigest(), replacement="我能接受少量香菜。",
            now=self.now, confirmed=True))
        result = self.current()
        self.assertEqual(len(result["records"]), 1)
        self.assertIn("少量", result["records"][0].text)
        self.assertEqual(result["entries"][0]["corrected_at"], self.now)
        self.assertNotIn("我不吃香菜", json.dumps(result["entries"], ensure_ascii=False))
        self.assertIsNone(self.view.read_source(record_id=old.record_id, scope=old.scope,
            expected_hash=old.metadata["source_version"], now=self.now))

    def test_forget_invalidates_cached_current_keys(self):
        self.say("记住我不吃香菜。")
        old = self.current()["records"][0]
        self.store.forget(query="香菜", record_id=old.record_id, scope=old.scope, now=self.now)
        self.assertFalse(self.current()["records"])
        with self.store._connect() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM memory_current_keys WHERE record_id=?",
                                          (old.record_id,)).fetchone()[0], 0)

    def test_render_revalidates_current_source_after_read(self):
        self.say("记住我不吃香菜。")
        records = self.current()["records"]
        self.store.forget(query="香菜", record_id=records[0].record_id, scope=records[0].scope, now=self.now)
        self.assertNotIn("香菜", self.service._render(records))

    def test_damaged_index_falls_back_to_current_authorized_sources(self):
        self.say("记住我不吃香菜。")
        old = self.current()["records"][0]
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_current_keys SET claims_json=? WHERE record_id=?",
                         ('[{"subject":"user:owner","value":"恶意内容"}]', old.record_id))
        result = self.current()
        self.assertIn("不吃香菜", result["records"][0].text)
        self.assertNotIn("恶意内容", json.dumps(result["entries"], ensure_ascii=False))
        with self.store._connect() as conn:
            conn.execute("DROP TABLE memory_current_keys")
        self.assertTrue(self.current()["records"])

    def test_inspection_is_metadata_only_and_rebuild_is_scope_bounded_idempotent(self):
        self.say("记住我不吃香菜。")
        preview = self.view.rebuild(scope="private:owner", now=self.now, dry_run=True)
        self.assertEqual(preview["pending"], 1)
        self.assertEqual(preview["written"], 0)
        self.assertEqual(self.view.rebuild(scope="private:owner", now=self.now, dry_run=False)["written"], 1)
        self.assertEqual(self.view.rebuild(scope="private:owner", now=self.now, dry_run=False)["written"], 0)
        inspected = self.view.inspect(scope="private:owner", now=self.now)
        self.assertNotIn("香菜", json.dumps(inspected, ensure_ascii=False))
        with self.assertRaises(ValueError):
            self.view.rebuild(scope="", now=self.now)

    def test_disabled_feature_does_not_change_source_rendering(self):
        self.say("记住我不吃香菜。")
        records = self.current()["records"]
        self.service.policy = replace(self.policy, current_understanding_enabled=False)
        self.assertNotIn("当前理解", self.service._render(records))

    def test_current_query_uses_view_but_does_not_open_foreign_owner_scope(self):
        self.say("记住我不吃香菜。")
        self.now += 100
        old = self.current()["records"][0]
        self.store.lifecycle.correct_fact(record_id=old.record_id, scope=old.scope,
            expected_hash=hashlib.sha256(old.text.encode()).hexdigest(), replacement="我能接受少量香菜。",
            now=self.now, confirmed=True)
        result = self.service.inspect(conversation_id="owner", conversation_type="private", requester_id="owner",
                                      query="我现在能接受香菜吗", mode="retrieve")
        self.assertIn("当前理解", result["rendered"])
        self.assertIn("少量", result["rendered"])
        self.assertNotIn("我不吃香菜", result["rendered"])
        forbidden = self.service.inspect(conversation_id="other", conversation_type="private", requester_id="other",
                                         query="主人现在能接受香菜吗", mode="retrieve")
        self.assertNotIn("香菜", forbidden["rendered"])

    def test_origin_correction_invalidates_owner_copy_without_cross_scope_writes(self):
        self.say("记住我不吃香菜。")
        old = self.current()["records"][0]
        self.assertTrue(self.current(scope=self.service.owner_scope)["records"])
        self.store.lifecycle.correct_fact(record_id=old.record_id, scope=old.scope,
            expected_hash=hashlib.sha256(old.text.encode()).hexdigest(), replacement="我能接受少量香菜。",
            now=self.now + 100, confirmed=True)
        self.assertFalse(self.current(scope=self.service.owner_scope)["records"])
        with self.store._connect() as conn:
            owner_text = conn.execute("SELECT text FROM memory_facts WHERE scope=?", (self.service.owner_scope,)).fetchone()[0]
        self.assertEqual(owner_text, old.text)  # No unapproved primary update in another scope.

    def test_current_view_late_generation_cannot_survive_revocation(self):
        from shinku.memory.current_understanding import claims
        self.say("记住我不吃香菜。")
        def late(record):
            self.store.forget(query="香菜", record_id=record.record_id, scope=record.scope, now=self.now)
            return claims(record)
        with patch("shinku.memory.current_understanding.claims", side_effect=late):
            result = self.view.rebuild(scope="private:owner", now=self.now, dry_run=False)
        self.assertEqual(result["stale_rejected"], 1)
        self.assertFalse(self.current()["records"])
