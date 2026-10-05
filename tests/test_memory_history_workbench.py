"""E workbench stores versioned handles, never old text or execution authority."""
from pathlib import Path
import tempfile
import unittest

from shinku.memory import MemoryStore
from shinku.memory.source_reader import MemorySourceReader
from shinku.memory.workbench import HistoryWorkbench, history_need_for


class HistoryWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.temp.name) / "memory.sqlite3")
        self.now = 1790900000
        self.reader = MemorySourceReader(self.store)
        self.workbench = HistoryWorkbench(self.store, ttl_seconds=120, max_entries=3)
        self.store.record_event(conversation_id="owner", scope="private:owner", source_id="one",
            text="我喜欢清淡面食。", created_at=self.now, metadata={"role": "user", "sender_id": "owner"})
        self.row = self.store.event_by_source_id(source_id="one", scope="private:owner")
        self.row = self.reader.read_source(record_id=self.row.record_id, scope=self.row.scope, now=self.now)

    def tearDown(self):
        self.temp.cleanup()

    def put(self):
        return self.workbench.put(scope="private:owner", requester="owner", topic="diet", records=(self.row,),
            allowed_scopes=("private:owner",), now=self.now)

    def get(self, *, now=None, requester="owner", scopes=("private:owner",), query="那换一种呢？"):
        return self.workbench.consider(scope="private:owner", requester=requester, query=query,
            allowed_scopes=scopes, now=self.now if now is None else now)

    def test_personal_recommendations_need_preferences_without_recall_words(self):
        self.assertEqual(history_need_for("帮我推荐晚饭。").topic, "diet")
        self.assertEqual(history_need_for("今天喝点什么茶？").topic, "drink")
        self.assertEqual(history_need_for("工作环境该怎么布置？").topic, "environment")

    def test_factual_questions_other_peoples_tasks_and_greetings_are_not_personal_needs(self):
        for query in ("你好", "为什么川菜那么辣？", "给我朋友推荐晚饭", "解释一下咖啡因", "帮我测试这个接口"):
            self.assertIsNone(history_need_for(query), query)

    def test_followup_revalidates_only_same_scope_and_subject(self):
        self.put()
        need, records = self.get()
        self.assertEqual(need.topic, "diet")
        self.assertEqual(records[0].text, self.row.text)
        self.assertIsNone(self.get(requester="other")[0])
        self.assertIsNone(self.get(scopes=("private:owner", "private:master"))[0])

    def test_expiry_and_backwards_clock_fail_closed(self):
        self.put()
        self.assertIsNone(self.get(now=self.now + 121)[0])
        self.put()
        self.assertIsNone(self.get(now=self.now - 1)[0])

    def test_changed_text_identity_and_revocation_invalidate_old_handles(self):
        self.put()
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_events SET text=? WHERE source_id='one'", ("我现在不喜欢面食。",))
        self.assertIsNone(self.get()[0])
        self.assertNotIn("我喜欢", str(self.workbench.inspect(scope="private:owner", requester="owner", now=self.now)))

    def test_forget_listener_drops_workbench_without_cached_text(self):
        self.put()
        self.store.forget(query="面食", record_id=self.row.record_id, scope=self.row.scope, now=self.now)
        self.assertIsNone(self.get()[0])

    def test_topic_change_and_greeting_clear_old_topic(self):
        self.put()
        self.get(query="换个话题，你记得星河项目方案A参数吗？")
        self.assertIsNone(self.get()[0])

    def test_deictic_project_continuation_is_not_misparsed_as_a_new_project_name(self):
        self.workbench.put(scope=self.row.scope, requester="owner", topic="project:星河项目", records=(self.row,),
            allowed_scopes=(self.row.scope,), now=self.now)
        need, rows = self.get(query="继续刚才那个项目")
        self.assertEqual(need.topic, "project:星河项目")
        self.assertTrue(rows)
        self.put()
        self.get(query="你好")
        self.assertIsNone(self.get()[0])

    def test_inspect_contains_only_bounded_handles_and_versions(self):
        self.put()
        result = self.workbench.inspect(scope="private:owner", requester="owner", now=self.now)
        self.assertEqual(result["topic"], "diet")
        self.assertFalse(result["includes_source_text"])
        self.assertNotIn(self.row.text, str(result))
        self.assertFalse(result["execution_authority"])

    def test_foreign_and_unbound_candidates_cannot_enter_workbench(self):
        from dataclasses import replace
        bad = replace(self.row, scope="private:foreign")
        self.assertFalse(self.workbench.put(scope="private:owner", requester="owner", topic="diet", records=(bad,),
            allowed_scopes=("private:owner",), now=self.now))
        bad = replace(self.row, metadata={})
        self.assertFalse(self.workbench.put(scope="private:owner", requester="owner", topic="diet", records=(bad,),
            allowed_scopes=("private:owner",), now=self.now))

    def test_natural_current_selection_merges_conflict_labels_across_authorized_scopes(self):
        from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService
        from shinku.memory.workbench import natural_preference_records
        service = MemoryService(store=self.store, router=MemoryRouter(), clock=lambda: self.now,
            policy=MemoryPolicy(current_understanding_enabled=True, master_user_ids=("owner",)))
        service.record_turn(conversation_id="owner", conversation_type="private", message_id="old",
            text="记住我不吃香菜。", metadata={"sender_id": "owner"})
        service.record_turn(conversation_id="room", conversation_type="group", message_id="new",
            text="记住我喜欢吃香菜。", metadata={"sender_id": "owner", "shinku_involved": True})
        rows = natural_preference_records(service, topic="diet", requester="owner",
            scopes=("private:owner", service.owner_scope), now=self.now, limit=5)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(entry["state"] == "conflict_pending_confirmation"
            for row in rows for entry in row.metadata["current_understanding"]))


if __name__ == "__main__":
    unittest.main()
