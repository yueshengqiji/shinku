"""Stage B integrity tests: real SQLite sources, deterministic local hints."""
from dataclasses import replace
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.search_index import hints


class SearchIndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.now = 1790860000
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3")
        self.policy = MemoryPolicy(search_keys_enabled=True, master_user_ids=("owner",))
        self.service = MemoryService(store=self.store, router=MemoryRouter(), policy=self.policy, clock=lambda: self.now)
        self.index = self.store.search_index

    def tearDown(self):
        self.tmp.cleanup()

    def event(self, text, source="turn:owner:m1", scope="private:owner", role="user", age=0):
        self.store.record_event(conversation_id="owner", scope=scope, text=text, source_id=source,
            created_at=self.now-age, metadata={"role": role, "sender_id": "owner" if role == "user" else "bot"})
        return self.store.event_by_source_id(source_id=source, scope=scope)

    def test_default_is_disabled_and_creates_no_derived_table(self):
        store = MemoryStore(Path(self.tmp.name) / "default.sqlite3")
        MemoryService(store=store, router=MemoryRouter())
        with store._connect() as conn:
            self.assertIsNone(conn.execute("SELECT name FROM sqlite_master WHERE name='memory_search_keys'").fetchone())
        self.assertFalse(MemoryPolicy.from_environment({}).search_keys_enabled)
        self.assertTrue(MemoryPolicy.from_environment({"SHINKU_MEMORY_SEARCH_KEYS_ENABLED": "true"}).search_keys_enabled)

    def test_tags_are_vocabulary_not_new_facts_and_original_is_unchanged(self):
        text = "这周胃不舒服，先别推荐辣的；平时我其实挺喜欢川菜。实际强度0.65，不是助手建议的0.8。"
        row = self.event(text)
        self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)
        with self.store._connect() as conn:
            key = conn.execute("SELECT * FROM memory_search_keys").fetchone()
            self.assertNotIn("0.65", key["hints_json"])
            self.assertNotIn("不吃辣", key["hints_json"])
        fresh = self.index.read_source(record_id=row.record_id, scope=row.scope, now=self.now)
        self.assertEqual(fresh.text, text)
        self.assertEqual(fresh.metadata["role"], "user")
        self.assertIn("饮食限制", hints(text))

    def test_synonym_ranking_improves_beyond_raw_lexical_with_noise(self):
        row = self.event("这周胃不舒服，先别推荐辣的；平时喜欢川菜。", age=3600)
        for i in range(8):
            self.event("整理无关文件和草稿。", source=f"noise:{i}")
        self.store.configure_search_keys(enabled=False)
        before = self.store.search(query="最近吃饭有什么限制", scope=row.scope, now=self.now, limit=1)
        self.assertNotEqual(before[0].record_id, row.record_id)
        self.store.configure_search_keys(enabled=True)
        after = self.store.search(query="最近吃饭有什么限制", scope=row.scope, now=self.now, limit=1)
        self.assertEqual(after[0].record_id, row.record_id)
        self.assertEqual(after[0].text, row.text)
        self.assertEqual(after[0].metadata["search_hints"], ["饮食限制"])

    def test_rebuild_is_idempotent_and_preview_writes_no_search_keys(self):
        row = self.event("我喜欢安静，不要背景音乐。")
        preview = self.index.rebuild(scope=row.scope, now=self.now)
        self.assertEqual(preview["pending"], 1)
        self.assertEqual(preview["written"], 0)
        self.assertEqual(self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)["written"], 1)
        self.assertEqual(self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)["written"], 0)
        with self.assertRaises(ValueError):
            self.index.rebuild(scope="", now=self.now)

    def test_generation_failure_retains_raw_retrieval(self):
        row = self.event("实际配置强度0.65。")
        with patch("shinku.memory.search_index.hints", side_effect=RuntimeError("synthetic-generation-failure")):
            result = self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)
            self.assertEqual(result["failed"], 1)
        self.assertEqual(self.store.search(query="实际配置", scope=row.scope, now=self.now)[0].text, row.text)

    def test_stale_completion_after_forget_cannot_commit(self):
        row = self.event("我喜欢安静。")
        original = hints
        def late(text):
            self.store.forget(query="安静", scope=row.scope, now=self.now)
            return original(text)
        with patch("shinku.memory.search_index.hints", side_effect=late):
            report = self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)
        self.assertEqual(report["stale_rejected"], 1)
        with self.store._connect() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM memory_search_keys").fetchone()[0], 0)

    def test_provenance_change_invalidates_old_key_and_read_handle(self):
        row = self.event("我喜欢安静。")
        self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)
        fresh = self.index.read_source(record_id=row.record_id, scope=row.scope, now=self.now)
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?", ('{"role":"assistant"}', row.record_id.removeprefix("event:")))
        self.assertIsNone(self.index.read_source(record_id=row.record_id, scope=row.scope, now=self.now,
                                                expected_hash=fresh.metadata["source_version"]))
        self.assertEqual(self.index.matches(records=[row], query="噪声", scope=row.scope, now=self.now), {})

    def test_foreign_scope_and_date_guards_remain_hard(self):
        row = self.event("我喜欢安静。", age=86400)
        self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)
        self.assertIsNone(self.index.read_source(record_id=row.record_id, scope="group:other", now=self.now))
        self.assertEqual(self.store.search(query="噪声", scope="group:other", now=self.now), [])
        self.assertEqual(self.store.search(query="噪声", scope=row.scope, now=self.now, start_at=self.now-100), [])
        self.assertEqual(self.store.search(query="噪声", scope=row.scope, now=self.now, require_term_overlap=True), [])

    def test_round_reading_only_pairs_explicit_source_and_keeps_roles(self):
        user = self.event("实际强度0.65。")
        self.event("建议下一次试0.8。", source=user.source_id + ":assistant", role="assistant")
        self.event("另一个人聊预算100元。", source="turn:owner:unrelated")
        result = self.index.read_rounds((user,), now=self.now, limit=3)
        self.assertEqual(len(result), 2)
        self.assertEqual([r.metadata["role"] for r in result], ["user", "assistant"])
        self.assertEqual(result[1].metadata["reading_relation"], "explicit_reply")
        rendered = self.service._render(result)
        self.assertIn("0.65", rendered)
        self.assertIn("0.8", rendered)
        self.assertNotIn("100元", rendered)

    def test_revoked_round_neighbor_is_not_resurrected_from_raw_turns(self):
        user = self.event("实际强度0.65。")
        assistant = self.event("建议0.8。", source=user.source_id + ":assistant", role="assistant")
        self.store.record_turn(conversation_id="owner", scope=user.scope, role="assistant", content=assistant.text,
                               source_id=assistant.source_id, created_at=self.now)
        self.store.forget(query="0.8", record_id=assistant.record_id, scope=user.scope, now=self.now)
        self.assertEqual(len(self.index.read_rounds((user,), now=self.now, limit=3)), 1)

    def test_lost_raw_stays_summary_and_is_labelled(self):
        row = self.event("当时试过两组；数值没有保留。")
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?", ('{"raw_detail_unavailable":true,"role":"user"}', row.record_id.removeprefix("event:")))
        records = self.index.read_rounds((row,), now=self.now, limit=2)
        self.assertEqual(records[0].metadata["reading_representation"], "summary")
        self.assertIn("原文不可用", self.service._render(records))

    def test_render_budget_omits_whole_message_never_claims_partial_quote(self):
        row = self.event("这周不要辣，" + "很长的条件" * 100 + "但下周恢复。")
        text = self.service._render((row,))
        self.assertIn("阅读预算不足", text)
        self.assertNotIn("source=", text)
        self.assertNotIn("这周不要辣", text)
        full = self.index.read_source(record_id=row.record_id, scope=row.scope, now=self.now)
        self.assertTrue(full.text.endswith("下周恢复。"))

    def test_correction_invalidates_keys_and_rebuild_reads_new_value(self):
        self.store.upsert_fact(conversation_id="owner", scope="private:owner", fact_key="tea", text="喜欢绿茶",
                               source_id="fact-source", now=self.now)
        row = next(r for r in self.store.search(query="绿茶", scope="private:owner", now=self.now) if r.layer == "fact")
        self.assertTrue(self.store.lifecycle.correct_fact(record_id=row.record_id, scope=row.scope,
            expected_hash=hashlib.sha256(row.text.encode()).hexdigest(), replacement="喜欢红茶", now=self.now, confirmed=True))
        self.assertIsNone(self.index.read_source(record_id=row.record_id, scope=row.scope, now=self.now,
                                                expected_hash=row.metadata["source_version"]))
        result = self.store.search(query="饮品偏好", scope=row.scope, now=self.now)
        self.assertEqual(result[0].text, "喜欢红茶")

    def test_derived_sql_failure_falls_back_without_losing_source(self):
        row = self.event("实际参数0.65。")
        with self.store._connect() as conn:
            conn.execute("DROP TABLE memory_search_keys")
        result = self.store.search(query="实际参数", scope=row.scope, now=self.now)
        self.assertEqual(result[0].text, row.text)
        self.assertEqual(self.store.search_keys_last_status["fallback"], "source_search")
        self.assertEqual(self.index.rebuild(scope=row.scope, now=self.now, force=True, dry_run=False)["written"], 1)
        self.assertTrue(self.index.matches(records=[row], query="设置参数", scope=row.scope, now=self.now))

    def test_source_mismatch_during_generation_rejected(self):
        row = self.event("我喜欢安静。")
        def mutate(text):
            with self.store._connect() as conn:
                conn.execute("UPDATE memory_events SET text='现在不喜欢安静' WHERE event_id=?", (row.record_id.removeprefix("event:"),))
            return hints(text)
        with patch("shinku.memory.search_index.hints", side_effect=mutate):
            result = self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)
        self.assertEqual(result["stale_rejected"], 1)

    def test_bad_json_and_tampered_labels_are_never_search_evidence(self):
        row = self.event("我喜欢安静。")
        self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)
        for payload in ('not-json', '{"饮食限制":true}', '["饮食限制"]'):
            with self.subTest(payload=payload):
                with self.store._connect() as conn:
                    conn.execute("UPDATE memory_search_keys SET hints_json=?", (payload,))
                self.assertEqual(self.index.matches(records=[row], query="吃饭限制", scope=row.scope, now=self.now), {})
        self.assertEqual(self.index.rebuild(scope=row.scope, now=self.now, force=True, dry_run=False)["written"], 1)
        self.assertTrue(self.index.matches(records=[row], query="环境噪声", scope=row.scope, now=self.now))

    def test_generator_version_is_part_of_validity_not_a_fact_upgrade(self):
        row = self.event("我喜欢安静。")
        self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_search_keys SET generator_version='obsolete'")
        self.assertEqual(self.index.matches(records=[row], query="环境噪声", scope=row.scope, now=self.now), {})
        self.assertEqual(self.index.inspect(scope=row.scope)["entries"][0]["state"], "stale")
        self.assertEqual(self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)["written"], 1)

    def test_inspector_has_no_text_and_requires_exact_scope(self):
        row = self.event("我喜欢安静。")
        self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)
        report = self.index.inspect(scope=row.scope)
        self.assertFalse(report["includes_source_text"])
        self.assertNotIn("喜欢", str(report))
        self.assertEqual(report["entries"][0]["state"], "current")
        self.assertEqual(self.index.inspect(scope="group:other")["entries"], [])
        with self.assertRaises(ValueError):
            self.index.inspect(scope="")

    def test_injected_role_or_origin_is_not_cross_paired(self):
        user = self.event("实际强度0.65。")
        neighbor = self.event("建议0.8。", source=user.source_id + ":assistant", role="assistant")
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_events SET conversation_id='another-origin' WHERE event_id=?",
                         (neighbor.record_id.removeprefix("event:"),))
        self.assertEqual(len(self.index.read_rounds((user,), now=self.now, limit=3)), 1)

    def test_complete_message_budget_never_exceeded(self):
        records = tuple(self.event("这个消息有完整的条件。" * 15, source=f"turn:owner:b{i}") for i in range(12))
        rendered = self.service._render(records)
        self.assertLessEqual(len(rendered), self.policy.render_total_chars)
        self.assertNotIn("这个消息有完整的条（", rendered)

    def test_forgetting_removes_warmed_keys_and_disabling_keeps_raw(self):
        row = self.event("我喜欢安静。")
        self.store.search(query="环境噪声", scope=row.scope, now=self.now)
        self.assertEqual(len(self.index.inspect(scope=row.scope)["entries"]), 1)
        self.store.configure_search_keys(enabled=False)
        self.assertEqual(self.store.search(query="安静", scope=row.scope, now=self.now)[0].text, row.text)
        self.assertNotIn("来源回读", self.service._render((row,)))
        self.store.forget(query="安静", scope=row.scope, now=self.now)
        self.assertEqual(self.index.inspect(scope=row.scope)["entries"], [])
        self.assertIsNone(self.index.read_source(record_id=row.record_id, scope=row.scope, now=self.now))

    def test_malformed_derived_schema_does_not_prevent_service_startup(self):
        store = MemoryStore(Path(self.tmp.name) / "malformed-index.sqlite3")
        store.record_event(conversation_id="owner", scope="private:owner", text="实际参数0.65。",
            source_id="turn:owner:startup", created_at=self.now, metadata={"role": "user"})
        with store._connect() as conn:
            conn.execute("CREATE TABLE memory_search_keys (record_id TEXT)")
        service = MemoryService(store=store, router=MemoryRouter(), policy=self.policy, clock=lambda: self.now)
        self.assertIsNone(store.search_index)
        self.assertFalse(store.search_keys_enabled)
        self.assertEqual(store.search_keys_last_status["fallback"], "source_search")
        self.assertEqual(store.search_keys_last_status["phase"], "initialize")
        context = service.prepare_turn(conversation_id="owner", conversation_type="private",
                                       user_message="你记得实际参数吗？")
        self.assertIn("0.65", context.rendered)
        inspected = service.inspect(conversation_id="owner", conversation_type="private", requester_id="owner",
                                    query="实际参数", mode="retrieve")
        self.assertIn("0.65", inspected["rendered"])

    def test_search_readback_preserves_fresh_author_not_stale_candidate_metadata(self):
        row = self.event("实际参数0.65。")
        original = self.index.read_sources
        def change_identity(**kwargs):
            if any(request[0] == row.record_id for request in kwargs["requests"]):
                with self.store._connect() as conn:
                    conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?",
                        (json.dumps({"role": "assistant", "sender_id": "new-author"}), row.record_id.removeprefix("event:")))
            return original(**kwargs)
        with patch.object(self.index, "read_sources", side_effect=change_identity):
            result = self.store.search(query="实际参数", scope=row.scope, now=self.now)
        self.assertEqual(result[0].metadata["role"], "assistant")
        self.assertEqual(result[0].metadata["sender_id"], "new-author")
        self.assertIn("source_version", result[0].metadata)

    def test_date_changed_during_readback_cannot_escape_query_window(self):
        row = self.event("实际参数0.65。")
        original = self.index.read_sources
        def change_date(**kwargs):
            with self.store._connect() as conn:
                conn.execute("UPDATE memory_events SET created_at=? WHERE event_id=?",
                             (self.now + 86400, row.record_id.removeprefix("event:")))
            return original(**kwargs)
        with patch.object(self.index, "read_sources", side_effect=change_date):
            result = self.store.search(query="实际参数", scope=row.scope, now=self.now,
                                      start_at=self.now - 100, end_at=self.now + 100)
        self.assertEqual(result, [])

    def test_neighbor_origin_changed_after_edge_check_is_not_delivered(self):
        user = self.event("实际参数0.65。")
        neighbor = self.event("建议试0.8。", source=user.source_id + ":assistant", role="assistant")
        original = self.index.read_source
        def change_origin(**kwargs):
            if kwargs["record_id"] == neighbor.record_id:
                with self.store._connect() as conn:
                    conn.execute("UPDATE memory_events SET conversation_id='different-origin' WHERE event_id=?",
                                 (neighbor.record_id.removeprefix("event:"),))
            return original(**kwargs)
        with patch.object(self.index, "read_source", side_effect=change_origin):
            result = self.index.read_rounds((user,), now=self.now, limit=3)
        self.assertEqual([record.record_id for record in result], [user.record_id])

    def test_assistant_suffix_without_qq_turn_prefix_is_not_a_reply_edge(self):
        self.event("实际参数0.65。", source="import:unrelated")
        assistant = self.event("建议0.8。", source="import:unrelated:assistant", role="assistant")
        result = self.index.read_rounds((assistant,), now=self.now, limit=3)
        self.assertEqual([record.record_id for record in result], [assistant.record_id])

    def test_representation_layer_change_invalidates_old_source_handle(self):
        row = self.event("只保留摘要：两组参数，原话未保留。")
        fresh = self.index.read_source(record_id=row.record_id, scope=row.scope, now=self.now)
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_events SET layer='legacy_summary' WHERE event_id=?",
                         (row.record_id.removeprefix("event:"),))
        self.assertIsNone(self.index.read_source(record_id=row.record_id, scope=row.scope, now=self.now,
                                                expected_hash=fresh.metadata["source_version"]))

    def test_agent_task_card_is_labelled_summary_not_original_dialogue(self):
        row = self.event("任务已完成：整理项目资料。", source="owner:agent-task-summary:demo")
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?",
                         (json.dumps({"record_type": "agent_task_summary"}), row.record_id.removeprefix("event:")))
        record = self.index.read_source(record_id=row.record_id, scope=row.scope, now=self.now)
        self.assertEqual(record.metadata["reading_representation"], "summary")
        self.assertIn("Agent任务摘要", self.service._render((record,)))

    def test_restore_import_commits_revocations_before_sql_index_listeners(self):
        row = self.event("恢复前记录：配置编号SL-472。")
        self.index.rebuild(scope=row.scope, now=self.now, dry_run=False)
        with self.store._connect() as conn:
            self.store.lifecycle.revoke(conn, scope=row.scope, source_id=row.source_id, text=row.text, now=self.now)
        payload = self.store.lifecycle.export_revocations(scope=row.scope)
        observed_states = []
        def observe_commit(**kwargs):
            with self.store._connect() as conn:
                observed_states.append(conn.execute("SELECT status FROM memory_events WHERE event_id=?",
                    (row.record_id.removeprefix("event:"),)).fetchone()[0])
        self.store.on_forget(observe_commit)
        original_connect = sqlite3.connect
        def short_timeout(*args, **kwargs):
            kwargs["timeout"] = 0.01  # Deterministic lock regression, not a 15-second test wait.
            return original_connect(*args, **kwargs)
        with patch("shinku.memory.store.sqlite3.connect", side_effect=short_timeout):
            self.store.lifecycle.import_revocations(payload, approved_scope=row.scope, confirmed=True)
        self.assertEqual(observed_states, ["forgotten"])
        with self.store._connect() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM memory_search_keys WHERE record_id=?",
                                         (row.record_id,)).fetchone()[0], 0)

    def test_bulk_sources_recheck_each_scope_version_and_revocation(self):
        valid = self.event("实际参数0.65。", source="turn:owner:valid")
        revoked = self.event("撤销标记Z-321。", source="turn:owner:revoked")
        stale = self.event("喜欢安静。", source="turn:owner:stale")
        previous = self.index.read_source(record_id=stale.record_id, scope=stale.scope, now=self.now)
        self.store.forget(query="Z-321", scope=revoked.scope, record_id=revoked.record_id, now=self.now)
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?",
                         ('{"role":"assistant"}', stale.record_id.removeprefix("event:")))
        requests = ((valid.record_id, valid.scope, ""),
                    (valid.record_id, "group:other", ""),
                    (revoked.record_id, revoked.scope, ""),
                    (stale.record_id, stale.scope, previous.metadata["source_version"]),
                    ("invalid-id", valid.scope, ""))
        with patch.object(self.store, "_connect", wraps=self.store._connect) as connects, \
             patch.object(self.store, "expire", wraps=self.store.expire) as expires:
            result = self.index.read_sources(requests=requests, now=self.now)
        self.assertEqual(result[0].text, valid.text)
        self.assertEqual(result[1:], (None, None, None, None))
        self.assertEqual(expires.call_count, 1)
        self.assertEqual(connects.call_count, 2)

    def test_empty_bulk_reads_do_not_scan_or_open_database(self):
        with patch.object(self.store, "_connect", wraps=self.store._connect) as connects:
            self.assertEqual(self.index.read_sources(requests=(), now=self.now), ())
        self.assertEqual(connects.call_count, 0)

    def test_search_connection_cost_does_not_grow_with_candidate_count(self):
        for i in range(40):
            self.event(f"实际设置参数{i}。", source=f"cost:{i}")
        self.index.rebuild(scope="private:owner", now=self.now, dry_run=False)
        with patch.object(self.store, "_connect", wraps=self.store._connect) as connects, \
             patch.object(self.store, "expire", wraps=self.store.expire) as expires:
            result = self.store.search(query="实际设置", scope="private:owner", now=self.now)
        self.assertTrue(result)
        self.assertLessEqual(connects.call_count, 8)
        self.assertLessEqual(expires.call_count, 3)

    def test_bulk_render_rechecks_revocation_after_round_read(self):
        row = self.event("需要撤销的配置0.65。")
        read = self.index.read_rounds((row,), now=self.now, limit=2)
        self.store.forget(query="0.65", scope=row.scope, record_id=row.record_id, now=self.now)
        self.assertNotIn("0.65", self.service._render(read))

    def test_bulk_read_does_not_bypass_expiration(self):
        row = self.event("短期配置0.65。")
        with self.store._connect() as conn:
            conn.execute("UPDATE memory_events SET expires_at=? WHERE event_id=?",
                         (self.now - 1, row.record_id.removeprefix("event:")))
        self.assertEqual(self.index.read_sources(requests=((row.record_id, row.scope, ""),), now=self.now), (None,))

    def test_cold_rebuild_shares_one_commit_without_holding_lock_during_generation(self):
        for i in range(40):
            self.event(f"实际参数{i}。", source=f"cold:{i}")
        original = hints
        def unlocked_generation(text):
            self.assertFalse(self.store._lock._is_owned())
            return original(text)
        with patch("shinku.memory.search_index.hints", side_effect=unlocked_generation), \
             patch.object(self.store, "_connect", wraps=self.store._connect) as connects:
            result = self.index.rebuild(scope="private:owner", now=self.now, dry_run=False)
        self.assertEqual(result["written"], 40)
        self.assertEqual(connects.call_count, 3)

    def test_rebuild_revalidates_early_generated_source_after_later_revocation(self):
        first = self.event("实际参数0.65。", source="turn:owner:first")
        self.event("安静环境。", source="turn:owner:second", age=100)
        original = hints
        def revoke_after_first(text):
            if "安静" in text:
                self.store.forget(query="0.65", record_id=first.record_id, scope=first.scope, now=self.now)
            return original(text)
        with patch("shinku.memory.search_index.hints", side_effect=revoke_after_first):
            result = self.index.rebuild(scope=first.scope, now=self.now, dry_run=False)
        self.assertEqual(result["stale_rejected"], 1)
        self.assertEqual(result["written"], 1)
        with self.store._connect() as conn:
            self.assertIsNone(conn.execute("SELECT record_id FROM memory_search_keys WHERE record_id=?",
                                         (first.record_id,)).fetchone())
