"""Evidence diagnostics count only exact authorized sources actually delivered."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shinku.memory import MemoryPolicy, MemoryRouter, MemoryService, MemoryStore
from shinku.memory.evidence import source_time
from shinku.memory.source_reader import MemorySourceReader
from test_memory_evidence_package import stamp


class EvidenceIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.now = stamp("2026-10-01T21:00:00+08:00")
        self.store = MemoryStore(Path(self.temp.name) / "memory.sqlite3")
        self.policy = MemoryPolicy(evidence_package_enabled=True)
        self.service = MemoryService(store=self.store, router=MemoryRouter(), policy=self.policy, clock=lambda: self.now)

    def tearDown(self):
        self.temp.cleanup()

    def add(self, text, *, source="a", scope="private:owner", at="2026-09-30T12:00:00+08:00"):
        self.store.record_event(conversation_id="conversation-" + source, scope=scope, source_id=source,
            text=text, created_at=stamp(at), metadata={"role": "user", "sender_id": "owner"})
        item = self.store.event_by_source_id(source_id=source, scope=scope)
        return MemorySourceReader(self.store).read_source(record_id=item.record_id, scope=scope, now=self.now)

    def ask(self, text):
        return self.service.prepare_turn(conversation_id="owner", conversation_type="private",
            user_message=text, message_meta={"sender_id": "owner"})

    def test_default_off(self):
        self.assertFalse(MemoryPolicy.from_environment({}).evidence_package_enabled)

    def test_current_validity_query_finds_source_outside_record_day(self):
        self.add("9月24日至9月27日暂时不喝咖啡，9月28日起照常。", at="2026-09-24T09:00:00+08:00")
        context = self.ask("你记得咖啡限制吗？今天10月1日还在限制期内吗？")
        self.assertIn("9月24日", context.rendered)
        self.assertIn("不在范围内", context.rendered)
        self.assertEqual(context.audit["evidence_package"]["coverage"][0]["state"], "covered")

    def test_explicit_record_day_does_not_cross_date(self):
        self.add("9月24日至9月27日暂时不喝咖啡。", at="2026-09-24T09:00:00+08:00")
        context = self.ask("只查10月1日那天的咖啡聊天记录。")
        self.assertNotIn("9月24日至", context.rendered)

    def test_summary_query_reads_original_parent_with_its_own_time_and_author(self):
        self.add("我上周五去看展了。", at="2026-09-30T12:00:00+08:00")
        self.store.rollup_daily_summaries(scope="private:owner", now=self.now)
        context = self.ask("你记得我看展是哪一天吗？按那条来源的日期解析上周五，不按今天倒推。")
        self.assertIn("2026-09-25", context.rendered)
        self.assertIn("a", [row["source_id"] for row in context.audit["evidence_package"]["sources"]])
        self.assertTrue(any(row.metadata.get("role") == "user" for row in context.records))

    def test_daily_summary_does_not_hide_original_validity_source(self):
        self.add("9月24日至9月27日暂时不喝咖啡，9月28日起照常。", at="2026-09-24T09:00:00+08:00")
        self.store.rollup_daily_summaries(scope="private:owner", now=self.now)
        context = self.ask("你记得咖啡限制吗？今天10月1日还在限制期内吗？")
        self.assertIn("a", [row["source_id"] for row in context.audit["evidence_package"]["sources"]])
        self.assertIn("不在范围内", context.rendered)

    def test_full_covered_candidates_do_not_add_facet_searches(self):
        a = self.add("星河项目方案A每月260元，延迟180毫秒。")
        b = self.add("星河项目方案B每月340元，延迟120毫秒。", source="b")
        with patch.object(self.store, "search", side_effect=AssertionError("redundant_facet_query")):
            rows = self.service._evidence_records((a,b), query="比较星河项目方案A和方案B费用与延迟",
                scopes=("private:owner",), requester="owner", now=self.now, limit=5)
        self.assertEqual(len(rows), 2)

    def test_cross_session_facets_and_missing_values_are_visible(self):
        self.add("星河项目方案A每月260元，延迟180毫秒。")
        self.add("星河项目方案B每月340元。", source="b")
        context = self.ask("你记得星河项目A、B方案费用与延迟吗？")
        package = context.audit["evidence_package"]
        facets = {row["key"]: row for row in package["coverage"]}
        self.assertEqual(facets["A.cost"]["state"], "covered")
        self.assertEqual(facets["B.latency"]["state"], "not_found_in_checked_sources")
        self.assertIn("B.latency：本次未找到证据", context.rendered)

    def test_oversize_source_does_not_count_as_covered(self):
        row = self.add("星河项目方案A每月260元。" + "长原文" * 120)
        text, package = self.service._render_evidence((row,), query="星河项目方案A费用", scopes=(row.scope,),
            requester="owner", mode="retrieve", now=self.now, limit=5)
        self.assertNotIn("260元", text)
        self.assertFalse(package["sources"])
        self.assertTrue(package["budget_exhausted"])
        self.assertIn("A.cost", package["missing"])

    def test_revoked_before_render_cannot_supply_coverage_or_text(self):
        row = self.add("星河项目方案A每月260元。")
        self.store.forget(query="星河", record_id=row.record_id, scope=row.scope, now=self.now)
        text, package = self.service._render_evidence((row,), query="星河项目方案A费用", scopes=(row.scope,),
            requester="owner", mode="retrieve", now=self.now, limit=5)
        self.assertFalse(package["sources"])
        self.assertNotIn("260元", text)

    def test_foreign_scope_is_removed_even_if_caller_supplies_candidate(self):
        row = self.add("星河项目方案A每月260元。", scope="private:foreign")
        text, package = self.service._render_evidence((row,), query="星河项目方案A费用", scopes=("private:owner",),
            requester="owner", mode="retrieve", now=self.now, limit=5)
        self.assertNotIn("260元", text)
        self.assertFalse(package["sources"])

    def test_total_render_budget_is_shared_with_coverage(self):
        self.service.policy = replace(self.policy, render_total_chars=350)
        self.add("星河项目方案A每月260元，延迟180毫秒。")
        self.add("星河项目方案B每月340元，延迟120毫秒。", source="b")
        context = self.ask("你记得星河项目A、B方案费用与延迟吗？")
        self.assertLessEqual(len(context.rendered), 350)
        self.assertTrue(context.audit["evidence_package"]["budget_exhausted"])

    def test_none_and_recent_do_not_trigger_evidence_search(self):
        with patch.object(self.store, "search", side_effect=AssertionError("irrelevant_search")):
            context = self.ask("你好")
        self.assertEqual(context.decision.history_mode, "none")
        self.assertFalse(context.rendered)
        self.assertNotIn("evidence_package", context.audit)

    def test_inspection_shares_same_evidence_budget_and_scope(self):
        self.add("星河项目方案A每月260元。")
        result = self.service.inspect(conversation_id="owner", conversation_type="private",
            requester_id="owner", query="星河项目方案A费用", mode="retrieve")
        self.assertEqual(result["evidence_package"]["coverage"][0]["state"], "covered")
        self.assertTrue(result["evidence_package"]["partial_scan"])


if __name__ == "__main__":
    unittest.main()
