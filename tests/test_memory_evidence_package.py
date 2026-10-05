"""D evidence coverage comes from retained, delivered sources, not a model claim."""
from dataclasses import replace
from datetime import datetime
import unittest

from shinku.memory.models import MemoryRecord
from shinku.memory.evidence import build_evidence_package, needs_for, source_time, validity_query


def stamp(value):
    return int(datetime.fromisoformat(value).timestamp())


def record(text, source="one", at="2026-09-30T12:00:00+08:00", **metadata):
    return MemoryRecord(record_id="event:" + source, layer="episodic", scope="private:owner", source_id=source,
        text=text, created_at=stamp(at), metadata={"role": "user", "sender_id": "owner", "source_version": source,
                                              "source_timezone": "Asia/Shanghai", **metadata})


class EvidencePackageTests(unittest.TestCase):
    now = stamp("2026-10-01T21:00:00+08:00")

    def package(self, query, records, **extra):
        return build_evidence_package(query=query, records=records, now=self.now,
            allowed_scopes=("private:owner",), requester_id="owner", checked_sources=len(records), **extra)

    def test_two_plans_require_each_cost_and_latency(self):
        query = "你记得星河项目A、B方案费用与延迟吗？"
        self.assertEqual({need["key"] for need in needs_for(query)}, {"A.cost", "A.latency", "B.cost", "B.latency"})
        result = self.package(query, (record("星河项目方案A每月260元，延迟180毫秒。", "a"),
                                      record("星河项目方案B每月340元，延迟120毫秒。", "b")))
        self.assertTrue(all(row["state"] == "covered" for row in result["coverage"]))
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(next(row for row in result["coverage"] if row["key"] == "B.cost")["quotes"], ["340元"])

    def test_missing_facet_is_not_filled_from_other_plan_or_similar_project(self):
        query = "你记得星河项目A、B方案费用与延迟吗？"
        result = self.package(query, (record("星河项目方案A每月260元，延迟180毫秒。", "a"),
                                      record("星河项目方案B每月340元。", "b"),
                                      record("别的项目方案B延迟120毫秒。", "other")))
        coverage = {row["key"]: row for row in result["coverage"]}
        self.assertEqual(coverage["B.latency"]["state"], "not_found_in_checked_sources")
        self.assertEqual(coverage["B.latency"]["source_ids"], [])
        self.assertTrue(result["partial_scan"])

    def test_negated_project_membership_cannot_supply_a_missing_facet(self):
        result = self.package("清泉项目甲、乙方案费用与延迟", (
            record("别的项目方案乙延迟75毫秒，不属于清泉项目。"),))
        self.assertEqual(next(row for row in result["coverage"] if row["key"] == "乙.latency")["state"],
                         "not_found_in_checked_sources")

    def test_relative_time_in_summary_cannot_be_anchored_to_summary_creation(self):
        result = source_time(record("用户：我上个月搬家了。", reading_representation="summary"), now=self.now)
        self.assertFalse(result["event_times"])
        self.assertEqual(result["time_state"], "unresolved")

    def test_same_record_with_two_targets_does_not_mix_the_values(self):
        result = self.package("比较星河项目方案A和方案B的费用和延迟", (
            record("星河项目方案A每月260元，延迟180毫秒；方案B每月340元，延迟120毫秒。"),))
        coverage = {row["key"]: row for row in result["coverage"]}
        self.assertEqual(coverage["A.cost"]["quotes"], ["260元"])
        self.assertEqual(coverage["B.latency"]["quotes"], ["120毫秒"])

    def test_multiple_values_for_same_target_are_marked_uncertain(self):
        result = self.package("比较星河项目方案A的费用", (record("星河项目方案A的报价有260元和300元。"),))
        self.assertEqual(result["coverage"][0]["state"], "ambiguous_in_checked_sources")

    def test_selection_alone_and_assistant_guess_do_not_prove_user_reason(self):
        result = self.package("我为什么放弃墨舟方案D选择E？", (
            record("墨舟项目我最终放弃方案D，选择E。", "choice"),
            record("我猜因为费用低才选E。", "guess", role="assistant", sender_id="bot")))
        self.assertEqual(next(row for row in result["coverage"] if row["key"] == "explicit_reason")["state"],
                         "not_found_in_checked_sources")

    def test_different_target_reason_and_someone_else_birthday_are_not_personal_evidence(self):
        result = self.package("我为什么放弃方案D？", (record("我放弃方案E，因为收费高。"),))
        self.assertEqual(result["coverage"][0]["state"], "not_found_in_checked_sources")
        result = self.package("我的生日是几月几号？", (record("我朋友的生日是5月3日。"),
            record("我猜你的生日是5月3日。", "guess", role="assistant", sender_id="bot")))
        self.assertEqual(next(row for row in result["coverage"] if row["key"] == "birthday")["state"],
                         "not_found_in_checked_sources")

    def test_unusual_labels_use_the_same_literal_facet_checks(self):
        result = self.package("比较星河项目方案M和方案Z的费用和延迟", (
            record("星河项目方案M每月260元，延迟180毫秒；方案Z每月340元，延迟120毫秒。"),))
        self.assertEqual({row["key"] for row in result["coverage"]}, {"M.cost", "M.latency", "Z.cost", "Z.latency"})
        self.assertTrue(all(row["state"] == "covered" for row in result["coverage"]))

    def test_relative_event_date_uses_record_date_not_question_date(self):
        times = source_time(record("我上周五去看展了。"), now=self.now)
        self.assertEqual(times["event_times"][0]["start"], "2026-09-25")
        month = source_time(record("我上个月搬家了。", at="2026-10-01T09:00:00+08:00"), now=self.now)
        self.assertEqual(month["event_times"][0]["start"], "2026-09-01")
        self.assertEqual(month["event_times"][0]["precision"], "month")

    def test_explicit_validity_is_distinct_from_record_date(self):
        result = source_time(record("9月24日至9月27日暂时不喝咖啡，9月28日起照常。",
                                    at="2026-09-24T09:00:00+08:00"), now=self.now)
        self.assertEqual(result["validity"]["start"], "2026-09-24")
        self.assertEqual(result["validity"]["end_exclusive"], "2026-09-28")
        self.assertFalse(result["validity"]["applies_at_request"])

    def test_invalid_dates_and_unknown_timezone_remain_uncertain(self):
        value = source_time(record("2月30日至2月31日不喝咖啡。"), now=self.now)
        self.assertEqual(value["time_state"], "unresolved")

    def test_disagreeing_event_dates_remain_ambiguous_not_latest_wins(self):
        result = self.package("我搬家发生在几月？", (
            record("我上个月搬家了。", "a", at="2026-09-30T12:00:00+08:00"),
            record("我上个月搬家了。", "b", at="2026-10-01T12:00:00+08:00")))
        self.assertEqual(result["coverage"][0]["state"], "ambiguous_in_checked_sources")
        value = source_time(record("我上周五看展。", source_timezone="Unknown/Zone"), now=self.now)
        self.assertEqual(value["time_state"], "unresolved")

    def test_validity_question_is_not_an_explicit_record_day_request(self):
        self.assertTrue(validity_query("你记得咖啡限制吗？今天10月1日还在限制期内吗？"))
        self.assertFalse(validity_query("只查10月1日那天的咖啡聊天记录。"))
        self.assertFalse(validity_query("昨天我问了你什么？"))

    def test_scope_and_source_handle_checks_do_not_accept_foreign_or_unbound_evidence(self):
        rows = (record("星河项目方案A每月260元。"),
                replace(record("星河项目方案B每月340元。", "b"), scope="private:foreign"),
                replace(record("星河项目方案B延迟120毫秒。", "unbound"), metadata={"role": "user"}))
        result = self.package("星河项目A、B费用与延迟", rows)
        self.assertEqual([row["source_id"] for row in result["sources"]], ["one"])
        self.assertTrue(any(row["state"] == "not_found_in_checked_sources" for row in result["coverage"]))

    def test_project_entity_stops_at_first_project_and_excludes_function_words(self):
        from shinku.memory.evidence import project_for
        for query, expected in (("你记得竹湾项目我在私聊和项目群说的参数吗？", "竹湾项目"),
            ("请比较我的北澜项目A、B费用", "北澜项目"),
            ("帮我检查星河项目的代码", "星河项目"), ("关于海桥项目和别的项目", "海桥项目")):
            self.assertEqual(project_for(query), expected, query)

    def test_duplicate_source_not_counted_twice_and_budget_missing_is_explicit(self):
        value = record("星河项目方案A每月260元，延迟180毫秒。")
        result = self.package("星河项目A、B费用与延迟", (value, value), budget_exhausted=True)
        self.assertEqual(len(result["sources"]), 1)
        self.assertTrue(result["budget_exhausted"])
        self.assertTrue(result["missing"])
        self.assertNotEqual(result["read_state"], "all_history_checked")
