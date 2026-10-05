"""Harness integrity checks, not scripted model-quality evidence."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage_a_harness", ROOT / "scripts/gray_memory_stage_a.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class MemoryStageAHarnessTests(unittest.TestCase):
    def test_fixture_is_frozen_balanced_and_covers_categories(self):
        data = bench.load_cases()
        self.assertEqual(len(data["cases"]), 24)
        self.assertEqual(sum(r["split"] == "dev" for r in data["cases"]), 12)
        self.assertEqual(sum(r["split"] == "holdout" for r in data["cases"]), 12)
        self.assertEqual(len({r["category"] for r in data["cases"]}), 12)

    def test_timestamps_and_expected_calendar(self):
        self.assertEqual(bench.datetime.fromtimestamp(bench.timestamp("2026-09-30T12:00:00+08:00")).weekday(), 2)
        with self.assertRaises(ValueError):
            bench.timestamp("2026-09-30T12:00:00")

    def test_scores_empty_answer_as_failure_and_marks_semantic_review(self):
        self.assertFalse(bench.score_answer("", {"all": []})["rule_pass"])
        self.assertTrue(bench.score_answer("最后选0.65", {"all": ["0[.]65"]})["rule_pass"])
        self.assertTrue(bench.score_answer("最后选0.65", {"all": ["0[.]65"]})["semantic_review_required"])
        self.assertFalse(bench.score_answer("蓝色 CANARY_ORCHID_731", {"all": ["蓝"], "forbidden": ["CANARY_ORCHID_731"]})["rule_pass"])

    def test_runtime_and_task_contract_failures_are_separate_from_reading(self):
        self.assertEqual(bench.failure_layer({"final_text": "……刚才那句我没接好。你再说一遍？"},
            [{"metrics_delta": {"errors": 1}}]), "model_transport_or_runtime_failure")
        self.assertEqual(bench.failure_layer({"reason": "task_contract_error:contract_repair_failed"}, []),
                         "host_task_contract_blocked")
        self.assertEqual(bench.failure_layer({"final_text": "选择0.65", "reason": "completed"},
            [{"metrics_delta": {"errors": 1}}, {"metrics_delta": {}}]), "")

    def test_supplement_has_actual_different_conversation_scopes(self):
        data = bench.load_cases(ROOT / "tests/fixtures/memory_stage_a_scope_v1.json")
        self.assertEqual(len(data["cases"]), 2)
        self.assertEqual({r["split"] for r in data["cases"]}, {"dev", "holdout"})
        self.assertEqual(data["cases"][0]["history"][1]["conversation_type"], "group")

    def test_stage_b_expansion_stays_separate_and_balanced(self):
        main = bench.load_cases()
        expansion = bench.load_cases(ROOT / "tests/fixtures/memory_stage_b_v1.json")
        self.assertEqual(len(expansion["cases"]), 4)
        self.assertEqual(sum(row["split"] == "dev" for row in expansion["cases"]), 2)
        self.assertEqual(sum(row["split"] == "holdout" for row in expansion["cases"]), 2)
        self.assertFalse({row["id"] for row in main["cases"]}.intersection(
            row["id"] for row in expansion["cases"]))
        self.assertEqual(expansion["question_date"], main["question_date"])
        self.assertTrue(all(row["required_sources"] for row in expansion["cases"]))

    def offline(self, case_id):
        data = bench.load_cases()
        case = next(r for r in data["cases"] if r["id"] == case_id)
        with tempfile.TemporaryDirectory(prefix="stage-a-harness-test-") as scratch:
            return bench.run_case(case, run_root=Path(scratch),
                policy=bench.MemoryPolicy(master_user_ids=(bench.OWNER,)),
                build=SimpleNamespace(provider=None, status="disabled"), runtime=None,
                config=None, persona="", question_time=bench.timestamp(data["question_date"]))

    def test_offline_has_no_fabricated_answer_or_qa_score(self):
        row = self.offline("dev-extract")
        self.assertEqual(row["reply"]["agent_status"], "not_run")
        self.assertEqual(row["reply"]["final_text"], "")
        self.assertIsNone(row["answer_check"])
        self.assertEqual(row["model_calls"], [])
        self.assertTrue(row["all_required_sources_retrieved"])

    def test_current_task_guard_is_measured_without_being_bypassed(self):
        row = self.offline("dev-natural")
        self.assertEqual(row["memory_contexts"][0]["decision"]["history_mode"], "none")
        self.assertFalse(row["all_required_sources_retrieved"])

    def test_confirmed_correction_uses_real_engine_and_keeps_source_alias(self):
        row = self.offline("hold-update")
        self.assertTrue(row["all_required_sources_retrieved"])
        self.assertEqual(row["actions"][0]["changed_count"], 2)
        records = row["memory_contexts"][0]["records"]
        corrected = [r for r in records if r["source_id"].startswith("correction:")]
        self.assertTrue(corrected)
        self.assertTrue(all("红茶" in r["text"] for r in corrected))

    def test_revocation_is_respected_after_index_backfill(self):
        row = self.offline("hold-revoke")
        self.assertFalse(row["evidence_leaks"])
        self.assertNotIn("park", row["retrieved_sources"])

    def test_restored_scratch_backup_applies_revocations(self):
        row = self.offline("hold-recovery")
        self.assertFalse(row["evidence_leaks"])
        self.assertNotIn("studio", row["retrieved_sources"])
        self.assertEqual(row["actions"][0]["type"], "restore_after_forget")

    def test_non_owner_cannot_read_owner_private_evidence(self):
        row = self.offline("dev-scope")
        self.assertFalse(row["evidence_leaks"])
        self.assertNotIn("private-book", row["retrieved_sources"])

    def test_oracle_is_explicit_and_does_not_change_normal_route(self):
        data = bench.load_cases()
        case = next(r for r in data["cases"] if r["id"] == "dev-natural")
        with tempfile.TemporaryDirectory(prefix="stage-a-oracle-test-") as scratch:
            row = bench.run_case(case, run_root=Path(scratch),
                policy=bench.MemoryPolicy(master_user_ids=(bench.OWNER,)),
                build=SimpleNamespace(provider=None, status="disabled"), runtime=None,
                config=None, persona="", question_time=bench.timestamp(data["question_date"]), oracle=True)
        self.assertTrue(row["oracle"])
        self.assertEqual(row["memory_contexts"][0]["decision"]["reason"], "fixture_oracle_reading_diagnostic")
        self.assertTrue(row["all_required_sources_retrieved"])

    def test_oracle_does_not_bypass_non_owner_scope(self):
        data = bench.load_cases()
        case = dict(next(r for r in data["cases"] if r["id"] == "dev-scope"))
        case["required_sources"] = ["private-book"]  # integrity attack on diagnostic input, not a scored fixture
        with tempfile.TemporaryDirectory(prefix="stage-a-oracle-scope-test-") as scratch:
            row = bench.run_case(case, run_root=Path(scratch),
                policy=bench.MemoryPolicy(master_user_ids=(bench.OWNER,)),
                build=SimpleNamespace(provider=None, status="disabled"), runtime=None,
                config=None, persona="", question_time=bench.timestamp(data["question_date"]), oracle=True)
        self.assertFalse(row["all_required_sources_visible"])
        self.assertFalse(row["evidence_leaks"])

    def test_actual_cross_scope_sources_are_found(self):
        data = bench.load_cases(ROOT / "tests/fixtures/memory_stage_a_scope_v1.json")
        for case in data["cases"]:
            with self.subTest(case=case["id"]), tempfile.TemporaryDirectory(prefix="stage-a-cross-scope-test-") as scratch:
                row = bench.run_case(case, run_root=Path(scratch),
                    policy=bench.MemoryPolicy(master_user_ids=(bench.OWNER,)),
                    build=SimpleNamespace(provider=None, status="disabled"), runtime=None,
                    config=None, persona="", question_time=bench.timestamp(data["question_date"]))
            self.assertTrue(row["all_required_sources_retrieved"])

    def test_summary_separates_positive_coverage_from_negative_controls(self):
        positive = self.offline("dev-extract")
        negative = self.offline("dev-revoke")
        report = bench.summarize([positive, negative])["all"]
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["required_evidence_cases"], 1)
        self.assertEqual(report["complete_evidence_cases"], 1)
        self.assertEqual(report["qa_checked"], 0)


if __name__ == "__main__":
    unittest.main()
