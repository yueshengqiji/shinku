from __future__ import annotations

import unittest

from shinku.tasks import payloads, status


class WorkflowStatusContractTests(unittest.TestCase):
    def test_aliases_and_defaults_are_normalized(self) -> None:
        self.assertEqual(status.normalize_workflow_status(" working "), "running")
        self.assertEqual(status.normalize_workflow_status("cancelled"), "canceled")
        self.assertEqual(status.normalize_workspace_status("done"), "completed")
        self.assertEqual(status.normalize_step_status("unknown"), "queued")
        self.assertEqual(status.normalize_event_status(None), "pending")

    def test_transition_matrix_allows_normal_progression(self) -> None:
        self.assertEqual(status.validate_workflow_transition("queued", "running"), "running")
        self.assertEqual(status.validate_workflow_transition("running", "completed"), "completed")
        self.assertEqual(status.validate_workflow_transition("failed", "cleaned"), "cleaned")

    def test_reopen_requires_explicit_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "completed -> queued"):
            status.validate_workflow_transition("completed", "queued")
        self.assertEqual(status.validate_workflow_transition("completed", "queued", allow_reopen=True), "queued")

    def test_cleanup_can_be_explicitly_allowed_from_any_state(self) -> None:
        self.assertEqual(status.validate_workflow_transition("running", "cleaned", allow_cleanup=True), "cleaned")

    def test_terminal_and_public_aliases_are_consistent(self) -> None:
        self.assertEqual(status.TERMINAL_WORKFLOW_STATUSES, status.TERMINAL_TASK_WORKFLOW_STATUSES)
        self.assertIs(status.TASK_WORKFLOW_TRANSITIONS, status._WORKFLOW_EDGES)


class PayloadContractTests(unittest.TestCase):
    def test_positive_int_and_text_list(self) -> None:
        self.assertEqual(payloads.coerce_positive_int("4"), 4)
        self.assertEqual(payloads.coerce_positive_int(-1), 0)
        self.assertEqual(payloads.coerce_positive_int("bad"), 0)
        self.assertEqual(payloads.normalize_text_list([" A ", "a", "B"]), ["A", "B"])

    def test_steps_use_stable_ids_and_bounded_fields(self) -> None:
        steps = payloads.normalize_steps(
            [
                {"name": " inspect ", "status": "bad", "note": "n" * 300},
                "write",
                {"title": ""},
            ],
            step_id_prefix="phase",
        )
        self.assertEqual(steps[0], {"id": "phase_1", "title": "inspect", "status": "queued", "note": "n" * 200})
        self.assertEqual(steps[1], {"id": "phase_2", "title": "write", "status": "queued", "note": ""})

    def test_artifact_normalization_deduplicates_by_identity(self) -> None:
        result = payloads.normalize_artifacts(
            [
                {"id": "a", "title": "first", "kind": "png"},
                {"id": "A", "title": "duplicate"},
                {"title": "untitled", "deliverable": 1},
            ]
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["source"], "task_worker")
        self.assertIs(result[1]["deliverable"], True)

    def test_merge_artifacts_and_steps_preserve_order(self) -> None:
        artifacts = payloads.merge_artifacts(
            [{"id": "a", "title": "old", "status": "ready"}],
            [{"id": "a", "status": "", "kind": "png"}, {"id": "b", "title": "new"}],
        )
        self.assertEqual(artifacts, [{"id": "a", "title": "old", "status": "ready", "kind": "png"}, {"id": "b", "title": "new"}])
        steps = payloads.merge_steps(
            [{"id": "s1", "title": "inspect", "status": "running"}],
            [{"title": "inspect", "status": "done"}, {"title": "write"}],
        )
        self.assertEqual(steps[0]["status"], "done")
        self.assertEqual(steps[1]["id"], "worker_step_2")


if __name__ == "__main__":
    unittest.main()
