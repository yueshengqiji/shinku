from __future__ import annotations

import unittest

from shinku.agent.task_state import TaskContractError
from shinku.tasks.handoff import TaskHandoff, TaskHandoffPhase


class TaskHandoffTests(unittest.TestCase):
    def test_explicit_none_is_not_an_admissible_strict_route(self):
        with self.assertRaises(TaskContractError) as caught:
            TaskHandoff.select(
                {"mode": "none"}, host_task_id="host:1", strict=True,
                allow_missing=True,
            )
        self.assertEqual(str(caught.exception), "task_handoff_requires_new_or_continue")
        self.assertEqual(caught.exception.field_path, "task_link.mode")
        self.assertEqual(caught.exception.phase, "routing")

    def test_missing_native_route_must_bind_before_preparation(self):
        pending = TaskHandoff.select(None, host_task_id="host:1", strict=True, allow_missing=True)
        self.assertEqual(pending.route, "pending")
        bound = pending.bind({"mode": "new"})
        self.assertEqual(bound.route, "new")
        self.assertEqual(bound.target_task_id, "host:1")
        with self.assertRaisesRegex(TaskContractError, "task_handoff_requires_new_or_continue"):
            pending.require_resolved()
        with self.assertRaisesRegex(TaskContractError, "task_handoff_requires_new_or_continue"):
            pending.bind(None)

    def test_selected_route_and_continue_target_cannot_be_rewritten(self):
        selected_new = TaskHandoff.select(
            {"mode": "new"}, host_task_id="host:1", strict=True,
        )
        with self.assertRaisesRegex(TaskContractError, "task_handoff_changed_after_selection"):
            selected_new.bind({"mode": "continue", "task_id": "task:2"})

        selected_continue = TaskHandoff.select(
            {"mode": "continue", "task_id": "task:1"}, host_task_id="host:2", strict=True,
        )
        self.assertEqual(selected_continue.target_task_id, "task:1")
        with self.assertRaisesRegex(TaskContractError, "task_handoff_changed_after_selection"):
            selected_continue.bind({"mode": "continue", "task_id": "task:2"})

    def test_phase_machine_rejects_queue_before_preparation(self):
        handoff = TaskHandoff.select({"mode": "new"}, host_task_id="host:1", strict=True)
        with self.assertRaisesRegex(TaskContractError, "task_handoff_invalid_transition"):
            handoff.advance(TaskHandoffPhase.QUEUED)

        prepared = handoff.advance(TaskHandoffPhase.PREPARED)
        queued = prepared.advance(TaskHandoffPhase.QUEUED)
        committed = queued.advance(TaskHandoffPhase.COMMITTED)
        running = committed.advance(TaskHandoffPhase.RUNNING)
        self.assertEqual(running.advance(TaskHandoffPhase.RUNNING), running)
        self.assertEqual(running.advance(TaskHandoffPhase.FINISHED).phase, TaskHandoffPhase.FINISHED)

    def test_legacy_none_mapping_is_kept_out_of_strict_route_semantics(self):
        legacy = TaskHandoff.select(
            {"mode": "none", "track": "agent"}, host_task_id="host:legacy", strict=False,
        )
        self.assertEqual(legacy.route, "legacy_none")
        self.assertEqual(legacy.target_task_id, "host:legacy")
        self.assertEqual(legacy.bind({"mode": "none"}), legacy)


if __name__ == "__main__":
    unittest.main()
