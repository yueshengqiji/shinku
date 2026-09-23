import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from shinku.agent.self_state import SelfStateTool
from shinku.tasks.coordinator import AgentTaskCoordinator


class P1SelfStateTests(unittest.TestCase):
    def test_self_state_is_read_only_and_redacts_secret_shaped_fields(self) -> None:
        tool = SelfStateTool(lambda: {"runtime": {"model": "demo"}, "api_key": "secret"})
        result = tool.execute(call={"scope": "all"}, context={})
        self.assertEqual(result["tool_type"], "read_self_state")
        self.assertIn("demo", result["followup_context"])
        self.assertIn("<redacted>", result["followup_context"])
        self.assertNotIn("secret", result["followup_context"])


class P1TaskCoordinatorTests(unittest.TestCase):
    def test_submit_status_cancel_and_retry_keep_task_history(self) -> None:
        coordinator = AgentTaskCoordinator(max_workers=1)
        try:
            attempts = []

            def handler(cancel_event):
                attempts.append(len(attempts) + 1)
                if len(attempts) == 1:
                    raise RuntimeError("first attempt failed")
                return "ok"

            created = coordinator.submit(handler, task_id="task-1", goal="测试", profile_user_id="u-1")
            self.assertEqual(created["status"], "queued")
            self._wait_for(coordinator, "task-1", "failed")
            retried = coordinator.retry("task-1", profile_user_id="u-1")
            self.assertTrue(retried["ok"])
            self._wait_for(coordinator, "task-1", "completed")
            current = coordinator.get("task-1", profile_user_id="u-1")
            self.assertEqual(current["attempt"], 2)
            self.assertEqual(current["result_summary"], "ok")
            self.assertFalse(coordinator.cancel("task-1", profile_user_id="u-1")["ok"])
        finally:
            coordinator.shutdown(wait=True)

    @staticmethod
    def _wait_for(coordinator, task_id: str, status: str) -> None:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            current = coordinator.get(task_id)
            if current and current["status"] == status:
                return
            time.sleep(0.01)
        raise AssertionError(f"task did not reach {status}")


class P2TaskPersistenceTests(unittest.TestCase):
    def test_completed_task_history_survives_coordinator_restart(self) -> None:
        with TemporaryDirectory() as root:
            state_path = Path(root) / "runtime" / "agent_tasks.sqlite3"
            first = AgentTaskCoordinator(max_workers=1, state_path=state_path)
            try:
                created = first.submit(lambda _cancel: "done", task_id="persisted", goal="完成检查")
                self.assertEqual(created["status"], "queued")
                P1TaskCoordinatorTests._wait_for(first, "persisted", "completed")
            finally:
                first.shutdown(wait=True)

            restored = AgentTaskCoordinator(max_workers=1, state_path=state_path)
            try:
                task = restored.get("persisted")
                self.assertIsNotNone(task)
                self.assertEqual(task["status"], "completed")
                self.assertEqual(task["result_summary"], "done")
                self.assertTrue(restored.storage_status()["persistent"])
            finally:
                restored.shutdown(wait=True)

    def test_incomplete_task_is_not_replayed_after_restart(self) -> None:
        with TemporaryDirectory() as root:
            state_path = Path(root) / "agent_tasks.sqlite3"
            release = Event()
            first = AgentTaskCoordinator(max_workers=1, state_path=state_path)
            first.submit(lambda _cancel: release.wait(2.0), task_id="interrupted", goal="不可自动重跑")
            P1TaskCoordinatorTests._wait_for(first, "interrupted", "running")

            recovered = AgentTaskCoordinator(max_workers=1, state_path=state_path)
            try:
                task = recovered.get("interrupted")
                self.assertEqual(task["status"], "failed")
                self.assertIn("不会自动重跑", task["error"])
                self.assertTrue(task["metadata"]["recovered_after_restart"])
                retry = recovered.retry("interrupted")
                self.assertFalse(retry["ok"])
                self.assertEqual(retry["reason"], "retry_handler_missing")
            finally:
                recovered.shutdown(wait=True)
                release.set()
                first.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
