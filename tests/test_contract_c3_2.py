from __future__ import annotations

import unittest

from shinku.tasks.workspace import TaskWorkspaceService


class MemoryWorkspace:
    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}
        self.events: list[dict] = []
        self.calls: list[tuple[str, dict]] = []
        self.next_id = 1

    def add_task_workspace(self, **fields):
        self.calls.append(("add", dict(fields)))
        task_id = fields.get("task_id") or f"task::{self.next_id}"
        self.next_id += 1
        task = {**fields, "task_id": task_id}
        task.setdefault("created_at", fields.get("timestamp"))
        task.setdefault("updated_at", fields.get("timestamp"))
        self.tasks[task_id] = task
        return dict(task)

    def get_task_workspace(self, task_id):
        task = self.tasks.get(task_id)
        return dict(task) if task else None

    def list_task_workspaces(self, **filters):
        result = list(self.tasks.values())
        if filters.get("profile_user_id") is not None:
            result = [x for x in result if x.get("profile_user_id") == filters["profile_user_id"]]
        if filters.get("session_id") is not None:
            result = [x for x in result if x.get("session_id") == filters["session_id"]]
        if filters.get("statuses") is not None:
            result = [x for x in result if x.get("status") in filters["statuses"]]
        return [dict(x) for x in result[: filters.get("limit", 20)]]

    def list_pending_agent_workspaces(self, **filters):
        return self.list_task_workspaces(statuses=["queued", "running", "waiting_user"], limit=filters.get("limit", 100))

    def update_task_workspace(self, **fields):
        task = self.tasks.get(fields["task_id"])
        if not task:
            return None
        self.calls.append(("update", dict(fields)))
        for key, value in fields.items():
            if key not in {"task_id", "allow_reopen", "allow_cleanup", "updated_at", "completed_at", "cleaned_at"}:
                task[key] = value
        if fields.get("updated_at") is not None:
            task["updated_at"] = fields["updated_at"]
        for key in ("completed_at", "cleaned_at"):
            if key in fields:
                task[key] = fields[key]
        return dict(task)

    def append_task_workspace_event(self, **fields):
        event = {"event_id": f"event::{len(self.events) + 1}", **fields}
        self.events.append(event)
        return dict(event)

    def list_task_workspace_events(self, **filters):
        result = [x for x in self.events if not filters.get("task_id") or x.get("task_id") == filters["task_id"]]
        if filters.get("status") is not None:
            result = [x for x in result if x.get("status") == filters["status"]]
        return [dict(x) for x in result[-filters.get("limit", 50) :]]

    def mark_task_workspace_event_handled(self, **fields):
        for event in self.events:
            if event["event_id"] == fields["event_id"]:
                event.update({"status": fields["status"], "handled_at": fields.get("handled_at")})
                return dict(event)
        return None


class TaskWorkspaceBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryWorkspace()
        self.service = TaskWorkspaceService(self.store)

    def test_create_normalizes_payload_and_records_creation_event(self) -> None:
        task = self.service.create_task(
            profile_user_id="master",
            session_id="qq",
            raw_request_text="  make a file  ",
            source_message_id=" msg-1 ",
            normalized_goal="  do it  ",
            success_criteria=["A", "a", "B"],
            constraints=["keep source"],
            steps=[{"title": "inspect", "status": "bad"}],
            artifacts=[{"id": "file-1", "kind": "md"}],
            timestamp=100,
        )
        self.assertEqual(task["raw_request"], {"text": "make a file", "source_message_id": "msg-1"})
        self.assertEqual(task["success_criteria"], ["A", "B"])
        self.assertEqual(task["steps"][0]["status"], "queued")
        self.assertEqual([x["event_type"] for x in self.store.events], ["task_created"])
        self.assertEqual(self.store.events[0]["status"], "handled")

    def test_update_normalizes_only_fields_that_are_present(self) -> None:
        task = self.service.create_task(profile_user_id="u", session_id="s", raw_request_text="x", timestamp=1)
        self.service.update_task(task_id=task["task_id"], status="working", normalized_goal=" next ", timestamp=2)
        updated = self.service.get_task(task["task_id"])
        assert updated is not None
        self.assertEqual(updated["status"], "running")
        self.assertEqual(updated["normalized_goal"], "next")
        self.assertEqual(updated["steps"], [])

    def test_append_requires_an_existing_task_and_normalizes_event(self) -> None:
        with self.assertRaisesRegex(ValueError, "task not found"):
            self.service.append_event(task_id="missing", event_type="x")
        task = self.service.create_task(profile_user_id="u", session_id="s", raw_request_text="x", timestamp=1)
        event = self.service.append_event(task_id=task["task_id"], event_type="question", status="bad", timestamp=2)
        self.assertEqual(event["status"], "pending")
        self.assertEqual(event["profile_user_id"], "u")

    def test_list_and_pending_delegate_with_normalized_filters(self) -> None:
        self.service.create_task(profile_user_id="u", session_id="s", raw_request_text="x", status="working", timestamp=1)
        self.assertEqual(len(self.service.list_tasks(profile_user_id="u", statuses=["working"])), 1)
        self.assertEqual(len(self.service.list_pending_agent_tasks(limit=5)), 1)

    def test_completion_and_cleanup_add_lifecycle_events(self) -> None:
        task = self.service.create_task(profile_user_id="u", session_id="s", raw_request_text="x", timestamp=1)
        completed = self.service.complete_task(task_id=task["task_id"], artifacts=[{"id": "a", "title": "result"}], timestamp=2)
        assert completed is not None
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["completed_at"], 2)
        cleaned = self.service.cleanup_task(task_id=task["task_id"], reason="done", timestamp=3)
        assert cleaned is not None
        self.assertEqual(cleaned["status"], "cleaned")
        self.assertEqual(cleaned["metadata"]["cleanup"]["cleaned_at"], 3)
        self.assertEqual([x["event_type"] for x in self.store.events], ["task_created", "task_completed", "task_cleaned"])

    def test_event_can_be_marked_handled(self) -> None:
        task = self.service.create_task(profile_user_id="u", session_id="s", raw_request_text="x", timestamp=1)
        event = self.service.append_event(task_id=task["task_id"], event_type="question", timestamp=2)
        handled = self.service.mark_event_handled(event_id=event["event_id"], timestamp=3)
        assert handled is not None
        self.assertEqual(handled["status"], "handled")
        self.assertEqual(handled["handled_at"], 3)


if __name__ == "__main__":
    unittest.main()
