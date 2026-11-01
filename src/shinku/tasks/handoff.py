"""Typed boundary between a selected tool route and Agent task admission.

The model's ``task_link`` is routing input, not a queue record or an executable
task.  This value freezes the selected route and target while host-only task
metadata is prepared, then records the legal admission phases used by the QQ
bridge and background coordinator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from shinku.agent.task_state import TaskContractError


class TaskHandoffPhase(str, Enum):
    ROUTED = "routed"
    PREPARED = "prepared"
    QUEUED = "queued"
    COMMITTED = "committed"
    RUNNING = "running"
    FINISHED = "finished"


_PHASE_TRANSITIONS = {
    TaskHandoffPhase.ROUTED: {TaskHandoffPhase.PREPARED, TaskHandoffPhase.FINISHED},
    TaskHandoffPhase.PREPARED: {
        TaskHandoffPhase.QUEUED,
        TaskHandoffPhase.COMMITTED,
        TaskHandoffPhase.FINISHED,
    },
    TaskHandoffPhase.QUEUED: {
        TaskHandoffPhase.COMMITTED,
        TaskHandoffPhase.RUNNING,
        TaskHandoffPhase.FINISHED,
    },
    TaskHandoffPhase.COMMITTED: {
        TaskHandoffPhase.QUEUED,
        TaskHandoffPhase.RUNNING,
        TaskHandoffPhase.FINISHED,
    },
    TaskHandoffPhase.RUNNING: {TaskHandoffPhase.FINISHED},
    TaskHandoffPhase.FINISHED: set(),
}


def _route_error(code: str, field_path: str) -> TaskContractError:
    return TaskContractError(code, field_path=field_path, phase="routing")


@dataclass(frozen=True, slots=True)
class TaskHandoff:
    """Immutable route/target token carried through preparation and execution."""

    route: str
    target_task_id: str
    strict: bool
    phase: TaskHandoffPhase = TaskHandoffPhase.ROUTED

    @classmethod
    def select(
        cls,
        task_link: Any,
        *,
        host_task_id: str,
        strict: bool,
        allow_missing: bool = False,
        phase: TaskHandoffPhase = TaskHandoffPhase.ROUTED,
    ) -> TaskHandoff:
        host_id = str(host_task_id or "").strip()
        if not host_id:
            raise _route_error("task_handoff_host_id_missing", "task_id")

        if task_link is None:
            if strict and allow_missing:
                return cls("pending", host_id, True, phase)
            if strict:
                raise _route_error("task_handoff_requires_new_or_continue", "task_link.mode")
            return cls("new", host_id, False, phase)

        if not isinstance(task_link, Mapping):
            raise _route_error("invalid_task_link", "task_link")

        mode = task_link.get("mode")
        if mode == "none":
            if strict:
                raise _route_error("task_handoff_requires_new_or_continue", "task_link.mode")
            # Compatibility mode historically treated a non-fast-track route
            # as a host-created task. Preserve that behavior without conflating
            # it with a strict, explicitly admitted task route.
            return cls("legacy_none", host_id, False, phase)
        if mode == "new":
            return cls("new", host_id, strict, phase)
        if mode == "continue":
            target = task_link.get("task_id")
            if not isinstance(target, str) or not target.strip():
                raise _route_error("task_handoff_continue_requires_task_id", "task_link.task_id")
            return cls("continue", target.strip(), strict, phase)
        raise _route_error("task_handoff_requires_new_or_continue", "task_link.mode")

    @classmethod
    def resume(cls, task_id: str) -> TaskHandoff:
        target = str(task_id or "").strip()
        if not target:
            raise _route_error("task_handoff_resume_requires_task_id", "task_id")
        # Approval resumes an already prepared and persisted Agent task.
        return cls("continue", target, True, TaskHandoffPhase.COMMITTED)

    def bind(self, task_link: Any) -> TaskHandoff:
        """Bind metadata once, or reject any rewrite of an explicit route."""
        if self.route == "pending":
            bound = self.select(
                task_link,
                host_task_id=self.target_task_id,
                strict=self.strict,
                allow_missing=False,
                phase=self.phase,
            )
            if bound.route == "pending":
                raise _route_error("task_handoff_requires_new_or_continue", "task_link.mode")
            return bound

        candidate = self.select(
            task_link,
            host_task_id=self.target_task_id,
            strict=self.strict,
            allow_missing=False,
            phase=self.phase,
        )
        if candidate.route != self.route or candidate.target_task_id != self.target_task_id:
            raise _route_error("task_handoff_changed_after_selection", "task_link")
        return self

    def require_resolved(self) -> TaskHandoff:
        if self.route == "pending":
            raise _route_error("task_handoff_requires_new_or_continue", "task_link.mode")
        return self

    def advance(self, phase: TaskHandoffPhase) -> TaskHandoff:
        target = TaskHandoffPhase(phase)
        if target == self.phase:
            return self
        if target not in _PHASE_TRANSITIONS[self.phase]:
            raise TaskContractError(
                "task_handoff_invalid_transition",
                field_path="handoff.phase",
                phase="admission",
            )
        return replace(self, phase=target)
