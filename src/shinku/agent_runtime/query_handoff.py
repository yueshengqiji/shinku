"""Process-local transfer of executed reads into the existing Agent checkpoint.

Models and JSON cannot construct this capability. A restarted host must inspect
the persisted operation instead of reconstructing runnable state from a summary.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType
from typing import Any, Callable

from shinku.agent.loop import AgentRunCheckpoint, AgentState, _call_signature
from shinku.agent.research import ResearchLedger
from shinku.hosts.tool_host import ToolHost
from shinku.tools.registry import ToolRegistry
from .contracts import AgentBoundaryError


def receipt_record(invocation, envelope):
    return {"name": invocation.name, "id": invocation.id,
        "arguments": copy.deepcopy(dict(invocation.arguments)), "status": envelope.status,
        "executed": True, "data": copy.deepcopy(envelope.data or {}), "phase": "foreground",
        "call_fingerprint": hashlib.sha256(_call_signature(invocation).encode()).hexdigest()}


class _TransferredRead:
    """Keep the audited adapter's original quota and forbid gratuitous rereads."""
    agent_repeatable_observation = False

    def __init__(self, handler, records):
        self.handler = handler
        self.versions = {}
        for row in records:
            if row["name"] != handler.tool_type or row["status"] != "ok":
                continue
            updates = row.get("data", {}).get("state_updates", {})
            version = updates.get("content_sha256")
            if isinstance(version, str) and len(version) == 64:
                signature = _call_signature(type("Read", (), {"name": row["name"], "arguments": row["arguments"]})())
                self.versions[signature] = version

    def __getattr__(self, name):
        return getattr(self.handler, name)

    def agent_previous_observation_changed(self, *, invocation, context):
        probe = getattr(self.handler, "agent_query_source_version", None)
        old = self.versions.get(_call_signature(invocation))
        if not old or not callable(probe):
            return False
        # Only an adapter's independent version observation can permit a
        # reread. An executor's reason string or requested version grants none.
        version = probe(arguments=dict(invocation.arguments), context=context)
        if not isinstance(version, str) or len(version) != 64 or version == old:
            return False
        context.setdefault("agent_query_source_refreshes", []).append({
            "invocation_id": invocation.id, "reason": "adapter_observed_source_version_change",
            "old_version": old, "new_version": version})
        return True

    def execute(self, *, call, context):
        result = self.handler.execute(call=call, context=context)
        if isinstance(result, Mapping):
            version = result.get("state_updates", {}).get("content_sha256")
            if isinstance(version, str) and len(version) == 64:
                args = {k: v for k, v in call.items() if k != "type" and not k.startswith("_tool_")}
                signature = _call_signature(type("Read", (), {"name": self.tool_type, "arguments": args})())
                self.versions[signature] = version
        return result


@dataclass(frozen=True)
class ForegroundQueryCheckpoint:
    operation_id: str
    work_id: str
    task_id: str
    requester: str
    session: str
    message_id: str
    original_request: str
    authority_revision: int
    permission_mode: str
    deadline: float
    clock: Callable[[], float] = field(repr=False, compare=False)
    calls: tuple[Any, ...] = field(repr=False)
    envelopes: tuple[Any, ...] = field(repr=False)
    handlers: Mapping[str, Any] = field(repr=False)
    read_context: Mapping[str, Any] = field(repr=False)
    remaining: tuple[str, ...]
    budget_at_transfer: Mapping[str, Any]
    research: ResearchLedger | None = field(default=None, repr=False, compare=False)
    _host_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if (not self.operation_id or not self.work_id or not self.task_id or not self.requester
                or not self.session or not self.message_id or not self.original_request
                or not callable(self.clock) or len(self.calls) != len(self.envelopes)
                or not self.calls or not self.handlers or not self.remaining):
            raise AgentBoundaryError("invalid_foreground_query_checkpoint")
        if any(call.id != envelope.invocation_id for call, envelope in zip(self.calls, self.envelopes)):
            raise AgentBoundaryError("query_receipt_binding_conflict")
        if any(getattr(handler, "agent_audited_query_read", False) is not True or handler.tool_type != name
               or not callable(getattr(getattr(handler, "capability", None), "admits", None))
               for name, handler in self.handlers.items()):
            raise AgentBoundaryError("query_audited_read_required")
        object.__setattr__(self, "handlers", MappingProxyType(dict(self.handlers)))
        object.__setattr__(self, "read_context", MappingProxyType(dict(self.read_context)))
        object.__setattr__(self, "budget_at_transfer", MappingProxyType(dict(self.budget_at_transfer)))
        object.__setattr__(self, "research", copy.deepcopy(self.research))

    def stop_reason(self):
        return "task_time_budget_exhausted" if self.clock() >= self.deadline else ""

    def projection(self):
        return {"operation_id": self.operation_id, "work_id": self.work_id,
            "requester": self.requester, "session": self.session, "message_id": self.message_id,
            "allowed_tools": sorted(self.handlers), "authority_revision": self.authority_revision,
            "permission_mode": self.permission_mode, "remaining": list(self.remaining),
            "budget_at_transfer": dict(self.budget_at_transfer),
            "receipt_ids": [row.invocation_id for row in self.envelopes],
            "runnable_checkpoint": "process_local_only"}

    def records(self):
        return [receipt_record(call, envelope) for call, envelope in zip(self.calls, self.envelopes)]

    def validate(self, task, grant, context):
        first = (task.get("requests") or [{}])[0]
        if ((task.get("task_id"), task.get("requester"), task.get("session")) !=
                (self.task_id, self.requester, self.session)
                or (str(first.get("message_id") or ""), first.get("text")) !=
                (self.message_id, self.original_request)
                or str(grant.authorization_ref) != str(self.authority_revision)
                or grant.permission_mode != self.permission_mode
                or context.get("agent_work_id") != self.work_id
                or not set(grant.allowed_tools) <= set(self.handlers)
                or task.get("foreground_query", {}).get("operation_id") != self.operation_id):
            raise AgentBoundaryError("foreground_query_scope_conflict")
        if self.stop_reason():
            raise AgentBoundaryError(self.stop_reason())
        if not grant.still_authorized():
            raise AgentBoundaryError("agent_permission_changed")

    def tool_host(self, host):
        if "host" not in self._host_cache:
            self._host_cache["host"] = ToolHost(ToolRegistry({name: _TransferredRead(handler, self.records())
                for name, handler in self.handlers.items()}), host_id=host.host_id)
        return self._host_cache["host"]

    def seed(self, context, handlers):
        """Retain real live receipts and duplicate guards, without a first call."""
        records = self.records()
        task = context["agent_task_state"]
        binding = {"task_id": self.task_id, "requester": self.requester, "session": self.session,
            "attempt": task["execution_attempt"], "work_id": self.work_id,
            "work_attempt": context["agent_work_attempt"]}
        for row in records:
            row["execution_binding"] = copy.deepcopy(binding)
            row["query_operation_id"] = self.operation_id
        state = AgentState(task_id=self.task_id, context=context, tool_calls=records)
        state.transcript = [{"role": "tool", "name": call.name, "status": envelope.status,
            "content": str(envelope.model_feedback or "")[:6000] +
                ("\n[原回执末尾在任务资料包中；本片段不是全文]" if len(str(envelope.model_feedback or "")) > 6000 else "")}
            for call, envelope in zip(self.calls, self.envelopes)]
        state.research.configure(handlers, context)
        if self.research is not None:
            # This capability transfers a live observation, unlike ordinary
            # restoration from JSON which still requires source refresh.
            prior = self.research
            state.research.candidates = copy.deepcopy(prior.candidates)
            state.research.sources = copy.deepcopy(prior.sources)
            state.research.intent = copy.deepcopy(prior.intent)
            state.research.review_calls = prior.review_calls
            state.research.answer_review_calls = prior.answer_review_calls
            state.research.answer_check = copy.deepcopy(prior.answer_check)
        state.local_answer.configure(context)
        state.history_answer.configure(context)
        for call, envelope in zip(self.calls, self.envelopes):
            state.local_answer.observe(call, envelope, handlers[call.name], context)
            state.history_answer.observe(envelope, handlers[call.name])
        state.observation_calls = len(self.calls)
        state.source_discovery_calls = sum(bool(getattr(handlers[call.name], "agent_research_candidate_field", ""))
            for call in self.calls)
        checkpoint = AgentRunCheckpoint(state=state, envelopes=copy.deepcopy(list(self.envelopes)))
        checkpoint.search_urls = {row["url"] for row in state.research.candidates.values()}
        for call, envelope in zip(self.calls, self.envelopes):
            signature = _call_signature(call)
            checkpoint.signature_attempts[signature] = checkpoint.signature_attempts.get(signature, 0) + 1
            if envelope.status == "ok":
                checkpoint.successful_signatures[signature] = ("observe", 0)
            else:
                key = hashlib.sha256(signature.encode()).hexdigest()
                state.progress.setdefault("failure_attempts", {})[key] = checkpoint.signature_attempts[signature]
        return checkpoint
