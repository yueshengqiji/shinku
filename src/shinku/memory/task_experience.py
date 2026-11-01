"""Opt-in auxiliary task experience projection from independently stored facts.

This consumes no expression lease, runs no model/tool and sends nothing. The
memory card and dedupe history commit together; task and memory databases do
not. Explicit retries and source-checked reads close that auxiliary boundary.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

from shinku.agent_runtime.contracts import AgentScope, canonical_hash
from shinku.agent_runtime.result_events import TERMINAL, ResultEventStore
from shinku.result_expression.facts import identity, validate_result
from .experience_facts import DeploymentEvidence, make_projection
from .experience_store import ExperienceStore


class TaskExperienceProjector:
    def __init__(self, memory_service, task_store, *, expression_archive=None, material_store=None,
                 max_versions=4096):
        if task_store.result_events is None:
            raise ValueError("experience_requires_result_events")
        if expression_archive is not None and expression_archive.store is not task_store:
            raise ValueError("experience_requires_same_expression_store")
        if material_store is not None and material_store.store is not task_store:
            raise ValueError("experience_requires_same_material_store")
        if getattr(memory_service.store, "task_experience", None) is not None:
            raise ValueError("experience_already_attached")
        self.memory = memory_service
        self.tasks = task_store
        self.events = task_store.result_events
        self.archive = expression_archive
        self.materials = material_store
        self.versions = ExperienceStore(memory_service.store, max_versions=max_versions)
        # SourceReader also fails closed when a restarted host omits this binding.
        memory_service.store.task_experience = self
        self._grants = {}

    @staticmethod
    def _authorized(check):
        try:
            return callable(check) and check() is True
        except Exception:
            return False

    def _owner(self, scope, metadata):
        kind, _, conversation = scope.session.partition(":")
        return kind in {"private", "group"} and bool(conversation) and self.memory.owner_memory_allowed(
            conversation_id=conversation, conversation_type=kind, sender_id=scope.requester,
            metadata=metadata, role="user")

    def _source_state(self, task, authorized):
        items = task.get("material_package", {}).get("items", [])
        reasons = set()
        bindings = []
        if task.get("material_package", {}).get("gaps"):
            reasons.add("material_package_gaps")
        for item in items[:128]:
            bindings.append({k: item.get(k) for k in ("reference", "content_sha256", "version", "kind", "complete", "required")})
            if self.materials is None:
                reasons.add("material_resolver_unavailable")
                continue
            try:
                self.materials.resolve(item["reference"], task=task, still_authorized=authorized)
            except Exception as exc:
                code = getattr(exc, "code", "material_source_unavailable")
                reasons.add(code if isinstance(code, str) and code.startswith("material_") else "material_source_unavailable")
            if item.get("complete") is not True:
                reasons.add("material_upstream_completeness_unknown")
        if len(items) > 128:
            reasons.add("material_reference_limit")
        status = "unavailable_or_incomplete" if reasons else "available" if items else "unknown"
        return {"status": status, "reasons": sorted(reasons), "binding_hash": canonical_hash(bindings)}

    def _load(self, task_id, scope, authorized, deployment):
        if not self._authorized(authorized):
            raise ValueError("experience_unauthorized")
        task = self.tasks.get(task_id, requester=scope.requester, session=scope.session)
        if not task:
            raise ValueError("experience_task_unavailable")
        result = task.get("execution_result")
        error = validate_result(result, scope)
        if error:
            raise ValueError("experience_result_invalid")
        if (result.get("task_id") != task_id or result["execution"].get("status") not in TERMINAL
                or (result["attempt"], result["version"]) != (task.get("execution_attempt", 0), task["revision"])
                or result["execution"]["status"] != task["status"]
                or result["execution"].get("source") != "agent_executor"
                or result["verification"].get("source") != "completion_verifier"
                or result["verification"].get("accepted") != task.get("completion", {}).get("accepted")):
            raise ValueError("experience_source_binding_invalid")
        if result["execution"]["status"] == "completed" and result["verification"].get("accepted") is not True:
            raise ValueError("experience_unverified_success")
        event = self.events.get(ResultEventStore.key(result), scope=scope)
        if (event is None or event["source_hash"] != result["result_hash"]
                or (event["attempt"], event["version"]) != (result["attempt"], result["version"])):
            raise ValueError("experience_result_event_unavailable")
        expression = self.archive.get(canonical_hash(identity(result)), scope=scope) if self.archive is not None else None
        if expression and expression["result_hash"] != result["result_hash"]:
            raise ValueError("experience_expression_binding_invalid")
        receipt = expression["delivery"] if expression else event["delivery"]
        if (expression and event["delivery"]["status"] != "not_sent"
                and event["delivery"] != expression["delivery"]):
            raise ValueError("experience_delivery_conflict")
        status = receipt.get("status", "uncertain")
        if status not in {"sent", "not_sent", "failed", "uncertain", "partial", "dry_run", "suppressed"}:
            status = "uncertain"
        sent = status == "sent" and receipt.get("sent") is True
        if status == "sent" and not sent:
            status = "uncertain"
        delivery = {"status": status, "sent": sent,
            "message_count": len(receipt.get("message_ids", [])),
            "reference": canonical_hash({k: receipt.get(k) for k in ("status", "sent", "message_ids")}),
            "source": "expression_archive" if expression else "result_event"}
        state = self._source_state(task, authorized)
        return make_projection(result, event, delivery, state, deployment)

    def project(self, *, task_id, scope, still_authorized, metadata=None, deployment=None):
        """Trusted host entry, using current persisted sources only.

        No caller-supplied success, summary, outcome or raw result is accepted.
        The host must bind its live grant on each restart/explicit retry.
        """
        if (not isinstance(scope, AgentScope) or not isinstance(task_id, str) or not task_id or len(task_id) > 240
                or metadata is not None and not isinstance(metadata, Mapping)
                or deployment is not None and not isinstance(deployment, DeploymentEvidence)):
            return {"status": "rejected", "reason": "experience_input_invalid"}
        meta = {"shinku_involved": (metadata or {}).get("shinku_involved") is True}
        if not self._owner(scope, meta):
            return {"status": "rejected", "reason": "experience_owner_scope_denied"}
        try:
            snapshot, text = self._load(task_id, scope, still_authorized, deployment)
            def current():
                if not self._owner(scope, meta):
                    return False
                try:
                    candidate, candidate_text = self._load(task_id, scope, still_authorized, deployment)
                    return candidate["projection_hash"] == snapshot["projection_hash"] and candidate_text == text
                except Exception:
                    return False
            result = self.versions.commit(snapshot, text, scope=self.memory.owner_scope,
                now=int(self.memory.clock()), check_current=current)
            if result["status"] in {"stored", "duplicate"}:
                self._grants[task_id] = (scope, still_authorized, meta, deployment)
            return result
        except Exception as exc:
            reason = str(exc)
            return {"status": "rejected", "reason": reason if reason.startswith("experience_")
                    and len(reason) < 128 else "experience_source_unavailable"}

    def validate_record(self, record):
        """Read-only revalidation; neither memory reinforcement nor a retry."""
        meta = record.metadata
        grant = self._grants.get(meta.get("task_id"))
        if (meta.get("experience_schema") != 1 or grant is None or record.scope != self.memory.owner_scope
                or record.source_id != "owner:agent-task:" + meta.get("task_id", "")
                or hashlib.sha256(record.text.encode()).hexdigest() != meta.get("text_hash")):
            return False
        scope, authorized, owner_meta, deployment = grant
        if not self._owner(scope, owner_meta):
            return False
        try:
            snapshot, text = self._load(meta["task_id"], scope, authorized, deployment)
            return (snapshot["projection_hash"] == meta.get("projection_hash") and text == record.text
                    and canonical_hash({k: v for k, v in meta["snapshot"].items() if k != "projection_hash"})
                    == snapshot["projection_hash"])
        except Exception:
            return False

    def bind(self, *, task_id, scope, still_authorized, metadata=None, deployment=None):
        """Rebind recall after restart without writing or re-running anything."""
        if (not isinstance(scope, AgentScope) or not isinstance(task_id, str) or not task_id or len(task_id) > 240
                or metadata is not None and not isinstance(metadata, Mapping)
                or deployment is not None and not isinstance(deployment, DeploymentEvidence)):
            return False
        meta = {"shinku_involved": (metadata or {}).get("shinku_involved") is True}
        if not self._owner(scope, meta) or not self._authorized(still_authorized):
            return False
        record = self.memory.store.event_by_source_id(source_id="owner:agent-task:" + task_id,
                                                    scope=self.memory.owner_scope)
        if record is None or record.metadata.get("source_scope") != {"requester": scope.requester, "session": scope.session}:
            return False
        if task_id not in self._grants and len(self._grants) >= self.versions.max_versions:
            return False
        self._grants[task_id] = (scope, still_authorized, meta, deployment)
        return True
