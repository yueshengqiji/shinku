"""Bounded result-event consumption; never an executor or a send retry loop."""
from __future__ import annotations

from .service import guard_reason


def _outcome(receipt):
    status = receipt.get("status", "uncertain")
    if status not in {"sent", "failed", "uncertain", "partial", "dry_run", "suppressed"}:
        status = "uncertain"
    confirmed = status == "sent" and receipt.get("sent") is True
    if status == "sent" and not confirmed:
        status = "uncertain"
    safe = {**receipt, "status": status, "sent": confirmed}
    state = ("done" if confirmed or status == "dry_run" else "uncertain" if status in {"uncertain", "partial"}
        else "failed" if status == "failed" else "suppressed")
    return safe, state, "" if state == "done" else "delivery_" + status


class ResultEventConsumer:
    def __init__(self, service):
        self.service = service
        self.events = service.archive.events

    def consume(self, event_id, *, scope, check_current, publish):
        """Trusted host supplies current authorization and its existing sender.

        `publish(source, prepared_record, lease_guard)` returns a delivery
        receipt and optional host value. No routing, secrets or executable
        handler are loaded from a persisted event.
        """
        try:
            claim = self.events.claim(event_id, scope=scope)
            if claim is None:
                return {"claimed": False, "event": self.events.get(event_id, scope=scope)}
        except Exception:
            return {"claimed": False, "error": "result_event_store_unavailable"}

        def guard():
            if not self.events.owns(claim, scope=scope):
                return "result_event_lease_lost"
            task = self.events.store.get(claim["task_id"], requester=scope.requester, session=scope.session)
            result = (task or {}).get("execution_result") or {}
            if (not task or (task.get("execution_attempt", 0), task["revision"]) != (claim["attempt"], claim["version"])
                    or result.get("result_hash") != claim["source_hash"]):
                return "stale_result_event"
            return guard_reason(check_current)

        value = None
        try:
            error = guard_reason(guard)
            if error:
                if error == "expression_current_state_unavailable":
                    self.events.retry(claim, scope=scope, reason=error)
                else:
                    self.events.finish(claim, scope=scope, status="stale", reason=error)
                return {"claimed": True, "error": error, "event": self.events.get(event_id, scope=scope)}
            task = self.events.store.get(claim["task_id"], requester=scope.requester, session=scope.session)
            source = task["execution_result"]
            record = self.service.prepare(source, scope=scope, check_current=guard, recover_interrupted=True)
            if record.get("status") == "processing":
                self.events.retry(claim, scope=scope, reason="expression_in_progress")
                return {"claimed": True, "error": "expression_in_progress"}
            if record.get("status") == "discarded":
                reason = record.get("fallback_reason") or "expression_discarded"
                if reason in {"expression_archive_unavailable", "expression_current_state_unavailable"}:
                    self.events.retry(claim, scope=scope, reason=reason)
                else:
                    self.events.finish(claim, scope=scope, status="suppressed", reason=reason)
                return {"claimed": True, "error": reason}
            # A recorded receipt is sufficient to settle a replay. Calling a
            # sender again would be a second external operation.
            receipt = record["delivery"]
            if receipt["status"] == "not_sent":
                receipt, value = publish(source, record, guard)
            receipt, outcome, reason = _outcome(receipt)
            finished = self.events.finish(claim, scope=scope, status=outcome, reason=reason,
                expression_id=record["expression_id"], delivery=receipt)
            return {"claimed": True, "event": self.events.get(event_id, scope=scope),
                "expression": record, "receipt": receipt, "value": value,
                "error": "" if finished else "result_event_lease_lost"}
        except Exception:
            # The sender adapter normally archives uncertainty. A failure
            # outside it must not create permission to repeat its effects.
            from shinku.agent_runtime.contracts import canonical_hash
            key = canonical_hash({"scope": claim["scope"], "task_id": claim["task_id"],
                "attempt": claim["attempt"], "version": claim["version"], "operation_id": ""})
            try:
                record = self.service.archive.get(key, scope=scope)
                if record and record["delivery"]["status"] != "not_sent":
                    receipt, outcome, reason = _outcome(record["delivery"])
                    finished = self.events.finish(claim, scope=scope, status=outcome, reason=reason,
                        expression_id=record["expression_id"], delivery=receipt)
                    if finished:
                        return {"claimed": True, "event": self.events.get(event_id, scope=scope),
                            "expression": record, "receipt": receipt, "value": value,
                            "error": "" if outcome == "done" else "result_consumer_error"}
                else:
                    self.events.retry(claim, scope=scope, reason="result_consumer_error")
            except Exception:
                # Leave the durable lease/claim intact when diagnostics also
                # fail. Recovery will inspect the publication record first.
                pass
            return {"claimed": True, "error": "result_consumer_error"}
