"""Bind a state answer to this turn's actual, bounded host observation.

The reply is rendered by the host from the observed snapshot. Chat prose cannot
mint evidence. These receipts certify a read at its recorded time, not service
health, deployment, delivery, or completeness outside the selected scope.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _binding(context):
    from .request_scope import RequestScope
    scope = context.get("agent_request_scope") if isinstance(context, Mapping) else None
    return (scope.requester, scope.session, scope.primary_message_id, scope.requests) if isinstance(scope, RequestScope) else None


class StateObservationStore:
    def __init__(self, *, clock=time.time):
        self.clock = clock
        self._reads = OrderedDict()
        self._lock = threading.Lock()

    def record(self, *, snapshot, scope, source, gaps, context):
        available = bool(snapshot) and snapshot.get("status") != "unavailable"
        complete = available and not gaps
        fresh = source in {"fresh_provider", "provider"}
        receipt = {"schema_version": 1, "read_id": uuid.uuid4().hex,
            "scope": scope, "source": source, "observed_at": self.clock(),
            "fresh": fresh, "complete": complete, "gaps": list(dict.fromkeys(gaps)),
            "snapshot_sha256": _digest(snapshot)}
        rendered = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2)
        limit = "本次只说明宿主提供的安全快照在读取时的内容；未核验其他范围、实际部署或消息送达。"
        if fresh and complete:
            answer = f"本次读取范围：{scope}。\n宿主状态快照如下：\n{rendered}\n{limit}"
        else:
            answer = f"本次读取范围：{scope}，尚未取得新鲜且完整的宿主状态快照，不能确认完整答案。\n{rendered[:8000]}\n{limit}"
        facts = [{"label": f"宿主状态快照（{scope}）", "value": snapshot,
            "condition": limit, "reference": "self_state:" + receipt["read_id"]}] if available and len(rendered) <= 8000 else []
        updates = {"self_state": snapshot, "self_state_receipt": receipt,
            "result_valid": fresh and complete, "result_facts": facts}
        if len(rendered) <= 8000:
            with self._lock:
                self._reads[receipt["read_id"]] = {"updates": copy.deepcopy(updates),
                    "binding": _binding(context), "answer": answer}
                while len(self._reads) > 128:
                    self._reads.popitem(last=False)
        return updates

    def resolve(self, *, updates, arguments, context):
        """No second IO: validate the original observation and its turn binding."""
        if not isinstance(updates, Mapping):
            return None
        receipt = updates.get("self_state_receipt")
        if not isinstance(receipt, Mapping):
            return None
        with self._lock:
            original = copy.deepcopy(self._reads.get(receipt.get("read_id")))
        if original is None or original["binding"] is None or original["binding"] != _binding(context):
            return None
        if original["updates"] != updates or receipt.get("scope") != str(arguments.get("scope") or "all").strip().casefold():
            return None
        age = self.clock() - receipt["observed_at"]
        if not math.isfinite(age) or not 0 <= age <= 180:
            return None
        return {"answer": original["answer"], "valid": updates.get("result_valid") is True}


def state_receipt_reply(invocations, envelopes, handlers, context):
    """Handle a pure state read; mixed observations require their own verifier."""
    from shinku.conversation import AnswerEvidence
    from shinku.conversation.models import text_digest
    if not invocations or len(invocations) != len(envelopes):
        return None
    sources = []
    for invocation, envelope in zip(invocations, envelopes):
        handler = handlers.get(invocation.name)
        if getattr(handler, "agent_self_state_answer_source", False) is not True:
            return None
        resolve = getattr(handler, "resolve_state_observation", None)
        result = resolve(updates=(envelope.data or {}).get("state_updates", {}),
            arguments=invocation.arguments, context=context) if (envelope.status == "ok"
                and envelope.invocation_id == invocation.id and callable(resolve)) else None
        if result is None:
            return ("未取得可核验的本回合宿主状态读取凭据，尚不能确认答案。", "partial",
                AnswerEvidence("state", "unverified", query_count=min(128, len(invocations))))
        sources.append(result)
    answer = "\n\n".join(row["answer"] for row in sources)
    valid = all(row["valid"] for row in sources)
    evidence = AnswerEvidence("state", "verified" if valid else "unverified", valid,
        min(128, len(invocations)), min(128, len(sources)), text_digest(answer) if valid else "")
    return answer, "completed" if valid else "partial", evidence
