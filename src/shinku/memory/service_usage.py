"""Service responsibilities: usage."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .models import MemoryContext, MemoryRecord
from .lifecycle import normalized

from .service_models import MemoryServiceUsageDependencies


def bind_usage_context(
    *,
    scope: str,
    message_id: str,
    requester_id: str,
    records: Sequence[MemoryRecord],
    user_text: str,
    deps: MemoryServiceUsageDependencies,
) -> None:
    """Keep only actually delivered current-turn candidates, not proofs.

        This bounded in-memory map is never restored as reinforcement evidence.
        Tools may add candidates under the same current request; mere binding
        changes no counters or levels.
        """
    if not deps.policy().lifecycle.enabled or not scope or not message_id or not requester_id:
        return
    now = int(deps.clock())
    with deps.usage_lock():
        deps.set_usage_contexts({key: item for key, item in deps.usage_contexts().items() if 0 <= now - item[0] <= 15 * 60})
        key = (scope, message_id)
        prior = deps.usage_contexts().get(key)
        combined = tuple(records) if prior is None or prior[1] != requester_id else (*prior[2], *tuple(records))
        by_id = {item.record_id: item for item in combined}
        deps.usage_contexts()[key] = (now, requester_id, tuple(list(by_id.values())[-24:]), str(user_text)[:2000])
        while len(deps.usage_contexts()) > 128:
            deps.usage_contexts().pop(next(iter(deps.usage_contexts())))


def forget_usage_contexts(
    *,
    scope: str,
    record_ids: set[str],
    deps: MemoryServiceUsageDependencies,
) -> None:
    with deps.usage_lock():
        if not scope:
            deps.usage_contexts().clear()
            return
        for key, entry in list(deps.usage_contexts().items()):
            remaining = tuple(record for record in entry[2] if not (record.scope == scope and record.record_id in record_ids))
            if not remaining:
                deps.usage_contexts().pop(key, None)
            elif remaining != entry[2]:
                deps.usage_contexts()[key] = (entry[0], entry[1], remaining, entry[3])


def finalize_usage(
    *,
    conversation_id: str,
    conversation_type: str,
    message_id: str,
    requester_id: str,
    final_text: str,
    deps: MemoryServiceUsageDependencies,
) -> dict[str, Any]:
    """Verify final source-bound use after generation, never on retrieval."""
    scope = deps.scope_for(conversation_id, conversation_type)
    now = int(deps.clock())
    with deps.usage_lock():
        pending = deps.usage_contexts().pop((scope, message_id), None)
    if pending is None or pending[1] != requester_id or not 0 <= now - pending[0] <= 15 * 60:
        return {"accepted": 0, "reason": "no_current_usage_context"}
    if re.search(r"记错|更正|不再|不是这样|没有说过|别记|忘记|忘掉", pending[3]):
        return {"accepted": 0, "reason": "correction_or_revocation_turn"}
    accepted = 0
    review_records: dict[str, list[MemoryRecord]] = {}
    for record in pending[2]:
        if record.scope != scope and not (record.scope == deps.owner_scope() and deps.is_master_user(requester_id)):
            continue
        # Conservative fast path: a complete source sentence is present
        # in the delivered answer. Arbitrary model-supplied IDs do not count.
        sentences = [part.strip() for part in re.split(r"(?<=[。！？!?\n])", record.text) if 12 <= len(part.strip()) <= 1000]
        quote = next((part for part in sentences if part in final_text), "")
        explicit_recall = bool(re.search(r"记得|回忆|说过|聊过|提过|喜欢什么", pending[3]))
        if not quote or not (explicit_recall or normalized(pending[3]) == normalized(record.text)):
            review_records.setdefault(record.scope, []).append(record)
            continue
        result = deps.store().lifecycle.reinforce_used(record=record, quote=quote, final_text=final_text, actor_id=requester_id, evidence_id=f"turn:{scope}:{message_id}", now=now, session_id=scope)
        accepted += int(bool(result.get("accepted")))
    queued = sum(deps.lifecycle_review().enqueue(scope=target_scope, actor=requester_id,
        evidence_id=f"turn:{scope}:{message_id}", kind="use", occurred=now,
        user_text=pending[3], final_text=final_text, records=records)
        for target_scope, records in review_records.items())
    return {"accepted": accepted, "semantic_reviews_queued": queued}


def reinforce_rementions(
    *,
    scope: str,
    body: str,
    sender_id: str,
    evidence_id: str,
    now: int,
    occurred_at: int | None=None,
    deps: MemoryServiceUsageDependencies,
) -> int:
    if not deps.policy().lifecycle.enabled or not sender_id or len(normalized(body)) < 8:
        return 0
    with deps.store()._lock, deps.store()._connect() as conn:
        rows = conn.execute("SELECT * FROM memory_events WHERE scope=? AND status='active' AND layer IN ('observation','episodic','legacy_raw') ORDER BY created_at DESC LIMIT ?", (scope, deps.policy().event_scan_limit)).fetchall()
    matched = [deps.store()._event_record(row, score=1.0) for row in rows if normalized(str(row["text"])) == normalized(body)]
    # Duplicates are references to the same experience, not independent
    # confirmations. Strengthen the earliest matching source once.
    matched.sort(key=lambda item: (item.created_at, item.record_id))
    for record in matched:
        result = deps.store().lifecycle.reinforce_remention(record=record, user_text=body, actor_id=sender_id, evidence_id=evidence_id, now=now, session_id=scope, occurred_at=occurred_at)
        if result.get("accepted"):
            return 1
    if deps.policy().lifecycle.semantic_review_enabled and deps.lifecycle_review().safe_turn(body, "remention"):
        candidates = deps.store().search(query=body, scope=scope, now=now, limit=deps.policy().lifecycle.review_candidates)
        by_text: dict[str, MemoryRecord] = {}
        for candidate in sorted(candidates, key=lambda item: (item.created_at, item.record_id)):
            by_text.setdefault(normalized(candidate.text), candidate)
        deps.lifecycle_review().enqueue(scope=scope, actor=sender_id, evidence_id=evidence_id,
            kind="remention", occurred=now if occurred_at is None else occurred_at,
            user_text=body, records=tuple(by_text.values()))
    return 0


def record_audit(
    *,
    conversation_id: str,
    scope: str,
    now: int,
    context: MemoryContext,
    deps: MemoryServiceUsageDependencies,
) -> None:
    try:
        deps.store().record_routing_audit(
            conversation_id=conversation_id,
            scope=scope,
            created_at=now,
            decision=context.decision.as_dict(),
            retrieved_count=context.retrieved_count,
            injected_chars=context.injected_chars,
        )
    except Exception:
        # Auditing is diagnostic only; a locked/older database must not
        # turn a valid model request into a memory or QQ failure.
        return
