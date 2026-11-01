"""Service responsibilities: recording."""

from __future__ import annotations

from collections.abc import Mapping

from .models import ForgetReport
from .lifecycle import sensitive
from .source_reader import fingerprint

from .service_models import MemoryServiceRecordingDependencies


from .service_recording_models import MemoryRecordedTurn


def prepare_recorded_turn(q: MemoryRecordedTurn, deps: MemoryServiceRecordingDependencies) -> bool:
    q.body = str(q.text or "").strip()
    if not q.body:
        return False
    q.now = int(deps.clock())
    q.scope = deps.scope_for(q.conversation_id, q.conversation_type)
    q.normalized_role = str(q.role or "user").strip().lower() or "user"
    q.effective_source_id = str(q.source_id or "").strip() or f"turn:{q.conversation_id}:{q.message_id or q.now}"
    q.turn_metadata = dict(q.metadata) if isinstance(q.metadata, Mapping) else {}
    q.turn_metadata.update({"source": "qq_turn", "role": q.normalized_role,
                            "conversation_type": q.conversation_type})
    q.owner_allowed = deps.owner_memory_allowed(
        conversation_id=q.conversation_id,
        conversation_type=q.conversation_type,
        sender_id=str(q.turn_metadata.get("sender_id") or ""),
        metadata=q.turn_metadata,
        role=q.normalized_role,
    )
    q.owner_source_id = f"owner:{q.effective_source_id}"
    q.owner_metadata = {
        **q.turn_metadata,
        "owner_memory": True,
        "owner_source_scope": q.scope,
    }
    return True


def note_turn_interaction(q: MemoryRecordedTurn, deps: MemoryServiceRecordingDependencies) -> None:
    if q.normalized_role == "user" and not deps.forget_pattern().match(q.body):
        sender_id = str(q.turn_metadata.get("sender_id") or (q.conversation_id if str(q.conversation_type).lower() != "group" else ""))
        # Preserve author identity even for simple private callers. Group
        # callers must supply the actual author; it is never inferred.
        q.turn_metadata.setdefault("sender_id", sender_id)
        q.owner_metadata.setdefault("sender_id", sender_id)
        evidence_id = f"turn:{q.scope}:{q.message_id}"
        occurred_at = q.turn_metadata.get("created_at")
        occurred_at = int(occurred_at) if isinstance(occurred_at, (int, float)) and not isinstance(occurred_at, bool) else q.now
        for target_scope in ((q.scope, deps.owner_scope()) if q.owner_allowed and q.scope != deps.owner_scope() else (q.scope,)):
            deps.store().lifecycle.note_interaction(scope=target_scope, evidence_id=evidence_id, now=q.now, occurred_at=occurred_at)
            deps.reinforce_rementions(scope=target_scope, body=q.body, sender_id=sender_id, evidence_id=evidence_id, now=q.now, occurred_at=occurred_at)


def write_turn_history(q: MemoryRecordedTurn, deps: MemoryServiceRecordingDependencies) -> None:
    deps.store().record_turn(
        conversation_id=q.conversation_id,
        scope=q.scope,
        role=q.normalized_role,
        content=q.body,
        source_id=q.effective_source_id,
        created_at=q.now,
        metadata=q.turn_metadata,
    )
    if q.owner_allowed and q.scope != deps.owner_scope():
        deps.store().record_turn(
            conversation_id="master",
            scope=deps.owner_scope(),
            role=q.normalized_role,
            content=q.body,
            source_id=q.owner_source_id,
            created_at=q.now,
            metadata=q.owner_metadata,
        )


def record_assistant_events(
    q: MemoryRecordedTurn,
    deps: MemoryServiceRecordingDependencies,
) -> None:
    deps.store().record_event(
        conversation_id=q.conversation_id,
        scope=q.scope,
        text=q.body,
        source_id=q.effective_source_id,
        layer="episodic",
        created_at=q.now,
        importance=0.18,
        confidence=0.42,
        expires_at=q.now + deps.policy().event_ttl_seconds if deps.policy().event_ttl_seconds else 0,
        metadata=q.turn_metadata,
    )
    if q.owner_allowed and q.scope != deps.owner_scope():
        deps.store().record_event(
            conversation_id="master",
            scope=deps.owner_scope(),
            text=q.body,
            source_id=q.owner_source_id,
            layer="episodic",
            created_at=q.now,
            importance=0.3,
            confidence=0.55,
            expires_at=q.now + deps.policy().event_ttl_seconds if deps.policy().event_ttl_seconds else 0,
            metadata=q.owner_metadata,
        )
    return None


def forget_requested_memory(
    q: MemoryRecordedTurn,
    deps: MemoryServiceRecordingDependencies,
) -> ForgetReport | None:
    forget_match = deps.forget_pattern().match(q.body)
    if forget_match:
        query = str(forget_match.group("query") or "").strip(" ：:，,。.!！")
        forgotten, purged = deps.store().forget(query=query, scope=q.scope, now=q.now)
        if q.owner_allowed and q.scope != deps.owner_scope():
            owner_forgotten, owner_purged = deps.store().forget(query=query, scope=deps.owner_scope(), now=q.now)
            forgotten += owner_forgotten
            purged += owner_purged
        return ForgetReport(query=query, scope=q.scope, forgotten_count=forgotten, purged_count=purged, reason="explicit_user_request")
    return None


def write_user_memory(q: MemoryRecordedTurn, deps: MemoryServiceRecordingDependencies) -> None:
    fact_match = deps.fact_pattern().match(q.body)
    # An explicit remember command is not a credential vault. Sensitive
    # material may remain in the separately retained turn log, never be
    # silently promoted into a non-expiring fact by this command.
    if fact_match and sensitive(str(fact_match.group("fact") or "")):
        fact_match = None
    if fact_match:
        fact = str(fact_match.group("fact") or "").strip()
        if fact:
            key = f"explicit:{deps.key(fact)}"
            fact_metadata = ({"metadata": {"role": "user", "sender_id": str(q.turn_metadata.get("sender_id") or ""),
                "confirmation_kind": "explicit_memory_request"}} if deps.policy().current_understanding_enabled else {})
            deps.store().upsert_fact(
                conversation_id=q.conversation_id,
                scope=q.scope,
                fact_key=key,
                text=fact,
                source_id=q.effective_source_id,
                now=q.now,
                **fact_metadata,
            )
            if q.owner_allowed and q.scope != deps.owner_scope():
                owner_fact_metadata = fact_metadata
                if deps.policy().current_understanding_enabled:
                    with deps.store()._lock, deps.store()._connect() as conn:
                        origin_row = conn.execute("SELECT * FROM memory_facts WHERE scope=? AND fact_key=?", (q.scope, key)).fetchone()
                        origin_record = deps.store()._fact_record(origin_row, score=0.0)
                    owner_fact_metadata = {"metadata": {**fact_metadata["metadata"],
                        "owner_origin_scope": q.scope, "owner_origin_record_id": origin_record.record_id,
                        "owner_origin_version": fingerprint(origin_record, str(origin_row["conversation_id"]))}}
                deps.store().upsert_fact(
                    conversation_id="master",
                    scope=deps.owner_scope(),
                    fact_key=key,
                    text=fact,
                    source_id=q.owner_source_id,
                    now=q.now,
                    **owner_fact_metadata,
                )
    else:
        deps.store().record_event(
            conversation_id=q.conversation_id,
            scope=q.scope,
            text=q.body,
            source_id=q.effective_source_id,
            layer="episodic",
            created_at=q.now,
            importance=0.22,
            confidence=0.45,
            expires_at=q.now + deps.policy().event_ttl_seconds if deps.policy().event_ttl_seconds else 0,
            metadata=q.turn_metadata,
        )
        if q.owner_allowed and q.scope != deps.owner_scope():
            deps.store().record_event(
                conversation_id="master",
                scope=deps.owner_scope(),
                text=q.body,
                source_id=q.owner_source_id,
                layer="episodic",
                created_at=q.now,
                importance=0.3,
                confidence=0.55,
                expires_at=q.now + deps.policy().event_ttl_seconds if deps.policy().event_ttl_seconds else 0,
                metadata=q.owner_metadata,
            )
