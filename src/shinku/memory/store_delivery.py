"""Store responsibilities: delivery."""

from __future__ import annotations

import json

from .models import MemoryRecord

from .store_models import MemoryStoreDeliveryDependencies


def confirm_assistant_delivery(
    *,
    scope: str,
    source_id: str,
    expected_text: str,
    delivered_at: int,
    status: str='sent',
    requester_name: str='',
    proof_id: str='',
    content_complete: bool=False,
    deps: MemoryStoreDeliveryDependencies,
) -> bool:
    """Attach a transport receipt to an existing, active assistant source.

        No new memory or raw speech is copied. Dry runs/total failures never
        call this; partial sends prove occurrence, not the complete body.
        """
    if not scope.startswith("group:") or status not in {"sent", "partial"}:
        return False
    with deps.lock(), deps.connect() as conn:
        row = conn.execute("SELECT * FROM memory_events WHERE scope=? AND source_id=? AND status='active' AND layer='episodic'",
                           (scope, source_id)).fetchone()
        if row is None or row["text"] != expected_text or int(row["created_at"]) > delivered_at:
            return False
        meta = deps.loads(row["metadata_json"])
        if (meta.get("role") != "assistant" or meta.get("source") != "qq_turn"
                or not meta.get("self_id") or meta.get("sender_id") != meta.get("self_id")
                or not source_id.startswith(f"turn:{row['conversation_id']}:")
                or not source_id.endswith(":assistant")
                or deps.lifecycle().revoked(conn, scope=scope, source_id=source_id, text=row["text"])):
            return False
        if meta.get("delivery_status") == "sent" and meta.get("delivery_proof") == "qq_transport":
            return True
        meta.update({"delivery_status": status, "delivered_at": int(delivered_at),
                     "delivery_content_complete": bool(content_complete and status == "sent"),
                     "delivery_proof": "qq_transport", "delivery_proof_id": str(proof_id)[:160]})
        if requester_name:
            meta["requester_name"] = str(requester_name).strip()[:80]
        encoded = json.dumps(meta, ensure_ascii=False)
        conn.execute("UPDATE memory_events SET metadata_json=? WHERE event_id=?", (encoded, row["event_id"]))
        conn.execute("UPDATE memory_turns SET metadata_json=? WHERE scope=? AND source_id=? AND role='assistant'",
                     (encoded, scope, source_id))
        return True


def delivered_group_activity(
    *,
    start_at: int,
    end_at: int,
    now: int,
    scope: str='',
    self_id: str='',
    limit: int=8,
    deps: MemoryStoreDeliveryDependencies,
) -> list[MemoryRecord]:
    """Candidates are ONLY the bot's receipt-backed group replies.

        The service must authenticate private cross-group readers. No user
        messages, observations, summaries, facts, or private turns are selected.
        """
    wanted = max(1, min(12, int(limit)))
    scope_clause = " AND scope=?" if scope else ""
    bot_clause = " AND json_extract(metadata_json,'$.self_id')=?" if self_id else ""
    params = [int(start_at), int(end_at), int(now), *([scope] if scope else []),
              *([self_id] if self_id else []), min(96, wanted * 8)]
    with deps.lock(), deps.connect() as conn:
        rows = conn.execute("""SELECT * FROM memory_events
                WHERE scope LIKE 'group:%' AND status='active' AND layer='episodic'
                  AND json_extract(metadata_json,'$.role')='assistant'
                  AND json_extract(metadata_json,'$.source')='qq_turn'
                  AND json_extract(metadata_json,'$.delivery_proof')='qq_transport'
                  AND json_extract(metadata_json,'$.delivery_status') IN ('sent','partial')
                  AND json_extract(metadata_json,'$.delivered_at') BETWEEN ? AND ?
                  AND (expires_at=0 OR expires_at>?)
                  AND json_extract(metadata_json,'$.self_id') IS NOT NULL
                  AND json_extract(metadata_json,'$.self_id') != ''
                  AND json_extract(metadata_json,'$.sender_id')=json_extract(metadata_json,'$.self_id')
                """ + scope_clause + bot_clause + " ORDER BY json_extract(metadata_json,'$.delivered_at') DESC,source_id DESC LIMIT ?", params).fetchall()
        return [deps.event_record(row, score=1.0) for row in rows
                if not deps.lifecycle().revoked(conn, scope=row["scope"], source_id=row["source_id"], text=row["text"])][:wanted]
