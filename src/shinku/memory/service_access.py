"""Service responsibilities: access."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .lifecycle import sensitive
from .self_activity import activity_window, activity_answer

from .service_models import MemoryServiceAccessDependencies


def recent_owner_handoff(
    *,
    requester_id: str,
    limit: int=4,
    max_age_seconds: int=4 * 60 * 60,
    deps: MemoryServiceAccessDependencies,
) -> tuple[Any, ...]:
    """Return a tiny, fresh topic cue for a master's private greeting.

        This is intentionally not the normal recall path: it reads only the
        isolated owner scope, requires an authenticated configured owner, and
        returns records from the most recently active source conversation.
        Group callers can still explicitly request cross-context recall via
        ``prepare_turn``; they never receive this unsolicited private cue.
        """

    if not deps.is_master_user(requester_id):
        return ()
    now = int(deps.clock())
    scan_limit = max(1, min(deps.policy().max_results, 16))
    candidates = deps.store().recent_events(
        scope=deps.owner_scope(),
        limit=scan_limit,
        now=now,
        max_age_seconds=max(1, int(max_age_seconds or 1)),
        layers=("episodic",),
        include_assistant=True,
    )
    trusted = [
        item
        for item in candidates
        if bool(item.metadata.get("owner_memory"))
        and str(item.metadata.get("role") or "").strip().lower() in {"user", "assistant"}
        and str(item.metadata.get("owner_source_scope") or "").strip()
    ]
    if not trusted:
        return ()
    latest_source = str(trusted[0].metadata.get("owner_source_scope") or "").strip()
    same_conversation = [
        item
        for item in trusted
        if str(item.metadata.get("owner_source_scope") or "").strip() == latest_source
    ]
    same_conversation.sort(key=lambda item: (int(item.created_at), item.record_id))
    return tuple(same_conversation[-max(1, min(8, int(limit or 4))) :])


def owner_memory_allowed(
    *,
    conversation_id: str,
    conversation_type: str,
    sender_id: str='',
    metadata: Mapping[str, Any] | None=None,
    role: str='user',
    deps: MemoryServiceAccessDependencies,
) -> bool:
    """Whether one handled turn belongs in the owner's shared memory.

        Private turns are owner-relevant only when the sender is configured as
        the owner.  A group turn additionally needs an explicit Shinku
        involvement marker; this prevents a user's unrelated group chatter
        from becoming owner memory merely because the owner was present.
        """

    if not deps.policy().master_user_ids:
        return False
    meta = dict(metadata or {})
    kind = str(conversation_type or "private").strip().lower()
    sender = str(sender_id or meta.get("sender_id") or "").strip()
    if str(role or "").strip().lower() == "assistant":
        # The speaker is Shinku, while owner-memory eligibility follows
        # the person whose turn she was answering.
        sender = str(meta.get("requester_id") or sender).strip()
    if not sender and kind in {"private", "friend", "direct"}:
        sender = str(conversation_id or "").strip()
    if kind in {"private", "friend", "direct", "desktop"}:
        return deps.is_master_user(sender)
    if "group" not in kind:
        return False
    if not bool(meta.get("shinku_involved") or meta.get("owner_memory_eligible")):
        return False
    # Mentioning/quoting the owner is not authority to write their private
    # library. Related third-party speech remains in the original group.
    return deps.is_master_user(sender)


def is_master_request(
    *,
    conversation_id: str,
    conversation_type: str,
    message_meta: Mapping[str, Any] | None,
    deps: MemoryServiceAccessDependencies,
) -> bool:
    meta = dict(message_meta or {})
    sender_id = str(meta.get("sender_id") or "").strip()
    kind = str(conversation_type or "private").strip().lower()
    if not sender_id and kind in {"private", "friend", "direct"}:
        sender_id = str(conversation_id or "").strip()
    return deps.is_master_user(sender_id)


def history_scopes(
    scope: str,
    *,
    owner_request: bool,
    mode: str,
    deps: MemoryServiceAccessDependencies,
) -> tuple[str, ...]:
    """Return current scope plus owner scope only for an owner recall."""

    current = str(scope or "").strip()
    if not owner_request or mode not in {"retrieve", "summary"} or current == deps.owner_scope():
        return (current,)
    return (current, deps.owner_scope())


def inspect_self_group_activity(
    *,
    conversation_id: str,
    conversation_type: str,
    requester_id: str,
    query: str,
    self_id: str='',
    limit: int=8,
    deps: MemoryServiceAccessDependencies,
) -> dict[str, Any]:
    """Separate bot-owned activity capability, never general cross-chat recall."""
    now = int(deps.clock())
    current = deps.scope_for(conversation_id, conversation_type)
    is_group = current.startswith("group:")
    owner_private = not is_group and deps.is_master_user(requester_id)
    requested_group = re.search(r"群\s*(\d{5,12})", query)
    foreign_group = requested_group and requested_group.group(1) != str(conversation_id)
    denied = not (is_group or owner_private) or bool(is_group and (foreign_group or re.search(r"另(?:一|个)|别的群|其他群", query)))
    scopes = [current] if is_group else []
    records = []
    start_at, end_at = activity_window(query, now=now, date_window=deps.request_date_window(query, now=now))
    # A private reader must also bind 'my speech' to the current bot's
    # authenticated identity; missing identity is uncertainty, not a scan
    # of every bot that may ever have used the database.
    if not denied and (is_group or self_id):
        selected_scope = current if is_group else ("group:" + requested_group.group(1) if requested_group else "")
        candidates = deps.store().delivered_group_activity(start_at=start_at, end_at=end_at, now=now,
            scope=selected_scope, self_id=self_id, limit=limit)
        fresh = deps.integrity_reader().read_sources(requests=tuple((row.record_id, row.scope, "") for row in candidates), now=now)
        for row in fresh:
            if row is None:
                continue
            meta = row.metadata
            safe_meta = {key: meta[key] for key in ("role", "sender_id", "self_id", "requester_id", "requester_name", "delivery_status", "delivered_at") if key in meta}
            can_quote = (meta.get("delivery_status") == "sent" and meta.get("delivery_content_complete")
                         and not meta.get("raw_detail_unavailable")
                         and meta.get("reading_representation") != "summary" and not sensitive(row.text)
                         and len(row.text) <= min(700, deps.policy().render_record_chars))
            records.append({"record_id": row.record_id, "source_id": row.source_id, "scope": row.scope,
                "created_at": row.created_at, "text": row.text if can_quote else "", "metadata": safe_meta})
        scopes = list(dict.fromkeys(row["scope"] for row in records))
    result = {"mode": "self_activity", "scope": current, "retrieval_scopes": scopes,
        "access": "denied" if denied else "allowed", "query": query, "count": len(records),
        "records": records, "coverage": "bounded_delivery_receipts", "absence_means": "unknown",
        "observed_at": now,
        "time_window": {"start_at": start_at, "end_at": end_at}}
    result["answer"] = activity_answer(result, query=query)
    result["rendered"] = (f"【{deps.character.name}自己的群聊发送证据】\n只证明{deps.character.name}自己的已发送回复；提问者不等于所有在场者。"
        "没有匹配记录只表示本轮无法确认，不证明没有说过话。禁止否认已有发送证据，禁止编造对象或原话。\n"
        + result["answer"])
    result["followup_context"] = result["rendered"]
    return result
