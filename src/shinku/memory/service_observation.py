"""Service responsibilities: recording."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping


from .service_models import MemoryServiceRecordingDependencies


from .service_recording_models import MemoryObservedMessage


def prepare_observation(q: MemoryObservedMessage, deps: MemoryServiceRecordingDependencies) -> bool:
    q.conversation_id = str(getattr(q.message, "conversation_id", "") or "").strip() or "unknown"
    q.conversation_type = str(getattr(q.message, "conversation_type", "private") or "private").strip().lower()
    if q.conversation_type == "group" and not deps.policy().passive_observe_group:
        return False
    if q.conversation_type != "group" and not deps.policy().passive_observe_private:
        return False
    q.text = str(getattr(q.message, "text", "") or "").strip()
    q.has_media = bool(getattr(q.message, "has_media", False))
    q.attachments = tuple(getattr(q.message, "attachments", ()) or ())
    q.quoted_text = str(getattr(q.message, "quoted_text", "") or "").strip()
    q.quoted_attachments = tuple(getattr(q.message, "quoted_attachments", ()) or ())
    q.has_quoted_media = bool(q.quoted_attachments)
    q.effective_has_media = q.has_media or q.has_quoted_media
    if not q.text and not q.effective_has_media and not q.quoted_text:
        return False
    now_value = getattr(q.message, "timestamp", None)
    try:
        q.now = int(float(now_value)) if now_value is not None else int(deps.clock())
    except (TypeError, ValueError):
        q.now = int(deps.clock())
    q.scope = deps.scope_for(q.conversation_id, q.conversation_type)
    q.message_id = str(getattr(q.message, "message_id", "") or "").strip()
    q.sender_id = str(getattr(q.message, "sender_id", "") or "").strip()
    return True


def read_observation_links(
    q: MemoryObservedMessage,
    deps: MemoryServiceRecordingDependencies,
) -> None:
    raw_event = getattr(q.message, "raw_event", {})
    raw_event = raw_event if isinstance(raw_event, Mapping) else {}
    sender = raw_event.get("sender") if isinstance(raw_event.get("sender"), Mapping) else {}
    q.sender_name = ""
    for key in ("card", "nickname", "display_name", "name"):
        value = sender.get(key) if isinstance(sender, Mapping) else None
        if value is not None and str(value).strip():
            q.sender_name = str(value).strip()
            break
    q.reply_to = ""
    q.at_targets: list[str] = []
    for segment in tuple(getattr(q.message, "segments", ()) or ()):
        payload = getattr(segment, "payload", {})
        payload = payload if isinstance(payload, Mapping) else {}
        data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
        sources = (payload, data)
        kind = str(getattr(segment, "kind", "") or "").strip().lower()
        if kind == "reply" and not q.reply_to:
            for source in sources:
                for key in ("id", "message_id", "messageId"):
                    if source.get(key) is not None and str(source.get(key)).strip():
                        q.reply_to = str(source.get(key)).strip()
                        break
                if q.reply_to:
                    break
        if kind == "at":
            for source in sources:
                for key in ("qq", "target", "user_id", "userId", "id"):
                    if source.get(key) is not None and str(source.get(key)).strip():
                        value = str(source.get(key)).strip()
                        if value not in q.at_targets:
                            q.at_targets.append(value)


def choose_observation_topic(
    q: MemoryObservedMessage,
    deps: MemoryServiceRecordingDependencies,
) -> None:
    previous = deps.store().latest_observation(scope=q.scope)
    previous_meta = previous.metadata if previous is not None else {}
    previous_text = previous.text if previous is not None else ""
    previous_message_id = str(previous_meta.get("message_id") or "").strip()
    gap = q.now - previous.created_at if previous is not None else 10**9
    overlap = deps.topic_overlap(q.text, previous_text)
    q.deictic_reference = bool(re.search(r"(?:那个|这件事|他|她|它|刚才|上一条|上面|继续|后来|然后呢)", q.text))
    q.continuation = bool(
        (q.reply_to and q.reply_to == previous_message_id)
        or (previous is not None and gap <= deps.policy().topic_window_seconds and overlap >= 1)
        or (previous is not None and gap <= 120 and q.sender_id and q.sender_id == str(previous_meta.get("sender_id") or ""))
        or (previous is not None and gap <= deps.policy().topic_window_seconds and q.deictic_reference)
    )
    q.topic_batch_id = (
        str(previous_meta.get("topic_batch_id") or "")
        if q.continuation and previous is not None
        else f"topic:{q.conversation_id}:{uuid.uuid4().hex[:12]}"
    )


def persist_observation(q: MemoryObservedMessage, deps: MemoryServiceRecordingDependencies) -> bool:
    from shinku.qq.attachment_labels import attachment_label
    body = q.text or attachment_label(q.attachments) or ("[附件]" if q.effective_has_media else "[未解析消息]")
    if q.quoted_text:
        body = f"{body}\n[引用消息]\n{q.quoted_text}".strip()
    elif q.has_quoted_media:
        body = f"{body}\n{attachment_label(q.quoted_attachments, quoted=True)}".strip()
    body = body[:2000]
    source_id = f"turn:{q.conversation_id}:{q.message_id}" if q.message_id else f"observation:{q.conversation_id}:{q.now}:{uuid.uuid4().hex[:8]}"
    attachment_refs = deps.attachment_refs(q.attachments, quoted=False) + deps.attachment_refs(
        q.quoted_attachments,
        quoted=True,
    )
    raw_event = getattr(q.message, "raw_event", {})
    raw_event = raw_event if isinstance(raw_event, Mapping) else {}
    metadata = {
        "source": "qq_passive_observation",
        "message_id": q.message_id,
        "sender_id": q.sender_id,
        "sender_name": q.sender_name,
        "self_id": str(raw_event.get("self_id") or ""),
        "reply_to_message_id": q.reply_to,
        "at_targets": q.at_targets,
        "has_image": bool(
            (q.has_media and any(str(getattr(item, "kind", "")) == "image" for item in tuple(getattr(q.message, "segments", ()) or ())))
            or any(
                str(item.get("kind") or "").strip().lower() == "image"
                for item in q.quoted_attachments
                if isinstance(item, Mapping)
            )
        ),
        "has_media": q.effective_has_media,
        "has_quoted_media": q.has_quoted_media,
        "attachment_refs": attachment_refs,
        "quoted_message_id": str(
            (getattr(q.message, "quoted_message", {}) or {}).get("message_id", "")
            if isinstance(getattr(q.message, "quoted_message", {}), Mapping)
            else ""
        ).strip(),
        "topic_batch_id": q.topic_batch_id,
        "topic_continuation": q.continuation,
        "deictic_reference": q.deictic_reference,
        "observed_at": q.now,
    }
    recorded = deps.store().record_event(
        conversation_id=q.conversation_id,
        scope=q.scope,
        text=body,
        source_id=source_id,
        layer="observation",
        created_at=q.now,
        importance=0.18,
        confidence=0.65,
        expires_at=q.now + deps.policy().event_ttl_seconds if deps.policy().event_ttl_seconds else 0,
        metadata=metadata,
    )
    from .participant_memory import promote_participant_event
    promote_participant_event(deps.store(), deps.policy(), scope=q.scope, source_id=source_id,
                              owner_scope=deps.owner_scope(), now=int(deps.clock()))
    return recorded
