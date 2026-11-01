"""Bind QQ mention markers to real members and render transport segments.

This module reads a turn only. It cannot send a message or grant permission.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .delivery import OutgoingSegment
from .turns import QQTurn

_SHINKU_AT_MARKER_RE = re.compile(r"\[\[\s*SHINKU_AT_(\d{1,2})\s*\]\]", re.IGNORECASE)
_SHINKU_AT_ANY_RE = re.compile(r"\[\[\s*SHINKU_AT_[^\]]*\]\]", re.IGNORECASE)


def group_mention_targets(
    turn: QQTurn,
    *,
    bot_ids: Sequence[str] = (),
    max_targets: int = 12,
) -> tuple[tuple[str, str], ...]:
    """Return only real, numeric, non-bot members visible in this group turn."""

    if str(turn.conversation_type or "").strip().casefold() != "group":
        return ()
    blocked_ids = {str(value or "").strip() for value in bot_ids if str(value or "").strip()}
    for message in tuple(turn.messages or ()):
        raw = getattr(message, "raw_event", {})
        if isinstance(raw, Mapping):
            self_id = str(raw.get("self_id") or raw.get("selfId") or "").strip()
            if self_id:
                blocked_ids.add(self_id)
    found: dict[str, str] = {}

    def add(identifier: Any, label: Any) -> None:
        user_id = str(identifier or "").strip()
        if not user_id.isascii() or not user_id.isdecimal() or user_id in blocked_ids or len(found) >= max_targets:
            return
        display = re.sub(r"[\x00-\x1f\x7f]", " ", str(label or "")).strip().lstrip("@").strip()
        if not display:
            display = f"QQ {user_id}"
        found.setdefault(user_id, display[:48])

    messages = tuple(turn.messages or ())
    for message in reversed(messages):
        add(getattr(message, "sender_id", ""), getattr(message, "sender_label", ""))
        for identifier, label in reversed(tuple(getattr(message, "at_targets", ()) or ())):
            add(identifier, label)

    for message in reversed(messages):
        raw = getattr(message, "raw_event", {})
        if not isinstance(raw, Mapping):
            continue
        context = raw.get("_shinku_group_context")
        if not isinstance(context, Mapping):
            continue

        def add_record(record: Any) -> None:
            if not isinstance(record, Mapping):
                return
            add(record.get("sender_id"), record.get("sender_label"))
            ids = record.get("at_target_ids")
            labels = record.get("at_target_labels")
            ids = ids if isinstance(ids, (list, tuple)) else ()
            labels = labels if isinstance(labels, (list, tuple)) else ()
            for index in range(min(len(ids), len(labels)) - 1, -1, -1):
                add(ids[index], labels[index])

        add_record(context.get("current"))
        current = context.get("current")
        if isinstance(current, Mapping):
            for parent in reversed(tuple(current.get("reply_parents") or ())):
                add_record(parent)
        for item in reversed(tuple(context.get("items") or ())):
            add_record(item)
        focus = context.get("focus")
        if isinstance(focus, Mapping):
            add(focus.get("root_sender_id"), focus.get("root_sender_label"))
            target_ids = focus.get("root_target_ids")
            target_labels = focus.get("root_target_labels")
            target_ids = target_ids if isinstance(target_ids, (list, tuple)) else ()
            target_labels = target_labels if isinstance(target_labels, (list, tuple)) else ()
            for index in range(min(len(target_ids), len(target_labels)) - 1, -1, -1):
                add(target_ids[index], target_labels[index])
            participant_ids = focus.get("participant_ids")
            participant_labels = focus.get("participant_labels")
            participant_ids = participant_ids if isinstance(participant_ids, (list, tuple)) else ()
            participant_labels = participant_labels if isinstance(participant_labels, (list, tuple)) else ()
            for index in range(min(len(participant_ids), len(participant_labels)) - 1, -1, -1):
                add(participant_ids[index], participant_labels[index])
    return tuple(found.items())


def sanitize_mention_markers(
    text: str,
    targets: Sequence[tuple[str, str]],
    *,
    allow_group: bool,
    max_mentions: int = 2,
) -> str:
    target_map = {index: target for index, target in enumerate(targets, start=1)}
    selected: set[int] = set()

    def replace_marker(match: re.Match[str]) -> str:
        if not allow_group:
            return ""
        try:
            index = int(match.group(1))
        except (TypeError, ValueError):
            return ""
        if index not in target_map or index in selected or len(selected) >= max_mentions:
            return ""
        selected.add(index)
        return f"[[SHINKU_AT_{index}]]"

    def replace_any_marker(match: re.Match[str]) -> str:
        parsed = _SHINKU_AT_MARKER_RE.fullmatch(match.group(0))
        return replace_marker(parsed) if parsed else ""

    return _SHINKU_AT_ANY_RE.sub(replace_any_marker, str(text or "")).strip()


def render_mention_markers(
    text: str,
    targets: Sequence[tuple[str, str]],
    *,
    allow_group: bool,
) -> str:
    if not allow_group:
        return _SHINKU_AT_ANY_RE.sub("", str(text or "")).strip()
    target_map = {index: target for index, target in enumerate(targets, start=1)}

    def render(match: re.Match[str]) -> str:
        target = target_map.get(int(match.group(1)))
        return f"@{target[1]}" if target else ""

    return _SHINKU_AT_MARKER_RE.sub(render, str(text or "")).strip()


def outgoing_text_segments(
    text: str,
    targets: Sequence[tuple[str, str]],
    *,
    allow_group: bool,
) -> tuple[OutgoingSegment, ...]:
    if not allow_group:
        clean = _SHINKU_AT_ANY_RE.sub("", str(text or ""))
        return (OutgoingSegment(kind="text", text=clean.strip()),) if clean.strip() else ()
    target_map = {index: target for index, target in enumerate(targets, start=1)}
    result: list[OutgoingSegment] = []
    cursor = 0
    for match in _SHINKU_AT_MARKER_RE.finditer(str(text or "")):
        before = str(text or "")[cursor : match.start()]
        if before.strip():
            result.append(OutgoingSegment(kind="text", text=before))
        target = target_map.get(int(match.group(1)))
        if target:
            result.append(OutgoingSegment(kind="at", target_id=target[0]))
        cursor = match.end()
    tail = str(text or "")[cursor:]
    if tail.strip():
        result.append(OutgoingSegment(kind="text", text=tail))
    if not result and str(text or "").strip():
        result.append(OutgoingSegment(kind="text", text=str(text).strip()))
    return tuple(result)
