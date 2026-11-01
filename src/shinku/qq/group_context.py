"""群聊说话人标签与短期话题交接。

群聊的旁观消息仍然可以被动入库，但只有当前消息已经叫到真红或明确回复
某条消息时，才把一小段本地话题快照交给 Agent。这里不调用模型，也不把
旁观者原话当成人设或长期事实。
"""

from __future__ import annotations

import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .message import IncomingMessage
from .routing import RouteDecision

__all__ = ["GroupContextTracker"]


_REFERENT_RE = re.compile(r"(?:他|她|它|那个|这个|刚才|前面|上面|继续|这件事|那件事|谁|哪句|哪一条)")


@dataclass(frozen=True)
class GroupContextItem:
    sender_id: str
    sender_label: str
    message_id: str
    text: str
    timestamp: float
    at_target_labels: tuple[str, ...] = ()
    at_target_ids: tuple[str, ...] = ()
    reply_message_ids: tuple[str, ...] = ()
    has_image: bool = False

    def render(self) -> str:
        label = self.sender_label or self.sender_id or "群成员"
        targets = "、".join(f"@{item}" for item in self.at_target_labels)
        recipient = f" → {targets}" if targets else ""
        return f"{label}（{self.sender_id or 'unknown'}）{recipient}：{self.text}"


class GroupContextTracker:
    """每个群保留一个有 TTL、有上限的本地话题快照。"""

    def __init__(self, *, ttl_seconds: float = 300.0, max_items: int = 16, max_chars: int = 2400) -> None:
        if ttl_seconds <= 0 or max_items <= 0 or max_chars <= 0:
            raise ValueError("group context limits must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.max_items = int(max_items)
        self.max_chars = int(max_chars)
        self._buckets: dict[str, list[GroupContextItem]] = {}
        self._focuses: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def enrich(
        self,
        message: IncomingMessage,
        route: RouteDecision,
        *,
        now: float,
        include_context: bool = False,
    ) -> None:
        """把当前消息的前置话题快照写回 raw event，再保存当前消息。"""

        if message.conversation_type != "group" or message.is_poke:
            return
        current = float(now)
        key = str(message.conversation_id or "").strip()
        if not key:
            return
        with self._lock:
            bucket = self._prune(key, current)
            focus = self._update_focus(message, route, current)
            context = self._select_context(
                message,
                route,
                bucket,
                include_context=include_context,
                focus=focus,
            )
            reply_parents = self._reply_parents(message, bucket)
            if isinstance(message.raw_event, dict):
                message.raw_event["_shinku_group_context"] = {
                    "items": [
                        {
                            "sender_id": item.sender_id,
                            "sender_label": item.sender_label,
                            "message_id": item.message_id,
                            "text": item.text,
                            "timestamp": item.timestamp,
                            "at_target_labels": list(item.at_target_labels),
                            "at_target_ids": list(item.at_target_ids),
                            "reply_message_ids": list(item.reply_message_ids),
                            "has_image": item.has_image,
                        }
                        for item in context
                    ],
                    "reason": self._reason(message, route, include_context=include_context, focus=focus),
                    "current": {
                        "message_id": str(message.message_id or "").strip(),
                        "sender_id": str(message.sender_id or "").strip(),
                        "sender_label": str(message.sender_label or "").strip(),
                        "text": self._message_text(message),
                        "at_target_ids": list(message.at_target_ids),
                        "at_target_labels": list(message.at_target_labels),
                        "reply_message_ids": list(message.reply_message_ids),
                        "has_image": any(segment.kind == "image" for segment in message.segments),
                        "reply_parents": [self._as_dict(item) for item in reply_parents],
                        "route_addressed": bool(route.addressed),
                        "route_reason": str(route.reason or ""),
                        "autonomous_candidate": bool(include_context and not route.addressed),
                    },
                    "focus": self._public_focus(focus),
                }
            text = self._message_text(message)
            if text:
                bucket.append(
                    GroupContextItem(
                        sender_id=str(message.sender_id or "").strip(),
                        sender_label=str(message.sender_label or "").strip(),
                        message_id=str(message.message_id or "").strip(),
                        text=text,
                        timestamp=current,
                        at_target_labels=tuple(message.at_target_labels),
                        at_target_ids=tuple(message.at_target_ids),
                        reply_message_ids=tuple(message.reply_message_ids),
                        has_image=any(segment.kind == "image" for segment in message.segments),
                    )
                )
                self._trim(bucket)
            self._buckets[key] = bucket

    def snapshot(self, conversation_id: str, *, limit: int = 8, now: float | None = None) -> list[dict[str, Any]]:
        current = float(now if now is not None else 0.0)
        key = str(conversation_id or "").strip()
        with self._lock:
            bucket = self._prune(key, current) if current else list(self._buckets.get(key, []))
            return [self._as_dict(item) for item in bucket[-max(1, int(limit or 8)) :]]

    def focus_snapshot(self, conversation_id: str, *, now: float) -> dict[str, Any]:
        """Read an existing directed exchange without mutating its turn state."""

        key = str(conversation_id or "").strip()
        with self._lock:
            focus = self._focuses.get(key)
            if not focus or float(now) - float(focus.get("addressed_at", focus.get("started_at", 0.0))) > self.ttl_seconds:
                return {}
            return self._public_focus(focus)

    def _select_context(
        self,
        message: IncomingMessage,
        route: RouteDecision,
        bucket: list[GroupContextItem],
        *,
        include_context: bool = False,
        focus: Mapping[str, Any] | None = None,
    ) -> list[GroupContextItem]:
        anchors: list[GroupContextItem] = []
        if include_context:
            focused_ids = {
                str(value or "").strip()
                for value in ((focus or {}).get("message_ids") or ())
                if str(value or "").strip()
            }
            if focused_ids:
                focused = [item for item in bucket if item.message_id in focused_ids]
                if focused:
                    anchors = focused[-2:]
        if not route.should_reply and not message.has_reply:
            if include_context:
                # An autonomous turn is still allowed to join a conversation,
                # but it must see the recent speaker/recipient sequence first.
                return self._merge_context(bucket, anchors)
            return []
        reply_ids = set(message.reply_message_ids)
        if reply_ids:
            exact = [item for item in bucket if item.message_id and item.message_id in reply_ids]
            if exact:
                return self._merge_context(bucket, exact[-2:])
        if route.addressed or _REFERENT_RE.search(message.text or ""):
            return self._merge_context(bucket, anchors)
        return self._merge_context(bucket, anchors) if include_context else []

    @staticmethod
    def _merge_context(
        bucket: list[GroupContextItem], anchors: list[GroupContextItem]
    ) -> list[GroupContextItem]:
        # A reference locates the topic; it must not hide subsequent corrections.
        selected = {id(item) for item in [*anchors, *bucket[-4:]]}
        return [item for item in bucket if id(item) in selected][-6:]

    def _reason(
        self,
        message: IncomingMessage,
        route: RouteDecision,
        *,
        include_context: bool = False,
        focus: Mapping[str, Any] | None = None,
    ) -> str:
        if focus and include_context:
            return "active_addressed_thread"
        if message.reply_message_ids:
            return "reply_chain"
        if route.addressed:
            return "addressed_group_turn"
        if _REFERENT_RE.search(message.text or ""):
            return "local_referent"
        return "recent_autonomous_group_context" if include_context else ""

    def _update_focus(
        self,
        message: IncomingMessage,
        route: RouteDecision,
        now: float,
    ) -> dict[str, Any] | None:
        """Track a short directed group exchange across separate webhook turns.

        The first explicit @ to another member establishes the exchange. Replies
        from either participant, including plain follow-up messages without a
        repeated @ or reply segment, inherit that exchange for a short TTL.
        """

        key = str(message.conversation_id or "").strip()
        focus = self._focuses.get(key)
        # 普通同一成员发言只能补上下文，不能无限延长“正在对别人说话”的硬拦截。
        # 明确 @ 或引用该对话中的消息才刷新定向证据的寿命。
        if focus and now - float(focus.get("addressed_at", focus.get("started_at", 0.0))) > self.ttl_seconds:
            self._focuses.pop(key, None)
            focus = None

        sender_id = str(message.sender_id or "").strip()
        sender_label = str(message.sender_label or sender_id or "群成员").strip()
        targets = list(message.at_targets)
        target_keys = [identifier or f"label:{label.casefold()}" for identifier, label in targets]

        if message.has_at_mention and not route.addressed and targets:
            existing_participants = set((focus or {}).get("participant_ids") or ())
            same_thread = bool(
                focus
                and sender_id in existing_participants
                and any(target_key in existing_participants for target_key in target_keys)
            )
            if not same_thread:
                focus = {
                    "reason": "explicit_other_member_mention",
                    "root_message_id": str(message.message_id or "").strip(),
                    "root_sender_id": sender_id,
                    "root_sender_label": sender_label,
                    "root_target_ids": [identifier for identifier, _ in targets],
                    "root_target_labels": [label for _, label in targets],
                    "participant_ids": list(dict.fromkeys(([sender_id] if sender_id else []) + target_keys)),
                    "participant_labels": list(dict.fromkeys(([sender_label] if sender_label else []) + [label for _, label in targets])),
                    "message_ids": [],
                    "started_at": now,
                    "addressed_at": now,
                    "last_at": now,
                }
                self._focuses[key] = focus
            else:
                focus["last_at"] = now
                focus["addressed_at"] = now
                for target_key, (_, label) in zip(target_keys, targets):
                    if target_key not in focus["participant_ids"]:
                        focus["participant_ids"].append(target_key)
                    if label and label not in focus["participant_labels"]:
                        focus["participant_labels"].append(label)

        if focus:
            focus_ids = set(str(value) for value in focus.get("message_ids", ()))
            reply_matches = bool(focus_ids.intersection(message.reply_message_ids))
            participant_matches = bool(sender_id and sender_id in set(focus.get("participant_ids", ())))
            if reply_matches or participant_matches:
                if reply_matches:
                    focus["addressed_at"] = now
                message_id = str(message.message_id or "").strip()
                if message_id and message_id not in focus["message_ids"]:
                    focus["message_ids"].append(message_id)
                    del focus["message_ids"][:-self.max_items]
                focus["last_at"] = now
                if sender_id and sender_id not in focus["participant_ids"]:
                    focus["participant_ids"].append(sender_id)
                if sender_label and sender_label not in focus["participant_labels"]:
                    focus["participant_labels"].append(sender_label)
                return focus
        return None

    @staticmethod
    def _reply_parents(
        message: IncomingMessage,
        bucket: list[GroupContextItem],
    ) -> list[GroupContextItem]:
        by_id = {item.message_id: item for item in bucket if item.message_id}
        return [by_id[identifier] for identifier in message.reply_message_ids if identifier in by_id]

    @staticmethod
    def _public_focus(focus: Mapping[str, Any] | None) -> dict[str, Any]:
        if not focus:
            return {}
        return {
            "reason": str(focus.get("reason") or ""),
            "root_message_id": str(focus.get("root_message_id") or ""),
            "root_sender_id": str(focus.get("root_sender_id") or ""),
            "root_sender_label": str(focus.get("root_sender_label") or ""),
            "root_target_ids": list(focus.get("root_target_ids") or ()),
            "root_target_labels": list(focus.get("root_target_labels") or ()),
            "participant_ids": list(focus.get("participant_ids") or ()),
            "participant_labels": list(focus.get("participant_labels") or ()),
            "message_ids": list(focus.get("message_ids") or ()),
        }

    def _prune(self, key: str, now: float) -> list[GroupContextItem]:
        cutoff = now - self.ttl_seconds
        bucket = [item for item in self._buckets.get(key, []) if item.timestamp >= cutoff]
        self._buckets[key] = bucket
        return bucket

    def _trim(self, bucket: list[GroupContextItem]) -> None:
        del bucket[:-self.max_items]
        while len("\n".join(item.render() for item in bucket)) > self.max_chars and len(bucket) > 1:
            bucket.pop(0)

    @staticmethod
    def _message_text(message: IncomingMessage) -> str:
        text = str(message.text_without_at_mentions or "").strip()
        if message.card_context:
            text = f"{text}\n{message.card_context}".strip()
        if message.has_media:
            text = f"{text}\n[附件]".strip()
        return text[:600]

    @staticmethod
    def _as_dict(item: GroupContextItem) -> dict[str, Any]:
        return {
            "sender_id": item.sender_id,
            "sender_label": item.sender_label,
            "message_id": item.message_id,
            "text": item.text,
            "timestamp": item.timestamp,
            "at_target_labels": list(item.at_target_labels),
            "at_target_ids": list(item.at_target_ids),
            "reply_message_ids": list(item.reply_message_ids),
            "has_image": item.has_image,
            "rendered": item.render(),
        }
