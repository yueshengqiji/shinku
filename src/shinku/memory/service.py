"""Memory lifecycle orchestration for one Shinku conversation."""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Mapping
from collections.abc import Callable
from typing import Any

from .models import ForgetReport, MemoryContext
from .policy import MemoryPolicy
from .router import MemoryRouter
from .store import MemoryStore

__all__ = ["MemoryService"]


class MemoryService:
    """Record history, ask the router, retrieve on demand, and forget explicitly."""

    # A logical library inside the same SQLite source of truth.  Keeping this
    # scope separate lets the owner recall a private/group conversation from
    # either side without making ordinary group spectators globally visible.
    OWNER_SCOPE = "private:master"

    _FACT_RE = re.compile(r"^(?:请)?记住(?:一下|了)?[：:，,、\s]*(?P<fact>.+)$")
    _FORGET_RE = re.compile(r"^(?:请)?(?:忘掉|忘记|别记|不要记|清除|删除)(?:关于|掉|掉关于)?\s*(?P<query>.+)$")
    _RECALL_FILLER_RE = re.compile(
        r"(?:今天|今日|昨晚|昨夜|昨天|前天|大前天|上周|上个月|前几天|最近几天|那天|哪天|几号|"
        r"你还记得|我还记得|记得吗|我问你|我们聊了什么|发生了什么|说过什么|提过什么|"
        r"谈了什么|哪件事|什么吗|吗[？?]?)"
    )
    _DATE_RE = re.compile(
        r"(?:(?P<year>20\d{2})\s*[年/-]\s*)?(?P<month>\d{1,2})\s*[月/-]\s*(?P<day>\d{1,2})\s*(?:日|号)?",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        store: MemoryStore,
        router: MemoryRouter,
        clock: Callable[[], float] | None = None,
        policy: MemoryPolicy | None = None,
    ) -> None:
        self.store = store
        self.router = router
        self.clock = clock or time.time
        self.policy = policy or MemoryPolicy()
        self.store.configure(
            max_query_terms=self.policy.max_query_terms,
            event_scan_limit=self.policy.event_scan_limit,
            fact_scan_limit=self.policy.fact_scan_limit,
            max_results=self.policy.max_results,
        )
        # A private scope is already owner-only, so it is safe to migrate its
        # existing records once into the cross-context owner library.  Group
        # scopes are deliberately not bulk-promoted because old group rows do
        # not reliably prove that Shinku participated.
        for master_id in self.policy.master_user_ids:
            try:
                self.store.promote_scope_to_owner(source_scope=f"private:{master_id}", owner_scope=self.OWNER_SCOPE)
            except Exception:
                # Historical migration is best-effort and must not block QQ.
                continue

    @property
    def owner_scope(self) -> str:
        return self.OWNER_SCOPE

    def is_master_user(self, user_id: str) -> bool:
        value = str(user_id or "").strip()
        return bool(value and value in self.policy.master_user_ids)

    def owner_status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.policy.master_user_ids),
            "scope": self.OWNER_SCOPE,
            "configured_master_count": len(self.policy.master_user_ids),
            "cross_context_modes": ["retrieve", "summary"],
        }

    def recent_owner_handoff(
        self,
        *,
        requester_id: str,
        limit: int = 4,
        max_age_seconds: int = 4 * 60 * 60,
    ) -> tuple[Any, ...]:
        """Return a tiny, fresh topic cue for a master's private greeting.

        This is intentionally not the normal recall path: it reads only the
        isolated owner scope, requires an authenticated configured owner, and
        returns records from the most recently active source conversation.
        Group callers can still explicitly request cross-context recall via
        ``prepare_turn``; they never receive this unsolicited private cue.
        """

        if not self.is_master_user(requester_id):
            return ()
        now = int(self.clock())
        scan_limit = max(1, min(self.policy.max_results, 16))
        candidates = self.store.recent_events(
            scope=self.OWNER_SCOPE,
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
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        sender_id: str = "",
        metadata: Mapping[str, Any] | None = None,
        role: str = "user",
    ) -> bool:
        """Whether one handled turn belongs in the owner's shared memory.

        Private turns are owner-relevant only when the sender is configured as
        the owner.  A group turn additionally needs an explicit Shinku
        involvement marker; this prevents a user's unrelated group chatter
        from becoming owner memory merely because the owner was present.
        """

        if not self.policy.master_user_ids:
            return False
        meta = dict(metadata or {})
        kind = str(conversation_type or "private").strip().lower()
        sender = str(sender_id or meta.get("sender_id") or "").strip()
        if not sender and kind in {"private", "friend", "direct"}:
            sender = str(conversation_id or "").strip()
        if kind in {"private", "friend", "direct"}:
            return self.is_master_user(sender)
        if "group" not in kind:
            return False
        if not bool(meta.get("shinku_involved") or meta.get("owner_memory_eligible")):
            return False
        at_targets = {str(item).strip() for item in tuple(meta.get("at_targets") or ()) if str(item).strip()}
        quoted_sender = str(meta.get("quoted_sender_id") or "").strip()
        return self.is_master_user(sender) or bool(at_targets.intersection(self.policy.master_user_ids)) or self.is_master_user(quoted_sender)

    def _is_master_request(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        message_meta: Mapping[str, Any] | None,
    ) -> bool:
        meta = dict(message_meta or {})
        sender_id = str(meta.get("sender_id") or "").strip()
        kind = str(conversation_type or "private").strip().lower()
        if not sender_id and kind in {"private", "friend", "direct"}:
            sender_id = str(conversation_id or "").strip()
        return self.is_master_user(sender_id)

    def _history_scopes(self, scope: str, *, owner_request: bool, mode: str) -> tuple[str, ...]:
        """Return current scope plus owner scope only for an owner recall."""

        current = str(scope or "").strip()
        if not owner_request or mode not in {"retrieve", "summary"} or current == self.OWNER_SCOPE:
            return (current,)
        return (current, self.OWNER_SCOPE)

    @staticmethod
    def _merge_by_source(primary: list[Any] | tuple[Any, ...], secondary: list[Any] | tuple[Any, ...], *, limit: int) -> tuple[Any, ...]:
        """Merge cross-scope records without showing the owner copy twice."""

        merged: dict[str, Any] = {}
        for record in (*tuple(primary), *tuple(secondary)):
            source = str(getattr(record, "source_id", "") or "").strip()
            key = f"source:{source}" if source else f"record:{getattr(record, 'record_id', '')}"
            previous = merged.get(key)
            if previous is None or float(getattr(record, "score", 0.0) or 0.0) > float(getattr(previous, "score", 0.0) or 0.0):
                merged[key] = record
        values = list(merged.values())
        values.sort(key=lambda item: (float(getattr(item, "score", 0.0) or 0.0), int(getattr(item, "created_at", 0) or 0)), reverse=True)
        return tuple(values[: max(1, int(limit or 1))])

    def prepare_turn(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        user_message: str,
        recent_context: str = "",
        message_meta: Mapping[str, Any] | None = None,
    ) -> MemoryContext:
        now = int(self.clock())
        self.store.expire(now=now)
        scope = self.scope_for(conversation_id, conversation_type)
        effective_meta = dict(message_meta or {})
        message_id = str(effective_meta.get("message_id") or "").strip()
        if message_id:
            observed = self.store.observation_for_message(scope=scope, message_id=message_id)
            if observed is not None:
                # The passive observer is the source of truth for reply links,
                # media markers and the local topic batch.
                effective_meta = {**observed.metadata, **effective_meta}
        latest = self.store.latest_observation(scope=scope)
        topic_state: dict[str, Any] = {}
        if latest is not None:
            topic_state = {
                "message_id": latest.metadata.get("message_id", ""),
                "created_at": latest.created_at,
                "topic_batch_id": latest.metadata.get("topic_batch_id", ""),
                **latest.metadata,
            }
        decision = self.router.decide(
            user_message=user_message,
            recent_context=recent_context,
            conversation_type=conversation_type,
            message_meta=effective_meta,
            topic_state=topic_state,
        )
        owner_request = self._is_master_request(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            message_meta=effective_meta,
        )
        history_scopes = self._history_scopes(
            scope,
            owner_request=owner_request,
            mode=decision.history_mode,
        )
        if decision.history_mode == "none":
            context = MemoryContext(
                decision=decision,
                audit={
                    "history_mode": "none",
                    "retrieved_count": 0,
                    "injected_chars": 0,
                    "retrieval_scopes": [scope],
                },
            )
            self._record_audit(
                conversation_id=conversation_id,
                scope=scope,
                now=now,
                context=context,
            )
            return context

        topic_batch_id = str(effective_meta.get("topic_batch_id") or "").strip()
        sender_id = str(effective_meta.get("sender_id") or "").strip()
        fallback = ""
        if decision.history_mode == "recent":
            records = tuple(
                self.store.recent_events(
                    scope=scope,
                    limit=self.policy.recent_limit,
                    now=now,
                    max_age_seconds=self.policy.recent_window_seconds,
                    topic_batch_id=topic_batch_id,
                    # Recent context is a continuity snapshot.  Do not feed
                    # generated Shinku replies back as voice examples; that
                    # creates a self-reinforcing loop for cold/defensive turns.
                    # Explicit retrieve/summary requests still include them.
                    include_assistant=False,
                )
            )
        elif decision.history_mode == "summary":
            for memory_scope in history_scopes:
                self.store.rollup_daily_summaries(scope=memory_scope, now=now)
            date_window = self._date_window(decision.query or user_message, now=now)
            if date_window is not None:
                start_at, end_at = date_window
                recall_query = self._recall_query(decision.query or user_message)
                focused_all: list[Any] = []
                neighbors_all: list[Any] = []
                for memory_scope in history_scopes:
                    focused = self.store.search(
                            query=recall_query,
                            scope=memory_scope,
                            limit=self.policy.retrieval_limit,
                            now=now,
                            # Date-scoped recall remains lexical and bounded;
                            # owner retrieval must not cross the requested day.
                            layers=(
                                "observation",
                                "episodic",
                                "legacy_raw",
                                "legacy_semantic",
                                "legacy_lesson",
                                "legacy_state",
                            ),
                            start_at=start_at,
                            end_at=end_at,
                            require_term_overlap=True,
                        )
                    focused_all.extend(focused)
                    for match in focused[: self.policy.retrieval_limit]:
                        neighbors_all.extend(
                            self.store.nearby_events(
                                scope=memory_scope,
                                center_at=match.created_at,
                                limit=4,
                                window_seconds=self.policy.topic_window_seconds,
                                start_at=start_at,
                                end_at=end_at,
                            )
                        )
                if focused_all:
                    records = self._merge_by_source(focused_all, neighbors_all, limit=self.policy.retrieval_limit)
                    fallback = "date_query_match"
                else:
                    summary_rows: list[Any] = []
                    for memory_scope in history_scopes:
                        summary_rows.extend(
                            self.store.timeline(
                                scope=memory_scope,
                                limit=self.policy.retrieval_limit,
                                now=now,
                                layers=("summary", "legacy_summary", "daily_summary"),
                                include_facts=False,
                                start_at=start_at,
                                end_at=end_at,
                            )
                        )
                    records = self._merge_by_source(summary_rows, (), limit=self.policy.retrieval_limit)
                    if not records:
                        evidence_rows: list[Any] = []
                        for memory_scope in history_scopes:
                            evidence_rows.extend(
                                self.store.timeline(
                                    scope=memory_scope,
                                    limit=self.policy.retrieval_limit,
                                    now=now,
                                    layers=(
                                        "observation",
                                        "episodic",
                                        "legacy_raw",
                                        "legacy_semantic",
                                        "legacy_lesson",
                                        "legacy_state",
                                    ),
                                    include_facts=False,
                                    start_at=start_at,
                                    end_at=end_at,
                                )
                            )
                        records = self._merge_by_source(evidence_rows, (), limit=self.policy.retrieval_limit)
                    fallback = "date_timeline" if records else "date_timeline_empty"
            else:
                summary_rows: list[Any] = []
                for memory_scope in history_scopes:
                    summary_rows.extend(
                        self.store.search(
                            query=f"{decision.query or user_message} {decision.time_range}".strip(),
                            scope=memory_scope,
                            limit=self.policy.retrieval_limit,
                            now=now,
                            layers={"summary", "legacy_summary", "daily_summary"},
                            topic_batch_id=topic_batch_id,
                            sender_id=sender_id,
                        )
                    )
                records = self._merge_by_source(summary_rows, (), limit=self.policy.retrieval_limit)
            # A summary may not exist yet for the current day or for a legacy
            # database that was never rolled up.  Use an audited evidence
            # fallback rather than pretending a summary exists.
            if not records and (
                date_window is None
                or not self._precise_day_request(decision.query or user_message)
            ):
                evidence_rows = []
                for memory_scope in history_scopes:
                    evidence_rows.extend(
                        self.store.search(
                            query=decision.query or user_message,
                            scope=memory_scope,
                            limit=self.policy.retrieval_limit,
                            now=now,
                            topic_batch_id=topic_batch_id,
                            sender_id=sender_id,
                        )
                    )
                records = self._merge_by_source(evidence_rows, (), limit=self.policy.retrieval_limit)
                fallback = "summary_to_evidence"
        else:
            evidence_rows: list[Any] = []
            for memory_scope in history_scopes:
                evidence_rows.extend(
                    self.store.search(
                        query=decision.query or user_message,
                        scope=memory_scope,
                        limit=self.policy.retrieval_limit,
                        now=now,
                        topic_batch_id=topic_batch_id,
                        sender_id=sender_id,
                    )
                )
            records = self._merge_by_source(evidence_rows, (), limit=self.policy.retrieval_limit)
        rendered = self._render(records, mode=decision.history_mode)
        context = MemoryContext(
            decision=decision,
            records=records,
            rendered=rendered,
            retrieved_count=len(records),
            injected_chars=len(rendered),
            audit={
                "history_mode": decision.history_mode,
                "retrieved_count": len(records),
                "injected_chars": len(rendered),
                "scope": scope,
                "retrieval_scopes": list(history_scopes),
                "owner_memory": owner_request,
                "fallback": fallback,
                "topic_batch_id": topic_batch_id,
            },
        )
        self._record_audit(
            conversation_id=conversation_id,
            scope=scope,
            now=now,
            context=context,
        )
        return context

    @classmethod
    def _recall_query(cls, text: str) -> str:
        """Keep the topic while removing temporal/request scaffolding."""

        value = cls._RECALL_FILLER_RE.sub(" ", str(text or ""))
        value = re.sub(r"\s+", " ", value).strip(" ，,。？！?!：:")
        return value or str(text or "").strip()

    @classmethod
    def _precise_day_request(cls, text: str) -> bool:
        """Whether an empty result must stay inside one exact calendar day."""

        return bool(cls._DATE_RE.search(str(text or "")) or re.search(r"今天|昨天|大前天|前天", str(text or "")))

    @staticmethod
    def _merge_records(primary: tuple[Any, ...], secondary: list[Any] | tuple[Any, ...], *, limit: int) -> tuple[Any, ...]:
        merged: dict[str, Any] = {}
        for record in (*primary, *tuple(secondary)):
            identifier = str(getattr(record, "record_id", "") or "")
            if identifier and identifier not in merged:
                merged[identifier] = record
        values = list(merged.values())
        values.sort(key=lambda item: (int(getattr(item, "created_at", 0) or 0), float(getattr(item, "score", 0.0) or 0.0)), reverse=True)
        return tuple(values[: max(1, int(limit or 1))])

    def observe_message(self, message: Any) -> bool:
        """Passively persist a normalized QQ message before reply routing.

        This method never calls an LLM and never decides whether Shinku should
        answer.  It only records bounded evidence for later local routing.
        """

        conversation_id = str(getattr(message, "conversation_id", "") or "").strip() or "unknown"
        conversation_type = str(getattr(message, "conversation_type", "private") or "private").strip().lower()
        if conversation_type == "group" and not self.policy.passive_observe_group:
            return False
        if conversation_type != "group" and not self.policy.passive_observe_private:
            return False
        text = str(getattr(message, "text", "") or "").strip()
        has_media = bool(getattr(message, "has_media", False))
        attachments = tuple(getattr(message, "attachments", ()) or ())
        quoted_text = str(getattr(message, "quoted_text", "") or "").strip()
        quoted_attachments = tuple(getattr(message, "quoted_attachments", ()) or ())
        has_quoted_media = bool(quoted_attachments)
        effective_has_media = has_media or has_quoted_media
        if not text and not effective_has_media and not quoted_text:
            return False
        now_value = getattr(message, "timestamp", None)
        try:
            now = int(float(now_value)) if now_value is not None else int(self.clock())
        except (TypeError, ValueError):
            now = int(self.clock())
        scope = self.scope_for(conversation_id, conversation_type)
        message_id = str(getattr(message, "message_id", "") or "").strip()
        sender_id = str(getattr(message, "sender_id", "") or "").strip()
        raw_event = getattr(message, "raw_event", {})
        raw_event = raw_event if isinstance(raw_event, Mapping) else {}
        sender = raw_event.get("sender") if isinstance(raw_event.get("sender"), Mapping) else {}
        sender_name = ""
        for key in ("card", "nickname", "display_name", "name"):
            value = sender.get(key) if isinstance(sender, Mapping) else None
            if value is not None and str(value).strip():
                sender_name = str(value).strip()
                break
        reply_to = ""
        at_targets: list[str] = []
        for segment in tuple(getattr(message, "segments", ()) or ()):
            payload = getattr(segment, "payload", {})
            payload = payload if isinstance(payload, Mapping) else {}
            data = payload.get("data") if isinstance(payload.get("data"), Mapping) else {}
            sources = (payload, data)
            kind = str(getattr(segment, "kind", "") or "").strip().lower()
            if kind == "reply" and not reply_to:
                for source in sources:
                    for key in ("id", "message_id", "messageId"):
                        if source.get(key) is not None and str(source.get(key)).strip():
                            reply_to = str(source.get(key)).strip()
                            break
                    if reply_to:
                        break
            if kind == "at":
                for source in sources:
                    for key in ("qq", "target", "user_id", "userId", "id"):
                        if source.get(key) is not None and str(source.get(key)).strip():
                            value = str(source.get(key)).strip()
                            if value not in at_targets:
                                at_targets.append(value)

        previous = self.store.latest_observation(scope=scope)
        previous_meta = previous.metadata if previous is not None else {}
        previous_text = previous.text if previous is not None else ""
        previous_message_id = str(previous_meta.get("message_id") or "").strip()
        gap = now - previous.created_at if previous is not None else 10**9
        overlap = self._topic_overlap(text, previous_text)
        deictic_reference = bool(re.search(r"(?:那个|这件事|他|她|它|刚才|上一条|上面|继续|后来|然后呢)", text))
        continuation = bool(
            (reply_to and reply_to == previous_message_id)
            or (previous is not None and gap <= self.policy.topic_window_seconds and overlap >= 1)
            or (previous is not None and gap <= 120 and sender_id and sender_id == str(previous_meta.get("sender_id") or ""))
            or (previous is not None and gap <= self.policy.topic_window_seconds and deictic_reference)
        )
        topic_batch_id = (
            str(previous_meta.get("topic_batch_id") or "")
            if continuation and previous is not None
            else f"topic:{conversation_id}:{uuid.uuid4().hex[:12]}"
        )
        body = text or ("[图片附件]" if effective_has_media else "[未解析消息]")
        if quoted_text:
            body = f"{body}\n[引用消息]\n{quoted_text}".strip()
        elif has_quoted_media:
            body = f"{body}\n[引用图片附件]".strip()
        body = body[:2000]
        source_id = f"turn:{conversation_id}:{message_id}" if message_id else f"observation:{conversation_id}:{now}:{uuid.uuid4().hex[:8]}"
        attachment_refs = self._attachment_refs(attachments, quoted=False) + self._attachment_refs(
            quoted_attachments,
            quoted=True,
        )
        metadata = {
            "source": "qq_passive_observation",
            "message_id": message_id,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "reply_to_message_id": reply_to,
            "at_targets": at_targets,
            "has_image": bool(
                (has_media and any(str(getattr(item, "kind", "")) == "image" for item in tuple(getattr(message, "segments", ()) or ())))
                or any(
                    str(item.get("kind") or "").strip().lower() == "image"
                    for item in quoted_attachments
                    if isinstance(item, Mapping)
                )
            ),
            "has_media": effective_has_media,
            "has_quoted_media": has_quoted_media,
            "attachment_refs": attachment_refs,
            "quoted_message_id": str(
                (getattr(message, "quoted_message", {}) or {}).get("message_id", "")
                if isinstance(getattr(message, "quoted_message", {}), Mapping)
                else ""
            ).strip(),
            "topic_batch_id": topic_batch_id,
            "topic_continuation": continuation,
            "deictic_reference": deictic_reference,
            "observed_at": now,
        }
        return self.store.record_event(
            conversation_id=conversation_id,
            scope=scope,
            text=body,
            source_id=source_id,
            layer="observation",
            created_at=now,
            importance=0.18,
            confidence=0.65,
            expires_at=now + self.policy.event_ttl_seconds if self.policy.event_ttl_seconds else 0,
            metadata=metadata,
        )

    def record_turn(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        message_id: str,
        text: str,
        role: str = "user",
        source_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> ForgetReport | None:
        body = str(text or "").strip()
        if not body:
            return None
        now = int(self.clock())
        scope = self.scope_for(conversation_id, conversation_type)
        normalized_role = str(role or "user").strip().lower() or "user"
        effective_source_id = str(source_id or "").strip() or f"turn:{conversation_id}:{message_id or now}"
        turn_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        turn_metadata.update({"source": "qq_turn", "role": normalized_role})
        owner_allowed = self.owner_memory_allowed(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            sender_id=str(turn_metadata.get("sender_id") or ""),
            metadata=turn_metadata,
            role=normalized_role,
        )
        owner_source_id = f"owner:{effective_source_id}"
        owner_metadata = {
            **turn_metadata,
            "owner_memory": True,
            "owner_source_scope": scope,
        }
        self.store.record_turn(
            conversation_id=conversation_id,
            scope=scope,
            role=normalized_role,
            content=body,
            source_id=effective_source_id,
            created_at=now,
            metadata=turn_metadata,
        )
        if owner_allowed and scope != self.OWNER_SCOPE:
            self.store.record_turn(
                conversation_id="master",
                scope=self.OWNER_SCOPE,
                role=normalized_role,
                content=body,
                source_id=owner_source_id,
                created_at=now,
                metadata=owner_metadata,
            )
        # Only an actual user turn may issue memory commands.  A generated
        # assistant reply must be historical evidence, never an instruction to
        # forget or promote a fact merely because it contains similar words.
        if normalized_role != "user":
            self.store.record_event(
                conversation_id=conversation_id,
                scope=scope,
                text=body,
                source_id=effective_source_id,
                layer="episodic",
                created_at=now,
                importance=0.18,
                confidence=0.42,
                expires_at=now + self.policy.event_ttl_seconds if self.policy.event_ttl_seconds else 0,
                metadata=turn_metadata,
            )
            if owner_allowed and scope != self.OWNER_SCOPE:
                self.store.record_event(
                    conversation_id="master",
                    scope=self.OWNER_SCOPE,
                    text=body,
                    source_id=owner_source_id,
                    layer="episodic",
                    created_at=now,
                    importance=0.3,
                    confidence=0.55,
                    expires_at=now + self.policy.event_ttl_seconds if self.policy.event_ttl_seconds else 0,
                    metadata=owner_metadata,
                )
            return None
        forget_match = self._FORGET_RE.match(body)
        if forget_match:
            query = str(forget_match.group("query") or "").strip(" ：:，,。.!！")
            forgotten, purged = self.store.forget(query=query, scope=scope, now=now)
            if owner_allowed and scope != self.OWNER_SCOPE:
                owner_forgotten, owner_purged = self.store.forget(query=query, scope=self.OWNER_SCOPE, now=now)
                forgotten += owner_forgotten
                purged += owner_purged
            return ForgetReport(query=query, scope=scope, forgotten_count=forgotten, purged_count=purged, reason="explicit_user_request")
        fact_match = self._FACT_RE.match(body)
        if fact_match:
            fact = str(fact_match.group("fact") or "").strip()
            if fact:
                key = f"explicit:{self._key(fact)}"
                self.store.upsert_fact(
                    conversation_id=conversation_id,
                    scope=scope,
                    fact_key=key,
                    text=fact,
                    source_id=effective_source_id,
                    now=now,
                )
                if owner_allowed and scope != self.OWNER_SCOPE:
                    self.store.upsert_fact(
                        conversation_id="master",
                        scope=self.OWNER_SCOPE,
                        fact_key=key,
                        text=fact,
                        source_id=owner_source_id,
                        now=now,
                    )
        else:
            self.store.record_event(
                conversation_id=conversation_id,
                scope=scope,
                text=body,
                source_id=effective_source_id,
                layer="episodic",
                created_at=now,
                importance=0.22,
                confidence=0.45,
                expires_at=now + self.policy.event_ttl_seconds if self.policy.event_ttl_seconds else 0,
                metadata=turn_metadata,
            )
            if owner_allowed and scope != self.OWNER_SCOPE:
                self.store.record_event(
                    conversation_id="master",
                    scope=self.OWNER_SCOPE,
                    text=body,
                    source_id=owner_source_id,
                    layer="episodic",
                    created_at=now,
                    importance=0.3,
                    confidence=0.55,
                    expires_at=now + self.policy.event_ttl_seconds if self.policy.event_ttl_seconds else 0,
                    metadata=owner_metadata,
                )
        return None

    @staticmethod
    def _attachment_refs(values: Any, *, quoted: bool) -> list[dict[str, str]]:
        """Keep bounded image/file references without persisting inline bytes."""

        result: list[dict[str, str]] = []
        for value in tuple(values or ())[:8]:
            if not isinstance(value, Mapping):
                continue
            item: dict[str, str] = {
                "kind": str(value.get("kind") or "").strip().lower()[:32],
                "quoted": "true" if quoted else "false",
            }
            for key in ("media_id", "url", "local_path"):
                raw = str(value.get(key) or "").strip()
                if not raw:
                    continue
                # A data URL is useful as a readiness marker, but storing its
                # base64 payload in the long-term DB is wasteful and unsafe.
                item[key] = "[inline-image]" if raw.lower().startswith("data:") else raw[:1024]
            if any(key in item for key in ("media_id", "url", "local_path")):
                result.append(item)
        return result

    def maintenance(self) -> dict[str, Any]:
        expired = self.store.expire(now=int(self.clock()))
        return {"expired": expired, "stats": self.store.stats()}

    def inspect(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        requester_id: str = "",
        query: str = "",
        mode: str = "recent",
        limit: int = 8,
    ) -> dict[str, Any]:
        """Read memory for an explicit user/tool request.

        Automatic routing and explicit inspection are intentionally separate:
        this method never changes persona context and never writes facts.  It
        only reads the current conversation scope and returns a bounded,
        provenance-preserving result for the Agent to summarize.
        """

        now = int(self.clock())
        self.store.expire(now=now)
        scope = self.scope_for(conversation_id, conversation_type)
        selected_mode = str(mode or "recent").strip().lower()
        if selected_mode not in {"recent", "retrieve", "summary"}:
            selected_mode = "recent"
        clean_query = str(query or "").strip()[: self.policy.max_query_chars]
        wanted = max(1, min(self.policy.max_results, int(limit or 8)))
        owner_request = self._is_master_request(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            message_meta={"sender_id": requester_id},
        )
        history_scopes = self._history_scopes(
            scope,
            owner_request=owner_request,
            mode=selected_mode,
        )
        records: tuple[Any, ...]
        fallback = ""

        if selected_mode == "recent" or not clean_query and selected_mode == "retrieve":
            records = tuple(
                self.store.timeline(
                    scope=scope,
                    limit=wanted,
                    now=now,
                    layers=("observation", "episodic", "summary"),
                    include_facts=True,
                )
            )
            if selected_mode == "retrieve" and not clean_query:
                fallback = "retrieve_without_query_to_recent"
                selected_mode = "recent"
        elif selected_mode == "summary":
            for memory_scope in history_scopes:
                self.store.rollup_daily_summaries(scope=memory_scope, now=now)
            date_window = self._date_window(clean_query, now=now)
            if date_window is not None:
                start_at, end_at = date_window
                summary_rows: list[Any] = []
                for memory_scope in history_scopes:
                    summary_rows.extend(
                        self.store.timeline(
                            scope=memory_scope,
                            limit=wanted,
                            now=now,
                            layers=("summary", "legacy_summary", "daily_summary"),
                            include_facts=False,
                            start_at=start_at,
                            end_at=end_at,
                        )
                    )
                records = self._merge_by_source(summary_rows, (), limit=wanted)
                if not records:
                    evidence_rows: list[Any] = []
                    for memory_scope in history_scopes:
                        evidence_rows.extend(
                            self.store.timeline(
                                scope=memory_scope,
                                limit=wanted,
                                now=now,
                                layers=("observation", "episodic"),
                                include_facts=False,
                                start_at=start_at,
                                end_at=end_at,
                            )
                        )
                    records = self._merge_by_source(evidence_rows, (), limit=wanted)
                fallback = "date_timeline" if records else "date_timeline_empty"
            else:
                summary_rows = []
                for memory_scope in history_scopes:
                    summary_rows.extend(
                        self.store.search(
                            query=clean_query,
                            scope=memory_scope,
                            limit=wanted,
                            now=now,
                            layers={"summary", "legacy_summary", "daily_summary"},
                        )
                    )
                records = self._merge_by_source(summary_rows, (), limit=wanted)
            if not records:
                timeline_rows: list[Any] = []
                for memory_scope in history_scopes:
                    timeline_rows.extend(
                        self.store.timeline(
                            scope=memory_scope,
                            limit=wanted,
                            now=now,
                            layers=("summary", "observation", "episodic"),
                            include_facts=False,
                        )
                    )
                records = self._merge_by_source(timeline_rows, (), limit=wanted)
                fallback = "summary_to_timeline"
        else:
            evidence_rows: list[Any] = []
            for memory_scope in history_scopes:
                evidence_rows.extend(
                    self.store.search(
                        query=clean_query,
                        scope=memory_scope,
                        limit=wanted,
                        now=now,
                    )
                )
            records = self._merge_by_source(evidence_rows, (), limit=wanted)
            if not records:
                fallback = "retrieve_empty"

        rendered = self._render(records, mode=selected_mode)
        if not rendered:
            rendered = ""
        inspection = {
            "scope": scope,
            "retrieval_scopes": list(history_scopes),
            "owner_memory": owner_request,
            "mode": selected_mode,
            "query": clean_query,
            "count": len(records),
            "fallback": fallback,
            "records": [record.as_dict() for record in records],
            "rendered": rendered,
        }
        return {
            **inspection,
            "followup_context": (
                f"已查看当前会话记忆（作用域：{scope}，模式：{selected_mode}，"
                f"结果：{len(records)} 条）。\n{rendered or '没有足够的可核实内容。'}"
            ),
        }

    def _record_audit(self, *, conversation_id: str, scope: str, now: int, context: MemoryContext) -> None:
        try:
            self.store.record_routing_audit(
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

    @staticmethod
    def scope_for(conversation_id: str, conversation_type: str) -> str:
        kind = str(conversation_type or "private").lower()
        prefix = "group" if "group" in kind else "private"
        return f"{prefix}:{str(conversation_id or 'unknown').strip() or 'unknown'}"

    def reset_all(self) -> dict[str, int]:
        """Clear the memory database bound to this service instance."""

        return self.store.reset_all()

    @classmethod
    def _date_window(cls, query: str, *, now: int) -> tuple[int, int] | None:
        """Turn an explicit calendar/relative date into a local-day range."""

        text = str(query or "").strip()
        local = time.localtime(int(now))
        match = cls._DATE_RE.search(text)
        if match is not None:
            year = int(match.group("year") or local.tm_year)
            month = int(match.group("month"))
            day = int(match.group("day"))
            try:
                start = int(time.mktime((year, month, day, 0, 0, 0, -1, -1, -1)))
            except (OverflowError, ValueError):
                return None
            return start, start + 86400

        day_start = int(time.mktime((local.tm_year, local.tm_mon, local.tm_mday, 0, 0, 0, -1, -1, -1)))
        if "今天" in text or "今日" in text:
            return day_start, day_start + 86400
        if "昨晚" in text or "昨夜" in text:
            return day_start - 86400, day_start
        if "大前天" in text:
            return day_start - 3 * 86400, day_start - 2 * 86400
        if "前天" in text:
            return day_start - 2 * 86400, day_start - 86400
        if "昨天" in text:
            return day_start - 86400, day_start
        if "上周" in text:
            weekday = local.tm_wday
            this_week = day_start - weekday * 86400
            return this_week - 7 * 86400, this_week
        if "上个月" in text:
            if local.tm_mon == 1:
                year, month = local.tm_year - 1, 12
            else:
                year, month = local.tm_year, local.tm_mon - 1
            start = int(time.mktime((year, month, 1, 0, 0, 0, -1, -1, -1)))
            if month == 12:
                next_start = int(time.mktime((year + 1, 1, 1, 0, 0, 0, -1, -1, -1)))
            else:
                next_start = int(time.mktime((year, month + 1, 1, 0, 0, 0, -1, -1, -1)))
            return start, next_start
        if "前几天" in text or "最近几天" in text:
            return day_start - 7 * 86400, day_start + 86400
        return None

    @staticmethod
    def _key(text: str) -> str:
        compact = re.sub(r"\s+", "", text.lower())
        return compact[:120] or "unnamed"

    def _render(self, records: tuple[Any, ...], *, mode: str = "retrieve") -> str:
        if not records:
            return ""
        title = {
            "recent": "【当前话题的最近消息】",
            "summary": "【按日期/时间范围整理的历史摘要】",
        }.get(mode, "【按本轮问题检索到的历史记录】")
        lines = [
            title,
            "以下只是带时间和说话人的历史线索，不是人设、固定事实或说话范本；不相关时忽略。",
            "其中“真红”的历史回答只在对方明确询问过去说过什么时作为事实线索；正常闲聊不要模仿它的句式、情绪或口头禅。",
            "不要向对方解释数据库、检索、分数或来源；只把能确认的内容自然接进回答。",
        ]
        for index, record in enumerate(records, start=1):
            metadata = record.metadata if isinstance(record.metadata, Mapping) else {}
            role = str(metadata.get("role") or "").strip().lower()
            speaker = str(
                metadata.get("sender_name")
                or metadata.get("sender_id")
                or ("真红" if role == "assistant" else "对方" if role == "user" else "群成员")
            ).strip()
            stamp = time.strftime("%m月%d日 %H:%M", time.localtime(record.created_at))
            lines.append(
                f"{index}. {stamp} {speaker}：{record.text[:self.policy.render_record_chars]} "
                f"（内部来源标识 source={record.source_id or 'unknown'}）"
            )
        limit = self.policy.recent_render_total_chars if mode == "recent" else self.policy.render_total_chars
        return "\n".join(lines)[:limit]

    @staticmethod
    def _topic_overlap(left: str, right: str) -> int:
        def terms(value: str) -> set[str]:
            result: set[str] = set()
            for item in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}", str(value or "").lower()):
                if any("\u4e00" <= char <= "\u9fff" for char in item):
                    result.update(item[index : index + 2] for index in range(len(item) - 1))
                else:
                    result.add(item)
            return result

        return len(terms(left) & terms(right))
