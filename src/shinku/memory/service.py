"""Memory lifecycle orchestration for one Shinku conversation."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from .models import ForgetReport, MemoryContext
from .policy import MemoryPolicy
from .router import MemoryRouter
from .store import MemoryStore

__all__ = ["MemoryService"]


class MemoryService:
    """Record history, ask the router, retrieve on demand, and forget explicitly."""

    _FACT_RE = re.compile(r"^(?:请)?记住(?:一下|了)?[：:，,、\s]*(?P<fact>.+)$")
    _FORGET_RE = re.compile(r"^(?:请)?(?:忘掉|忘记|别记|不要记|清除|删除)(?:关于|掉|掉关于)?\s*(?P<query>.+)$")

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

    def prepare_turn(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        user_message: str,
        recent_context: str = "",
    ) -> MemoryContext:
        now = int(self.clock())
        self.store.expire(now=now)
        decision = self.router.decide(
            user_message=user_message,
            recent_context=recent_context,
            conversation_type=conversation_type,
        )
        if not decision.need_retrieval:
            return MemoryContext(decision=decision)
        scope = self.scope_for(conversation_id, conversation_type)
        records = tuple(
            self.store.search(
                query=decision.query or user_message,
                scope=scope,
                limit=self.policy.retrieval_limit,
                now=now,
            )
        )
        return MemoryContext(decision=decision, records=records, rendered=self._render(records))

    def record_turn(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        message_id: str,
        text: str,
        role: str = "user",
    ) -> ForgetReport | None:
        body = str(text or "").strip()
        if not body:
            return None
        now = int(self.clock())
        scope = self.scope_for(conversation_id, conversation_type)
        source_id = f"turn:{conversation_id}:{message_id or now}"
        self.store.record_turn(
            conversation_id=conversation_id,
            scope=scope,
            role=role,
            content=body,
            source_id=source_id,
            created_at=now,
        )
        forget_match = self._FORGET_RE.match(body)
        if forget_match:
            query = str(forget_match.group("query") or "").strip(" ：:，,。.!！")
            forgotten, purged = self.store.forget(query=query, scope=scope, now=now)
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
                    source_id=source_id,
                    now=now,
                )
        else:
            self.store.record_event(
                conversation_id=conversation_id,
                scope=scope,
                text=body,
                source_id=source_id,
                layer="episodic",
                created_at=now,
                importance=0.22,
                confidence=0.45,
                expires_at=now + self.policy.event_ttl_seconds if self.policy.event_ttl_seconds else 0,
                metadata={"role": role, "source": "qq_turn"},
            )
        return None

    def maintenance(self) -> dict[str, Any]:
        expired = self.store.expire(now=int(self.clock()))
        return {"expired": expired, "stats": self.store.stats()}

    @staticmethod
    def scope_for(conversation_id: str, conversation_type: str) -> str:
        kind = str(conversation_type or "private").lower()
        prefix = "group" if "group" in kind else "private"
        return f"{prefix}:{str(conversation_id or 'unknown').strip() or 'unknown'}"

    @staticmethod
    def _key(text: str) -> str:
        compact = re.sub(r"\s+", "", text.lower())
        return compact[:120] or "unnamed"

    def _render(self, records: tuple[Any, ...]) -> str:
        if not records:
            return ""
        lines = [
            "【按本轮问题检索到的历史记录】",
            "以下只是带来源的历史记录，不是当前事实，也不是说话范本；不相关时忽略。",
        ]
        for index, record in enumerate(records, start=1):
            lines.append(
                f"{index}. [{record.layer}] {record.text[:self.policy.render_record_chars]} "
                f"(score={record.score:.2f}, source={record.source_id or 'unknown'})"
            )
        return "\n".join(lines)[: self.policy.render_total_chars]
