"""真红会话记忆生命周期编排服务（MemoryService）。

职责与上下游：
- 职责：统一协调对话历史记录（`record_turn`）、被动观察（`observe_message`）、
  意图路由与检索准备（`prepare_turn`）、上下文证据聚合以及显式遗忘（`forget`）。
- 上游：由 `turn_preparation.py` 在回合准备阶段调用 `prepare_turn` 与 `record_turn`；
  由 `agent_bridge.py` 在被动监听消息时调用 `observe_message`。
- 下游：管理底层 `MemoryStore`（SQLite 存储与向量索引）、`MemoryRouter`（检索决策器）
  以及各类特化阅读器（`MemorySourceReader`、`HistoryWorkbench`）。

主人作用域（`OWNER_SCOPE = "private:master"`）：
- 跨上下文逻辑库：允许在群聊中安全回想私聊中主人的偏好和事实，
  同时严格隔离普通群成员的发言，防止跨群或跨作用域泄露。

阅读入口：
- 回合检索门禁与准备：`prepare_turn`
- 消息入库持久化：`record_turn`、`observe_message`
- 显式遗忘与级联失效：`forget`
"""

from __future__ import annotations

from shinku.character import CharacterProfile, SHINKU_CHARACTER

import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .models import ForgetReport, MemoryContext
from .policy import MemoryPolicy
from .router import MemoryRouter
from .store import MemoryStore
from .models import MemoryRecord
from .lifecycle_review import LifecycleReviewQueue
from .source_reader import MemorySourceReader

from . import service_access, service_queries, service_evidence, service_recording, service_tasks, service_usage
from .service_models import (
    MemoryServiceAccessDependencies,
    MemoryServiceQueriesDependencies,
    MemoryServiceEvidenceDependencies,
    MemoryServiceRecordingDependencies,
    MemoryServiceTasksDependencies,
    MemoryServiceUsageDependencies,
)

__all__ = ["MemoryService"]


class MemoryService:
    """Record history, ask the router, retrieve on demand, and forget explicitly."""

    # A logical library inside the same SQLite source of truth.  Keeping this
    # scope separate lets the owner recall a private/group conversation from
    # either side without making ordinary group spectators globally visible.
    OWNER_SCOPE = "private:master"

    _FACT_RE = re.compile(r"^(?:请)?记住(?:一下|了)?[：:，,、\s]*(?P<fact>.+)$")
    _FORGET_RE = re.compile(r"^(?:请)?(?:忘掉|忘记|别记|不要记|清除|删除)(?:关于|掉|掉关于)?\s*(?P<query>.+)$")
    _TASK_SECRET_ASSIGNMENT_RE = re.compile(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|refresh[_ -]?token|client[_ -]?secret|password|cookie)\b(\s*[:=]\s*)([^\s,;]+)"
    )
    _TASK_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*")
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
        lifecycle_reviewer: Any | None = None,
        projection_model: Any | None = None,
        character: CharacterProfile = SHINKU_CHARACTER,
    ) -> None:
        self.character = character
        self.store = store
        self.router = router
        self.clock = clock or time.time
        self.policy = policy or MemoryPolicy()
        self.projection = None
        self.projection_status: dict[str, Any] = {}
        self.store.configure_lifecycle(self.policy.lifecycle)
        self.store.configure_search_keys(enabled=self.policy.search_keys_enabled)
        self.current_understanding = None
        self.evidence_reader = MemorySourceReader(self.store) if self.policy.evidence_package_enabled else None
        self.integrity_reader = MemorySourceReader(self.store)
        self.history_workbench = None
        if self.policy.history_needs_enabled:
            from .workbench import HistoryWorkbench
            self.history_workbench = HistoryWorkbench(self.store, ttl_seconds=self.policy.workbench_ttl_seconds,
                max_entries=self.policy.workbench_max_entries)
        self.current_understanding_status: dict[str, Any] = {}
        if self.policy.current_understanding_enabled:
            from .current_understanding import CurrentUnderstanding
            try:
                self.current_understanding = CurrentUnderstanding(self.store)
            except Exception as exc:
                self.current_understanding_status = {"fallback": "source_search", "error_type": type(exc).__name__}
        self.store.lifecycle.wall_time = self.clock
        self._usage_lock = threading.RLock()
        self._usage_contexts: dict[tuple[str, str], tuple[int, str, tuple[MemoryRecord, ...], str]] = {}
        self.lifecycle_review = LifecycleReviewQueue(self, lifecycle_reviewer)
        self.store.on_forget(self._forget_usage_contexts)
        self.store.on_forget(self.lifecycle_review.invalidate)
        self.store.configure(
            max_query_terms=self.policy.max_query_terms,
            event_scan_limit=self.policy.event_scan_limit,
            fact_scan_limit=self.policy.fact_scan_limit,
            max_results=self.policy.max_results,
        )
        if self.policy.projection.enabled:
            from .projection.engine import ProjectionEngine
            try:
                self.projection = ProjectionEngine(self.store, self.policy.projection, projection_model,
                    understanding_enabled=self.policy.current_understanding_enabled)
            except Exception as exc:
                self.projection_status = {"state":"unavailable", "error_type":type(exc).__name__}
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

    @property
    def _search_keys_ready(self) -> bool:
        return bool(self.policy.search_keys_enabled and self.store.search_keys_enabled
                    and self.store.search_index is not None)

    @property
    def _source_reader(self):
        if self._search_keys_ready:
            return self.store.search_index
        if self.policy.current_understanding_enabled and self.current_understanding is not None:
            return self.current_understanding
        if self.policy.evidence_package_enabled:
            return self.evidence_reader
        if self.policy.history_needs_enabled and self.history_workbench is not None:
            return self.history_workbench.reader
        return None

    def _current_records(self, records, *, scopes, subject, query, now, limit):
        return service_evidence.current_records(
            records,
            scopes=scopes,
            subject=subject,
            query=query,
            now=now,
            limit=limit,
            deps=self._service_evidence_dependencies(),
        )

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
        return service_access.recent_owner_handoff(
            requester_id=requester_id,
            limit=limit,
            max_age_seconds=max_age_seconds,
            deps=self._service_access_dependencies(),
        )

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
        return service_access.owner_memory_allowed(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            sender_id=sender_id,
            metadata=metadata,
            role=role,
            deps=self._service_access_dependencies(),
        )

    def _is_master_request(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        message_meta: Mapping[str, Any] | None,
    ) -> bool:
        return service_access.is_master_request(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            message_meta=message_meta,
            deps=self._service_access_dependencies(),
        )

    def _history_scopes(self, scope: str, *, owner_request: bool, mode: str) -> tuple[str, ...]:
        """Return current scope plus owner scope only for an owner recall."""
        return service_access.history_scopes(
            scope,
            owner_request=owner_request,
            mode=mode,
            deps=self._service_access_dependencies(),
        )

    @staticmethod
    def _merge_by_source(
        primary: list[Any] | tuple[Any, ...],
        secondary: list[Any] | tuple[Any, ...],
        *,
        limit: int,
    ) -> tuple[Any, ...]:
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

    def inspect_self_group_activity(
        self, *, conversation_id: str, conversation_type: str, requester_id: str,
        query: str, self_id: str = "", limit: int = 8,
    ) -> dict[str, Any]:
        """Separate bot-owned activity capability, never general cross-chat recall."""
        return service_access.inspect_self_group_activity(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            requester_id=requester_id,
            query=query,
            self_id=self_id,
            limit=limit,
            deps=self._service_access_dependencies(),
        )

    def prepare_turn(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        user_message: str,
        recent_context: str = "",
        message_meta: Mapping[str, Any] | None = None,
    ) -> MemoryContext:
        return service_queries.prepare_turn(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            user_message=user_message,
            recent_context=recent_context,
            message_meta=message_meta,
            deps=self._service_queries_dependencies(),
        )

    def read_recent_owner_sources(self, *, conversation_id, conversation_type,
                                  requester, self_id, message_id, selection_at=None):
        """公开读取主人短期来源；身份、窗口和原件校验由存储适配层每次重核。

        返回本轮有效记录及版本依赖；选材时刻只固定本轮范围，不继承旧授权，不写入或缓存正文。
        展示数量与字数由 owner_continuity 独立处理，不能影响来源失效判定。
        """
        from .owner_sources import read_recent_owner_sources
        return read_recent_owner_sources(self, conversation_id=conversation_id,
            conversation_type=conversation_type, requester=requester,
            self_id=self_id, message_id=message_id, selection_at=selection_at)

    def project_context(self, *, conversation_id, conversation_type, user_message,
                        message_meta, memory_context=None, flow_reason="", continuity_handles=None):
        """Authorize anew at every prompt build, including tool follow-ups."""
        from .projection.service import project_context
        return project_context(self, conversation_id=conversation_id, conversation_type=conversation_type,
            user_message=user_message, message_meta=message_meta, memory_context=memory_context,
            flow_reason=flow_reason, continuity_handles=continuity_handles)

    def _private_recent_dialogue(
        self, *, scope: str, conversation_id: str, conversation_type: str,
        requester: str, self_id: str, now: int,
    ) -> str:
        """Read one exact, receipt-backed private pair, even after a restart.

        Never enter retrieval candidates, usage reinforcement or voice examples.
        A guessed identity, lost source or unsent draft is not dialogue evidence.
        """
        return service_evidence.private_recent_dialogue(
            scope=scope,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            requester=requester,
            self_id=self_id,
            now=now,
            deps=self._service_evidence_dependencies(),
        )

    def bind_usage_context(
        self,
        *,
        scope: str,
        message_id: str,
        requester_id: str,
        records: Sequence[MemoryRecord],
        user_text: str,
    ) -> None:
        """Keep only actually delivered current-turn candidates, not proofs.

        This bounded in-memory map is never restored as reinforcement evidence.
        Tools may add candidates under the same current request; mere binding
        changes no counters or levels.
        """
        return service_usage.bind_usage_context(
            scope=scope,
            message_id=message_id,
            requester_id=requester_id,
            records=records,
            user_text=user_text,
            deps=self._service_usage_dependencies(),
        )

    def _forget_usage_contexts(self, *, scope: str, record_ids: set[str]) -> None:
        return service_usage.forget_usage_contexts(
            scope=scope,
            record_ids=record_ids,
            deps=self._service_usage_dependencies(),
        )

    def finalize_usage(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        message_id: str,
        requester_id: str,
        final_text: str,
    ) -> dict[str, Any]:
        """Verify final source-bound use after generation, never on retrieval."""
        return service_usage.finalize_usage(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            message_id=message_id,
            requester_id=requester_id,
            final_text=final_text,
            deps=self._service_usage_dependencies(),
        )

    def _reinforce_rementions(
        self,
        *,
        scope: str,
        body: str,
        sender_id: str,
        evidence_id: str,
        now: int,
        occurred_at: int | None = None,
    ) -> int:
        return service_usage.reinforce_rementions(
            scope=scope,
            body=body,
            sender_id=sender_id,
            evidence_id=evidence_id,
            now=now,
            occurred_at=occurred_at,
            deps=self._service_usage_dependencies(),
        )

    @staticmethod
    def _evidence_ngrams(value: str, size: int) -> set[str]:
        chunks = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9_]{3,}", str(value or "").lower())
        return {chunk[index:index + size] for chunk in chunks for index in range(len(chunk) - size + 1)}

    def _own_utterance_evidence(
        self, *, scope: str, now: int, user_message: str, recent_records: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        """Select at most two relevant original replies, never third-party claims."""
        return service_evidence.own_utterance_evidence(
            scope=scope,
            now=now,
            user_message=user_message,
            recent_records=recent_records,
            deps=self._service_evidence_dependencies(),
        )

    @staticmethod
    def _render_own_utterances(records: tuple[Any, ...], *, character=SHINKU_CHARACTER) -> str:
        lines = [
            f"【{character.name}本人近期发言原文核对】",
            "这些是本人发言的记录，只证明当时说过这些话，不证明内容一定正确；不要模仿措辞。"
            "若与群友转述冲突，先按时间核对原话；没有匹配原话时，不要凭转述断言自己说过或没说过。",
        ]
        for record in records:
            stamp = time.strftime("%H:%M:%S", time.localtime(record.created_at))
            lines.append(f"{stamp} {service_evidence.assistant_label(record.metadata, character)}：{record.text[:180]}（source={record.source_id}）")
        return "\n".join(lines)[:600]

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

    def observe_message(self, message: Any) -> bool:
        """Passively persist a normalized QQ message before reply routing.

        This method never calls an LLM and never decides whether Shinku should
        answer.  It only records bounded evidence for later local routing.
        """
        return service_recording.observe_message(
            message,
            deps=self._service_recording_dependencies(),
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
        return service_recording.record_turn(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            message_id=message_id,
            text=text,
            role=role,
            source_id=source_id,
            metadata=metadata,
            deps=self._service_recording_dependencies(),
        )

    def record_agent_task_summary(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        requester_id: str,
        task_id: str,
        goal: str,
        status: str,
        outcome: str,
        reason: str = "",
        rounds: int = 0,
        tool_outcomes: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Persist a compact, owner-only index card for a terminal Agent task.

        This compact index card stores the goal, final user-facing outcome,
        status, reason, tool names/statuses, and task ID. A separate, expiring
        detail record may keep the ordered tool names/statuses. Raw tool
        arguments, outputs, and model reasoning are intentionally excluded.
        The index card is non-expiring; the owner's normal forget/reset path
        remains its retention control.
        """
        return service_tasks.record_agent_task_summary(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            requester_id=requester_id,
            task_id=task_id,
            goal=goal,
            status=status,
            outcome=outcome,
            reason=reason,
            rounds=rounds,
            tool_outcomes=tool_outcomes,
            metadata=metadata,
            deps=self._service_tasks_dependencies(),
        )

    def inspect_agent_task(self, *, requester_id: str, task_id: str) -> dict[str, Any]:
        """Open an exact owner task card without searching or exposing other scopes."""
        return service_tasks.inspect_agent_task(
            requester_id=requester_id,
            task_id=task_id,
            deps=self._service_tasks_dependencies(),
        )

    def record_agent_task_detail(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        requester_id: str,
        task_id: str,
        status: str,
        rounds: int,
        steps: Sequence[Mapping[str, Any]] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> bool:
        """Store a bounded owner-only tool sequence without raw arguments/results."""
        return service_tasks.record_agent_task_detail(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            requester_id=requester_id,
            task_id=task_id,
            status=status,
            rounds=rounds,
            steps=steps,
            metadata=metadata,
            deps=self._service_tasks_dependencies(),
        )

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
            for key in ("media_id", "url", "local_path", "file_id", "name", "size"):
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
        now = int(self.clock())
        expired = self.store.expire(now=now)
        return {"expired": expired, "stats": self.store.stats(), "lifecycle": self.store.lifecycle.scan(now=now)}

    def inspect(
        self,
        *,
        conversation_id: str,
        conversation_type: str,
        requester_id: str = "",
        query: str = "",
        mode: str = "recent",
        limit: int = 8,
        message_id: str = "",
        user_text: str = "",
        self_id: str = "",
        recorded_time: Mapping[str, str] | None = None,
        source_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Read memory for an explicit user/tool request.

        Automatic routing and explicit inspection are intentionally separate:
        this method never changes persona context and never writes facts.  It
        only reads the current conversation scope and returns a bounded,
        provenance-preserving result for the Agent to summarize.
        """
        if self.policy.projection.enabled and self.policy.projection.retrieval_strategy == "model":
            if mode == "activity":
                if recorded_time is not None:
                    from .query_time import normalize_recorded_time
                    if any(normalize_recorded_time(recorded_time).values()):
                        raise ValueError("activity_recorded_time_not_supported")
                return self.inspect_self_group_activity(conversation_id=conversation_id,
                    conversation_type=conversation_type, requester_id=requester_id,
                    query=query or user_text, self_id=self_id, limit=limit)
            from .projection.inspection import inspect_context
            return inspect_context(self, conversation_id=conversation_id, conversation_type=conversation_type,
                requester_id=requester_id, query=query, mode=mode, limit=limit,
                message_id=message_id, user_text=user_text, self_id=self_id, recorded_time=recorded_time,
                source_ids=source_ids)
        if recorded_time is not None or source_ids:
            raise ValueError("recorded_time_requires_model_retrieval")
        return service_queries.inspect(
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            requester_id=requester_id,
            query=query,
            mode=mode,
            limit=limit,
            message_id=message_id,
            user_text=user_text,
            self_id=self_id,
            deps=self._service_queries_dependencies(),
        )

    def _record_audit(
        self,
        *,
        conversation_id: str,
        scope: str,
        now: int,
        context: MemoryContext,
    ) -> None:
        return service_usage.record_audit(
            conversation_id=conversation_id,
            scope=scope,
            now=now,
            context=context,
            deps=self._service_usage_dependencies(),
        )

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

    def _request_date_window(self, query: str, *, now: int):
        if self.policy.evidence_package_enabled:
            from .evidence import validity_query, event_time_query
            if validity_query(query) or event_time_query(query):
                return None
        return self._date_window(query, now=now)

    def _evidence_records(
        self,
        records,
        *,
        query: str,
        scopes,
        requester: str,
        now: int,
        limit: int,
    ):
        """Bounded facet searches inside the already authorized history scopes.

        This doesn't route new tasks or infer missing data. Exact calendar record
        requests never broaden their window. Primary facet sources share the same
        record budget with existing hits and explicit reply neighbours.
        """
        return service_evidence.evidence_records(
            records,
            query=query,
            scopes=scopes,
            requester=requester,
            now=now,
            limit=limit,
            deps=self._service_evidence_dependencies(),
        )

    def _render_evidence(self, records, *, query: str, scopes, requester: str, mode: str, now: int, limit: int,
                         history_need=None, delivered_records=None):
        return service_evidence.render_evidence(
            records,
            query=query,
            scopes=scopes,
            requester=requester,
            mode=mode,
            now=now,
            limit=limit,
            history_need=history_need,
            delivered_records=delivered_records,
            deps=self._service_evidence_dependencies(),
        )

    def _render(
        self,
        records: tuple[Any, ...],
        *,
        mode: str = "retrieve",
        delivered=None,
        reserved_chars: int = 0,
    ) -> str:
        return service_evidence.render(
            records,
            mode=mode,
            delivered=delivered,
            reserved_chars=reserved_chars,
            deps=self._service_evidence_dependencies(),
        )

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

    def _service_access_dependencies(self) -> MemoryServiceAccessDependencies:
        return MemoryServiceAccessDependencies(
            character=self.character,
            owner_scope=lambda: self.OWNER_SCOPE,
            request_date_window=lambda *args, **kwargs: self._request_date_window(*args, **kwargs),
            clock=lambda *args, **kwargs: self.clock(*args, **kwargs),
            integrity_reader=lambda: self.integrity_reader,
            is_master_user=lambda *args, **kwargs: self.is_master_user(*args, **kwargs),
            policy=lambda: self.policy,
            scope_for=lambda *args, **kwargs: self.scope_for(*args, **kwargs),
            store=lambda: self.store,
        )

    def _service_queries_dependencies(self) -> MemoryServiceQueriesDependencies:
        return MemoryServiceQueriesDependencies(
            character=self.character,
            current_records=lambda *args, **kwargs: self._current_records(*args, **kwargs),
            evidence_records=lambda *args, **kwargs: self._evidence_records(*args, **kwargs),
            history_scopes=lambda *args, **kwargs: self._history_scopes(*args, **kwargs),
            is_master_request=lambda *args, **kwargs: self._is_master_request(*args, **kwargs),
            merge_by_source=lambda *args, **kwargs: self._merge_by_source(*args, **kwargs),
            own_utterance_evidence=lambda *args, **kwargs: self._own_utterance_evidence(*args, **kwargs),
            precise_day_request=lambda *args, **kwargs: self._precise_day_request(*args, **kwargs),
            private_recent_dialogue=lambda *args, **kwargs: self._private_recent_dialogue(*args, **kwargs),
            recall_query=lambda *args, **kwargs: self._recall_query(*args, **kwargs),
            record_audit=lambda *args, **kwargs: self._record_audit(*args, **kwargs),
            render_evidence=lambda *args, **kwargs: self._render_evidence(*args, **kwargs),
            render_own_utterances=lambda *args, **kwargs: self._render_own_utterances(*args, character=self.character, **kwargs),
            request_date_window=lambda *args, **kwargs: self._request_date_window(*args, **kwargs),
            source_reader=lambda: self._source_reader,
            bind_usage_context=lambda *args, **kwargs: self.bind_usage_context(*args, **kwargs),
            clock=lambda *args, **kwargs: self.clock(*args, **kwargs),
            history_workbench=lambda: self.history_workbench,
            inspect_self_group_activity=lambda *args, **kwargs: self.inspect_self_group_activity(*args, **kwargs),
            policy=lambda: self.policy,
            router=lambda: self.router,
            scope_for=lambda *args, **kwargs: self.scope_for(*args, **kwargs),
            store=lambda: self.store,
            route_history_need=lambda routing_function, **kwargs: routing_function(self, **kwargs),
        )

    def _service_evidence_dependencies(self) -> MemoryServiceEvidenceDependencies:
        return MemoryServiceEvidenceDependencies(
            character=self.character,
            evidence_ngrams=lambda *args, **kwargs: self._evidence_ngrams(*args, **kwargs),
            merge_by_source=lambda *args, **kwargs: self._merge_by_source(*args, **kwargs),
            render=lambda *args, **kwargs: self._render(*args, **kwargs),
            request_date_window=lambda *args, **kwargs: self._request_date_window(*args, **kwargs),
            source_reader=lambda: self._source_reader,
            clock=lambda *args, **kwargs: self.clock(*args, **kwargs),
            current_understanding=lambda: self.current_understanding,
            integrity_reader=lambda: self.integrity_reader,
            policy=lambda: self.policy,
            store=lambda: self.store,
            set_current_understanding_status=lambda value: self._set_memory_current_understanding_status(value),
        )

    def _service_recording_dependencies(self) -> MemoryServiceRecordingDependencies:
        return MemoryServiceRecordingDependencies(
            owner_scope=lambda: self.OWNER_SCOPE,
            fact_pattern=lambda: self._FACT_RE,
            forget_pattern=lambda: self._FORGET_RE,
            attachment_refs=lambda *args, **kwargs: self._attachment_refs(*args, **kwargs),
            key=lambda *args, **kwargs: self._key(*args, **kwargs),
            reinforce_rementions=lambda *args, **kwargs: self._reinforce_rementions(*args, **kwargs),
            topic_overlap=lambda *args, **kwargs: self._topic_overlap(*args, **kwargs),
            clock=lambda *args, **kwargs: self.clock(*args, **kwargs),
            owner_memory_allowed=lambda *args, **kwargs: self.owner_memory_allowed(*args, **kwargs),
            policy=lambda: self.policy,
            scope_for=lambda *args, **kwargs: self.scope_for(*args, **kwargs),
            store=lambda: self.store,
        )

    def _service_tasks_dependencies(self) -> MemoryServiceTasksDependencies:
        return MemoryServiceTasksDependencies(
            owner_scope=lambda: self.OWNER_SCOPE,
            task_bearer_pattern=lambda: self._TASK_BEARER_RE,
            task_secret_pattern=lambda: self._TASK_SECRET_ASSIGNMENT_RE,
            clock=lambda *args, **kwargs: self.clock(*args, **kwargs),
            is_master_user=lambda *args, **kwargs: self.is_master_user(*args, **kwargs),
            owner_memory_allowed=lambda *args, **kwargs: self.owner_memory_allowed(*args, **kwargs),
            policy=lambda: self.policy,
            scope_for=lambda *args, **kwargs: self.scope_for(*args, **kwargs),
            store=lambda: self.store,
        )

    def _service_usage_dependencies(self) -> MemoryServiceUsageDependencies:
        return MemoryServiceUsageDependencies(
            owner_scope=lambda: self.OWNER_SCOPE,
            usage_contexts=lambda: self._usage_contexts,
            usage_lock=lambda: self._usage_lock,
            clock=lambda *args, **kwargs: self.clock(*args, **kwargs),
            is_master_user=lambda *args, **kwargs: self.is_master_user(*args, **kwargs),
            lifecycle_review=lambda: self.lifecycle_review,
            policy=lambda: self.policy,
            scope_for=lambda *args, **kwargs: self.scope_for(*args, **kwargs),
            store=lambda: self.store,
            set_usage_contexts=lambda value: self._set_memory_usage_contexts(value),
        )

    def _set_memory_usage_contexts(
        self,
        value: dict[tuple[str, str], tuple[int, str, tuple[MemoryRecord, ...], str]],
    ) -> None:
        self._usage_contexts = value

    def _set_memory_current_understanding_status(self, value: dict[str, Any]) -> None:
        self.current_understanding_status = value
