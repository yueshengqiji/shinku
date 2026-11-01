"""共享会话运行状态边界（ConversationRuntime）。

职责与上下游：
- 职责：为前台回合（Foreground Turns）、后台任务（Background Workers）以及主动输出（Proactive Outputs）
  提供统一的操作状态账本，以及主动输出的并发准入检查。
- 上游：由 `QQAgentBridge` 在回合进入（`begin`）、步进（`step`）与结束（`finish`）时驱动；
  由主动问候调度器在生成前后申请门禁（`proactive_permit` / `reserve_proactive`）。
- 下游：通过 `prompt_context` 和 `snapshot` 为模型注入客观的会话事实上下文，
  防止模型基于幻觉捏造执行证据。

状态所有权与持久化边界：
- 状态归属：`ConversationRuntime` 负责会话元数据的读写，`_lock` 只保护当前实例的数据库操作；
  多个实例可以访问同一个 SQLite 文件，跨实例写入由 `_database` 的 `BEGIN IMMEDIATE` 事务协调；
- 数据边界：仅存储有界的结构化元数据（租约、状态码、核验证据类型、送达回执、版本号），
  绝对不存储对话正文、完整提示词、工具参数或认证密钥；
- 租约隔离：每个回合拥有独立唯一的 `turn_id` 和租约有效期（`lease_until`），
  旧任务的执行完成绝不能意外关闭或清除更新回合的状态。

阅读入口：
- 核心生命周期：`begin`（开启回合租约）、`step`（更新阶段与证据）、`finish`（终结并记录送达回执）
- 事实投影导出：`prompt_context`（导出提示词事实上下文）、`snapshot`
- 主动问候并发门禁：`proactive_permit`、`validate_proactive`、`reserve_proactive`
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from .models import (
    AnswerEvidence,
    ConversationChannel,
    ConversationScope,
    ConversationTurnState,
    ProactivePermit,
    safe_code,
)

CONVERSATION_STATE_GUIDANCE = (
    "【共享会话运行事实】conversation_runtime来自宿主操作账本，不是历史记忆、模型推测或新的授权。"
    "候选记忆数量不证明历史问题已核对；只有实际查询与答案绑定的validated/verified记录才是核验结果。"
    "failed/blocked/interrupted表示未完成，not_found只表示已查范围未找到，read_failed不是没有发生。"
    "普通聊天回复完成不会消除旧的查询失败，也不能据此断言记忆完整、正常或已经核对清楚。"
    "单条历史查询成功也不证明整个记忆系统正常、没有紊乱或自动修复；失败也不证明整个记忆库损坏。"
    "对方追问未完成的核对时，承认当前不确定，按需调用只读查询；可以自然聊天，但不维护先前未经核实的说法。"
    "scheduled来源只是独立主动问候，不是对上一请求的查询结果；先前助手的话也不能自证事件或动机。"
    "sent=false、dry_run、partial或uncertain不证明用户收到了完整答复。不要复述内部字段或诊断代码。"
)


class ConversationStoreError(RuntimeError):
    pass


class ConversationRuntime:
    """会话运行状态运行时。

    设计原因：
    1. 真实操作账本：记录宿主传入的回合状态、核验证据（AnswerEvidence）与送达回执，
       模型回复不能自行替代这些运行事实；
    2. 独立回合租约：同一频道可有多个有效回合；`snapshot` 将租约过期的活跃回合显示为
       `interrupted`，主动消息门禁也不再把该租约视为有效占用，不会自动恢复或重跑原任务；
    3. 主动消息防打扰：有效活跃回合和最近 `quiet_seconds` 内的入站消息会阻止主动输出；
       非 scheduled、非 continued 的等待用户或未决核验状态仅在 `unresolved_quiet_seconds`
       窗口内阻止主动输出（默认 600 秒）。窗口过后仍须满足其他门禁，生成后的投递还需校验频道版本。
    """
    def __init__(
        self, path: str | Path | None = None, *, clock: Callable[[], float] = time.time,
        lease_seconds: float = 1920, retention_seconds: float = 86400,
        max_turns: int = 512, quiet_seconds: float = 90,
        unresolved_quiet_seconds: float = 600,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.clock = clock
        if any(not math.isfinite(float(value)) for value in (
            lease_seconds, retention_seconds, max_turns, quiet_seconds, unresolved_quiet_seconds,
        )):
            raise ValueError("invalid_conversation_runtime_limits")
        self.lease_seconds = max(30, min(86400, float(lease_seconds)))
        self.retention_seconds = max(600, min(7 * 86400, float(retention_seconds)))
        self.max_turns = max(16, min(4096, int(max_turns)))
        self.quiet_seconds = max(0, min(600, float(quiet_seconds)))
        self.unresolved_quiet_seconds = max(0, min(3600, float(unresolved_quiet_seconds)))
        self._lock = threading.RLock()
        self._memory = None
        if self.path is None:
            self._memory = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS conversation_channels (
                channel TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                last_inbound REAL NOT NULL, updated_at REAL NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS conversation_turns (
                turn_id TEXT PRIMARY KEY, scope TEXT NOT NULL, channel TEXT NOT NULL,
                updated_at REAL NOT NULL, payload TEXT NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS conversation_turn_scope ON conversation_turns(scope, updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS conversation_turn_channel ON conversation_turns(channel, updated_at)")

    @contextmanager
    def _database(self):
        with self._lock:
            try:
                conn = self._memory or sqlite3.connect(self.path, timeout=3)
            except sqlite3.Error as exc:
                raise ConversationStoreError("conversation_state_unavailable") from exc
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.commit()
            except (sqlite3.Error, ValueError, TypeError, KeyError) as exc:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise ConversationStoreError("conversation_state_unavailable") from exc
            except Exception:
                conn.rollback()
                raise
            finally:
                if self._memory is None:
                    conn.close()

    def close(self) -> None:
        with self._lock:
            if self._memory is not None:
                self._memory.close()

    @staticmethod
    def _load(conn, turn_id: str) -> ConversationTurnState | None:
        row = conn.execute("SELECT payload FROM conversation_turns WHERE turn_id=?", (turn_id,)).fetchone()
        return ConversationTurnState.from_dict(json.loads(row[0])) if row else None

    @staticmethod
    def _rows(conn, *, key: str, column: str, limit: int | None = None) -> list[ConversationTurnState]:
        if column not in {"scope", "channel"}:
            raise ValueError("invalid_conversation_lookup")
        rows = conn.execute(
            f"SELECT payload FROM conversation_turns WHERE {column}=? ORDER BY updated_at DESC"
            + (" LIMIT ?" if limit is not None else ""),
            (key, limit) if limit is not None else (key,),
        ).fetchall()
        return [ConversationTurnState.from_dict(json.loads(row[0])) for row in rows]

    @staticmethod
    def _revision(conn, channel: ConversationChannel, now: float, *, inbound: bool = False) -> None:
        conn.execute("""INSERT INTO conversation_channels VALUES (?,1,?,?)
            ON CONFLICT(channel) DO UPDATE SET revision=revision+1,
            last_inbound=CASE WHEN ? THEN excluded.last_inbound ELSE last_inbound END,
            updated_at=excluded.updated_at""", (channel.key, now if inbound else 0, now, int(inbound)))

    @staticmethod
    def _save(conn, state: ConversationTurnState) -> None:
        state.__post_init__()
        conn.execute("""INSERT INTO conversation_turns VALUES (?,?,?,?,?)
            ON CONFLICT(turn_id) DO UPDATE SET updated_at=excluded.updated_at,payload=excluded.payload""",
            (state.turn_id, state.scope.key, state.scope.channel.key, state.updated_at,
             json.dumps(state.as_dict(), ensure_ascii=False, separators=(",", ":"))))

    def notice_inbound(self, channel: ConversationChannel) -> None:
        now = float(self.clock())
        with self._database() as conn:
            self._revision(conn, channel, now, inbound=True)

    def begin(self, scope: ConversationScope, turn_id: str, *, origin: str = "user") -> None:
        """开启新回合生命周期并分配租约。

        设计原因与关键边界：
        - 幂等性：若同名 `turn_id` 已经存在，校验作用域一致后直接返回，防止重试导致租约漂移；
        - 租约上限：初始化租约截止时间为 `now + self.lease_seconds`，确保即使进程异常崩溃，
          租约超时后状态也会被判定为 `interrupted`，不会永久阻塞会话；
        - 频道活跃更新：非定时调度消息（origin != "scheduled"）将更新 `last_inbound` 时间戳，
          触发对主动问候的安静期保护（`quiet_seconds`）。
        """
        now = float(self.clock())
        with self._database() as conn:
            old = self._load(conn, turn_id)
            if old is not None:
                if old.scope != scope:
                    raise ValueError("conversation_scope_changed")
                return
            state = ConversationTurnState(turn_id, scope, now, now, now + self.lease_seconds, origin=origin)
            self._save(conn, state)
            self._revision(conn, scope.channel, now, inbound=origin != "scheduled")
            self._prune(conn, now)

    def _prune(self, conn, now: float) -> None:
        rows = conn.execute("SELECT turn_id,payload FROM conversation_turns ORDER BY updated_at DESC").fetchall()
        removed = []
        for index, (identifier, payload) in enumerate(rows):
            state = ConversationTurnState.from_dict(json.loads(payload))
            if state.active and state.lease_until > now:
                continue
            if index >= self.max_turns or now - state.updated_at > self.retention_seconds:
                removed.append((identifier,))
        conn.executemany("DELETE FROM conversation_turns WHERE turn_id=?", removed)
        conn.execute("""DELETE FROM conversation_channels WHERE updated_at < ? AND channel NOT IN
            (SELECT channel FROM conversation_turns)""", (now - self.retention_seconds,))

    def step(self, turn_id: str, *, phase: str | None = None, route: str | None = None,
             evidence: AnswerEvidence | None = None, candidate_count: int | None = None,
             task_id: str | None = None, task_attempt: int | None = None, task_version: int | None = None,
             queued: bool = False, lease_seconds: float | None = None) -> bool:
        now = float(self.clock())
        with self._database() as conn:
            state = self._load(conn, turn_id)
            if state is None or not state.active:
                return False
            if phase is not None:
                state.phase = phase
            if route is not None:
                state.route = route
            if evidence is not None:
                if not isinstance(evidence, AnswerEvidence):
                    raise TypeError("conversation_evidence_requires_host_receipt")
                state.evidence = evidence
            if candidate_count is not None:
                state.candidate_count = max(0, min(128, int(candidate_count)))
            if task_id is not None:
                if state.task_id != task_id:
                    state.task_attempt, state.task_version = 0, 0
                state.task_id = str(task_id)[:160]
            if task_attempt is not None:
                state.task_attempt = task_attempt
            if task_version is not None:
                state.task_version = task_version
            if queued:
                state.status = "queued"
            elif state.status == "queued":
                state.status = "running"
            lease = self.lease_seconds if lease_seconds is None else float(lease_seconds)
            if not math.isfinite(lease):
                raise ValueError("invalid_conversation_lease")
            state.updated_at, state.lease_until = now, now + max(30, min(86400, lease))
            self._save(conn, state)
            self._revision(conn, state.scope.channel, now)
            return True

    def finish(self, turn_id: str, *, status: str, reason: str = "", delivery_status: str = "not_sent",
               sent: bool = False, answer: str = "", message_ids=()) -> bool:
        """终结回合状态，记录实际送达回执与核验证据。

        设计原因与并发边界：
        1. 终态不可逆与 Worker 优先：
           - 若状态已经非 active（例如后台 Worker 已经先于前台超时包装器写入了完成回执），
             前台包装器返回 False，严禁用未完成状态覆写真实的 Worker 产物；
        2. 真实发送回执边界：
           - `state.sent` 必须同时满足 `sent == True` 且 `delivery_status == "sent"`，
             严禁在 dry_run、partial 或消息发送抛出异常时标记为已发送；
        3. 证据绑定（`evidence.bound_to`）：
           - 比较已核验正文的 SHA-256；若最终正文不同，则将旧证据降为未核验，
             防止正文改写后沿用旧核验结果。这不是签名，也不提供存储防篡改保证；
        4. 旧任务审批闭环：
           - 只有显式关联了相同 `task_id` 的后续回合（且非等待用户状态）才能将先前处于 `waiting_user` 的回合标记为 `continued`，
             普通的无关闲聊绝不能随手关掉未决的审批。
        """
        now = float(self.clock())
        with self._database() as conn:
            state = self._load(conn, turn_id)
            if state is None or not state.active:
                return False  # 前台包装器严禁覆写后台 Worker 已生成的终态回执
            state.status = status
            state.reason = safe_code(reason)
            state.delivery_status = delivery_status if delivery_status in {
                "not_sent", "queued", "dry_run", "sent", "partial", "suppressed", "failed", "uncertain",
            } else "uncertain"
            state.sent = bool(sent and state.delivery_status == "sent")
            state.message_ids = tuple(str(v)[:256] for v in tuple(message_ids)[:32])
            state.evidence = state.evidence.bound_to(answer)
            state.phase = "finished"
            state.updated_at, state.lease_until = now, 0
            self._save(conn, state)
            # 只有明确绑定同一任务的后续回合才能关闭旧的等待审批；普通闲聊绝不解决旧任务
            if state.task_id and status != "waiting_user":
                for prior in self._rows(conn, key=state.scope.key, column="scope"):
                    if (prior.turn_id != turn_id and prior.task_id == state.task_id and prior.status == "waiting_user"
                            and state.task_attempt >= prior.task_attempt):
                        prior.status, prior.updated_at = "continued", now
                        self._save(conn, prior)
            self._revision(conn, state.scope.channel, now)
            return True

    def snapshot(self, scope: ConversationScope, *, exclude_turn: str = "") -> dict:
        now = float(self.clock())
        with self._database() as conn:
            rows = self._rows(conn, key=scope.key, column="scope")
        values = []
        for state in rows:
            if state.turn_id == exclude_turn or now - state.updated_at > self.retention_seconds:
                continue
            status = "interrupted" if state.active and state.lease_until <= now else state.status
            evidence = state.evidence
            values.append({"turn_id": state.turn_id, "origin": state.origin, "route": state.route,
                "phase": state.phase, "status": status, "reason": state.reason,
                "evidence_kind": evidence.kind, "disposition": evidence.disposition,
                "validated": evidence.validated, "verified": evidence.verified,
                "query_count": evidence.query_count, "source_count": evidence.source_count,
                "candidate_count": state.candidate_count, "delivery_status": state.delivery_status,
                "sent": state.sent, "task_id": state.task_id,
                "task_attempt": state.task_attempt, "task_version": state.task_version,
                "updated_at": state.updated_at, "lease_until": state.lease_until})
        # Keep unresolved verification visible even if ordinary turns intervened.
        verifications = [row for row in values if row["evidence_kind"] != "none"][:4]
        return {"available": True, "recent_turns": values[:4], "recent_verifications": verifications}

    def prompt_context(self, scope: ConversationScope, *, exclude_turn: str = "") -> dict:
        try:
            return self.snapshot(scope, exclude_turn=exclude_turn)
        except ConversationStoreError:
            return {"available": False, "reason": "conversation_state_unavailable"}

    def task_receipts(self, scope: ConversationScope) -> dict:
        """Bounded host metadata; no dialogue, task goals or model reports."""
        now = float(self.clock())
        try:
            with self._database() as conn:
                states = self._rows(conn, key=scope.key, column="scope", limit=self.max_turns)
            return {"available": True, "records": [{
                "turn_id": s.turn_id, "task_id": s.task_id, "task_attempt": s.task_attempt,
                "task_version": s.task_version, "status": s.status, "route": s.route,
                "phase": s.phase, "delivery_status": s.delivery_status, "sent": s.sent,
                "reason": s.reason, "updated_at": s.updated_at, "lease_until": s.lease_until,
                "message_ids": list(s.message_ids),
                "started_at": s.started_at,
            } for s in states if now - s.updated_at <= self.retention_seconds]}
        except ConversationStoreError:
            return {"available": False, "records": [], "reason": "conversation_state_unavailable"}

    def _proactive(self, conn, channel: ConversationChannel, now: float, *, exclude_turn: str = "") -> ProactivePermit:
        row = conn.execute("SELECT revision,last_inbound FROM conversation_channels WHERE channel=?", (channel.key,)).fetchone()
        revision, last_inbound = row if row else (0, 0)
        states = [state for state in self._rows(conn, key=channel.key, column="channel") if state.turn_id != exclude_turn]
        if any(state.active and state.lease_until > now for state in states):
            return ProactivePermit(channel, revision, now, False, "conversation_active")
        if last_inbound and now - last_inbound < self.quiet_seconds:
            return ProactivePermit(channel, revision, now, False, "conversation_recently_active")
        for state in states:
            age = now - state.updated_at
            if age >= self.unresolved_quiet_seconds or state.origin == "scheduled" or state.status == "continued":
                continue
            if state.status == "waiting_user" or state.evidence.kind != "none" and not state.evidence.resolved:
                return ProactivePermit(channel, revision, now, False, "conversation_verification_unresolved")
        return ProactivePermit(channel, revision, now, True)

    def proactive_permit(self, channel: ConversationChannel, *, exclude_turn: str = "") -> ProactivePermit:
        now = float(self.clock())
        try:
            with self._database() as conn:
                return self._proactive(conn, channel, now, exclude_turn=exclude_turn)
        except ConversationStoreError:
            return ProactivePermit(channel, -1, now, False, "conversation_state_unavailable")

    def validate_proactive(self, permit: ProactivePermit, *, exclude_turn: str = "") -> bool:
        if not permit.allowed or not 0 <= float(self.clock()) - permit.checked_at <= 120:
            return False
        current = self.proactive_permit(permit.channel, exclude_turn=exclude_turn)
        return current.allowed and current.revision == permit.revision

    def reserve_proactive(self, permit: ProactivePermit, scope: ConversationScope,
                          turn_id: str) -> ProactivePermit | None:
        """Atomically validate a generated message and publish its delivery intent."""
        now = float(self.clock())
        if not permit.allowed or permit.channel != scope.channel or not 0 <= now - permit.checked_at <= 120:
            return None
        try:
            with self._database() as conn:
                current = self._proactive(conn, scope.channel, now)
                if not current.allowed or current.revision != permit.revision or self._load(conn, turn_id):
                    return None
                state = ConversationTurnState(turn_id, scope, now, now, now + self.lease_seconds,
                    origin="scheduled", route="scheduled", phase="delivery")
                self._save(conn, state)
                self._revision(conn, scope.channel, now)
                self._prune(conn, now)
                return self._proactive(conn, scope.channel, now, exclude_turn=turn_id)
        except ConversationStoreError:
            return None

    def storage_status(self) -> dict:
        return {"persistent": self.path is not None, "path": str(self.path or ""),
                "max_turns": self.max_turns, "retention_seconds": self.retention_seconds}
