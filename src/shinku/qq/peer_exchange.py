"""已登记 Agent 的群聊及私聊互答限额，独立于亲属身份和工具权限。

observe 接收经宿主验证的原始入站消息；claim 在模型调用前原子占用最多两次接话。
群聊限额按实例、群和真人接话周期持久化；私聊按对方账号持久化。
私聊没有第三方真人入站，因此不会被其他私聊或群里的发言自动重置。
重启、等待、改昵称或机器人换话题不能刷新额度；宿主可显式豁免指定账号的私聊上限。
私聊豁免仍验证接收账号并去重，不影响该账号在群聊中的额度或主人权限。
失败、静默及未知送达也消耗已占额度，避免通过重试放大模型请求。这里不发送消息。
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from collections.abc import Sequence


class PeerExchangeGuard:
    """短连接事务持有计数；仅登记的 Agent 受限，其他真实账号的入站发言恢复额度。

平台不提供可靠的人类/机器人证明，因此所有自动账号都应加入 peer_ids。
当前账号、通知、引用正文和重放的真人消息不能恢复额度。配置不完整时拒绝启用。
"""

    def __init__(self, path: str | Path, *, peer_ids: Sequence[str], self_ids: Sequence[str],
                 private_unlimited_ids: Sequence[str] = (), clock=time.time):
        self.peer_ids = frozenset(str(v).strip() for v in peer_ids)
        self.self_ids = frozenset(str(v).strip() for v in self_ids)
        self.private_unlimited_ids = frozenset(str(v).strip() for v in private_unlimited_ids)
        if (not self.peer_ids or not self.self_ids or self.peer_ids & self.self_ids
                or len(self.peer_ids) > 32 or any(not v or len(v) > 128 for v in self.peer_ids | self.self_ids)):
            raise ValueError("peer_exchange_identity_invalid")
        if not self.private_unlimited_ids <= self.peer_ids:
            raise ValueError("peer_private_unlimited_not_peer")
        if not str(path).strip():
            raise ValueError("peer_exchange_path_required")
        self.path, self.clock = Path(path), clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS peer_windows (
                    conversation TEXT PRIMARY KEY, used INTEGER NOT NULL DEFAULT 0,
                    human_at REAL NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS peer_events (
                    conversation TEXT NOT NULL, sender TEXT NOT NULL, message TEXT NOT NULL,
                    observed REAL NOT NULL, PRIMARY KEY(conversation,sender,message));
            """)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def _trusted(self, message) -> bool:
        raw = getattr(message, "raw_event", {})
        return (getattr(message, "conversation_type", "") in {"group", "private"}
                and bool(getattr(message, "conversation_id", ""))
                and bool(getattr(message, "message_id", ""))
                and isinstance(raw, dict) and str(raw.get("self_id") or "") in self.self_ids)

    def is_peer_turn(self, turn) -> bool:
        """整批都来自登记的 Agent 才允许额外静默；混有真人时保留原请求处理。"""
        return (getattr(turn, "conversation_type", "") in {"group", "private"}
                and bool(getattr(turn, "messages", ()))
                and all(self._trusted(m) and str(m.sender_id) in self.peer_ids
                        and str(m.conversation_id) == str(turn.conversation_id) for m in turn.messages))

    def observe(self, message) -> None:
        """只用新的真实发言恢复额度；无可靠时间的消息不能刷新已耗尽的窗口。"""
        if (getattr(message, "conversation_type", "") != "group"
                or not self._trusted(message) or str(message.sender_id) in self.peer_ids | self.self_ids
                or getattr(message, "event_kind", "message") != "message"
                or getattr(message, "is_poke", False)
                or not (str(getattr(message, "text", "")).strip() or getattr(message, "has_media", False))):
            return
        now = self.clock()
        timestamp = getattr(message, "timestamp", None)
        if not isinstance(timestamp, (int, float)) or not 0 <= now - timestamp <= 120:
            return
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM peer_events WHERE observed < ?", (now - 300,))
            added = db.execute("INSERT OR IGNORE INTO peer_events VALUES(?,?,?,?)",
                (str(message.conversation_id), str(message.sender_id), str(message.message_id), now)).rowcount
            if not added:
                return
            db.execute("INSERT OR IGNORE INTO peer_windows(conversation) VALUES(?)", (str(message.conversation_id),))
            db.execute("UPDATE peer_windows SET used=0,human_at=? WHERE conversation=? AND human_at<=?",
                (timestamp, str(message.conversation_id), timestamp))

    def claim(self, turn) -> str:
        """返回空串表示允许，否则是静默原因；重复消息和存储错误均不能开启新轮。"""
        if (getattr(turn, "conversation_type", "") in {"group", "private"}
                and any(str(getattr(m, "sender_id", "")) in self.peer_ids and not self._trusted(m)
                        for m in getattr(turn, "messages", ()))):
            return "peer_exchange_receiver_mismatch"
        if not self.is_peer_turn(turn):
            return ""
        # 豁免由宿主账号名单决定，只适用于本人发来的私聊整批消息；
        # 群号碰巧同号、其他 Agent 混入或正文自称身份均不能获得豁免。
        private_unlimited = (turn.conversation_type == "private"
            and str(turn.conversation_id) in self.private_unlimited_ids
            and all(str(m.sender_id) == str(turn.conversation_id) for m in turn.messages))
        try:
            primary = next(m for m in turn.messages if str(m.message_id) == str(turn.primary_message_id))
            now = self.clock()
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                # 保留既有群聊键，避免升级刷新已经耗尽的额度；私聊加命名空间，
                # 防止恰好同号的群消息恢复机器人私聊额度。
                group = ("private:" if turn.conversation_type == "private" else "") + str(turn.conversation_id)
                db.execute("INSERT OR IGNORE INTO peer_windows(conversation) VALUES(?)", (group,))
                used, human_at = db.execute("SELECT used,human_at FROM peer_windows WHERE conversation=?", (group,)).fetchone()
                # 真人已开新话轮时，不把排队中的旧机器人消息当作新一轮互聊。
                if human_at and (not isinstance(primary.timestamp, (int, float)) or primary.timestamp < human_at):
                    return "peer_exchange_stale"
                if not private_unlimited and used >= 2:
                    return "peer_exchange_limit"
                added = db.execute("INSERT OR IGNORE INTO peer_events VALUES(?,?,?,?)",
                    (group, str(primary.sender_id), str(primary.message_id), now)).rowcount
                if not added:
                    return "peer_exchange_duplicate"
                # 保留旧累计值，撤销豁免时仍沿用原限额；无限私聊也必须写事件去重。
                if not private_unlimited:
                    db.execute("UPDATE peer_windows SET used=used+1 WHERE conversation=?", (group,))
                return ""
        except (OSError, sqlite3.Error, StopIteration):
            return "peer_exchange_unavailable"
