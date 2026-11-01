"""将显式登记参与者的原始对话收录到私用库，不扩大读取和控制权限。

观察入口覆盖未回复的群聊/私聊；回合入口覆盖宿主已发送的答复。
只复制同库原始事件，保留作者、时间、原会话和来源版本。历史补录也复用此入口，
以来源 ID 幂等；原件撤销、过期或改写后，来源读取器拒绝继续使用副本。
"""
from __future__ import annotations

import json

from .source_reader import MemorySourceReader


def participant_id(policy, *, scope, metadata):
    """身份仅来自宿主元数据和本机配置，不从昵称、引用或正文提取。"""
    if not policy.master_user_ids:
        return ""
    role = metadata.get("role", "user")
    if role not in {"user", "assistant"}:
        return ""
    sender = str(metadata.get("requester_id" if role == "assistant" else "sender_id") or "")
    if sender not in policy.owner_participant_ids or sender in policy.master_user_ids:
        return ""
    if scope.startswith("group:") or scope == "private:" + sender:
        return sender
    return ""


def promote_participant_event(store, policy, *, scope, source_id, owner_scope="private:master", now):
    """复制一条有效原件；返回是否新建副本，不修改原件或生成长期事实。"""
    if not policy.owner_participant_ids or not policy.master_user_ids or scope == owner_scope:
        return False
    with store._lock, store._connect() as conn:
        row = conn.execute("SELECT * FROM memory_events WHERE scope=? AND source_id=?",
                           (scope, source_id)).fetchone()
        if row is None:
            return False
        snapshot = MemorySourceReader(store)._snapshot(conn, "event:" + row["event_id"], scope, _now=now)
        if snapshot is None:
            return False
        origin, version, _ = snapshot
        meta = dict(origin.metadata)
        if meta.get("source") not in {"qq_turn", "qq_passive_observation"}:
            return False
        # 观察事件先入库时保留观察正文；正式回合只补充同一消息的作者和角色元数据。
        turn = conn.execute("SELECT metadata_json FROM memory_turns WHERE scope=? AND source_id=?",
                            (scope, source_id)).fetchone()
        if turn:
            detail = json.loads(turn["metadata_json"])
            if meta.get("sender_id") and detail.get("sender_id") != meta["sender_id"]:
                return False
            meta.update(detail)
        meta.setdefault("role", "user")
        participant = participant_id(policy, scope=scope, metadata=meta)
        if not participant:
            return False
        if meta["role"] == "assistant" and meta.get("delivery_status") != "sent":
            return False
        meta.update(owner_memory=True, owner_participant_id=participant,
                    owner_participant_label=policy.owner_participant_labels.get(participant, ""),
                    owner_source_scope=scope,
                    owner_origin_scope=scope, owner_origin_record_id=origin.record_id,
                    owner_origin_version=version)
        # 副本的事实性与原件一致；自动账号的“记住/忘记”只作为对话来源，不能变成主人指令。
        return store.record_event(conversation_id="master", scope=owner_scope, text=origin.text,
            source_id="owner:" + source_id, layer="episodic", created_at=origin.created_at,
            importance=origin.importance, confidence=origin.confidence,
            expires_at=row["expires_at"], metadata=meta)


def recall_participant_events(store, policy, *, query, scopes, mode, now, limit, date_window=None):
    """显式问到登记参与者时，先限定人物、原会话类型和日期，再选原始经历。

    通用摘要可能主要命中主人的旧聊天，故不能先选摘要再尝试寻找该人物。
    这里只在上游已经授权私用库的检索/摘要中生效；None 表示普通查询，空元组
    表示本次明确人物范围内没有有效来源，不能用无关经历补足。
    """
    from .query_context import mentions
    from dataclasses import replace
    import re

    owner_scope = "private:master"
    if mode not in {"retrieve", "summary"} or owner_scope not in scopes:
        return None
    peers = [peer for peer in policy.owner_participant_ids
             if mentions(query, peer) or (policy.owner_participant_labels.get(peer)
                 and mentions(query, policy.owner_participant_labels[peer]))]
    if not peers:
        return None
    clauses = ["scope=?", "status='active'", "layer='episodic'",
               "json_extract(metadata_json,'$.owner_participant_id') IN (" + ",".join("?" for _ in peers) + ")"]
    values = [owner_scope, *peers]
    if date_window is not None:
        clauses += ["created_at>=?", "created_at<?"]
        values.extend(date_window)
    private = bool(re.search(r"私聊|私信", query))
    group = bool(re.search(r"群聊|群里|群内", query))
    if private != group:
        clauses.append("json_extract(metadata_json,'$.owner_source_scope') LIKE ?")
        values.append("private:%" if private else "group:%")
    with store._lock, store._connect() as conn:
        rows = conn.execute("SELECT * FROM memory_events WHERE " + " AND ".join(clauses)
            + " ORDER BY created_at DESC,event_id DESC LIMIT ?",
            (*values, policy.event_scan_limit)).fetchall()
    ranked = [store._score_event(row, store._terms(query), now) for row in rows]
    ranked.sort(key=(lambda r: (r.created_at, r.score)) if mode == "summary"
                else (lambda r: (r.score, r.created_at)), reverse=True)
    # 来源校验放在截断之前，旧副本失效不能挤掉仍有效的记录。
    fresh = MemorySourceReader(store).read_sources(
        requests=tuple((row.record_id, owner_scope, "") for row in ranked), now=now)
    return tuple(replace(current, score=old.score) for old, current in zip(ranked, fresh)
                 if current is not None)[:limit]
