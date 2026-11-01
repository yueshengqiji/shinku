"""主人短期来源的存储适配层，由 MemoryService 的公开入口调用。

本模块持有查询结构与原件关系的知识，复用 ProjectionSources 的有效来源校验。
上层只接收本次读取的记录和版本依赖；快照不可当作可跨轮复用的授权或文本缓存。
读取使用有截止时间的单一只读事务，不写入、强化、过期清理或创建后台任务。
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import MemoryRecord
from .projection.contracts import Deadline, ProjectionRequest
from .projection.sources import ProjectionSources, connection
from .projection.view import admissible


@dataclass(frozen=True)
class OwnerSourceBatch:
    """按新到旧排列的有效来源；版本依赖包含所有候选，不受展示预算裁剪。"""

    read_at: int
    window_seconds: int
    records: tuple[MemoryRecord, ...]
    versions: tuple[tuple[str, str, str], ...]
    selection_at: int = 0


def read_recent_owner_sources(service, *, conversation_id, conversation_type,
                              requester, self_id, message_id, selection_at=None):
    """按当前身份和策略重读主人来源；群聊、非主人或关闭策略返回 None。

    SQL 和读取器内部接口只在此适配边界使用，调用方不提供额外授权范围。
    五分钟默认窗口仅限制自动承接；本轮选材时刻可固定，真实TTL、身份与原件仍按当前时间重核。
    """
    if (conversation_type not in {"private", "desktop"}
            or not service.is_master_user(requester)):
        return None
    seconds = max(0, min(1800, int(service.policy.owner_continuity_seconds)))
    if not seconds:
        return None
    now = int(service.clock())
    selected_at = now if selection_at is None else selection_at
    if type(selected_at) is not int or not 0 <= selected_at <= now:
        raise ValueError("invalid_owner_selection_time")
    scope = service.scope_for(conversation_id, conversation_type)
    owner_scope = service.OWNER_SCOPE
    request = ProjectionRequest(scope, tuple(dict.fromkeys((scope, owner_scope))),
                                requester, self_id, "", message_id, now)
    reader, deadline = ProjectionSources(service.store), Deadline(200)
    current = f"owner:turn:{conversation_id}:{message_id}"
    records, versions = [], []
    with connection(service.store, deadline) as conn:
        # 选材下界固定在本轮初读时刻，避免生成十几秒就因自然滑窗失去原依赖。
        # 消息到达上界继续防排队串轮；新增来源仍参与版本比较，TTL/撤权按真实now检查。
        position = reader.turn_position(conn, request)
        cutoff = min(selected_at, position["created_at"]) if position is not None else selected_at
        rows = conn.execute("""SELECT event_id FROM memory_events
            WHERE scope=? AND status='active' AND layer IN ('episodic','observation')
              AND created_at>? AND created_at<=? AND source_id<>?
              AND source_id LIKE 'owner:turn:%'
              AND json_extract(metadata_json,'$.owner_memory')=1
              AND ((json_extract(metadata_json,'$.role')='user'
                    AND json_extract(metadata_json,'$.sender_id')=?)
                OR (json_extract(metadata_json,'$.role')='assistant'
                    AND json_extract(metadata_json,'$.requester_id')=?))
            ORDER BY created_at DESC,rowid DESC LIMIT 16""",
            (owner_scope, selected_at - seconds, now, current, requester, requester)).fetchall()
        for item in rows:
            deadline.check()
            snapshot = reader._snapshot(conn, "event:" + item["event_id"], owner_scope, _now=now)
            record = reader._reading_source(snapshot) if snapshot else None
            if (record is None or not admissible(record, request)
                    or record.metadata.get("reading_representation") != "source"
                    or not record.metadata.get("owner_source_scope")
                    or record.created_at <= selected_at - seconds):
                continue
            # 共享副本仍依赖原件的署名、送达状态和版本；相同正文不足以授权使用。
            origin = conn.execute("SELECT event_id FROM memory_events WHERE scope=? AND source_id=?",
                (record.metadata["owner_source_scope"], record.source_id.removeprefix("owner:"))).fetchone()
            original = reader._snapshot(conn, "event:" + origin["event_id"],
                record.metadata["owner_source_scope"], _now=now) if origin else None
            receipt_keys = ("role", "sender_id", "requester_id", "self_id", "delivery_status",
                            "delivery_message_ids", "delivery_content_complete", "delivery_channel")
            if (original is None or not admissible(original[0], request)
                    or any(record.metadata.get(key) != original[0].metadata.get(key) for key in receipt_keys)):
                continue
            # 迟于到达时间的记录不进入提示或来源句柄，避免与校验上界冲突。
            # 仍跟踪所有有效候选的版本：生成期间新增更正不能被展示过滤隐藏。
            if record.created_at <= cutoff:
                records.append(record)
            versions.append((record.record_id, record.metadata["source_version"], original[1]))
    return OwnerSourceBatch(now, seconds, tuple(records), tuple(versions), selected_at)
