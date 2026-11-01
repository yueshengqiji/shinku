"""模型检索的来源目录：发现资料位置，不替模型猜话题或生成事实。

读取授权由 service 提供；本模块只返回经过原件校验的定位符和署名/时间。
目录不携带截断正文，跨轮锚点仍由同一投影的话题存储持有。
"""
from dataclasses import replace
import sqlite3

from .contracts import Deadline
from .sources import connection
from .view import admissible, time_iso


def source_catalog(projector, request, deadline, *, limit=8):
    """主人私聊的相关/近期来源目录；发现位置不等于读取正文或确认事实。"""
    owner_scope = "private:master"
    if not request.scope.startswith("private:") or owner_scope not in request.allowed_scopes or limit <= 0:
        return [], (), 0
    with connection(projector.store, deadline) as conn:
        current = projector.sources.turn_position(conn, request)
        cutoff = min(request.now, current["created_at"]) if current else request.now
        # 以原会话分组选少量候选，避免私聊新消息把另一端的资料位置全部挤掉。
        people = tuple(dict.fromkeys((request.requester, *request.participant_ids)))
        placeholders = ','.join('?' for _ in people)
        author_filter = ("(json_extract(metadata_json,'$.sender_id') IN ("+placeholders+") OR "
                         "json_extract(metadata_json,'$.requester_id') IN ("+placeholders+"))")
        origins = conn.execute("""SELECT json_extract(metadata_json,'$.owner_source_scope') AS origin,
                max(created_at) AS latest FROM memory_events
            WHERE scope=? AND status='active' AND created_at BETWEEN ? AND ?
              AND layer IN ('episodic','observation') AND source_id LIKE 'owner:turn:%'
              AND json_extract(metadata_json,'$.owner_memory')=1
              AND json_extract(metadata_json,'$.owner_source_scope')<>?
              AND """+author_filter+" GROUP BY origin ORDER BY latest DESC,origin LIMIT 4",
            (owner_scope, cutoff-projector.policy.recent_seconds, cutoff, request.scope,
             *people, *people)).fetchall()
        # 先保留各会话一个入口，避免同一会话的两条消息挤掉其他端的目录。
        primary, secondary = [], []
        for origin, _ in origins:
            rows = conn.execute("""SELECT event_id FROM memory_events WHERE scope=? AND status='active'
                AND source_id LIKE 'owner:turn:%' AND created_at BETWEEN ? AND ?
                AND json_extract(metadata_json,'$.owner_memory')=1
                AND json_extract(metadata_json,'$.owner_source_scope')=?
                AND """+author_filter+" ORDER BY created_at DESC,rowid DESC LIMIT 2",
                (owner_scope, cutoff-projector.policy.recent_seconds, cutoff, origin, *people, *people)).fetchall()
            if rows:
                primary.append(("event:"+rows[0][0], owner_scope, ""))
                secondary.extend(("event:"+row[0], owner_scope, "") for row in rows[1:])
    recent = (primary+secondary)[:min(8, limit)]
    related = ()
    # 当前主模型没有提出读取时，也给它少量跨时间的相关位置。复用本地索引，
    # 不做远端语义判别、索引维护或替模型生成答案；失败仍留下原有近期目录。
    # 紧预算也给相关位置留空间，同时保留各端的近期入口；无效候选不能吃掉全部回退位置。
    reserved_recent = min(len(primary), max(1, limit//2))
    remaining = max(0, min(8, limit-reserved_recent))
    if remaining and request.search_query.strip() and deadline.remaining > .05:
        search_deadline = Deadline(min(250, deadline.remaining*250))
        bounded = replace(request, end_at=min(request.end_at, cutoff+1) if request.end_at is not None else cutoff+1)
        try:
            related, _, _ = projector.archive.discover(bounded, search_deadline, limit=remaining)
        except (sqlite3.Error, TimeoutError, ValueError):
            related = ()
    related = related[:remaining]
    handles = tuple(dict.fromkeys([*related, *recent]))[:limit]
    rows = projector.sources.read(tuple(handles), request, deadline)
    live = {row.record_id:row for row in rows if row is not None and admissible(row, request)}
    related_ids = [h[0] for h in related if h[0] in live]
    # 窗口外的本人原话优先提供位置，助手原话仍可供命名/旧答复核对，不能升级为用户事实。
    rank = {rid:i for i,rid in enumerate(related_ids)}
    related_ids.sort(key=lambda rid:(live[rid].created_at>=cutoff-projector.policy.recent_seconds,
        live[rid].metadata.get('role')=='assistant',rank[rid]))
    selected = list(dict.fromkeys([*related_ids[:3],*(h[0] for h in recent if h[0] in live)]))[:min(8,limit)]
    catalog, dependencies = [], []
    for rid in selected:
        row=live[rid]
        meta = row.metadata
        # 原文必须由 memory_search 补读；目录不生成主题摘要，也不暗示已理解其内容。
        catalog.append(dict(record_id=row.record_id, scope=row.scope, source_version=meta["source_version"],
            origin_scope=meta.get("owner_source_scope",row.scope), saved_at=time_iso(row.created_at),
            author=str(meta.get("sender_id") or ""), role=str(meta.get("role") or "user"),
            discovery='query_related' if rid in related_ids[:3] else 'recent',
            source_kind=('assistant_utterance' if meta.get('role')=='assistant' else
                         'user_statement' if meta.get('role')=='user' else 'unknown_author'),
            content_available=True, content_not_read_in_prompt=True, execution_authority=False))
        dependencies.append((row.record_id, row.scope, meta["source_version"]))
    return catalog, tuple(dependencies), len(handles)
