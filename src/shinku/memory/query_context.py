"""服务层的有界查询衔接：前文只定位，别名只桥接一次，不写入事实。

入口为 service_history_selection；候选仍由 Store 搜索并经现有来源回读交付。
工作台持有短期版本句柄，本模块不持久化查询、原文或别名图。
"""
from __future__ import annotations

from dataclasses import replace
import re

from .source_reader import MemorySourceReader

_REFERENCE = re.compile(r"^(?:那|这)?(?:刚才(?:的|那个|那件事)?|前面那个|那个|它|接着刚才|继续刚才)")
_SWITCH = re.compile(r"另外|换个话题|不说这个|顺便|忘记|忘掉|删除|清除")
_NAME = r"[\w\u4e00-\u9fff-]{2,24}"
_ALIAS = re.compile(rf"(?:^|[。；;\n])\s*(?:原来的)?(?P<old>{_NAME}?)(?:后来)?(?:改称|改名为|更名为|又名|也叫)\s*(?P<new>{_NAME})")
_UNCERTAIN = re.compile(r"不是|并非|不叫|不再|没有|不要|未改|未曾|可能|也许|据说|听说|他说|她说|你说|转述|比如|如果|假如|假设|是否|吗|[？?“”\"「」]")


def short_reference(query: str) -> bool:
    """只承接短指代；显式换题、日期回顾和删除请求仍走原路由。"""
    text = str(query).strip()
    return bool(len(text) <= 80 and _REFERENCE.match(text) and not _SWITCH.search(text)
                and not re.search(r"昨天|前天|上周|那天|\d{1,2}月\d{1,2}|历史记录", text))


def mentions(text: str, name: str) -> bool:
    # ASCII 编号须守住边界，K8 不关联 K80；中文名称允许自然助词衔接。
    pattern = re.escape(name)
    if name.isascii():
        pattern = r"(?<![\w.-])" + pattern + r"(?![A-Za-z0-9_.-])"
        return bool(re.search(pattern, text, re.ASCII))
    return name in text


def alias_queries(query: str, record):
    """只接受本人直接陈述的明确命名，不把推测、引用或助手说法当关联。"""
    if (record.metadata.get("role") != "user" or not record.metadata.get("sender_id")
            or record.metadata.get("reading_representation") != "source"
            or _UNCERTAIN.search(record.text)):
        return ()
    pairs = []
    for match in _ALIAS.finditer(record.text):
        old, new = match.group("old", "new")
        if old == new:
            continue
        for present, other in ((old, new), (new, old)):
            if mentions(query, present) and not mentions(query, other):
                pattern = (r"(?<![A-Za-z0-9_.-])" + re.escape(present) + r"(?![A-Za-z0-9_.-])"
                           if present.isascii() else re.escape(present))
                pairs.append((present, other, re.sub(pattern, lambda _: other, query)))
    return tuple(pairs[:2])


def search_context(store, *, query, scopes, now, limit, anchors=(), **kwargs):
    """原查询优先，一次前文扩展、最多两条明确别名扩展，共享结果预算。

    不改编码器查询指令。扩展文本只是检索线索；所有输出都是原始来源。
    只在同一已授权 scope 内桥接，别名原文和关联资料必须成对保留。
    """
    reader = MemorySourceReader(store)
    selected = {}
    base = []
    for scope in scopes:
        base.extend(store.search(query=query, scope=scope, now=now, limit=limit, **kwargs))
    base.sort(key=lambda row: (row.score, row.created_at), reverse=True)
    anchors = tuple(row for row in anchors if row.scope in scopes)[:limit]
    if anchors:
        fresh = reader.read_sources(requests=tuple((r.record_id, r.scope, r.metadata.get("source_version", ""))
                                                   for r in anchors), now=now)
        anchors = tuple(r for r in fresh if r is not None)
        # 多个前文可能指向不同对象，只保留证据供回答澄清，不擅自挑选一个实体。
        if len(anchors) == 1:
            anchor = anchors[0]
            expanded = (query[:80] + " " + anchor.text[:160]).strip()
            rows = store.search(query=expanded, scope=anchor.scope, now=now, limit=limit, **kwargs)
            if reader.read_source(record_id=anchor.record_id, scope=anchor.scope, now=now,
                                  expected_hash=anchor.metadata["source_version"]) is not None:
                base = list(rows) + base
        for row in anchors:
            selected.setdefault(row.record_id, row)
    fresh = reader.read_sources(requests=tuple((r.record_id, r.scope, r.metadata.get("source_version", ""))
                                               for r in base[:limit]), now=now)
    plans = []
    for row in fresh:
        if row is not None:
            plans.extend((row, *item) for item in alias_queries(query, row))
    # 同一新名称出现多种旧名称时，不猜测其中哪一个正确。
    ambiguous = {present for _, present, _, _ in plans
                 if len({(r.scope, other) for r, p, other, _ in plans if p == present}) > 1}
    for bridge, present, other, expanded in plans[:2]:
        if present in ambiguous or limit - len(selected) < 2:
            continue
        rows = store.search(query=expanded, scope=bridge.scope, now=now, limit=limit, **kwargs)
        if reader.read_source(record_id=bridge.record_id, scope=bridge.scope, now=now,
                              expected_hash=bridge.metadata["source_version"]) is None:
            continue
        # 一个名字往往对应多条资料，不能只留第一条就丢掉真正的条件。
        # 桥接及资料最多占本轮的一半再加一个槽，仍给原查询保留空间。
        linked_rows = [row for row in rows if row.record_id != bridge.record_id
                       and mentions(row.text, other) and not _ALIAS.search(row.text)]
        room = min(max(1, limit // 2), limit - len(selected) - 1)
        fresh_links = reader.read_sources(requests=tuple((row.record_id, row.scope,
            row.metadata.get("source_version", "")) for row in linked_rows[:room]), now=now)
        if any(row is not None for row in fresh_links):
            selected.setdefault(bridge.record_id, bridge)
            for linked in fresh_links:
                if linked is not None:
                    # 临时依赖随候选流转；交付时仍须展示同版本桥接原文。
                    selected.setdefault(linked.record_id, replace(linked, metadata={**linked.metadata,
                        "query_bridge": (bridge.record_id, bridge.scope, bridge.metadata["source_version"])}))
    for row in base:
        selected.setdefault(row.record_id, row)
    return tuple(selected.values())[:limit]


def recent_reference(service, *, scope, requester, query, recent_context, message_meta, now):
    """只接纳前文中逐字出现的本人来源或明确回复目标，裸上下文不成为证据。"""
    if not requester or not short_reference(query):
        return ()
    meta = dict(message_meta or {})
    reply_id = str(meta.get("reply_to_message_id") or "").strip()
    if reply_id:
        row = service.store.observation_for_message(scope=scope, message_id=reply_id)
        rows = (row,) if row is not None else ()
    else:
        rows = service.store.recent_events(scope=scope, now=now, limit=8,
            max_age_seconds=service.policy.workbench_ttl_seconds, include_assistant=False)
        rows = tuple(row for row in rows if row.text and row.text in str(recent_context)[-1800:])
    rows = tuple(row for row in rows if row.metadata.get("sender_id") == requester
                 and row.metadata.get("role") in {None, "", "user"}
                 and row.text != query and 0 <= now - row.created_at <= service.policy.workbench_ttl_seconds)
    return tuple(row for row in service.integrity_reader.read_sources(
        requests=tuple((r.record_id, scope, "") for r in rows[:2]), now=now) if row is not None)
