"""在既有消息/送达记录上承接 QQ 文件，定位符不授予新权限。

原始附件仍由平台读取器读取；本模块只沿同一账号、会话的真实回复边查找来源。
宿主将有界文件标识和版本写入已送达回复的 metadata，不另建全局“最近文件”库，
不保存文件正文。每次承接重新检查记忆原件的撤销/过期及实际文件字节版本。
"""
from collections.abc import Mapping
import re
import sqlite3

from .message import IncomingMessage

MAX_AGE = 1800
RECENT_AGE = 120


def _items(value, limit=2):
    return value[:limit] if isinstance(value, (list, tuple)) else ()


def _age(now, value, limit):
    return type(value) is int and 0 <= now - value <= limit


class FileSources:
    """仅持有本回合的读取预算与已核对节点，不跨请求缓存正文或授权。"""
    def __init__(self, turn, service, reader):
        self.turn, self.service, self.reader = turn, service, reader
        self.primary = next(m for m in turn.messages if m.message_id == turn.primary_message_id)
        self.scope = f"{turn.conversation_type}:{turn.conversation_id}"
        self.self_id = str(self.primary.raw_event.get("self_id") or "")
        self.now = int(service.clock())
        self.platform_reads = 0
        self.legacy_refs = {}
        self.gaps = []
        self.recent = service.store.recent_assistant_events(scope=self.scope, now=self.now,
            max_age_seconds=86400, limit=64)

    def fresh(self, record):
        if record is None:
            return None
        fresh = self.service.integrity_reader.read_source(record_id=record.record_id,
            scope=self.scope, now=self.now)
        return fresh if fresh is not None and fresh.metadata.get("reading_representation") == "source" else None

    def own_reply(self, record):
        meta = record.metadata
        return bool(self.self_id and meta.get("source") == "qq_turn" and meta.get("role") == "assistant"
            and meta.get("sender_id") == meta.get("self_id") == self.self_id
            and meta.get("delivery_status") in {"sent", "partial"}
            and isinstance(meta.get("delivery_message_ids"), (tuple, list))
            and meta["delivery_message_ids"] and meta.get("requester_id")
            and all(isinstance(mid, str) and mid.strip() and len(mid) <= 80 for mid in meta["delivery_message_ids"])
            and record.source_id == f"turn:{self.turn.conversation_id}:{meta.get('message_id')}:assistant"
            and (self.turn.conversation_type != "private" or meta["requester_id"] == self.primary.sender_id))

    def original(self, mid):
        return self.fresh(self.service.store.event_by_source_id(
            scope=self.scope, source_id=f"turn:{self.turn.conversation_id}:{mid}"))

    def refs(self, record):
        """兼容旧引用记录缺 file_id；只能回取同一个平台消息，不能按文件名扫描缓存。"""
        meta = record.metadata
        if (meta.get("source") not in {"qq_turn", "qq_passive_observation"}
                or meta.get("role") not in {None, "", "user"}
                or meta.get("self_id", self.self_id) != self.self_id):
            return []
        rows = [dict(row) for row in _items(meta.get("attachment_refs"), 12) if isinstance(row, Mapping)
            and row.get("kind") == "file" and row.get("quoted") != "true"
            and row.get("reference_kind", "direct") == "direct"][:2]
        cache_key = None
        if rows and any(not row.get("file_id") for row in rows):
            cache_key = (record.record_id, meta.get("source_version", ""))
            if cache_key in self.legacy_refs:
                return [dict(row) for row in self.legacy_refs[cache_key]]
            unavailable = {"status": "unavailable", "reason": "qq_file_reference_unavailable", "read_complete": False,
                "name": str(rows[0].get("name") or rows[0].get("media_id") or "")[:255],
                "source_message_id": str(meta.get("message_id") or "")}
            fetch = getattr(self.reader, "message_fetcher", None)
            if not callable(fetch) or self.platform_reads >= 2:
                self.gaps.append(unavailable)
                return []
            self.platform_reads += 1
            try:
                raw = fetch(str(meta.get("message_id") or ""))
            except (OSError, ValueError, TypeError):
                raw = None
            if not isinstance(raw, Mapping):
                self.gaps.append(unavailable)
                return []
            message = IncomingMessage.from_event(raw)
            if (message.conversation_type != self.turn.conversation_type
                    or message.conversation_id != self.turn.conversation_id
                    or message.message_id != str(meta.get("message_id") or "")
                    or message.sender_id != meta.get("sender_id")
                    or str(message.raw_event.get("self_id") or "") != self.self_id):
                self.gaps.append(unavailable)
                return []
            rows = [dict(row) for row in message.attachments if row.get("kind") == "file"][:2]
        result = [{**row, "source_record_id": record.record_id,
            "source_message_id": str(meta.get("message_id") or ""),
            "source_sender_id": str(meta.get("sender_id") or ""), "self_id": self.self_id,
            "scope": self.scope} for row in rows]
        if cache_key is not None:
            # 同一回合复用同一原件版本的已核对描述；不缓存正文，不借此跳过原件重验。
            self.legacy_refs[cache_key] = result
        return result

    def validate(self, source):
        if (not isinstance(source, Mapping) or source.get("scope") != self.scope
                or source.get("self_id") != self.self_id
                or not isinstance(source.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"])
                or not _age(self.now, source.get("first_read_at"), MAX_AGE)):
            return None
        record = self.service.integrity_reader.read_source(record_id=source.get("source_record_id", ""),
            scope=self.scope, now=self.now)
        if record is None:
            return None
        for ref in self.refs(record):
            if all(ref.get(key) == source.get(key) for key in
                   ("file_id", "name", "size", "source_message_id", "source_sender_id")):
                return dict(source)
        return None

    def linked(self, mid, depth=0, seen=()):
        if depth >= 4 or mid in seen:
            return []
        # 只把宿主实际送达的回复ID反查为助手节点；模型文字中的ID和文件名不参与查找。
        matches = [row for row in self.recent if self.own_reply(row)
            and mid in row.metadata["delivery_message_ids"]]
        nodes = [self.fresh(row) for row in matches] if matches else [self.original(mid)]
        result = []
        for node in filter(None, nodes):
            if self.own_reply(node):
                refs = node.metadata.get("qq_file_sources", ())
                if "qq_file_sources" in node.metadata:
                    if not isinstance(refs, (list, tuple)) or not refs:
                        self.gaps.append({"status": "unavailable", "reason": "qq_file_source_unavailable", "read_complete": False})
                    for row in _items(refs):
                        ref = self.validate(row)
                        if ref is not None:
                            result.append(ref)
                        else:
                            self.gaps.append({"status": "unavailable", "reason": "qq_file_source_unavailable", "read_complete": False})
                    continue  # 撤销/失效的已知来源不能降级沿旧引用重新铸造权限。
            else:
                declared_file = any(isinstance(row, Mapping) and row.get("kind") == "file"
                    for row in _items(node.metadata.get("attachment_refs"), 12))
                result.extend(self.refs(node))
                if declared_file:
                    continue
            parent = str(node.metadata.get("reply_to_message_id") or "")
            if parent:
                result.extend(self.linked(parent, depth + 1, (*seen, mid)))
        return result

    def select(self):
        if not self.self_id:
            return [], ""
        mids = tuple(self.primary.reply_message_ids)
        refs = [ref for mid in mids[:2] for ref in self.linked(mid)]
        relation = "continued"
        if not mids:
            relation = "recent_candidate"
            for old in self.recent:
                if (not self.own_reply(old) or old.metadata.get("requester_id") != self.primary.sender_id
                        or not _items(old.metadata.get("qq_file_sources"))):
                    continue
                node = self.fresh(old)
                if node is None:
                    continue
                for row in _items(node.metadata.get("qq_file_sources")):
                    if (isinstance(row, Mapping) and _age(self.now, row.get("first_read_at"), RECENT_AGE)
                            and (ref := self.validate(row)) is not None):
                        refs.append(ref)
        unique = {(row.get("source_record_id"), row.get("file_id"), row.get("sha256")): row for row in refs}
        return list(unique.values()), relation


def prepare_file_inputs(turn, reader, memory_service):
    """直接附件优先；承接只返回经原件重验的来源，多个候选不猜最新一份。"""
    rows = reader.read_turn(turn)
    primary = next((m for m in turn.messages if m.message_id == turn.primary_message_id), None)
    if primary is None or memory_service is None:
        return rows, ()
    try:
        return _with_sources(turn, reader, memory_service, rows)
    except (sqlite3.Error, OSError, ValueError):
        # 可选承接失败不破坏本轮已经取得的附件，也不能让普通无附件聊天整体报错。
        for row in rows:
            row.pop("_source", None)
        return rows, ()


def _with_sources(turn, reader, memory_service, rows):
    resolver = FileSources(turn, memory_service, reader)
    if not rows:
        refs, relation = resolver.select()
        if len(refs) > 1:
            return [{"status": "ambiguous", "reason": "qq_file_multiple_candidates", "read_complete": False,
                "candidates": [{"name": row.get("name", ""), "sha256": row.get("sha256", "")} for row in refs[:4]]}], ()
        if refs:
            rows = reader.read_turn(turn, continuation=refs, continuation_relation=relation)
        rows.extend(resolver.gaps[:2])
    sources = []
    for row in rows:
        source = row.pop("_source", None)
        if not source:
            continue
        record = resolver.original(source["source_message_id"])
        if record is None:
            continue
        # 新来源必须与实际入站记录一致；旧版本保留首次读取时间，闲聊不能无限续期。
        refs = resolver.refs(record)
        if not any(ref.get("file_id") == source.get("file_id") for ref in refs):
            continue
        sources.append({**source, "source_record_id": record.record_id, "scope": resolver.scope,
            "self_id": resolver.self_id, "first_read_at": source.get("first_read_at", resolver.now),
            "origin_version": record.metadata["source_version"]})
    return rows, tuple(sources)


def file_sources_current(turn, service):
    """发送前只重验原件撤销/账号边界，不重放平台下载，也不把历史读取当作新授权。"""
    if not turn.file_input_sources:
        return ""
    if service is None:
        return "qq_file_source_unavailable"
    primary = next((m for m in turn.messages if m.message_id == turn.primary_message_id), None)
    if primary is None:
        return "qq_file_source_unavailable"
    scope = f"{turn.conversation_type}:{turn.conversation_id}"
    for source in turn.file_input_sources:
        if (source.get("scope") != scope
                or source.get("self_id") != str(primary.raw_event.get("self_id") or "")):
            return "qq_file_source_unavailable"
        record = service.integrity_reader.read_source(record_id=source.get("source_record_id", ""),
            scope=scope, now=int(service.clock()))
        if (record is None or record.source_id != f"turn:{turn.conversation_id}:{source.get('source_message_id')}"
                or record.metadata.get("sender_id") != source.get("source_sender_id")
                or record.metadata.get("source_version") != source.get("origin_version")):
            return "qq_file_source_unavailable"
    return ""
