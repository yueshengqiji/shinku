"""Explicit task-scoped retention and reads in the existing business SQLite.

    No URL fetching, model paths, executable state or automatic replay.
"""
from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

from .contracts import AgentBoundaryError, bounded_text, canonical_hash
from .information import content_hash, safe_source, source_text


@dataclass(frozen=True)
class MaterialPolicy:
    max_entries: int = 4096
    max_source_bytes: int = 1048576
    max_storage_bytes: int = 33554432
    max_refs: int = 64
    ttl_seconds: float = 3600
    inline_chars: int = 6000
    page_chars: int = 12000
    max_reads: int = 16
    max_read_chars: int = 98304
    read_deadline_seconds: float = 300
    max_images: int = 8
    max_image_bytes: int = 4194304

    def __post_init__(self):
        for name in ("max_entries", "max_source_bytes", "max_storage_bytes", "max_refs", "inline_chars",
                     "page_chars", "max_reads", "max_read_chars", "max_images", "max_image_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise AgentBoundaryError("invalid_material_policy")
        if (type(self.ttl_seconds) not in (int, float) or type(self.read_deadline_seconds) not in (int, float)
                or not math.isfinite(self.ttl_seconds) or not math.isfinite(self.read_deadline_seconds)
                or not 0 < self.ttl_seconds <= 7 * 86400 or not 0 < self.read_deadline_seconds <= 86400
                or self.max_refs > 128 or self.page_chars > 24000 or self.max_reads > 128
                or self.max_read_chars > 1048576 or self.max_source_bytes > 4194304
                or self.max_entries > 16384 or self.max_storage_bytes > 134217728
                or self.max_images > 16 or self.max_image_bytes > 8388608 or self.inline_chars > 24000):
            raise AgentBoundaryError("invalid_material_policy")


@dataclass(frozen=True)
class MaterialSource:
    """Only a trusted host constructs these; an AgentRequest cannot supply one."""
    source_id: str
    kind: str
    content: object = None
    provenance: Mapping = field(default_factory=dict)
    required: bool = True
    version: str = ""
    complete: bool = True
    local_path: str | Path | None = None
    revalidate: Callable[[], bool] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        bounded_text(self.source_id, 256, "invalid_material_source")
        if (not isinstance(self.kind, str) or self.kind not in _KINDS or type(self.required) is not bool or type(self.complete) is not bool
                or not isinstance(self.provenance, Mapping) or not isinstance(self.version, str)
                or len(self.version) > 256 or (self.revalidate is not None and not callable(self.revalidate))):
            raise AgentBoundaryError("invalid_material_source")


_KINDS = {"related_message", "document", "attachment", "image", "executor_result", "clarification", "caller_data"}
_PROVENANCE = {"message_id", "sender_id", "relation", "record_id", "source_version", "invocation_id"}


class TaskMaterialStore:
    def __init__(self, task_store, *, policy=None, approved_roots=()):
        self.store = task_store
        self.policy = policy or MaterialPolicy()
        if not isinstance(self.policy, MaterialPolicy):
            raise AgentBoundaryError("invalid_material_policy")
        self.approved_roots = tuple(Path(path).expanduser().resolve() for path in approved_roots)
        self._validators = {}
        with self.store._transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS agent_task_materials (
                reference TEXT PRIMARY KEY, registration_key TEXT UNIQUE NOT NULL,
                scope TEXT NOT NULL, task_id TEXT NOT NULL, expires REAL NOT NULL,
                content_bytes INTEGER NOT NULL, payload TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS agent_task_material_reads (
                read_id TEXT PRIMARY KEY, scope TEXT NOT NULL, task_id TEXT NOT NULL,
                reference TEXT NOT NULL, payload TEXT NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS agent_task_material_limits (
                task_id TEXT PRIMARY KEY, scope TEXT NOT NULL, payload TEXT NOT NULL)""")

    @staticmethod
    def scope(task):
        return canonical_hash([task["requester"], task["session"]])

    @staticmethod
    def encode(record):
        record["record_hash"] = canonical_hash({k: v for k, v in record.items() if k != "record_hash"})
        return json.dumps(record, ensure_ascii=False, allow_nan=False)

    @staticmethod
    def decode(payload):
        record = json.loads(payload)
        if canonical_hash({k: v for k, v in record.items() if k != "record_hash"}) != record.get("record_hash"):
            raise AgentBoundaryError("material_integrity_failed")
        return record

    def current(self, task, still_authorized):
        return self._current(task, still_authorized, allow_canceled=False)

    def _current(self, task, still_authorized, *, allow_canceled, require_same_revision=True):
        try:
            if still_authorized() is not True:
                raise AgentBoundaryError("material_authorization_denied")
            current = self.store.get(task["task_id"], requester=task["requester"], session=task["session"])
        except AgentBoundaryError:
            raise
        except Exception as exc:
            raise AgentBoundaryError("material_state_unavailable") from exc
        if current is None:
            raise AgentBoundaryError("material_task_unavailable")
        if current.get("execution_attempt", 0) != task.get("execution_attempt", 0):
            raise AgentBoundaryError("material_attempt_conflict")
        if not allow_canceled and (current.get("cancel_requested") or current.get("status") == "canceled"):
            raise AgentBoundaryError("material_task_canceled")
        if allow_canceled and require_same_revision and current.get("revision") != task.get("revision"):
            raise AgentBoundaryError("material_revision_conflict")
        return current

    def _file_content(self, path):
        target, text, digest, _ = self._file_observation(path)
        return target, text, digest

    def _file_observation(self, path):
        target = Path(path).expanduser().resolve()
        if not self.approved_roots or not any(target.is_relative_to(root) for root in self.approved_roots):
            raise AgentBoundaryError("material_path_outside_scope")
        try:
            with target.open("rb") as stream:
                raw = stream.read(self.policy.max_source_bytes + 1)
            if len(raw) > self.policy.max_source_bytes:
                raise AgentBoundaryError("material_source_oversized")
            text = raw.decode("utf-8-sig")
        except AgentBoundaryError:
            raise
        except UnicodeError as exc:
            raise AgentBoundaryError("material_attachment_format_unsupported") from exc
        except OSError as exc:
            raise AgentBoundaryError("material_source_unavailable") from exc
        return target, text, hashlib.sha256(raw).hexdigest(), len(raw)

    def _prepare(self, source):
        if not isinstance(source, MaterialSource) or source.kind not in _KINDS:
            raise AgentBoundaryError("host_material_source_required")
        bounded_text(source.source_id, 256, "invalid_material_source")
        if type(source.required) is not bool or type(source.complete) is not bool:
            raise AgentBoundaryError("invalid_material_source")
        origin = {}
        value = source.content
        if source.local_path is not None:
            path, value, original_hash = self._file_content(source.local_path)
            origin = {"kind": "local_file", "path": str(path), "sha256": original_hash}
        if value is None:
            raise AgentBoundaryError("material_source_unavailable")
        if source.kind == "image":
            if not isinstance(value, Mapping) or set(value) != {"data_url"}:
                raise AgentBoundaryError("material_image_invalid")
            match = re.fullmatch(r"data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=]+)", str(value["data_url"]))
            if not match or len(match[2]) > (self.policy.max_image_bytes + 2) * 4 // 3 + 4:
                raise AgentBoundaryError("material_image_invalid_or_oversized")
            try:
                raw = base64.b64decode(match[2], validate=True)
            except ValueError as exc:
                raise AgentBoundaryError("material_image_invalid") from exc
            if not raw or len(raw) > self.policy.max_image_bytes:
                raise AgentBoundaryError("material_image_invalid_or_oversized")
            media = {"mime": match[1], "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            safe = {"data_url": str(value["data_url"])}
        else:
            safe, media = safe_source(value), None
        encoded = source_text(safe)
        limit = self.policy.max_image_bytes * 2 if media else self.policy.max_source_bytes
        # 已由请求入口限制到16MiB的会话背景允许整份留存以供分页回读；总存储、
        # requester/session、版本、TTL及授权重验不变，普通附件不获得这项资源额度。
        if source.kind == "related_message" and source.source_id == "conversation-history" and not source.required and source.local_path is None:
            limit = min(16 * 1024 * 1024, self.policy.max_storage_bytes)
        if len(encoded.encode("utf-8")) > limit:
            raise AgentBoundaryError("material_source_oversized")
        provenance = {str(k): str(v) for k, v in source.provenance.items() if k in _PROVENANCE}
        if any(len(v) > 256 for v in provenance.values()) or len(source.version) > 256:
            raise AgentBoundaryError("invalid_material_source")
        return safe, safe_source(provenance), origin, media

    def retain(self, task, source, *, authorization_ref):
        self.prune()
        self.current(task, lambda: True)  # Registration is already a trusted host action.
        safe, provenance, origin, media = self._prepare(source)
        digest = content_hash(safe)
        registration = canonical_hash([self.scope(task), task["task_id"], source.source_id, source.kind,
                                       digest, source.version, origin, source.required, source.complete])
        now = self.store.clock()
        with self.store._transaction() as db:
            row = db.execute("SELECT payload FROM agent_task_materials WHERE registration_key=?", (registration,)).fetchone()
            if row:
                record = self.decode(row[0])
                if record["revoked"] or record["expires"] <= now:
                    raise AgentBoundaryError("material_source_expired_or_revoked")
            else:
                used, size = db.execute("SELECT count(*),coalesce(sum(content_bytes),0) FROM agent_task_materials").fetchone()
                if used >= self.policy.max_entries or size + len(source_text(safe).encode()) > self.policy.max_storage_bytes:
                    raise AgentBoundaryError("material_capacity")
                record = {"schema_version": 1, "reference": "tm:" + uuid.uuid4().hex,
                    "scope": self.scope(task), "task_id": task["task_id"], "source_id": source.source_id,
                    "kind": source.kind, "content": safe, "content_sha256": digest,
                    "version": source.version or digest, "origin": origin, "media": media,
                    "provenance": provenance, "required": source.required, "complete": source.complete,
                    "requires_revalidation": source.revalidate is not None,
                    "grant_ref": str(authorization_ref), "expires": now + min(self.policy.ttl_seconds, self.store.ttl_seconds),
                    "revoked": False}
                db.execute("INSERT INTO agent_task_materials VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (record["reference"], registration, self.scope(task), task["task_id"], record["expires"],
                     len(source_text(safe).encode()), self.encode(record)))
        if source.revalidate is not None:
            self._validators[record["reference"]] = source.revalidate
        return self.descriptor(record)

    def descriptor(self, record):
        text = "" if record["kind"] == "image" else source_text(record["content"])
        content = text[:self.policy.inline_chars]
        return {k: copy.deepcopy(record[k]) for k in ("reference", "source_id", "kind", "content_sha256", "version",
                "media", "provenance", "required", "complete", "expires")} | {
            "content": content, "content_chars": len(text), "omitted_chars": max(0, len(text) - len(content)),
            "retrieval": "available", "representation": "retained_safe_source"}

    def package(self, task, sources, *, authorization_ref, previous=None, gaps=()):
        if (not isinstance(sources, (tuple, list)) or len(sources) > self.policy.max_refs
                or any(not isinstance(row, MaterialSource) for row in sources)
                or not isinstance(gaps, (tuple, list)) or len(gaps) > 128):
            raise AgentBoundaryError("invalid_material_package")
        items = copy.deepcopy((previous or {}).get("items", []))
        # 当前描述符有界；旧的可选工具原件仅缩成身份索引，不删除原文或读取记录。
        archived = copy.deepcopy((previous or {}).get("archived_tool_refs", []))
        missing = []
        for row in gaps:
            if (not isinstance(row, Mapping) or set(row) - {"source_id", "required", "reason", "reference", "remaining_chars"}
                    or not isinstance(row.get("source_id"), str) or not isinstance(row.get("reason"), str)
                    or not re.fullmatch(r"[a-zA-Z0-9_:.-]{1,128}", row["reason"])
                    or type(row.get("required", True)) is not bool):
                raise AgentBoundaryError("invalid_material_gap")
            missing.append(safe_source(dict(row)))
        seen = {row["reference"] for row in items}
        for source in sources:
            try:
                evict = next((r for r in items if r.get("kind") == "executor_result"
                              and r.get("required") is False), None)
                if len(items) >= self.policy.max_refs and evict is None:
                    raise AgentBoundaryError("material_reference_limit")
                if len(items) >= self.policy.max_refs and len(archived) >= self.policy.max_entries:
                    raise AgentBoundaryError("material_archive_limit")
                if source.kind == "image" and sum(row["kind"] == "image" for row in items) >= self.policy.max_images:
                    raise AgentBoundaryError("material_image_limit")
                row = self.retain(task, source, authorization_ref=authorization_ref)
                if row["reference"] not in seen:
                    if len(items) >= self.policy.max_refs:
                        # 先成功保留新原件，再移动旧描述符；存储失败不得丢掉现有索引。
                        archived.append({k: evict[k] for k in ("reference", "content_sha256", "version")})
                        items.remove(evict)
                    items.append(row)
                    seen.add(row["reference"])
            except AgentBoundaryError as exc:
                missing.append({"source_id": getattr(source, "source_id", "unknown"),
                    "required": bool(getattr(source, "required", True)), "reason": exc.code})
        # 省略计数保留必需性；旧记录没有分类时仍视为未知，不能默认为可选。
        omitted_required = (previous or {}).get("omitted_required_gap_records",
                                                (previous or {}).get("omitted_gap_records", 0))
        omitted_required += sum(row.get("required", True) for row in missing[128:])
        return {"schema_version": 1, "task_id": task["task_id"], "scope_hash": self.scope(task),
                "items": items, "archived_tool_refs": archived, "gaps": missing[:128],
                "omitted_gap_records": max(0, len(missing) - 128) + (previous or {}).get("omitted_gap_records", 0),
                "omitted_required_gap_records": omitted_required,
                "source_count": len(items) + len(archived) + len(missing)}

    def prune(self):
        """Delete expired full sources/read bodies; manifests retain missing IDs.

        Called by host retention/reads, with a public explicit maintenance hook.
        No task revisions, model calls or background scheduler are involved.
        """
        # 待回答任务保留原件和读取回执供恢复/审计；只延长存储，不延长来源
        # 授权。_record仍拒绝过期/撤销材料，回答不能使旧登录或授权自动复活。
        pinned = "SELECT task_id FROM agent_conversation_tasks WHERE json_extract(payload, '$.status') = 'waiting_user'"
        condition = f"(expires<=? AND task_id NOT IN ({pinned})) OR task_id NOT IN (SELECT task_id FROM agent_conversation_tasks)"
        with self.store._transaction() as db:
            expired = [r[0] for r in db.execute("SELECT reference FROM agent_task_materials WHERE " + condition,
                                               (self.store.clock(),)).fetchall()]
            db.execute("DELETE FROM agent_task_materials WHERE " + condition, (self.store.clock(),))
            db.execute("DELETE FROM agent_task_material_reads WHERE reference NOT IN (SELECT reference FROM agent_task_materials) OR task_id NOT IN (SELECT task_id FROM agent_conversation_tasks)")
            db.execute("DELETE FROM agent_task_material_limits WHERE task_id NOT IN (SELECT task_id FROM agent_conversation_tasks)")
        for reference in expired:
            self._validators.pop(reference, None)
        return len(expired)

    def _record(self, reference, task):
        self.prune()
        if not isinstance(reference, str) or not re.fullmatch(r"tm:[a-f0-9]{32}", reference):
            raise AgentBoundaryError("material_reference_invalid")
        with self.store._transaction() as db:
            row = db.execute("SELECT payload FROM agent_task_materials WHERE reference=? AND scope=? AND task_id=?",
                (reference, self.scope(task), task["task_id"])).fetchone()
        if not row:
            raise AgentBoundaryError("material_reference_unavailable")
        record = self.decode(row[0])
        if record["scope"] != self.scope(task) or record["task_id"] != task["task_id"]:
            raise AgentBoundaryError("material_scope_conflict")
        if record["revoked"] or record["expires"] <= self.store.clock():
            raise AgentBoundaryError("material_source_expired_or_revoked")
        if content_hash(record["content"]) != record["content_sha256"]:
            raise AgentBoundaryError("material_content_integrity_failed")
        return record

    def resolve(self, reference, *, task, still_authorized):
        return self._resolve(reference, task=task, still_authorized=still_authorized, recovery=False)

    def resolve_for_read_only_view(self, reference, *, task, still_authorized):
        """仅供用户侧只读展示（执行日志面板）取已保留回执，不向调用者授予执行能力。

        与 resolve / check_retained_for_recovery 的区别与理由：
        - resolve 在任务已取消时拒绝（material_task_canceled），因为"能给模型读"必须跟随放弃执行；
          但用户回看自己刚被取消那轮的日志并不属于继续推进任务，不应因此变成空白。
        - check_retained_for_recovery 要求 revision 完全相等，因为它要支撑"重新排队"预检；
          而取消动作本身就会推进 revision（save 每次 +1），所以它不适用于取消后的展示。
        本条仍然保留服务端所有硬校验：调用方授权回调、作用域、任务存在性、当前 attempt、
        引用属于本任务、正文摘要与版本、未过期未撤销。只放宽两项：不因已取消而拒，
        不要求 revision 相等。返回正文的副本，不写库、不计数、不改任务状态。
        """
        return self._resolve(reference, task=task, still_authorized=still_authorized,
                             recovery=True, allow_canceled=True, require_same_revision=False)

    def check_retained_for_recovery(self, reference, *, task, still_authorized):
        """仅供宿主显式恢复预检：校验保留材料，不向调用者返回正文。

        已取消任务尚未重新排队，不能使用普通读取入口。此处只跳过取消标志，
        仍核对当前授权、任务轮次/版本、引用作用域、有效期及原文完整性；
        不改任务状态、读取回执或预算，也不赋予模型读取已取消材料的能力。
        """
        self._resolve(reference, task=task, still_authorized=still_authorized, recovery=True)

    def _resolve(self, reference, *, task, still_authorized, recovery, allow_canceled=None, require_same_revision=True):
        # allow_canceled 默认必须跟随 recovery：恢复预检（check_retained_for_recovery）本来就
        # 要能在已取消任务上校验保留材料，若默认成 False，恢复流程会被误判为 material_task_canceled。
        if allow_canceled is None:
            allow_canceled = recovery
        current = self._current(task, still_authorized, allow_canceled=allow_canceled,
                                require_same_revision=require_same_revision)
        package = current.get("material_package", {})
        references = {item["reference"]: item for item in package.get("items", [])}
        archived = {item["reference"]: item for item in package.get("archived_tool_refs", [])}
        references.update(archived)
        # Registration callers may use the package being saved; readers never
        # obtain arbitrary registered blobs merely by knowing another ID.
        if reference not in references:
            raise AgentBoundaryError("material_reference_not_in_task")
        record = self._record(reference, task)
        # 归档通道只容纳同任务的非必需工具结果，作用域、撤权及正文摘要仍照常核验。
        if reference in archived and (record["kind"] != "executor_result" or record["required"] is not False):
            raise AgentBoundaryError("material_archive_invalid")
        item = references[reference]
        if item.get("content_sha256") != record["content_sha256"] or item.get("version") != record["version"]:
            raise AgentBoundaryError("material_version_conflict")
        if record["requires_revalidation"]:
            validator = self._validators.get(reference)
            try:
                valid = callable(validator) and validator() is True
            except Exception:
                valid = False
            if not valid:
                raise AgentBoundaryError("material_origin_unavailable_or_changed")
        if record["origin"]:
            _, value, digest = self._file_content(record["origin"]["path"])
            if digest != record["origin"]["sha256"] or content_hash(safe_source(value)) != record["content_sha256"]:
                raise AgentBoundaryError("material_version_conflict")
        self._current(task, still_authorized, allow_canceled=recovery)
        return None if recovery else copy.deepcopy(record["content"])

    def revoke(self, reference, *, task):
        record = self._record(reference, task)
        record["revoked"] = True
        record["content"] = None
        with self.store._transaction() as db:
            db.execute("UPDATE agent_task_materials SET payload=?,content_bytes=0 WHERE reference=? AND scope=? AND task_id=?",
                (self.encode(record), reference, self.scope(task), task["task_id"]))
            db.execute("DELETE FROM agent_task_material_reads WHERE reference=? AND scope=? AND task_id=?",
                       (reference, self.scope(task), task["task_id"]))
        self._validators.pop(reference, None)

    def limits(self, task):
        with self.store._transaction() as db:
            row = db.execute("SELECT scope,payload FROM agent_task_material_limits WHERE task_id=?", (task["task_id"],)).fetchone()
            if row:
                if row[0] != self.scope(task):
                    raise AgentBoundaryError("material_scope_conflict")
                return self.decode(row[1])
            record = {"schema_version": 1, "reads": 0, "chars": 0,
                      "max_reads": self.policy.max_reads, "max_read_chars": self.policy.max_read_chars,
                      "page_chars": self.policy.page_chars,
                      "deadline": self.store.clock() + self.policy.read_deadline_seconds}
            db.execute("INSERT INTO agent_task_material_limits VALUES (?, ?, ?)",
                       (task["task_id"], self.scope(task), self.encode(record)))
            return record

    def read(self, reference, *, task, still_authorized, offset=0, length=6000, provided=True):
        if (type(offset) is not int or offset < 0 or type(length) is not int
                or not 1 <= length <= self.policy.page_chars):
            raise AgentBoundaryError("material_read_range_invalid")
        source = self.resolve(reference, task=task, still_authorized=still_authorized)
        record = self._record(reference, task)
        if record["kind"] == "image":
            raise AgentBoundaryError("material_image_requires_media_input")
        text = source_text(source)
        if offset > len(text):
            raise AgentBoundaryError("material_read_range_invalid")
        read_id = canonical_hash([self.scope(task), task["task_id"], reference, offset, length])
        self.limits(task)
        with self.store._transaction() as db:
            budget = self.decode(db.execute("SELECT payload FROM agent_task_material_limits WHERE task_id=?",
                                           (task["task_id"],)).fetchone()[0])
            if self.store.clock() > budget["deadline"]:
                raise AgentBoundaryError("material_read_deadline")
            row = db.execute("SELECT payload FROM agent_task_material_reads WHERE read_id=? AND scope=? AND task_id=?",
                             (read_id, self.scope(task), task["task_id"])).fetchone()
            if row:
                receipt = self.decode(row[0])
            else:
                content = text[offset:offset + length]
                if (budget["reads"] >= min(self.policy.max_reads, budget.get("max_reads", self.policy.max_reads))
                        or budget["chars"] + len(content) > min(self.policy.max_read_chars, budget.get("max_read_chars", self.policy.max_read_chars))
                        or length > budget.get("page_chars", self.policy.page_chars)):
                    raise AgentBoundaryError("material_read_budget_exhausted")
                receipt = {"schema_version": 1, "read_id": read_id, "reference": reference,
                    "provided": provided is True,
                    "content_sha256": record["content_sha256"], "version": record["version"],
                    "offset": offset, "content": content, "returned_chars": len(content), "total_chars": len(text),
                    "next_offset": offset + len(content) if offset + len(content) < len(text) else None,
                    "remaining_chars": max(0, len(text) - offset - len(content)),
                    "page_sha256": hashlib.sha256(content.encode()).hexdigest(), "source": "host_material_reader"}
                budget["reads"] += 1
                budget["chars"] += len(content)
                db.execute("UPDATE agent_task_material_limits SET payload=? WHERE task_id=? AND scope=?",
                           (self.encode(budget), task["task_id"], self.scope(task)))
                db.execute("INSERT INTO agent_task_material_reads VALUES (?, ?, ?, ?, ?)",
                           (read_id, self.scope(task), task["task_id"], reference, self.encode(receipt)))
        self.resolve(reference, task=task, still_authorized=still_authorized)
        return receipt

    def mark_provided(self, receipt, *, task, still_authorized):
        self.resolve(receipt["reference"], task=task, still_authorized=still_authorized)
        with self.store._transaction() as db:
            row = db.execute("SELECT payload FROM agent_task_material_reads WHERE read_id=? AND scope=? AND task_id=?",
                             (receipt["read_id"], self.scope(task), task["task_id"])).fetchone()
            if not row:
                raise AgentBoundaryError("material_read_receipt_unavailable")
            record = self.decode(row[0])
            if record["page_sha256"] != receipt["page_sha256"]:
                raise AgentBoundaryError("material_read_integrity_failed")
            record["provided"] = True
            db.execute("UPDATE agent_task_material_reads SET payload=? WHERE read_id=?", (self.encode(record), receipt["read_id"]))

    def read_receipts(self, task):
        self.prune()
        with self.store._transaction() as db:
            rows = db.execute("SELECT payload FROM agent_task_material_reads WHERE scope=? AND task_id=?",
                              (self.scope(task), task["task_id"])).fetchall()
        return [self.decode(row[0]) for row in rows]

    def verified_file_read(self, receipt, *, task, still_authorized):
        """Attest a complete, delivered local-file read from host-owned pages.

        Material references and caller-supplied receipt fields cannot select a
        path or supply read evidence. The retained origin and durable pages do.
        """
        if not isinstance(receipt, Mapping) or not isinstance(receipt.get("read_id"), str):
            return {}
        reference = receipt.get("reference")
        source = self.resolve(reference, task=task, still_authorized=still_authorized)
        record = self._record(reference, task)
        origin = record.get("origin", {})
        if origin.get("kind") != "local_file" or record.get("complete") is not True:
            return {}
        text = source_text(source)
        target, original, digest, size = self._file_observation(origin["path"])
        # A redacted or transformed representation is not the complete file.
        if original != text or digest != origin.get("sha256"):
            return {}
        pages = [page for page in self.read_receipts(task) if page.get("reference") == reference]
        observed = next((page for page in pages if page.get("read_id") == receipt["read_id"]), None)
        fields = ("read_id", "reference", "content_sha256", "version", "offset", "returned_chars",
                  "total_chars", "page_sha256", "source")
        if (not observed or observed.get("provided") is not True or any(
                type(receipt.get(key)) is not type(observed.get(key)) or receipt.get(key) != observed.get(key)
                for key in fields)):
            return {}
        ranges = []
        for page in pages:
            if page.get("provided") is not True:
                continue
            start, count, content = page.get("offset"), page.get("returned_chars"), page.get("content")
            if (type(start) is not int or type(count) is not int or start < 0 or count < 0
                    or start + count > len(text) or not isinstance(content, str)
                    or page.get("total_chars") != len(text) or page.get("source") != "host_material_reader"
                    or page.get("version") != record["version"]
                    or page.get("content_sha256") != record["content_sha256"]
                    or count != len(content) or content != text[start:start + count]
                    or page.get("page_sha256") != hashlib.sha256(content.encode("utf-8")).hexdigest()):
                return {}
            ranges.append((start, start + count))
        end = 0
        for start, stop in sorted(ranges):
            if start > end:
                return {}
            end = max(end, stop)
        if not ranges or end != len(text):
            return {}
        if self.resolve(reference, task=task, still_authorized=still_authorized) != source:
            return {}
        return {"verified_effects": ["read"], "completion_targets": [str(target)],
                "sha256": digest, "bytes": size, "read_complete": True,
                "result_valid": True, "file_text": text,
                "material_read_ids": [page["read_id"] for page in pages if page.get("provided") is True]}


from .material_session import TaskMaterialSession  # noqa: E402 - public convenience export
