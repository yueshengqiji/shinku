"""Non-authoritative expression archive in the existing task store.

It does not change business task revisions or schedule work. Creation is
explicit; default hosts never migrate a production DB for this feature.
"""
from __future__ import annotations

import json

from shinku.agent_runtime.contracts import canonical_hash
from .facts import archive_result
from .models import PARAGRAPH_PRESENTATION_VERSION


class ExpressionArchive:
    def __init__(self, task_store, *, max_entries=4096):
        self.store = task_store
        from shinku.agent_runtime.result_events import ResultEventStore
        self.events = task_store.result_events or ResultEventStore(task_store)
        self.max_entries = max(1, min(16384, max_entries))
        with self.store._transaction() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS agent_result_expressions (
                expression_id TEXT PRIMARY KEY, scope TEXT NOT NULL, payload TEXT NOT NULL)""")

    @staticmethod
    def _scope(scope):
        return canonical_hash({"requester": scope.requester, "session": scope.session})

    @staticmethod
    def _encode(record):
        record["record_hash"] = canonical_hash({k: v for k, v in record.items() if k != "record_hash"})
        return json.dumps(record, ensure_ascii=False)

    @staticmethod
    def _decode(payload):
        record = json.loads(payload)
        if (canonical_hash({k: v for k, v in record.items() if k != "record_hash"}) != record.get("record_hash")
                or canonical_hash(record["canonical_result"]) != record.get("archive_projection_hash")):
            raise ValueError("expression_archive_integrity_failed")
        if "presentation_projection" in record and (
                canonical_hash(record["presentation_projection"]) != record.get("presentation_projection_hash")
                or record["presentation_projection"].get("source_result_hash") != record["result_hash"]):
            raise ValueError("expression_presentation_integrity_failed")
        return record

    def get(self, expression_id, *, scope):
        with self.store._transaction() as db:
            row = db.execute("SELECT payload FROM agent_result_expressions WHERE expression_id=? AND scope=?",
                             (expression_id, self._scope(scope))).fetchone()
        return self._decode(row[0]) if row else None

    def claim(self, expression_id, *, scope, result, config_hash, binding_hash, processing_seconds=140,
              format_version=1, detail_policy_version=0,
              paragraph_presentation_version=PARAGRAPH_PRESENTATION_VERSION, semantic_review_version=0,
              content_policy="complete"):
        if content_policy not in {"complete", "selective"}:
            raise ValueError("invalid_expression_content_policy")
        if (type(semantic_review_version) is not int or semantic_review_version not in (0, 1)
                or semantic_review_version and format_version != 2):
            raise ValueError("invalid_expression_semantic_review")
        with self.store._transaction() as db:
            row = db.execute("SELECT scope, payload FROM agent_result_expressions WHERE expression_id=?",
                             (expression_id,)).fetchone()
            if row:
                if row[0] != self._scope(scope):
                    raise ValueError("expression_scope_collision")
                return self._decode(row[1]), False
            if db.execute("SELECT count(*) FROM agent_result_expressions").fetchone()[0] >= self.max_entries:
                # Never evict the tombstone of a current result and silently
                # purchase another call. Event retention belongs to P4a.
                raise ValueError("expression_archive_capacity")
            canonical = archive_result(result)
            record = {"expression_id": expression_id, "result_hash": result["result_hash"],
                "binding_hash": binding_hash, "config_hash": config_hash, "canonical_result": canonical,
                "archive_projection_hash": canonical_hash(canonical), "status": "processing", "text": "",
                "check": {"accepted": False, "reason": "pending"}, "fallback_reason": "",
                "candidate_hash": "", "candidate": None, "usage": {},
                "delivery": {"status": "not_sent", "sent": False}, "created": self.store.clock(),
                "processing_until": self.store.clock() + processing_seconds}
            if format_version == 2:
                record["format_version"] = 2
                record["paragraph_presentation_version"] = paragraph_presentation_version
            if detail_policy_version:
                record["detail_policy_version"] = detail_policy_version
            if semantic_review_version:
                record["semantic_review_version"] = semantic_review_version
            if content_policy != "complete":
                record["content_policy"] = content_policy
            db.execute("INSERT INTO agent_result_expressions VALUES (?, ?, ?)",
                       (expression_id, self._scope(scope), self._encode(record)))
            return record, True

    def update(self, expression_id, *, scope, change):
        with self.store._transaction() as db:
            row = db.execute("SELECT payload FROM agent_result_expressions WHERE expression_id=? AND scope=?",
                             (expression_id, self._scope(scope))).fetchone()
            if not row:
                raise ValueError("expression_record_missing")
            record = self._decode(row[0])
            change(record)
            db.execute("UPDATE agent_result_expressions SET payload=? WHERE expression_id=? AND scope=?",
                       (self._encode(record), expression_id, self._scope(scope)))
            return record

    def finish(self, expression_id, *, scope, **fields):
        def update(record):
            # A late response may add usage, but never replace timeout fallback
            # text or revive a discarded candidate.
            if record["status"] == "processing":
                record.update(fields)
        return self.update(expression_id, scope=scope, change=update)

    def bind_presentation(self, expression_id, *, scope, projection, binding_hash):
        def change(record):
            if record["status"] != "processing" or "presentation_projection" in record:
                raise ValueError("expression_presentation_already_bound")
            record.update(presentation_projection=projection, presentation_projection_hash=canonical_hash(projection),
                          binding_hash=binding_hash)
        return self.update(expression_id, scope=scope, change=change)

    def usage(self, expression_id, *, scope, channel, usage):
        def change(record):
            previous = record["usage"].get(channel, {})
            merged = {**previous, **usage}
            # A host timeout can race a late SDK completion. Keep known counts
            # and the first failure while status/candidate remain terminal.
            if previous.get("known") is True and usage.get("known") is False:
                merged.update(previous)
            if previous.get("failure"):
                merged["failure"] = previous["failure"]
            record["usage"][channel] = merged
        return self.update(expression_id, scope=scope, change=change)

    def reserve_delivery(self, expression_id, *, scope, text):
        accepted = False
        def change(record):
            nonlocal accepted
            if (record["status"] in {"ready", "fallback"} and record["text"] == text
                    and record["delivery"]["status"] == "not_sent"):
                record["delivery"] = {"status": "sending", "sent": False}
                record["delivery_started"] = self.store.clock()
                accepted = True
        self.update(expression_id, scope=scope, change=change)
        return accepted

    def recover_interrupted(self, expression_id, *, scope):
        """Expired generation falls back once; an unknown send never resends.

        Called only by the leased event consumer. Ordinary concurrent prepare
        calls cannot take over an active generation or publication.
        """
        from .facts import fact_blocks, template
        from .natural import paragraph_blocks
        def change(record):
            if record["status"] == "processing" and self.store.clock() >= record.get("processing_until", float("inf")):
                source, projection = record["canonical_result"], record.get("presentation_projection")
                from .selective import fallback_text
                if record.get("content_policy") == "selective":
                    text = fallback_text(source)
                else:
                    blocks = (paragraph_blocks(source, projection, detail_policy_version=record.get("detail_policy_version", 0),
                              presentation_version=record.get("paragraph_presentation_version", 0))
                              if record.get("format_version") == 2 else fact_blocks(source, projection))
                    text = template(blocks)
                record.update(status="fallback", text=text,
                    check={"accepted": False, "reason": "expression_interrupted", "mode": "immutable_template"},
                    fallback_reason="expression_interrupted", candidate=None)
                record["usage"].setdefault("expression", {"calls": 1, "known": False, "cost_known": False})
            if record["delivery"]["status"] == "sending":
                record["delivery"] = {"status": "uncertain", "sent": False,
                    "reason": "publication_interrupted", "message_ids": []}
        return self.update(expression_id, scope=scope, change=change)

    def delivery(self, expression_id, *, scope, receipt):
        status = str(receipt.get("status") or "uncertain")
        if status not in {"sent", "failed", "uncertain", "partial", "dry_run", "suppressed"}:
            status = "uncertain"
        confirmed = status == "sent" and receipt.get("sent") is True
        if status == "sent" and not confirmed:
            status = "uncertain"
        safe = {"status": status, "sent": confirmed,
                "message_ids": [str(v)[:256] for v in receipt.get("message_ids", [])[:32]]}
        def change(record):
            if record["delivery"]["status"] == "sending":
                record["delivery"] = safe
        return self.update(expression_id, scope=scope, change=change)
