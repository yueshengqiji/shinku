"""Bounded conversational task state, separate from persona/long-term memory.

模型解释不授予权限或执行证明。用户目标与硬条件只能追加；宿主登记的
说明性调查计划允许修订，旧原件由保存事务保留，不能抹掉执行回执。
Restoring this database does not replay a tool or restore permission grants.
"""

from __future__ import annotations

import copy
import json
import re
import sqlite3
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class TaskContractError(ValueError):
    def __init__(self, code: str, *, field_path: str = "", phase: str = "validation"):
        super().__init__(code)
        self.field_path = field_path
        self.phase = phase


def _text(value: Any, limit: int = 2000, *, field_path: str = "") -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise TaskContractError("invalid_or_oversized_text", field_path=field_path)
    return value.strip()


def _texts(value: Any, *, limit: int = 16, nonempty: bool = False, field_path: str = "") -> list[str]:
    if not isinstance(value, list) or len(value) > limit or (nonempty and not value):
        raise TaskContractError("invalid_text_list", field_path=field_path)
    return list(dict.fromkeys(_text(item, field_path=f"{field_path}[{i}]" if field_path else "") for i, item in enumerate(value)))


def validate_contract(value: Any) -> dict[str, Any]:
    """Validate shape/bounds, not claim perfect semantic goal coverage."""
    if not isinstance(value, Mapping):
        raise TaskContractError("contract_missing", field_path="task_contract")
    raw = value.get("requirements")
    if not isinstance(raw, list) or not raw or len(raw) > 24:
        raise TaskContractError("requirements_missing_or_oversized", field_path="requirements")
    requirements = []
    for index, item in enumerate(raw):
        path = f"requirements[{index}]"
        if not isinstance(item, Mapping):
            raise TaskContractError("invalid_requirement", field_path=path)
        requirements.append({
            "id": _text(item.get("id"), 64, field_path=path + ".id"),
            "description": _text(item.get("description"), field_path=path + ".description"),
            "effect": _text(item.get("effect"), 64, field_path=path + ".effect"),
            "targets": _texts(item.get("targets", []), field_path=path + ".targets"),
        })
        if "target_selector" in item:
            selector = item["target_selector"]
            if (not isinstance(selector, Mapping) or set(selector) - {"kind", "value", "suffixes"}
                    or selector.get("kind") not in {"file_name", "file_stem"}
                    or (requirements[-1]["effect"] not in {"delete", "write", "send", "document_translation"}
                        and not requirements[-1]["effect"].startswith("tool:"))
                    or requirements[-1]["targets"]):
                raise TaskContractError("invalid_target_selector")
            name = _text(selector.get("value"), 255)
            if any(c in name for c in '/\\:*?"<>|') or any(ord(c) < 32 for c in name) or name in {".", ".."}:
                raise TaskContractError("invalid_target_selector")
            normalized = {"kind": selector["kind"], "value": name}
            if "suffixes" in selector:
                suffixes = _texts(selector["suffixes"], limit=16, nonempty=True)
                if any(not re.fullmatch(r"(?:\.[A-Za-z0-9]{1,16}){1,3}", suffix) for suffix in suffixes):
                    raise TaskContractError("invalid_target_selector")
                normalized["suffixes"] = suffixes
            requirements[-1]["target_selector"] = normalized
        if "checks" in item:
            checks = item["checks"]
            allowed = {"contains", "sha256", "min_bytes", "http_status", "page_contains", "receipt_required", "required_fields", "min_sources", "require_dates", "date_kinds", "source_pages", "unchanged", "command", "exit_code"}
            if not isinstance(checks, Mapping) or set(checks) - allowed:
                raise TaskContractError("invalid_result_checks", field_path=path + ".checks")
            for key, val in checks.items():
                check_path = path + ".checks." + key
                if key in {"contains", "page_contains", "sha256", "command"}:
                    _text(val, 2000, field_path=check_path)
                elif key == "required_fields":
                    _texts(val, limit=8, nonempty=True, field_path=check_path)
                elif key == "date_kinds":
                    kinds = _texts(val, limit=3, nonempty=True, field_path=check_path)
                    if (set(kinds) - {"event", "publication", "updated"} or checks.get("require_dates") is not True
                            or requirements[-1]["effect"] != "research_answer"):
                        raise TaskContractError("invalid_result_checks", field_path=check_path)
                elif key in {"receipt_required", "require_dates", "unchanged"}:
                    if type(val) is not bool:
                        raise TaskContractError("invalid_result_checks", field_path=check_path)
                    if key == "receipt_required" and val and requirements[-1]["effect"] != "send":
                        raise TaskContractError("receipt_check_requires_send_effect", field_path=check_path)
                    if key == "unchanged" and requirements[-1]["effect"] != "read":
                        raise TaskContractError("unchanged_check_requires_read_effect", field_path=check_path)
                elif key == "min_sources":
                    if type(val) is not int or not 1 <= val <= (32 if requirements[-1]["effect"] == "local_answer" else 8):
                        raise TaskContractError("invalid_result_checks", field_path=check_path)
                elif key == "source_pages":
                    if type(val) is not int or not 1 <= val <= 100 or requirements[-1]["effect"] != "document_translation":
                        raise TaskContractError("invalid_document_source_pages_check", field_path=check_path)
                elif type(val) is not int or val < 0:
                    raise TaskContractError("invalid_result_checks", field_path=check_path)
            requirements[-1]["checks"] = dict(checks)
        if requirements[-1]["effect"] == "command_run":
            checks = requirements[-1].get("checks", {})
            if (requirements[-1]["targets"] or set(checks) != {"command", "exit_code"}
                    or checks.get("exit_code") != 0 or type(checks.get("exit_code")) is not int):
                raise TaskContractError("command_run_requires_exact_command_and_zero_exit")
        elif any(k in requirements[-1].get("checks", {}) for k in ("command", "exit_code")):
            raise TaskContractError("command_check_requires_command_run_effect")
        if requirements[-1]["effect"] == "research_answer" and requirements[-1]["targets"]:
            raise TaskContractError("research_answer_requires_evidence_bindings_not_placeholder_targets")
        if requirements[-1]["effect"] == "local_answer":
            checks = requirements[-1].get("checks", {})
            if (requirements[-1]["targets"] or set(checks) - {"required_fields", "min_sources"}
                    or not checks.get("required_fields")):
                raise TaskContractError("local_answer_requires_fields_and_read_bindings")
        if requirements[-1]["effect"] == "history_answer" and (requirements[-1]["targets"]
                or set(requirements[-1].get("checks", {})) - {"required_fields", "min_sources"}):
            raise TaskContractError("history_answer_requires_history_bindings", field_path=path)
        if requirements[-1]["effect"] == "read" and not requirements[-1]["targets"]:
            raise TaskContractError("read_requires_explicit_targets")
    if len({item["id"] for item in requirements}) != len(requirements):
        raise TaskContractError("duplicate_requirement_id")
    result = {
        "goal": _text(value.get("goal"), field_path="goal"),
        "deliverables": _texts(value.get("deliverables"), nonempty=True, field_path="deliverables"),
        "scope": _texts(value.get("scope", []), field_path="scope"),
        "requirements": requirements,
    }
    if len(json.dumps(result, ensure_ascii=False).encode("utf-8")) > 12000:
        raise TaskContractError("contract_too_large")
    return result


def extend_contract(existing: Mapping[str, Any], proposal: Any) -> dict[str, Any]:
    """Corrections/additions must not erase original goals or weaken conditions.

Cancellation/replacement of an original requirement needs a future explicit
scope-revision workflow; this batch deliberately fails closed on removals.
"""
    old = validate_contract(existing)
    new = validate_contract(proposal)
    proposed = {item["id"]: item for item in new["requirements"]}
    if any(proposed.get(item["id"]) != item for item in old["requirements"]):
        raise TaskContractError("contract_requirement_removed_or_changed")
    if not set(old["deliverables"]).issubset(new["deliverables"]):
        raise TaskContractError("contract_deliverable_removed")
    # Retain the initial goal and scope; added instructions remain traceable in
    # requests. A new objective belongs to a new task, not a renamed old task.
    new["goal"] = old["goal"]
    new["scope"] = list(dict.fromkeys([*old["scope"], *new["scope"]]))
    return validate_contract(new)


_EXCLUDED_TOKEN_KEYS = frozenset({
    "tokens",
    "max_tokens",
    "reported_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "token_accounting",
    "token_budget",
    "estimated_tokens",
})
_SECRET_KEY = re.compile(
    r"(?:api.?key|authorization|cookie|password|passwd|secret|access_token|refresh_token|\btoken\b)",
    re.I,
)
_SECRET_TEXT = re.compile(
    r"(?i)(\b(?:bearer)\s+)[\w.\-/+=]+|"
    r"(\b(?:api[_-]?key|password|cookie|access_token|refresh_token)\s*[:=]\s*)[^\s,;]+"
)


def _is_secret_key(key: Any, value: Any) -> bool:
    name = str(key or "").strip().lower()
    # 持续任务的累计上限用null表示；无内容可泄露，必须原样落盘才能恢复。
    # 仅豁免这个空容量值，字符串形式的同名凭据仍照常脱敏。
    if name == "max_tokens" and value is None:
        return False
    # Only numeric usage metrics are harmless; a credential named "tokens"
    # must not inherit a global key-name exemption.
    if (name in _EXCLUDED_TOKEN_KEYS or name.endswith("_tokens")) and type(value) in (int, float):
        return False
    if name == "token_accounting" and value == "reported_usage_or_conservative_reservation":
        return False
    return bool(_SECRET_KEY.search(name)) or name in _EXCLUDED_TOKEN_KEYS or name.endswith("_tokens")


def redact(value: Any, *, depth: int = 0, losses=None, path=()) -> Any:
    if depth > 10:
        if losses is not None:
            losses.append({"path": list(path), "reason": "snapshot_depth_limit", "omitted": 1})
        return "[omitted]"
    if isinstance(value, Mapping):
        if losses is not None and len(value) > 80:
            losses.append({"path": list(path), "reason": "snapshot_mapping_limit",
                           "total": len(value), "retained": 80, "omitted": len(value) - 80})
        return {str(key): "[redacted]" if _is_secret_key(key, item) else redact(item, depth=depth + 1,
                    losses=losses, path=(*path, str(key)))
                for key, item in list(value.items())[:80]}
    if isinstance(value, (list, tuple)):
        return [redact(item, depth=depth + 1, losses=losses, path=(*path, index)) for index, item in enumerate(value)]
    if isinstance(value, str):
        return _SECRET_TEXT.sub(lambda match: (match.group(1) or match.group(2)) + "[redacted]", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "[unsupported]"


class AgentTaskStore:
    """Small SQLite ledger; each task is scoped to requester AND conversation."""

    def __init__(self, path: str | Path | None = None, *, ttl_seconds: float = 7 * 86400,
                 max_entries: int = 128, clock=time.time) -> None:
        self.path = Path(path).expanduser().resolve() if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.ttl_seconds = max(1.0, float(ttl_seconds))
        self.max_entries = max(1, min(2048, int(max_entries)))
        self._lock = threading.RLock()
        self._closed = False
        self.result_events = None  # Explicit host opt-in; no default migration.
        self._db = None if self.path else sqlite3.connect(":memory:", check_same_thread=False, timeout=10)
        with self._transaction() as db:
            from .work_state import init_table
            init_table(db)
            db.execute("""CREATE TABLE IF NOT EXISTS agent_conversation_tasks (
                task_id TEXT PRIMARY KEY, requester TEXT NOT NULL, session TEXT NOT NULL,
                updated REAL NOT NULL, payload TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS agent_conversation_tasks_scope ON agent_conversation_tasks(requester, session, updated)")

    @contextmanager
    def _transaction(self):
        # File-backed connections are short-lived (Windows must be able to
        # release/backup the DB even when an assembly object remains alive).
        with self._lock:
            if self._closed:
                raise RuntimeError("task_store_closed")
            db = sqlite3.connect(str(self.path), timeout=10) if self.path else self._db
            try:
                with db:
                    db.execute("BEGIN IMMEDIATE")
                    yield db
            finally:
                if self.path:
                    db.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._db is not None:
                self._db.close()

    def _prune(self, db, now: float) -> None:
        if self.result_events is not None:
            self.result_events.prune_tasks_in_transaction(db, now)
            db.execute("DELETE FROM agent_task_work_revisions WHERE task_id NOT IN (SELECT task_id FROM agent_conversation_tasks)")
            return
        # 人工等待不计入历史淘汰；由显式取消/完成解除保留，不能让迟到回答丢失任务。
        pinned = "json_extract(payload, '$.status') = 'waiting_user'"
        db.execute(f"DELETE FROM agent_conversation_tasks WHERE NOT ({pinned}) AND (updated < ? OR updated > ?)",
                         (now - self.ttl_seconds, now + 300))
        db.execute(f"""DELETE FROM agent_conversation_tasks WHERE task_id IN
            (SELECT task_id FROM agent_conversation_tasks WHERE NOT ({pinned}) ORDER BY updated DESC, rowid DESC LIMIT -1 OFFSET ?)""",
                         (self.max_entries,))
        db.execute("DELETE FROM agent_task_work_revisions WHERE task_id NOT IN (SELECT task_id FROM agent_conversation_tasks)")

    def save(self, state: Mapping[str, Any], *, now: float | None = None, work_state=None) -> dict[str, Any]:
        losses = []
        source = copy.deepcopy(dict(state))
        sealed = source.get("execution_result")
        if isinstance(sealed, dict) and sealed.get("schema_version") == 1:
            import hashlib
            digest = hashlib.sha256(json.dumps({k: v for k, v in sealed.items() if k != "result_hash"},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if digest == sealed.get("result_hash") and redact(sealed) == sealed:
                source.pop("execution_result")
            else:
                sealed = None
        else:
            sealed = None
        safe = redact(source, losses=losses)
        if sealed is not None:
            # Already sealed/scrubbed at its own boundary; nesting it under a
            # task must not clip its information metadata and invalidate hash.
            safe["execution_result"] = sealed
        if losses:
            safe["snapshot_information"] = {"schema_version": 1, "status": "incomplete", "gap_count": len(losses),
                "gaps": losses[:128], "omitted_gap_records": max(0, len(losses) - 128),
                "source_completeness": "unknown", "retrieval": "retained_material_or_unavailable"}
        for name in ("task_id", "requester", "session"):
            _text(safe.get(name), 256)
        # Never trim immutable original requests/contracts to fit a snapshot.
        # Reject oversize input instead of silently losing objectives.
        if len(safe.get("requests", [])) > 32:
            raise TaskContractError("task_request_limit")
        if len(json.dumps(safe.get("requests", []), ensure_ascii=False).encode("utf-8")) > 65536:
            raise TaskContractError("task_snapshot_too_large")
        for name in ("tool_calls", "evidence", "failed_attempts"):
            safe[name] = list(safe.get(name, []))
            if len(safe[name]) > 2048:
                raise TaskContractError("task_ledger_limit")
        for call in [*safe["tool_calls"], *safe.get("inflight", [])]:
            arguments = call.get("arguments", {})
            if isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and key not in {"path", "url", "name", "pid", "message"} and len(value) > 512:
                        arguments[key] = "[执行参数正文已省略；不能作为结果证明]"
            updates = call.get("data", {}).get("state_updates", {})
            if isinstance(updates, dict) and "page_text" in updates:
                updates.pop("page_text")
                updates["page_content_requires_refresh"] = True
        safe["updated"] = self.clock() if now is None else float(now)
        payload = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
        with self._transaction() as db:
            existing = db.execute("SELECT requester, session, payload FROM agent_conversation_tasks WHERE task_id=?",
                                        (safe["task_id"],)).fetchone()
            if existing and existing[:2] != (safe["requester"], safe["session"]):
                raise TaskContractError("task_scope_collision")
            previous = json.loads(existing[2]) if existing else None
            if previous is not None and safe.get("revision") != previous.get("revision"):
                raise TaskContractError("task_state_conflict")
            if previous is not None:
                old_requests = previous.get("requests", [])
                if safe.get("requests", [])[:len(old_requests)] != old_requests:
                    raise TaskContractError("original_requests_removed_or_changed")
                if previous.get("input_mode") != safe.get("input_mode"):
                    raise TaskContractError("task_input_mode_changed")
                if previous.get("contract") is not None:
                    # 持久化层同样验证已登记计划的修订，不能只让内存循环放行。
                    # 原请求新增严格要求后，不沿用旧策略降低条件。
                    from .analysis_contract import merge_plan
                    basis = {**previous, "requests": safe.get("requests", [])}
                    preserved = merge_plan(basis, safe.get("contract"))
                    if preserved != safe.get("contract"):
                        raise TaskContractError("original_goal_or_scope_changed")
                if safe.get("contract") is not None:
                    from .analysis_contract import origin_record
                    # 不相信提交快照自报的归属；依据库内旧记录重新生成。
                    basis = {**previous, "requests": safe.get("requests", [])}
                    safe["analysis_plan_origin"] = origin_record(basis, safe["contract"])
                safe["contract_history"] = copy.deepcopy(previous.get("contract_history", []))
                if safe.get("contract") != previous.get("contract"):
                    safe["contract_history"].append(copy.deepcopy(previous.get("contract")))
            elif safe.get("analysis_plan_origin"):
                # 导入旧任务只能保留硬条件，不通过外部快照伪造可删改来源。
                safe.pop("analysis_plan_origin", None)
            safe["revision"] = int((previous or {}).get("revision", 0)) + 1
            from .work_state import persist, compact_plan_calls
            persist(db, safe, previous, work_state)
            compact_plan_calls(db, safe)
            payload = json.dumps(safe, ensure_ascii=False, separators=(",", ":"))
            if len(payload.encode("utf-8")) > 1048576:
                raise TaskContractError("task_snapshot_too_large")
            db.execute("INSERT OR REPLACE INTO agent_conversation_tasks VALUES (?, ?, ?, ?, ?)",
                             (safe["task_id"], safe["requester"], safe["session"], safe["updated"], payload))
            if self.result_events is not None:
                self.result_events.append_in_transaction(db, safe)
            self._prune(db, safe["updated"])
        return safe

    def candidates(self, *, requester: str, session: str, limit: int = 3, now: float | None = None) -> list[dict[str, Any]]:
        with self._transaction() as db:
            self._prune(db, self.clock() if now is None else float(now))
            rows = db.execute("""SELECT payload FROM agent_conversation_tasks
                WHERE requester=? AND session=? ORDER BY updated DESC, rowid DESC LIMIT ?""",
                (requester, session, max(1, min(3, limit)))).fetchall()
        return [json.loads(row[0]) for row in rows]

    def get(self, task_id: str, *, requester: str, session: str, now: float | None = None) -> dict[str, Any] | None:
        with self._transaction() as db:
            self._prune(db, self.clock() if now is None else float(now))
            row = db.execute("SELECT payload FROM agent_conversation_tasks WHERE task_id=? AND requester=? AND session=?",
                                   (task_id, requester, session)).fetchone()
        return json.loads(row[0]) if row else None

    def get_work_state(self, task_id, *, requester, session, revision=None):
        """当前双重作用域和 TTL 内读取导航修订；读取不刷新任务寿命。"""
        from .work_state import read
        with self._transaction() as db:
            now = self.clock()
            row = db.execute("SELECT payload FROM agent_conversation_tasks WHERE task_id=? AND requester=? AND session=? AND updated>=? AND updated<=?",
                (task_id, requester, session, now - self.ttl_seconds, now + 300)).fetchone()
            return read(db, json.loads(row[0]), revision) if row else None

    def read_scoped_page(self, *, requester: str, session: str, offset: int = 0,
                         limit: int = 10, task_id: str = "", exclude: str = "") -> dict[str, Any]:
        """只读诊断分页：按双重作用域及TTL筛选，不清理、刷新或改写旧证据。

        正常get/candidates拥有清理生命周期；诊断不能因观察而删除过期原件。
        offset分页描述本次读取时的账本，新任务写入后需从首页重新定位。
        """
        if type(offset) is not int or not 0 <= offset <= 2048 or type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("task_record_page_invalid")
        now = self.clock()
        where = "requester=? AND session=? AND updated>=? AND updated<=? AND task_id!=?"
        params = [requester, session, now - self.ttl_seconds, now + 300, exclude]
        if task_id:
            where += " AND task_id=?"
            params.append(task_id)
        with self._transaction() as db:
            total = db.execute("SELECT COUNT(*) FROM agent_conversation_tasks WHERE " + where, params).fetchone()[0]
            rows = db.execute("SELECT payload FROM agent_conversation_tasks WHERE " + where +
                " ORDER BY updated DESC, task_id DESC LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
        return {"items": [json.loads(row[0]) for row in rows], "total": total, "observed_at": now}

    def for_message_ids(self, message_ids, *, requester: str, session: str) -> list[dict[str, Any]]:
        """Read source-request bindings across the retained scoped ledger."""
        identifiers = set(str(v) for v in tuple(message_ids)[:32])
        if not identifiers:
            return []
        with self._transaction() as db:
            self._prune(db, self.clock())
            rows = db.execute("SELECT payload FROM agent_conversation_tasks WHERE requester=? AND session=? ORDER BY updated DESC",
                              (requester, session)).fetchall()
        tasks = [json.loads(row[0]) for row in rows]
        return [t for t in tasks if any(r.get("message_id") in identifiers for r in t.get("requests", []))][:3]


def new_task(*, task_id: str, requester: str, session: str, request: str,
             message_id: str, contract: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "task_id": task_id, "requester": requester, "session": session,
        "requests": [{"message_id": message_id, "text": _text(request, 12000)}],
        "contract": validate_contract(contract) if contract is not None else None,
        "status": "running", "tool_calls": [], "evidence": [], "failed_attempts": [],
        "notes": {"hypotheses": [], "ruled_out": [], "remaining": [], "blocker": ""},
        "completion": {}, "last_report": "", "inflight": [], "uncertain_attempts": [],
    }


def update_notes(state: dict[str, Any], notes: Any) -> None:
    if not isinstance(notes, Mapping):
        return
    current = copy.deepcopy(state.get("notes", {}))
    for name in ("hypotheses", "ruled_out", "remaining"):
        if name in notes:
            values = _texts(notes[name])
            # Model-labelled elimination is a hypothesis, never executor proof.
            if name == "remaining":
                current[name] = values
            else:
                current[name] = list(dict.fromkeys([*current.get(name, []), *values]))[-16:]
    if "blocker" in notes:
        current["blocker"] = _text(notes["blocker"]) if notes["blocker"] else ""
    state["notes"] = current


def task_prompt(state: Mapping[str, Any], *, compact: bool = False) -> dict[str, Any]:
    keys = ("task_id", "status", "contract", "requests", "notes", "last_report", "completion", "inflight", "uncertain_attempts")
    result = {name: copy.deepcopy(state.get(name)) for name in keys}
    if isinstance(result.get("completion"), dict):
        # The full receipt ledger remains in storage/trace; the planner needs
        # satisfied/missing conditions, not another copy of every tool record.
        result["completion"].pop("evidence", None)
    result["evidence"] = list(state.get("evidence", []))[-(2 if compact else 6):]
    result["failed_attempts"] = list(state.get("failed_attempts", []))[-(2 if compact else 6):]
    research = state.get("research", {})
    if isinstance(research, Mapping) and research.get("enabled"):
        result["research"] = {"enabled": True, "intent": research.get("intent", {}),
            "candidates": [{key: row.get(key) for key in ("id", "url", "relevance", "read_attempts")}
                           for row in list(research.get("candidates", []))[-(3 if compact else 6):]],
            "source_receipts_require_refresh": True}
    requests = result.get("requests") or []
    result["requests"] = [{"message_id": item.get("message_id"), "text": str(item.get("text") or "")[:(600 if compact else 1600)]}
                          for item in (requests[:1] + requests[-1:] if len(requests) > 1 else requests)]
    result["request_count"] = len(requests)
    result["uncertain_attempts"] = list(result.get("uncertain_attempts") or [])[-4:]
    result["excerpts_only"] = True
    result["notes"] = {key: ([str(item)[:200] for item in value[-(2 if compact else 4):]]
                            if isinstance(value, list) else str(value)[:400])
                       for key, value in (result.get("notes") or {}).items()}
    for name in ("evidence", "failed_attempts"):
        result[name] = [{**item, "summary": str(item.get("summary") or "")[:(240 if compact else 500)]} for item in result[name]]
    if compact:
        contract = result.get("contract")
        if isinstance(contract, Mapping):
            result["contract"] = {"goal": str(contract.get("goal") or "")[:1000],
                                  "deliverables": [str(item)[:80] for item in contract.get("deliverables", [])[:6]]}
    result["last_report"] = str(result.get("last_report") or "")[:(400 if compact else 1000)]
    budget = 4096 if compact else 16000
    # Preserve the complete active contract; only supporting excerpts shrink.
    for key in ("evidence", "failed_attempts", "research", "notes", "last_report", "requests", "inflight"):
        if len(json.dumps(result, ensure_ascii=False)) <= budget:
            break
        result[key] = [] if isinstance(result[key], list) else {} if isinstance(result[key], dict) else ""
        result["supporting_excerpts_omitted"] = True
    return redact(result)
