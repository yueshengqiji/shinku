"""Executor evidence, rather than read count or model assertions, measures progress."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

VOLATILE = {"timestamp", "updated_at", "created_at", "elapsed_ms", "duration", "invocation_id", "tool_type", "tool", "request_id"}


def stable(value):
    if isinstance(value, Mapping):
        return {str(key): stable(item) for key, item in value.items() if str(key).lower() not in VOLATILE}
    if isinstance(value, (list, tuple)):
        return [stable(item) for item in value]
    return value


def obstacle(envelope) -> str:
    data = envelope.data or {}
    updates = data.get("state_updates", {})
    if isinstance(updates, Mapping) and updates.get("http_status") in {403, 429}:
        return "target_denied"
    explicit = updates.get("obstacle") if isinstance(updates, Mapping) else None
    known = {"timeout", "dependency_missing", "permission", "network", "proxy", "target_denied", "invalid_arguments"}
    if explicit in known:
        return explicit
    text = " ".join([str(data.get("code", "")), str(data.get("error_type", "")), envelope.model_feedback]).lower()
    for category, pattern in (
        ("target_denied", r"http.?403|http.?429|access denied|captcha|验证码|目标站.*拒绝"),
        ("proxy", r"proxy|代理"),
        ("dependency_missing", r"modulenotfound|importerror|not installed|缺少.*依赖|找不到模块"),
        ("permission", r"permission|unauthorized|forbidden|tool_blocked|权限|未授权"),
        ("timeout", r"timeout|timed out|超时"),
        ("network", r"connection|dns|network|网络|连接失败"),
        ("invalid_arguments", r"bad_args|参数.*格式|invalid.argument"),
    ):
        if re.search(pattern, text):
            return category
    return "unknown"


class ProgressLedger:
    def __init__(self, previous=(), *, stalled=0):
        self.seen = dict.fromkeys(previous)
        self.stalled = stalled if type(stalled) is int and 0 <= stalled <= 1024 else 0
        self.events = []

    def observe(self, invocation, envelope, *, progress_class, candidate_only=False, execution_event=None):
        data = envelope.data or {}
        updates = data.get("state_updates", {})
        updates = updates if isinstance(updates, Mapping) else {}
        # Failed mutations NEVER reset the stall budget. Novel failures are
        # useful diagnostic evidence, but not proof of state change.
        mutation = envelope.status == "ok" and progress_class == "change" and updates.get("changed") is True
        content = updates.get("progress_evidence", envelope.model_feedback)
        payload = stable({"status": envelope.status, "content": content, "updates": updates})
        if execution_event is not None:
            # Only a registered executor's authenticated receipt reaches this
            # branch. Timestamps, diagnostics and repeated prose are not motion.
            payload = {"status": envelope.status, "execution": execution_event}
        fingerprint = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
        novel = fingerprint not in self.seen and bool(content or updates)
        self.seen[fingerprint] = None
        waiting = False
        if execution_event is not None:
            kind = execution_event["kind"]
            waiting = envelope.status == "ok" and kind in {"waiting", "output"}
            novel = novel and kind != "waiting"
            progress = envelope.status == "ok" and novel and kind in {"output", "command_completed", "artifact_changed"}
            mutation = progress and kind == "artifact_changed"
        elif candidate_only:
            progress = False
        elif envelope.status != "ok" and progress_class == "change":
            progress = False
        elif progress_class == "change":
            progress = mutation or (envelope.status == "ok" and novel and "browser_access" in updates.get("verified_effects", []))
        else:
            progress = novel
        self.stalled = 0 if progress else self.stalled if waiting else self.stalled + 1
        event = {"tool": invocation.name, "status": envelope.status, "new_evidence": novel and not candidate_only,
                 "verified_change": mutation, "progress": progress, "stalled": self.stalled,
                 "fingerprint": fingerprint, "obstacle": obstacle(envelope) if envelope.status != "ok" or updates.get("http_status") in {403, 429} else ""}
        self.events.append(event)
        if execution_event is not None:
            event.update(execution_event=execution_event["kind"], waiting=waiting)
        if candidate_only:
            event.update(stage="candidate_discovery", new_candidates=novel)
        return event
