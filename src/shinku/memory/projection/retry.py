"""Opt-in experiment only. Typed control is never inferred from normal prose."""
from __future__ import annotations

from .contracts import Deadline


def compare_main_retry(*, call, request, view, sources, max_read=4):
    """At most one validated source read and one retry; no Judge, no transport.

    `call` must produce a structured envelope {control, draft}. Production reply
    parsing is deliberately unchanged until this candidate protocol is accepted.
    """
    available = {row["record_id"]:(row["record_id"],row["scope"],row["source_version"])
                 for row in view.records}
    first = call({"phase":"initial", "query":request.query, "context":view.rendered,
                  "allowed_read_ids":list(available), "protocol":"context-retry-experiment-v1"})
    control = first.get("control") if isinstance(first,dict) else None
    if control is None:
        return {"draft":first.get("draft", "") if isinstance(first,dict) else "", "retries":0,
                "status":"answer", "publish":False}
    if not isinstance(control,dict) or control.get("type") != "need_context":
        return {"draft":"", "retries":0, "status":"invalid_control", "publish":False}
    ids = control.get("record_ids")
    if not isinstance(ids,list) or not 1<=len(ids)<=max_read or any(i not in available for i in ids):
        return {"draft":"", "retries":0, "status":"unauthorized_handle", "publish":False}
    records = sources.read(tuple(available[i] for i in ids), request, Deadline(300))
    if any(r is None for r in records):
        return {"draft":"", "retries":0, "status":"source_unavailable", "publish":False}
    second = call({"phase":"retry", "query":request.query, "context":view.rendered,
                   "sources":[{"record_id":r.record_id,"text":r.text[:2400],
                               "excerpt":len(r.text)>2400} for r in records], "allow_retry":False})
    fresh = sources.read(tuple(available[i] for i in ids), request, Deadline(300))
    if (any(r is None for r in fresh) or not isinstance(second,dict) or second.get("control") is not None):
        return {"draft":"", "retries":1, "status":"unresolved", "publish":False}
    return {"draft":str(second.get("draft") or ""), "retries":1, "status":"candidate_answer", "publish":False}
