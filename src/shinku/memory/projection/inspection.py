"""Model-requested reads reuse projection discovery and canonical validation.

The query is interpretation supplied by the existing conversation model. The
host retains ownership of requester, scopes, delivery, source versions and time.
This path does not classify language, rebuild an index, or confirm new facts.
"""
from dataclasses import replace
from datetime import datetime
import sqlite3
import time

from .contracts import Deadline, ProjectionRequest, ProjectionView
from .sources import connection
from .view import admissible, render, source_item
from ..query_time import normalize_recorded_time, recorded_time_window


def _timeline(projector, request, deadline, limit):
    scopes = ",".join("?" for _ in request.allowed_scopes)
    parts, values = [], []
    for kind, table, key in (("event", "memory_events", "event_id"), ("fact", "memory_facts", "fact_id")):
        parts.append(f"""SELECT '{kind}:'||{key} AS rid,scope,created_at AS stamp FROM {table}
            WHERE scope IN ({scopes}) AND status='active' AND created_at<=?
              AND source_id NOT LIKE 'owner:agent-task-detail:%'
              AND (? IS NULL OR created_at>=?) AND (? IS NULL OR created_at<?)""")
        values.extend((*request.allowed_scopes, request.now, request.start_at, request.start_at,
                       request.end_at, request.end_at))
    with connection(projector.store, deadline) as conn:
        rows = conn.execute(" UNION ALL ".join(parts)+" ORDER BY stamp DESC,rid LIMIT ?",
                            (*values, limit+1)).fetchall()
    return tuple((r["rid"], r["scope"], "") for r in rows[:limit]), {
        "state":"bounded_timeline", "candidate_count":min(limit, len(rows)),
        "truncated":len(rows)>limit, "complete":False, "maintenance_in_request":0,
        "absence_means":"unknown"}, []


def _related(left, right):
    if left.scope != right.scope:
        return False
    family = left.source_id.removesuffix(":assistant")
    if family.startswith(("turn:", "owner:turn:")) and right.source_id.removesuffix(":assistant") == family:
        return True
    for source, other in ((left, right), (right, left)):
        meta = source.metadata
        mids = meta.get("reply_message_ids", ())
        mids = list(mids) if isinstance(mids, (list, tuple)) else []
        if meta.get("reply_to_message_id"):
            mids.append(meta["reply_to_message_id"])
        if other.metadata.get("message_id") and other.metadata["message_id"] in mids:
            return True
        if other.record_id in meta.get("source_record_ids", ()):
            return True
    return False


def inspect_context(service, *, conversation_id, conversation_type, requester_id, query,
                    mode, limit, message_id, user_text, self_id, recorded_time=None, source_ids=()):
    started = time.monotonic()
    if not isinstance(query, str) or len(query) > 2000 or mode not in {"recent", "retrieve", "summary"}:
        raise ValueError("invalid_model_memory_query")
    if (not isinstance(source_ids, (tuple,list)) or len(source_ids)>8
            or any(not isinstance(v,str) or len(v)>128 or not v.startswith(('event:','fact:')) for v in source_ids)
            or source_ids and mode!='retrieve'):
        raise ValueError('invalid_model_memory_sources')
    now = int(service.clock())
    scope = service.scope_for(conversation_id, conversation_type)
    requester = str(requester_id or (conversation_id if scope.startswith("private:") else ""))
    if not requester:
        raise ValueError("memory_requester_required")
    owner = service._is_master_request(conversation_id=conversation_id, conversation_type=conversation_type,
                                       message_meta={"sender_id":requester})
    # 主人私聊的主动 recent 读取也是明确的读请求；不能再被旧 recent 路由截断共享范围。
    scope_mode = "retrieve" if owner and scope.startswith("private:") else mode
    scopes = service._history_scopes(scope, owner_request=owner, mode=scope_mode)
    # Explicit tool intent separates a record's saved time from the subject's
    # event time. The host parses only the separately declared date expression;
    # it does not infer that every date mentioned in a query restricts records.
    # None is retained for older internal callers, not offered by the new tool.
    if recorded_time is None:
        window = service._request_date_window(query, now=now) if mode != "recent" else None
        time_filter_source = "legacy_query"
    else:
        recorded_time = normalize_recorded_time(recorded_time)
        if any(recorded_time.values()) and mode == "recent":
            raise ValueError("recent_recorded_time_not_supported")
        bounds = recorded_time_window(recorded_time)
        window = bounds if any(bound is not None for bound in bounds) else None
        time_filter_source = "explicit_recorded_time"
    request = ProjectionRequest(scope=scope, allowed_scopes=scopes, requester=requester, self_id=self_id,
        query=user_text or query, retrieval_query=query.strip(), message_id=message_id, now=now,
        history_mode=mode, start_at=window[0] if window else None, end_at=window[1] if window else None,
        participant_ids=service.policy.owner_participant_ids if owner and "private:master" in scopes else ())
    projector = service.projection
    if projector is None:
        raise RuntimeError("projection_reader_unavailable")
    wanted = max(1, min(12, service.policy.max_results, int(limit)))
    # The tool returns structured sources plus a readable projection. Bound
    # both representations under the audited tool's serialized result budget.
    policy = replace(projector.policy, max_chars=min(4500, projector.policy.max_chars),
                     record_chars=min(2400, projector.policy.record_chars, projector.policy.max_chars), source_limit=wanted)
    deadline = Deadline(max(1, policy.retrieval_ms-300))
    view = ProjectionView()
    selected, attempted = [], set()
    error = ""
    checkpoint = None
    semantic = {"status": "not_needed", "calls": 0}
    try:
        if source_ids:
            # 定位符只缩小已授权候选；模型不能指定 scope 或借此打开其他人的资料。
            with connection(projector.store, deadline) as conn:
                handles=[]
                for rid in dict.fromkeys(source_ids):
                    kind, _, key=rid.partition(':')
                    table, pk=('memory_events','event_id') if kind=='event' else ('memory_facts','fact_id')
                    row=conn.execute(f"SELECT scope FROM {table} WHERE {pk}=? AND scope IN ("+
                        ','.join('?' for _ in scopes)+')', (key,*scopes)).fetchone()
                    handles.append((rid,row[0] if row else scope,''))
            handles=tuple(handles)
            coverage=dict(state='exact_source_handles',candidate_count=len(handles),complete=False,
                          maintenance_in_request=0,absence_means='unknown')
            gaps=[]
        elif mode == "recent" or not query.strip() or mode == "summary" and window:
            handles, coverage, gaps = _timeline(projector, request, deadline, policy.candidate_limit)
        else:
            handles, coverage, gaps = projector.archive.discover(request, deadline, limit=policy.candidate_limit)
        view.coverage.update(coverage)
        view.gaps.extend(gaps)
        reserve = min(16, wanted, policy.candidate_limit//4)
        primary_handles = handles[:max(1, policy.candidate_limit-reserve)]
        if len(primary_handles) < len(handles):
            view.gaps.append({"reason":"candidate_budget", "omitted_count":len(handles)-len(primary_handles)})
        attempted.update(primary_handles)
        rows = projector.sources.read(primary_handles, request, deadline)
        primary = [r for r in rows if r is not None and admissible(r, request)]
        unavailable = len(rows)-len(primary)
        if unavailable:
            view.gaps.append({"reason":"source_unavailable_or_unverified", "count":unavailable})
            if source_ids:
                error='source_handles_unavailable'
        related, gaps = projector.sources.linked(primary[:wanted], request, deadline, reserve)
        view.gaps.extend(gaps)
        related = tuple(h for h in related if h not in attempted)[:policy.candidate_limit-len(attempted)]
        attempted.update(related)
        neighbors = [r for r in projector.sources.read(related, request, deadline)
                     if r is not None and admissible(r, request)] if related else []
        origins = set()
        for root in primary:
            for row in (root, *(r for r in [*primary, *neighbors] if r.record_id != root.record_id and _related(root,r))):
                origin = (row.metadata.get("owner_source_scope",row.scope),
                          row.source_id.removeprefix("owner:") if row.metadata.get("owner_memory") else row.source_id)
                if origin not in origins:
                    origins.add(origin)
                    selected.append(row)
        if len(selected) > wanted:
            view.gaps.append({"reason":"source_limit", "omitted_count":len(selected)-wanted})
        selected = selected[:wanted]
        # Reuse extraction only when this real read would otherwise lose raw
        # evidence to the display budget. Ordinary small reads add no AUX call.
        preview = ProjectionView(records=[source_item(r) for r in selected],
                                 coverage=dict(view.coverage), gaps=list(view.gaps))
        render(preview, request, policy)
        pressure = ({r["record_id"] for r in preview.records if r["text"]}
                    != {r.record_id for r in selected}
                    or any(len(r.text) > policy.record_chars for r in selected))
        if selected and pressure:
            remaining_ms = int(policy.read_timeout_seconds * 1000 -
                               (time.monotonic()-started)*1000 - 350)
            if remaining_ms > 0:
                checkpoint, semantic = projector.semantic.process("distill", request, selected, [],
                    timeout_ms=remaining_ms, max_chars=policy.max_chars)
            else:
                semantic = {"status": "read_budget", "calls": 0}
            if semantic["status"] not in {"completed", "cache_hit", "coalesced"}:
                view.gaps.append({"reason": "semantic_" + semantic["status"]})
            if semantic.get("omitted_input_records"):
                view.gaps.append({"reason": "semantic_input_budget", "omitted_count": semantic["omitted_input_records"]})
        # Extraction, including a cache hit, cannot resurrect a changed source.
        fresh = projector.sources.read(tuple((r.record_id,r.scope,r.metadata["source_version"]) for r in selected),
                                       request, Deadline(min(300, policy.retrieval_ms)))
        if any(r is None for r in fresh):
            view.gaps.append({"reason":"source_changed_during_projection"})
            error = "sources_changed_during_read"
            checkpoint = None
            projector.semantic.invalidate()
        selected = [r for r in fresh if r is not None and admissible(r,request)]
    except (sqlite3.Error, TimeoutError, ValueError) as exc:
        error = "retrieval_budget_exhausted" if isinstance(exc, TimeoutError) or not deadline.remaining else "source_read_failed"
        selected = []
        checkpoint = None
        view.gaps.append({"reason":error, "error_type":type(exc).__name__})
    view.records = [source_item(r) for r in selected]
    view.topic = {"binding":"model_requested_sources", "unresolved":True,
                  "historical_promises_are_unverified":True}
    view.coverage.update(complete=False, absence_means="unknown", sources_validated=len(selected),
                         candidate_reads=len(attempted), requested_time_window=[request.start_at,request.end_at],
                         time_filter_source=time_filter_source, recorded_time=recorded_time,
                         maintenance_in_request=0)
    render(view,request,policy,checkpoint)
    indexed = {r.record_id:r for r in selected}
    records = []
    delivered = []
    for item in view.records:
        if not item["text"]:
            continue
        row = indexed[item["record_id"]]
        metadata = {**item["attribution"], "source_version":item["source_version"],
                    "reading_representation":item["representation"]}
        for key in ("owner_memory", "owner_source_scope", "owner_origin_scope"):
            if key in row.metadata:
                metadata[key] = row.metadata[key]
        records.append(replace(row,text=item["text"],metadata=metadata).as_dict())
        delivered.append(row)
    if selected and not records:
        error = error or "source_budget_exhausted"
    if not records and any(g.get("reason") in {"source_read_failed", "budget_exhausted",
                                               "source_unavailable_or_unverified"} for g in view.gaps):
        # Discovery can preserve a working ranker after another fails. If no
        # validated evidence survives, that partial failure is not a completed
        # negative lookup. Keep the unknown coverage and a failed-read receipt.
        error = error or "source_read_failed"
    # Candidate binding alone never reinforces a source or writes a fact.
    service.bind_usage_context(scope=scope,message_id=message_id,requester_id=requester,
                                records=tuple(delivered),user_text=user_text)
    result = {"scope":scope, "retrieval_scopes":list(scopes), "owner_memory":owner, "mode":mode,
              "query":query, "count":len(records), "records":records, "rendered":view.rendered,
              "read_at":{"unix":now, "local_iso":datetime.fromtimestamp(now).astimezone().isoformat()},
              "coverage":view.coverage, "gaps":view.gaps, "fallback":"",
              "execution_authority":False, "model_calls":semantic["calls"],
              "semantic":semantic, "source_relations":view.source_relations,
              "followup_context":view.rendered}
    if error:
        result["error"] = error
    elif delivered:
        try:
            # 派生缓存只记真实读取来源，失败/空结果不清掉另一项仍有效的讨论。
            projector.topics.remember_read(request, delivered, Deadline(100))
        except (sqlite3.Error, TimeoutError):
            result["coverage"]["anchor_state"] = "unavailable"
    return result
