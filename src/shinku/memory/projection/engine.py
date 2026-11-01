"""Bounded, revocable source projection; no changes to source facts or authority."""
from __future__ import annotations

from dataclasses import replace
import re
import sqlite3
import time

from .archive import ArchiveLocator
from .contracts import Deadline, ProjectionView, digest
from .semantics import SemanticProcessor
from .sources import ProjectionSources
from .view import admissible, entity_candidates, render, source_item, reference_answer_basis
from .topics import TopicState
from .. import retrieval_trace


def underspecified_reference(query, entities):
    """A bare follow-up supplies no basis for an LLM to choose among people."""
    rest = query
    for entity in entities:
        rest = rest.replace(entity["name"], "")
    rest = re.sub(r"后来|然后|到底|当时|现在|之前|刚才|哪里|哪儿|怎么样|有没有|是不是|是否|怎样|已经|还是|那个人|哪个|哪位|她|他|这|那|个|呢|吗|呀|啊|了|的|在|去|来|会|能|吧|有|没|做|说|谁|是|怎么", "", rest)
    return len(re.sub(r"[\W_]", "", rest)) < 2


class ProjectionEngine:
    def __init__(self, store, policy, model=None, *, understanding_enabled=False):
        self.store, self.policy = store, policy
        self.understanding_enabled = understanding_enabled
        self.sources = ProjectionSources(store)
        self.topics = TopicState(store, policy)
        self.archive = ArchiveLocator(store, policy)
        self.semantic = SemanticProcessor(policy, model, store=store)
        self.last_diagnostics = {}
        store.on_forget(self.invalidate)

    def invalidate(self, *, scope, record_ids):
        self.archive.invalidate(scope=scope, record_ids=record_ids)
        self.semantic.invalidate()

    def start(self):
        if self.policy.enabled:
            self.archive.start()

    def stop(self):
        self.archive.stop()
        self.semantic.stop()

    def project(self, request):
        retrieval_trace.query("projection.request", request.query, mode=request.history_mode)
        if not self.policy.enabled:
            retrieval_trace.observe("projection.skipped", reason="disabled")
            return None
        started = time.monotonic()
        deadline = Deadline(max(1, self.policy.retrieval_ms-300))
        view = ProjectionView()
        stages = {}
        records = {}
        relevant_rank = {}
        attempted = set()
        continuing = request.continuity_handles is not None
        model_retrieval = self.policy.retrieval_strategy == "model"
        anchor_rank, catalog_dependencies, catalog_reads = {}, (), 0

        def stage(name, operation):
            stamp = time.monotonic()
            try:
                return operation()
            except (sqlite3.Error, TimeoutError, ValueError) as exc:
                view.gaps.append({"reason":"budget_exhausted" if isinstance(exc, TimeoutError) or not deadline.remaining
                                  else "source_read_failed", "stage":name, "error_type":type(exc).__name__})
                return None
            finally:
                stages[name] = round((time.monotonic()-stamp)*1000, 3)

        def read(handles):
            handles = tuple(h for h in dict.fromkeys(handles) if h not in attempted)
            remaining = max(0,self.policy.candidate_limit-len(attempted)-catalog_reads)
            if len(handles)>remaining:
                view.gaps.append({"reason":"candidate_budget", "omitted_count":len(handles)-remaining})
            handles = handles[:remaining]
            attempted.update(handles)
            if not handles:
                return
            batch = self.sources.read(handles, request, deadline)
            for handle, row in zip(handles, batch):
                if row is None:
                    view.gaps.append({"reason":"source_unavailable", "source_id":handle[0]})
                elif (request.message_id and row.metadata.get("message_id") == request.message_id
                        and row.metadata.get("role", "") in {"", "user"}
                        and row.metadata.get("sender_id") == request.requester
                        and row.metadata.get("owner_source_scope", row.scope) == request.scope):
                    continue
                elif ((request.start_at is not None and row.created_at < request.start_at)
                        or (request.end_at is not None and row.created_at >= request.end_at)):
                    view.gaps.append({"reason":"outside_requested_time_window", "source_id":handle[0]})
                elif admissible(row, request):
                    if row.metadata.get("reading_representation") != "source":
                        view.gaps.append({"reason":"raw_detail_unavailable", "source_id":row.record_id})
                    records.setdefault(row.record_id, row)
                else:
                    view.gaps.append({"reason":"unverified_attribution_or_delivery", "source_id":handle[0]})

        stage("continuity_validation", lambda: read(request.continuity_handles or ()))
        if model_retrieval and not continuing:
            # 既读锚点无需匹配续聊词；多候选交给主模型理解，原件变更不能沿用旧版。
            anchors = stage("read_anchors", lambda: self.topics.read_anchors(request, deadline)) or ()
            anchor_rank = {h[0]: i for i, h in enumerate(anchors)}
            stage("anchor_validation", lambda: read(anchors))
            from .continuity import source_catalog
            catalog = stage("source_catalog", lambda: source_catalog(self, request, deadline,
                limit=max(0, self.policy.candidate_limit-len(attempted))))
            if catalog:
                view.source_catalog, catalog_dependencies, catalog_reads = catalog
        dialogue = () if continuing else stage("dialogue_candidates", lambda: self.sources.dialogue(
            request, deadline, pairs=max(1, min(3, self.policy.source_limit // 2)),
            seconds=self.policy.recent_seconds, recent_limit=self.policy.recent_limit,
            preserve_addressed=model_retrieval)) or ()
        # Read before archive seeds can spend the entire candidate budget.
        stage("dialogue_read", lambda: read(dialogue))
        stage("seed_validation", lambda: read(()) if continuing else read(request.seed_handles))
        previous = None if continuing or model_retrieval else stage("topic_state", lambda: self.topics.continuation(request, deadline))
        parent_topic = None
        if previous:
            handles, parent_topic = previous
            stage("topic_source_validation", lambda: read(handles))
        recent = () if continuing else stage("recent_candidates", lambda: self.sources.recent(
            request, deadline, self.policy.recent_limit, self.policy.recent_seconds)) or ()
        if request.start_at is not None or request.end_at is not None:
            recent = ()  # Exact calendar recall must not acquire today's unrelated slice.
        # Chronological neighbours belong to a topic only if explicit edges
        # prove that link. The recent slice is still useful as attributed raw context.
        stage("recent_read", lambda: read(recent))
        history_needed = not continuing and (request.history_mode in {"retrieve", "summary"} or bool(request.seed_handles)
                          or request.start_at is not None or request.end_at is not None
                          or (not model_retrieval and (request.history_mode != "none" or
                              bool(re.search(r"之前|以前|那件|那个|上次|后来|还记得|旧|项目|参数", request.query)))))
        if history_needed:
            found = stage("archive_discovery", lambda: self.archive.discover(request, deadline, limit=self.policy.candidate_limit))
            if found:
                handles, coverage, gaps = found
                relevant_rank.update({h[0]:i for i,h in enumerate(handles)})
                view.coverage.update(coverage)
                view.gaps.extend(gaps)
                view.coverage["candidate_reads_remaining"] = max(0, self.policy.candidate_limit-len(attempted))
                stage("fused_source_read", lambda: read(handles))
        else:
            view.coverage.update(state="turn_sources_revalidated" if continuing else "recent_only",
                                 complete=False, maintenance_in_request=0)
        # Other incoming group messages belong to the next turn. A tool result
        # is appended by its caller; it must not silently replace the dialogue
        # selection (or grow entity headers enough to crowd its evidence out).
        linked = None if continuing else stage("predecessor_discovery", lambda: self.sources.linked(
            tuple(records.values())[:self.policy.source_limit], request, deadline, min(16,self.policy.source_limit)))
        if linked:
            handles, gaps = linked
            view.gaps.extend(gaps)
            stage("predecessor_read", lambda: read(handles))

        # Deduplicate only canonical origin links; identical text can be two
        # independent speakers and must not be merged by similarity.
        unique = []
        origins = set()
        for row in records.values():
            origin_scope = row.metadata.get("owner_source_scope", row.scope)
            origin_source = row.source_id.removeprefix("owner:") if row.metadata.get("owner_memory") else row.source_id
            key = (origin_scope, origin_source)
            if key not in origins:
                origins.add(key)
                unique.append(row)
        seeds = {r[0] for r in request.seed_handles}
        continuity_rank = {h[0]:i for i,h in enumerate(request.continuity_handles or ())}
        dialogue_rank = {h[0]:i for i,h in enumerate(dialogue)}
        recent_ids = {r[0] for r in recent}
        # Retain relevant archive hits before unrelated recent noise, without
        # erasing the recent adjacent exchange on a follow-up.
        unique.sort(key=lambda r:(continuity_rank.get(r.record_id, len(continuity_rank)),
                                 dialogue_rank.get(r.record_id, len(dialogue_rank)),
                                 anchor_rank.get(r.record_id, len(anchor_rank)),
                                 r.record_id not in seeds,
                                 relevant_rank.get(r.record_id, self.policy.candidate_limit+1) if history_needed else 0,
                                 (r.record_id in recent_ids) if history_needed else False,
                                 -r.created_at, r.record_id))
        if len(unique) > self.policy.source_limit:
            view.gaps.append({"reason":"source_limit", "omitted_count":len(unique)-self.policy.source_limit})
        selected = unique[:self.policy.source_limit]
        view.entities = [] if model_retrieval else entity_candidates(selected)
        pronoun = not model_retrieval and bool(re.search(r"她|他(?!们)|那个人|哪一个|哪个小", request.query))
        candidates = [e for e in view.entities if e["scope"] == request.scope]
        explicit = [e for e in candidates if e["name"] in request.query]
        active = explicit or candidates
        ambiguous = (pronoun and len(active) != 1) or len(explicit) > 1
        selected_entity = active[0]["candidate_id"] if pronoun and len(active)==1 else ""
        if ambiguous:
            view.gaps.append({"reason":"ambiguous_referent", "candidate_count":len(active)})
        topic_sources = [(r.record_id, r.metadata["source_version"]) for r in selected]
        view.topic = {"topic_id":"topic:"+digest([request.scope, topic_sources])[:20],
                      "event_start":min((r.created_at for r in selected),default=None),
                      "source_ids":[r.record_id for r in selected], "parent_topic":parent_topic,
                      "predecessor_ids":[h[0] for h in linked[0]] if linked else [],
                      "continuation":bool(request.reply_message_ids or pronoun),
                      "last_subject":selected_entity, "binding":"source_bound_current_turn",
                      "unresolved":bool(ambiguous), "historical_promises_are_unverified":True}
        pressure = sum(len(r.text)+350 for r in selected)>self.policy.max_chars
        checkpoint = None
        slow_diag = {"status":"not_needed", "calls":0}
        if model_retrieval:
            # 候选人物和指代均交给当前对话模型；不再用旧代词表强选或额外调用判断模型。
            view.topic.update(last_subject="", unresolved=True, binding="model_interprets_sources")
        # 原有的超长证据摘录仍独立受配置与预算控制，不能被“关闭指代判断”一并跳过。
        if ambiguous and underspecified_reference(request.query, active):
            slow_diag = {"status":"clarification_needed", "calls":0}
            view.topic["clarification_reason"] = "no_disambiguating_query_evidence"
        elif ambiguous and self.policy.ambiguity_strategy == "judge":
            locator_handles = {r.record_id:(r.record_id,r.scope,r.metadata["source_version"])
                               for r in records.values()}
            for r in selected:
                for parent in r.metadata.get("source_record_ids",[])[:8]:
                    locator_handles.setdefault(parent,(parent,r.scope,""))
            outcome, slow_diag = self.semantic.process("judge", request, selected, view.entities,
                                                       locator_ids=tuple(locator_handles))
            if outcome:
                view.topic["judge"] = outcome
                view.topic["last_subject"] = outcome["selected_entity"]
                # The selection is a defeasible interpretation, not a fact.
                view.topic["unresolved"] = outcome["uncertain"] or not outcome["selected_entity"]
                wanted = [locator_handles[rid] for rid in outcome.get("read_ids",[]) if rid in locator_handles]
                if wanted:
                    try:
                        extra = self.sources.read(tuple(wanted[:4]),request,Deadline(200))
                        additions = [r for r in extra if r is not None and admissible(r,request)
                                     and r.record_id not in {s.record_id for s in selected}]
                        if any(r is None for r in extra):
                            view.gaps.append({"reason":"judge_requested_source_unavailable"})
                        selected = (additions+selected)[:self.policy.source_limit]
                        slow_diag["targeted_reads"] = len(wanted)
                    except (sqlite3.Error, TimeoutError):
                        view.gaps.append({"reason":"judge_targeted_read_failed"})
        elif ambiguous:
            slow_diag = {"status":"main_retry_comparison_only", "calls":0}
        elif pressure and (history_needed or request.reply_message_ids or pronoun or parent_topic):
            checkpoint, slow_diag = self.semantic.process("distill", request, selected, view.entities)
        if slow_diag["status"] not in {"not_needed", "completed", "cache_hit", "coalesced", "clarification_needed"}:
            view.gaps.append({"reason":"semantic_"+slow_diag["status"]})
        # Always revalidate after semantic work, including cache hits; exact
        # versions prevent late replies from resurrecting changed sources.
        final_deadline = Deadline(min(300, self.policy.retrieval_ms))
        try:
            fresh = self.sources.read(tuple((r.record_id,r.scope,r.metadata["source_version"]) for r in selected),
                                      request, final_deadline)
            if any(r is None for r in fresh):
                view.gaps.append({"reason":"source_changed_during_projection"})
                checkpoint = None
                view.topic.pop("judge", None)
                view.topic.update(last_subject="", unresolved=True)
                self.semantic.invalidate()
            selected = [r for r in fresh if r is not None and admissible(r,request)]
        except (sqlite3.Error, TimeoutError):
            selected = []
            checkpoint = None
            view.topic.pop("judge", None)
            view.topic.update(last_subject="", unresolved=True)
            view.gaps.append({"reason":"final_validation_failed"})
        live_ids = {r.record_id for r in selected}
        view.entities = [e for e in view.entities if set(e["sources"]) <= live_ids]
        view.topic["source_ids"] = [r.record_id for r in selected]
        referent = next((e for e in view.entities if e['candidate_id']==view.topic['last_subject']),None)
        if referent and not view.topic['unresolved']:
            view.topic['answer_basis'] = reference_answer_basis(referent, selected)
        # Reuse the existing deterministic current-understanding recognizer,
        # without synchronously rebuilding its index or copying stale annotations.
        from ..current_understanding import claims
        from ..evidence import source_time
        claim_groups = {}
        annotations = {}
        for row in selected:
            entries = claims(row) if self.understanding_enabled and not model_retrieval and row.metadata.get("reading_representation")=="source" else []
            for entry in entries:
                entry["state"] = "confirmed" if entry["confirmation"] == "explicit_memory_request" else "unconfirmed"
                key = (entry["subject"],entry["attribute"],entry["condition"])
                claim_groups.setdefault(key, set()).add(entry["value"])
            annotations[row.record_id] = entries
        selected = [replace(row, metadata={**row.metadata, "current_understanding":[
            {**entry,"state":"conflict_pending_confirmation"} if len(claim_groups[(entry["subject"],entry["attribute"],entry["condition"])])>1
            else entry for entry in annotations[row.record_id]], "source_time":source_time(row,now=request.now)}) for row in selected]
        view.records = [source_item(r, dialogue=r.record_id in dialogue_rank) for r in selected]
        # 语义选择和最终原件校验已完成；从此处起，显示省略不能抹掉依赖。
        # 提示来源契约最多64项；目录与正文共同占用，不能显示后再静默丢掉依赖。
        catalog_dependencies = catalog_dependencies[:max(0,64-len(selected))]
        catalog_ids = {h[0] for h in catalog_dependencies}
        view.source_catalog = [row for row in view.source_catalog if row['record_id'] in catalog_ids]
        view.source_dependencies = tuple(dict.fromkeys([
            *((r.record_id, r.scope, r.metadata["source_version"]) for r in selected), *catalog_dependencies]))
        retrieval_trace.observe("projection.validated", selected)
        view.coverage.update(absence_means="unknown", sources_validated=len(selected),
                             recent_window_seconds=self.policy.recent_seconds,
                             requested_time_window=[request.start_at, request.end_at])
        render(view, request, self.policy, checkpoint)
        if retrieval_trace.enabled():
            # 投影可展示摘录：只记录可见来源，不能把可见片段冒充完整原文。
            retrieval_trace.observe("projection.visible",
                tuple((r["record_id"], r["scope"]) for r in view.records if r["text"]),
                rendered_chars=len(view.rendered), source_count=len(view.records))
        topic_saved = False
        try:
            before = self.archive.before_commit()
            self.topics.save(request, view.topic, selected, Deadline(100))
            self.archive.after_commit(before, False)
            topic_saved = True
        except (sqlite3.Error, TimeoutError):
            view.gaps.append({"reason":"topic_state_write_failed"})
        view.diagnostics = {"policy_version":self.policy.version, "stages_ms":stages,
                            "elapsed_ms":round((time.monotonic()-started)*1000,3),
                            "semantic":slow_diag, "source_count":len(view.records),
                            "candidate_reads":len(attempted)+catalog_reads,
                            "read_anchor_candidates":len(anchor_rank),
                            "source_catalog_count":len(view.source_catalog),
                            "query_catalog_count":sum(row.get('discovery')=='query_related' for row in view.source_catalog),
                            "continuity_active":continuing,
                            "continuity_requested":len(request.continuity_handles or ()),
                            "continuity_retained":sum(r["record_id"] in continuity_rank for r in view.records if r["text"]),
                            "recent_dialogue_requested":len(dialogue),
                            "recent_dialogue_injected":sum(r["record_id"] in dialogue_rank for r in view.records if r["text"]),
                            "injected_source_ids":[r["record_id"] for r in view.records if r["text"]],
                            "injected_chars":len(view.rendered), "gap_reasons":sorted({g["reason"] for g in view.gaps}),
                            "maintenance_in_request":0, "topic_state_written":topic_saved}
        self.last_diagnostics = view.diagnostics
        return view
