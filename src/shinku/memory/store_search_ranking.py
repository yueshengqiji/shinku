"""候选评分、去重和最终排序；来源准入与持久化仍由原查询链负责。

语义扫描与最终合并共用同一评分函数，防止词法裸分与归一化混合分直接比较。
本模块不改变检索范围、向量门槛或来源状态，也不把相关性当作事实可信度。
"""

from __future__ import annotations

from dataclasses import replace

from .models import MemoryRecord
from .store_relevance import QueryRelevance, query_relevance, attributed_search_text

from .store_models import MemoryStoreQueriesDependencies


from .store_search_models import MemoryStoreSearch


def calibrated_candidate(
    record: MemoryRecord, *, terms: list[str], similarity: float | None = None,
    features: QueryRelevance | None = None,
) -> MemoryRecord:
    """相关性与辅助因素分开计分，跨检索路使用同一尺度。

    语义/词法/精确/辅助的预算分别为 0.55/0.15/0.25/0.05，缺失项不重新
    分配给其他项；精确命中仍保留语义区分，避免多个引用者的相似度被抹平。
    辅助贡献与查询长度无关。similarity=None 保留词法回退，不伪造语义观测。
    此函数不使用旧 record.score，避免其中重要性/时间被重复计入。
    """
    lexical_score, exact_score = (features or query_relevance("", terms)).scores(attributed_search_text(record.text, record.metadata))
    semantic = max(0.0, min(1.0, similarity)) if similarity is not None else 0.0
    auxiliary = record.metadata.get("ranking_auxiliary", {})
    prior = (0.5 * auxiliary.get("importance", 0.0) + 0.2 * auxiliary.get("recency", 0.0)
             + 0.2 * auxiliary.get("topic", 0.0) + 0.1 * auxiliary.get("sender", 0.0))
    components = {"semantic": 0.55 * semantic, "exact": 0.25 * exact_score,
                  "lexical": 0.15 * lexical_score, "auxiliary": 0.05 * max(0.0, min(1.0, prior))}
    score = sum(components.values())
    metadata = dict(record.metadata)
    metadata["retrieval_method"] = "hybrid" if similarity is not None else "lexical"
    metadata["lexical_score"] = round(float(lexical_score), 6)
    metadata["exact_match_score"] = exact_score
    metadata["ranking_components"] = components
    if similarity is not None:
        metadata["semantic_similarity"] = round(float(similarity), 6)
    return replace(record, score=score, metadata=metadata)


def rank_search_candidates(q: MemoryStoreSearch, deps: MemoryStoreQueriesDependencies) -> None:
    merged: dict[str, MemoryRecord] = {}
    for item in q.candidates:
        previous = merged.get(item.record_id)
        is_hybrid = item.metadata.get("retrieval_method") == "hybrid"
        previous_is_hybrid = previous is not None and previous.metadata.get("retrieval_method") == "hybrid"
        if previous is None or is_hybrid or (not previous_is_hybrid and item.score > previous.score):
            merged[item.record_id] = item
    q.candidates = list(merged.values())
    # 同一语义查询的各个 scope 必须使用相同尺度，即使其中一个 scope 没有
    # 向量命中或编码失败，也不能让它的词法裸分在上层跨 scope 合并时压过证据。
    # 语义已评分的候选不重复归一化；关闭语义与显式词法日期查询保持原有评分。
    if (deps.semantic_enabled() and deps.embedding_provider() is not None
            and not q.require_term_overlap and q.query_text):
        features = query_relevance(q.query_text, q.terms)
        q.candidates = [item if item.metadata.get("retrieval_method") == "hybrid"
                        else calibrated_candidate(item, terms=q.terms, features=features) for item in q.candidates]
        _recheck_exact_sources(q, deps)
    if deps.search_keys_available() and deps.search_index() is not None:
        try:
            deps.set_search_keys_last_status(deps.search_index().rebuild(scope=str(q.scope), now=q.stamp, limit=deps.event_scan_limit(), dry_run=False))
            matches = deps.search_index().matches(records=q.candidates, query=q.query_text, scope=str(q.scope), now=q.stamp)
        except Exception as exc:
            matches = {}  # Source search still works if the derived index is unavailable.
            deps.set_search_keys_last_status({'scope': str(q.scope), 'fallback': 'source_search', 'error_type': type(exc).__name__, 'remote_calls': 0})
        q.candidates = [replace(item, score=item.score + min(0.30, 0.15 * len(matches[item.record_id])),
                      metadata={**item.metadata, "search_hints": list(matches[item.record_id]),
                                "search_generator": "local-topic-hints-v1"})
                      if item.record_id in matches else item for item in q.candidates]
        # Bind ALL returned sources, not just boosted hits, to this reading version.
        fresh_candidates = []
        fresh_sources = deps.search_index().read_sources(requests=tuple(
            (item.record_id, str(q.scope), "") for item in q.candidates), now=q.stamp)
        for item, fresh in zip(q.candidates, fresh_sources):
            if (fresh is not None and fresh.text == item.text and fresh.source_id == item.source_id
                    and fresh.created_at == item.created_at and fresh.layer == item.layer):
                # Never overwrite current provenance with an older candidate.
                diagnostics = {key: value for key, value in item.metadata.items()
                               if key in {"search_hints", "search_generator", "retrieval_method",
                                          "semantic_similarity", "lexical_score", "exact_match_score",
                                          "ranking_components", "ranking_auxiliary"}}
                fresh_candidates.append(replace(fresh, score=item.score, metadata={**fresh.metadata, **diagnostics}))
        q.candidates = fresh_candidates
    q.candidates.sort(key=lambda item: (item.score, item.importance, item.created_at), reverse=True)
    q.selected = q.candidates[: max(1, min(deps.max_results(), int(q.limit or 5)))]


def _recheck_exact_sources(q: MemoryStoreSearch, deps: MemoryStoreQueriesDependencies) -> None:
    """精确通道无需向量，也必须在其他检索工作结束后复核原文和撤销状态。"""
    kept = []
    with deps.lock(), deps.connect() as conn:
        for item in q.candidates:
            if not item.metadata.get("exact_match_score"):
                kept.append(item)
                continue
            kind, bare_id = item.record_id.split(":", 1)
            table, key = ("memory_events", "event_id") if kind == "event" else ("memory_facts", "fact_id")
            row = conn.execute(f"SELECT * FROM {table} WHERE {key}=? AND scope=? AND status='active'",
                               (bare_id, str(q.scope))).fetchone()
            if row is None:
                continue
            fresh = deps.event_record(row, score=item.score) if kind == "event" else deps.fact_record(row, score=item.score)
            if (fresh.text == item.text and fresh.source_id == item.source_id and fresh.created_at == item.created_at
                    and fresh.layer == item.layer):
                kept.append(item)
    q.candidates = kept
