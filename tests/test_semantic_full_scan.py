"""Semantic budgets select by relevance after scanning the allowed source set."""
from __future__ import annotations

import sqlite3

import pytest

from shinku.memory import MemoryStore
from shinku.memory.lifecycle_policy import LifecyclePolicy


class Provider:
    name = "full-scan-fixture"
    version = "v1"
    dimension = 2

    def collection_key(self):
        return "full_scan_fixture_v1_2"

    def embed_text(self, text):
        return [1.0, 0.0] if any(t in text for t in ("安静", "清静", "宁静")) else [0.0, 1.0]

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


def make_store(tmp_path, *, event_limit=800, fact_limit=120):
    return MemoryStore(tmp_path / "memory.sqlite3", embedding_provider=Provider(), semantic_enabled=True,
                       event_scan_limit=event_limit, fact_scan_limit=fact_limit,
                       lifecycle_policy=LifecyclePolicy(mode="off"))


def event(store, source, text="修理自行车", *, scope="private:one", at=1000, layer="episodic"):
    store.record_event(conversation_id=scope, scope=scope, text=text, source_id=source,
                       created_at=at, layer=layer)


def fill(store, count):
    for i in range(count):
        event(store, f"new-{i}", at=1000+i)
    while store.backfill_embeddings(limit=1000):
        pass


def test_old_semantic_match_beyond_800_recent_events_is_found(tmp_path):
    store = make_store(tmp_path)
    event(store, "old-target", "我需要清静的工作环境", at=100)
    fill(store, 850)
    results = store.search(query="宁静的地方", scope="private:one", now=3000, limit=1)
    assert results[0].source_id == "old-target"
    assert results[0].metadata["retrieval_method"] == "hybrid"


def test_old_fact_beyond_fact_scan_budget_is_found(tmp_path):
    store = make_store(tmp_path, fact_limit=10)
    store.upsert_fact(conversation_id="one", scope="private:one", fact_key="quiet",
                      text="我需要清静的环境", source_id="old-fact", now=100)
    for i in range(20):
        store.upsert_fact(conversation_id="one", scope="private:one", fact_key=f"new-{i}",
                          text="修理自行车", source_id=f"fact-{i}", now=1000+i)
    store.backfill_embeddings(limit=100)
    results = store.search(query="宁静的地方", scope="private:one", now=3000, limit=1)
    assert results[0].source_id == "old-fact"


def test_full_scan_keeps_scope_layer_date_and_exclusive_end_guards(tmp_path):
    store = make_store(tmp_path, event_limit=10)
    for source, scope, at, layer in [
        ("allowed", "private:one", 101, "observation"),
        ("other-scope", "private:two", 101, "observation"),
        ("at-end", "private:one", 102, "observation"),
        ("before-start", "private:one", 99, "observation"),
        ("other-layer", "private:one", 101, "summary"),
    ]:
        event(store, source, "清静的环境", scope=scope, at=at, layer=layer)
    fill(store, 30)
    results = store.search(query="宁静的地方", scope="private:one", now=3000,
                           start_at=100, end_at=102, layers={"observation"})
    assert [r.source_id for r in results] == ["allowed"]


def test_stale_forgotten_and_task_detail_vectors_do_not_fill_relevance_budget(tmp_path):
    store = make_store(tmp_path, event_limit=10)
    event(store, "allowed", "清静的环境", at=100)
    for i in range(20):
        event(store, f"stale-{i}", "清静的环境", at=1000+i)
        event(store, f"forgotten-{i}", "清静的环境", at=2000+i)
        event(store, f"owner:agent-task-detail:{i}", "清静的环境", at=3000+i)
    store.backfill_embeddings(limit=1000)
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE memory_events SET text='旧向量与当前内容不同' WHERE source_id LIKE 'stale-%'")
        conn.execute("UPDATE memory_events SET status='forgotten' WHERE source_id LIKE 'forgotten-%'")
    # Exercise existing indexed vectors without repairing the stale distractors first.
    results = store._semantic_candidates(query="宁静的地方", terms=[], scope="private:one",
                                         selected_layers=(), start_at=None, end_at=None,
                                         topic_batch_id="", sender_id="", now=4000)
    assert [r.source_id for r in results] == ["allowed"]


def test_deferred_l1_vectors_cannot_hide_an_old_allowed_l2_match(tmp_path):
    store = make_store(tmp_path, event_limit=10)
    event(store, "allowed-l2", "清静的环境", at=100)
    fill(store, 30)
    store.lifecycle.policy = LifecyclePolicy(mode="active", defer_l1_embeddings=True)
    with store._connect() as conn:
        for row in conn.execute("SELECT * FROM memory_events").fetchall():
            store.lifecycle._enroll(conn, "event", row, now=3000)
        conn.execute("UPDATE memory_lifecycle SET tier=1,pinned=0")
        conn.execute("UPDATE memory_lifecycle SET tier=2 WHERE record_id IN "
                     "(SELECT event_id FROM memory_events WHERE source_id='allowed-l2')")
    results = store._semantic_candidates(query="宁静的地方", terms=[], scope="private:one",
                                         selected_layers=(), start_at=None, end_at=None,
                                         topic_batch_id="", sender_id="", now=3000)
    assert results[0].source_id == "allowed-l2"


def test_semantic_candidates_remain_bounded_after_full_scan(tmp_path):
    store = make_store(tmp_path, event_limit=10, fact_limit=10)
    for i in range(100):
        event(store, f"match-{i}", "清静的环境", at=100+i)
    store.backfill_embeddings(limit=1000)
    rows = store._semantic_candidates(query="宁静的地方", terms=[], scope="private:one",
                                      selected_layers=(), start_at=None, end_at=None,
                                      topic_batch_id="", sender_id="", now=3000)
    assert len(rows) <= 20
    assert rows[0].source_id == "match-99"


def test_retained_budget_uses_final_hybrid_score_instead_of_cosine_only(tmp_path):
    class HybridProvider(Provider):
        def embed_text(self, text):
            return [0.8, 0.6] if "anchor" in text and text != "anchor" else [1.0, 0.0]

    store = make_store(tmp_path, event_limit=10, fact_limit=10)
    store._embedding_provider = HybridProvider()
    event(store, "older-with-topic", "anchor 是这条记忆的主题", at=100)
    fill(store, 30)
    rows = store.search(query="anchor", scope="private:one", now=3000, limit=1)
    assert rows[0].source_id == "older-with-topic"


def test_source_revoked_during_scan_is_checked_before_return(tmp_path, monkeypatch):
    store = make_store(tmp_path, event_limit=10)
    event(store, "revoked", "清静的工作环境", at=100)
    event(store, "newer", "清静的房间", at=200)
    store.backfill_embeddings(limit=100)
    original = store._hybrid_record

    def revoke(record, similarity, *, terms, features=None):
        match = original(record, similarity, terms=terms, features=features)
        if record.source_id == "newer":
            with sqlite3.connect(store.path) as conn:
                conn.execute("UPDATE memory_events SET status='forgotten' WHERE source_id='revoked'")
        return match

    monkeypatch.setattr(store, "_hybrid_record", revoke)
    rows = store._semantic_candidates(query="宁静的地方", terms=[], scope="private:one",
                                      selected_layers=(), start_at=None, end_at=None,
                                      topic_batch_id="", sender_id="", now=3000)
    assert [r.source_id for r in rows] == ["newer"]


def test_numpy_and_dependency_free_scans_agree_and_reject_invalid_vectors(monkeypatch):
    import math
    import random
    import sys

    from shinku.memory.semantic import pack_vector
    from shinku.memory.semantic_scan import cosine_scores

    randomizer = random.Random(9821)
    query = [randomizer.uniform(-1, 1) for _ in range(64)]
    rows = [{"dimension": 64, "vector_blob": pack_vector([randomizer.uniform(-1, 1) for _ in range(64)])}
            for _ in range(20)]
    rows.extend([
        {"dimension": 64, "vector_blob": b"broken"},
        {"dimension": 63, "vector_blob": pack_vector([1.0] * 64)},
        {"dimension": 64, "vector_blob": pack_vector([math.nan] * 64)},
        {"dimension": 64, "vector_blob": pack_vector([math.inf] * 64)},
        {"dimension": 64, "vector_blob": pack_vector([0.0] * 64)},
    ])
    with monkeypatch.context() as patcher:
        patcher.setitem(sys.modules, "numpy", None)
        fallback = cosine_scores(query, rows)
    np = pytest.importorskip("numpy")
    accelerated = cosine_scores(query, rows)
    assert accelerated[:20] == pytest.approx(fallback[:20], abs=1e-12)
    assert accelerated[20:] == fallback[20:] == [None] * 5
    assert cosine_scores([math.nan] * 64, rows) == [None] * len(rows)


def test_scoring_failure_releases_cursor_and_store_lock(tmp_path, monkeypatch):
    import threading

    store = make_store(tmp_path)
    event(store, "match", "清静的环境")
    store.backfill_embeddings()

    def fail(*args, **kwargs):
        raise RuntimeError("fixture_scoring_failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(store, "_hybrid_record", fail)
        with pytest.raises(RuntimeError, match="fixture_scoring_failure"):
            store._semantic_candidates(query="宁静的地方", terms=[], scope="private:one",
                                        selected_layers=(), start_at=None, end_at=None,
                                        topic_batch_id="", sender_id="", now=3000)
    completed = threading.Event()

    def write_after_failure():
        event(store, "after-failure")
        completed.set()

    thread = threading.Thread(target=write_after_failure, daemon=True)
    thread.start()
    thread.join(2)
    assert completed.is_set()
