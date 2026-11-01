"""覆盖后台 L1 索引、无固定前缀的投影，以及编码/读取期间的来源失效。"""
from __future__ import annotations

import pytest

from shinku.memory import MemoryStore
from shinku.memory.lifecycle_policy import LifecyclePolicy
from shinku.memory.projection.archive import ArchiveLocator
from shinku.memory.projection.contracts import Deadline, ProjectionPolicy, ProjectionRequest
from shinku.memory.semantic import pack_vector
from test_memory_partial_vector_coverage import NOW, PARAPHRASE, SCOPE, Provider, put, vector


def make_store(tmp_path, provider=None):
    return MemoryStore(tmp_path / "memory.sqlite3", semantic_enabled=True,
                       embedding_provider=provider or Provider(), semantic_min_score=0.5,
                       event_scan_limit=10, semantic_backfill_batch_size=8,
                       lifecycle_policy=LifecyclePolicy(mode="active", defer_l1_embeddings=True))


def enroll(store, source, text=PARAPHRASE, *, at=NOW, scope=SCOPE):
    row = put(store, source, text, at=at, scope=scope)
    store.lifecycle.enroll(record_id=row.record_id, now=NOW)
    return row


def request(**changes):
    return ProjectionRequest(**{
        "scope": SCOPE, "allowed_scopes": (SCOPE,), "requester": "owner",
        "self_id": "bot", "query": "宁静", "now": NOW,
        "message_id": "current", "history_mode": "retrieve", **changes})


def run_worker(store, monkeypatch, cycles):
    """同步驱动真实 worker 回调，保留实际回填链路，不靠墙钟等待线程。"""
    class Stop:
        stopped = False
        count = 0
        def clear(self):
            self.stopped = False
        def set(self):
            self.stopped = True
        def is_set(self):
            return self.stopped
        def wait(self, interval):
            self.count += 1
            self.stopped = self.stopped or self.count >= cycles
    class Thread:
        def __init__(self, *, target, **kwargs):
            self.target = target
        def is_alive(self):
            return False
        def start(self):
            self.target()
    monkeypatch.setattr(store, "_embedding_worker_stop", Stop())
    with monkeypatch.context() as patch:
        patch.setattr("shinku.memory.store_semantic.threading.Thread", Thread)
        store.start_embedding_worker()


def test_background_worker_covers_l1_beyond_recent_and_cache_windows(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    target = enroll(store, "target", at=NOW - 1000)
    for i in range(160):
        enroll(store, f"noise-{i}", "无关散步", at=NOW - 200 + i)
    before = store.lifecycle.state(target.record_id)
    assert store.backfill_embeddings(limit=1000) == 0  # 前台仍延后 L1 持久化。
    run_worker(store, monkeypatch, 21)
    batches = list(store.embedding_provider.batches)
    assert len(batches) == 21 and max(map(len, batches)) <= 8
    assert store.semantic_status()["vectors"] == 161
    rows = store.search(query="宁静", scope=SCOPE, now=NOW, limit=1)
    assert rows[0].record_id == target.record_id
    assert store.embedding_provider.batches == batches  # 已覆盖来源无需再临时编码。
    assert store.lifecycle.state(target.record_id) == before
    # 新 Store 不依赖旧进程的 RAM 缓存；投影复用同一来源索引。
    restarted = MemoryStore(store.path, semantic_enabled=True, embedding_provider=Provider(),
                            lifecycle_policy=store.lifecycle.policy, semantic_min_score=0.5)
    archive = ArchiveLocator(restarted, ProjectionPolicy(vector_scan_limit=2))
    try:
        handles, coverage, _ = archive.discover(request(), Deadline(2000), limit=3)
        assert (target.record_id, SCOPE, "") in handles
        assert not coverage["vector_truncated"]
    finally:
        archive.stop()


@pytest.mark.parametrize("mutation", ["forget", "correct", "delete", "scope", "source", "model"])
def test_background_encoding_rechecks_source_and_model_before_commit(tmp_path, mutation):
    provider = Provider()
    store = make_store(tmp_path, provider)
    target = enroll(store, "target")
    original = provider.embed_texts
    def encode(texts):
        result = original(texts)
        if mutation == "forget":
            store.forget(query="", scope=SCOPE, record_id=target.record_id, now=NOW)
        elif mutation == "model":
            provider.collection_key = lambda: "new-model"
        else:
            with store._connect() as conn:
                if mutation == "delete":
                    conn.execute("DELETE FROM memory_events WHERE source_id='target'")
                elif mutation == "scope":
                    conn.execute("UPDATE memory_events SET scope='private:other' WHERE source_id='target'")
                elif mutation == "source":
                    conn.execute("UPDATE memory_events SET source_id='owner:agent-task-detail:hidden' WHERE source_id='target'")
                else:
                    conn.execute("UPDATE memory_events SET text='已更正内容' WHERE source_id='target'")
        return result
    provider.embed_texts = encode
    assert store.backfill_embeddings(limit=8, include_deferred_l1=True) == 0
    with store._connect() as conn:
        assert conn.execute("SELECT count(*) FROM memory_embeddings").fetchone()[0] == 0


def test_background_repair_is_idempotent_and_forgotten_l1_does_not_reappear(tmp_path):
    store = make_store(tmp_path)
    target = enroll(store, "target")
    assert store.backfill_embeddings(include_deferred_l1=True) == 1
    assert store.backfill_embeddings(include_deferred_l1=True) == 0
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET text='a tranquil workspace revised' WHERE source_id='target'")
    assert store.backfill_embeddings(include_deferred_l1=True) == 1
    store.forget(query="", scope=SCOPE, record_id=target.record_id, now=NOW)
    assert store.backfill_embeddings(include_deferred_l1=True) == 0
    assert store.semantic_status()["vectors"] == 0
    assert store.search(query="宁静", scope=SCOPE, now=NOW) == []


def test_worker_recovers_after_encoding_failure_without_foreground_rebuild(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    enroll(store, "target")
    provider = store.embedding_provider
    original = provider.embed_texts
    attempts = []
    def encode(texts):
        attempts.append(tuple(texts))
        if len(attempts) == 1:
            raise RuntimeError("synthetic encoder unavailable")
        return original(texts)
    provider.embed_texts = encode
    run_worker(store, monkeypatch, 3)
    assert len(attempts) == 2
    assert store.semantic_status()["vectors"] == 1
    assert store.backfill_embeddings() == 0


def test_worker_stop_during_batch_does_not_start_another_batch(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    for i in range(12):
        enroll(store, f"row-{i}")
    original = store.embedding_provider.embed_texts
    def encode(texts):
        store.stop_embedding_worker()
        return original(texts)
    store.embedding_provider.embed_texts = encode
    run_worker(store, monkeypatch, 4)
    assert len(store.embedding_provider.batches) == 1
    assert store.semantic_status()["vectors"] == 8


@pytest.mark.parametrize("batch_size", [1, 2, 256])
def test_projection_scores_all_batches_before_top_k(tmp_path, batch_size):
    store = make_store(tmp_path)
    for i in range(7):
        row = enroll(store, f"row-{i}", PARAPHRASE if i == 6 else "无关散步", at=NOW - i)
        vector(store, row, identity=f"ordered-{i}", values=store.embedding_provider.embed_text(row.text))
    archive = ArchiveLocator(store, ProjectionPolicy(vector_scan_limit=batch_size))
    try:
        handles, coverage, gaps = archive.discover(request(), Deadline(2000), limit=2)
        assert (row.record_id, SCOPE, "") in handles
        assert coverage["vector_rows"] == 7 and not coverage["vector_truncated"]
        assert not gaps and not store.embedding_provider.batches
    finally:
        archive.stop()


def test_projection_full_scan_keeps_scope_time_hash_and_invalid_vector_guards(tmp_path):
    store = make_store(tmp_path)
    for i in range(8):
        row = enroll(store, f"row-{i}", at=NOW - 10 if i != 3 else NOW,
                     scope="private:other" if i == 0 else SCOPE)
        vector(store, row, identity=f"ordered-{i}", values=[0.9, 0.435889894])
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET status='forgotten' WHERE source_id='row-1'")
        conn.execute("UPDATE memory_events SET text='已更正' WHERE source_id='row-2'")
        conn.execute("UPDATE memory_embeddings SET vector_blob=? WHERE embedding_id='ordered-4'", (b"broken",))
        conn.execute("UPDATE memory_embeddings SET vector_blob=? WHERE embedding_id='ordered-5'", (pack_vector([float("nan"), 1]),))
        conn.execute("UPDATE memory_embeddings SET vector_blob=? WHERE embedding_id='ordered-6'", (pack_vector([0, 0]),))
    archive = ArchiveLocator(store, ProjectionPolicy(vector_scan_limit=2))
    try:
        handles, coverage, _ = archive.discover(request(start_at=NOW - 20, end_at=NOW), Deadline(2000), limit=5)
        assert handles == ((row.record_id, SCOPE, ""),)
        assert not coverage["vector_truncated"]
    finally:
        archive.stop()


def test_projection_deadline_returns_partial_coverage_without_claiming_absence(tmp_path, monkeypatch):
    import shinku.memory.projection.archive as module
    store = make_store(tmp_path)
    for i in range(6):
        row = enroll(store, f"row-{i}")
        vector(store, row, identity=f"ordered-{i}", values=[0.9, 0.435889894])
    archive = ArchiveLocator(store, ProjectionPolicy(vector_scan_limit=2))
    deadline = Deadline(2000)
    original = module.cosine_scores
    def score(query, rows):
        scores = original(query, rows)
        deadline.until = 0
        return scores
    monkeypatch.setattr(module, "cosine_scores", score)
    try:
        _, coverage, gaps = archive.discover(request(), deadline, limit=2)
        assert coverage["vector_truncated"] and coverage["vector_rows"] == 2
        assert coverage["absence_means"] == "unknown"
        assert any(gap["reason"] == "budget_exhausted" for gap in gaps)
    finally:
        archive.stop()
