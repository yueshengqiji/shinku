"""用真实 SQLite 和可控向量定位候选缺口，不宣称自然语义质量。"""
from __future__ import annotations

import hashlib
import math

import pytest

from shinku.memory import MemoryStore
from shinku.memory.lifecycle_policy import LifecyclePolicy
from shinku.memory.projection.archive import ArchiveLocator
from shinku.memory.projection.contracts import Deadline, ProjectionPolicy, ProjectionRequest
from shinku.memory.retrieval_trace import capture_retrieval
from shinku.memory.semantic import pack_vector

NOW = 1_800_000_000
SCOPE = "private:owner"
QUERY = "枫桥资料"
TARGET = "枫桥资料只保存在离线磁盘，联网安装尚未得到同意。"
PARAPHRASE = "a tranquil workspace"


class Provider:
    """文字目标刻意低于向量门槛；确保文字路不会借用伪造的语义分数。"""
    dimension = 2

    def __init__(self):
        self.batches = []

    def collection_key(self):
        return "partial_vector_fixture"

    def embed_text(self, text):
        return [0.9, math.sqrt(0.19)] if text in (QUERY, "宁静", PARAPHRASE) else [0.0, 1.0]

    def embed_texts(self, texts):
        self.batches.append(tuple(texts))
        return [self.embed_text(text) for text in texts]


def put(store, source, text, *, kind="event", at=NOW, scope=SCOPE):
    if kind == "event":
        store.record_event(conversation_id="owner", scope=scope, source_id=source,
                           text=text, created_at=at, metadata={"role": "user", "sender_id": "owner"})
        return store.event_by_source_id(scope=scope, source_id=source)
    store.upsert_fact(conversation_id="owner", scope=scope, source_id=source,
                      fact_key=source, text=text, now=at)
    with store._connect() as conn:
        row = conn.execute("SELECT * FROM memory_facts WHERE scope=? AND fact_key=?", (scope, source)).fetchone()
    return store._fact_record(row, score=0.0)


def vector(store, record, *, identity, values, stale=False):
    kind, bare = record.record_id.split(":", 1)
    with store._connect() as conn:
        conn.execute("INSERT INTO memory_embeddings VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (identity, kind, bare, record.scope, store.embedding_model_key,
                      "stale" if stale else hashlib.sha256(record.text.encode()).hexdigest(),
                      2, pack_vector(values), record.created_at, record.created_at))


def event_at(trace, stage):
    return next(item for item in trace.events if item["stage"] == stage)


@pytest.mark.parametrize("kind", ["event", "fact"])
@pytest.mark.parametrize("index_state", ["missing", "stale", "below_threshold"])
@pytest.mark.parametrize("search_keys", [False, True])
def test_old_literal_source_survives_other_semantic_hits(tmp_path, kind, index_state, search_keys):
    provider = Provider()
    store = MemoryStore(tmp_path / "memory.sqlite3", semantic_enabled=True, embedding_provider=provider,
                        semantic_backfill_batch_size=8, semantic_min_score=0.5,
                        event_scan_limit=10, fact_scan_limit=10, lifecycle_policy=LifecyclePolicy(mode="off"))
    target = put(store, "target", TARGET, kind=kind, at=NOW - 40 * 86400)
    other = put(store, "semantic", PARAPHRASE, at=NOW - 1)
    vector(store, other, identity="semantic-vector", values=provider.embed_text(other.text))
    if index_state != "missing":
        vector(store, target, identity="target-vector", values=[0.0, 1.0], stale=index_state == "stale")
    for i in range(25):
        put(store, f"noise-{i}", f"今日安排散步与整理书桌第{i}项", kind=kind, at=NOW - 100 + i)
    store.configure_search_keys(enabled=search_keys)
    # 保留真实的有界回填：它只处理较新的 8 项，目标仍有缺口。
    with capture_retrieval(watch=((target.record_id, SCOPE),)) as trace:
        rows = store.search(query=QUERY, scope=SCOPE, now=NOW, limit=3)
    assert event_at(trace, "store.semantic")["count"] > 0
    assert not event_at(trace, "store.lexical")["watch"][0]["present"]
    assert not event_at(trace, "store.semantic")["watch"][0]["present"]
    assert event_at(trace, "store.ranked")["watch"][0]["present"]
    result = next(row for row in rows if row.record_id == target.record_id)
    assert result.text == TARGET and "semantic_similarity" not in result.metadata
    assert result.metadata["retrieval_method"] == "lexical"
    assert provider.batches and all(TARGET not in batch for batch in provider.batches)


def test_independent_lexical_channel_keeps_semantic_scope_date_layer_and_revocation(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3", semantic_enabled=True, embedding_provider=Provider(),
                        semantic_min_score=0.5, event_scan_limit=10,
                        lifecycle_policy=LifecyclePolicy(mode="off"))
    target = put(store, "target", TARGET, at=NOW - 200)
    put(store, "foreign", TARGET, at=NOW - 200, scope="private:other")
    put(store, "outside", TARGET, at=NOW)
    put(store, "owner:agent-task-detail:hidden", TARGET, at=NOW - 200)
    summary = put(store, "summary", TARGET, at=NOW - 200)
    revoked = put(store, "revoked", TARGET, at=NOW - 200)
    with store._connect() as conn:
        conn.execute("UPDATE memory_events SET layer='summary' WHERE event_id=?",
                     (summary.record_id.removeprefix("event:"),))
    for i in range(25):
        put(store, f"noise-{i}", "今日天气", at=NOW - 100 + i)
    other = put(store, "semantic", PARAPHRASE, at=NOW - 1)
    store.backfill_embeddings(limit=100)
    store.forget(query="", scope=SCOPE, record_id=revoked.record_id, now=NOW)
    with capture_retrieval(watch=((target.record_id, SCOPE),)) as trace:
        rows = store.search(query=QUERY, scope=SCOPE, now=NOW, limit=2,
                            start_at=NOW - 300, end_at=NOW, layers={"episodic"})
    assert {row.record_id for row in rows} == {target.record_id, other.record_id}
    assert event_at(trace, "store.lexical_fallback")["reason"] == "independent_lexical_channel"


@pytest.mark.parametrize("mode,tier", [("off", 1), ("shadow", 1), ("active", 2), ("active", 3)])
def test_persisted_vectors_keep_cold_paraphrase_coverage(tmp_path, mode, tier):
    """受控层级只验证候选准入；不把人工设置层级当作实际晋级实验。"""
    provider = Provider()
    store = MemoryStore(tmp_path / "memory.sqlite3", semantic_enabled=True, embedding_provider=provider,
                        semantic_min_score=0.5, event_scan_limit=10,
                        lifecycle_policy=LifecyclePolicy(mode=mode, defer_l1_embeddings=True))
    target = put(store, "target", PARAPHRASE, at=NOW - 1000)
    if mode != "off":
        store.lifecycle.enroll(record_id=target.record_id, now=NOW)
        with store._connect() as conn:
            conn.execute("UPDATE memory_lifecycle SET tier=? WHERE record_id=?",
                         (tier, target.record_id.removeprefix("event:")))
    vector(store, target, identity="target-vector", values=provider.embed_text(target.text))
    for i in range(25):
        put(store, f"noise-{i}", "无关天气", at=NOW - 100 + i)
    with capture_retrieval(watch=((target.record_id, SCOPE),)) as trace:
        rows = store.search(query="宁静", scope=SCOPE, now=NOW)
    assert rows[0].record_id == target.record_id
    assert event_at(trace, "store.semantic")["watch"][0]["present"]
    assert rows[0].metadata["retrieval_method"] == "hybrid"


@pytest.mark.parametrize("within_window", [False, True])
def test_deferred_l1_paraphrase_window_is_explicit_and_bounded(tmp_path, within_window):
    """窗口外纯转述仍是已知缺口；不可把未编码来源误报成已经语义比对。"""
    provider = Provider()
    store = MemoryStore(tmp_path / "memory.sqlite3", semantic_enabled=True, embedding_provider=provider,
                        semantic_min_score=0.5, event_scan_limit=10,
                        lifecycle_policy=LifecyclePolicy(mode="active", defer_l1_embeddings=True))
    target = put(store, "target", PARAPHRASE, at=NOW if within_window else NOW - 1000)
    store.lifecycle.enroll(record_id=target.record_id, now=NOW)
    for i in range(64):
        row = put(store, f"noise-{i}", f"今日散步第{i}次", at=NOW - 100 + i)
        store.lifecycle.enroll(record_id=row.record_id, now=NOW)
    with capture_retrieval(watch=((target.record_id, SCOPE),)) as trace:
        rows = store.search(query="宁静", scope=SCOPE, now=NOW)
    stage = event_at(trace, "store.l1_candidates")
    assert stage["vector_truncated"] and stage["count"] == 64
    assert stage["watch"][0]["present"] is within_window
    assert (target.record_id in {row.record_id for row in rows}) is within_window
    assert len(provider.batches) == 1 and len(provider.batches[0]) == 64
    assert store.semantic_status()["vectors"] == 0


@pytest.mark.parametrize("scan_limit", [2, 3])
def test_projection_vector_batches_do_not_exclude_later_matches(tmp_path, scan_limit):
    provider = Provider()
    store = MemoryStore(tmp_path / "memory.sqlite3", semantic_enabled=True, embedding_provider=provider,
                        semantic_min_score=0.5, lifecycle_policy=LifecyclePolicy(mode="off"))
    for i in range(3):
        row = put(store, f"record-{i}", PARAPHRASE if i == 2 else "无关天气", at=NOW - i)
        vector(store, row, identity=f"ordered-{i}", values=provider.embed_text(row.text))
        if i == 2:
            target = row
    # 外部作用域的向量排在最前，也不能占用当前 scope 的扫描预算。
    foreign = put(store, "foreign", PARAPHRASE, scope="private:other")
    vector(store, foreign, identity="aaa-foreign", values=provider.embed_text(foreign.text))
    archive = ArchiveLocator(store, ProjectionPolicy(vector_scan_limit=scan_limit))
    request = ProjectionRequest(scope=SCOPE, allowed_scopes=(SCOPE,), requester="owner", self_id="bot",
                                query="宁静", now=NOW, message_id="current", history_mode="retrieve")
    try:
        handles, coverage, gaps = archive.discover(request, Deadline(1000), limit=3)
        # 旧参数现在限制每批读取，相关目标不能被固定前缀排除。
        assert (target.record_id, SCOPE, "") in handles
        assert not coverage["vector_truncated"]
        assert coverage["vector_rows"] == 3
        assert coverage["vector_batch_size"] == scan_limit
        assert coverage["absence_means"] == "unknown" and not gaps
        assert not provider.batches  # 投影只编码查询，不维护或编码来源库。
        assert all(handle[1] == SCOPE for handle in handles)
    finally:
        archive.stop()
