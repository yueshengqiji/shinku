"""混排通过真实 SQLite 检索验证；夹具只模拟可控向量，不接触本机记忆。"""
from __future__ import annotations

import math

import pytest

from shinku.memory import MemoryService, MemoryStore
from shinku.memory.lifecycle_policy import LifecyclePolicy


class RankingProvider:
    name = "ranking-fixture"
    version = "v1"
    dimension = 2

    def collection_key(self):
        return "ranking_fixture_v1_2"

    def embed_text(self, text):
        if text == "alpha beta":
            return [1.0, 0.0]
        if "quiet workspace" in text:
            return [0.9, math.sqrt(0.19)]
        return [0.0, 1.0]

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


def make_store(tmp_path):
    return MemoryStore(tmp_path / "memory.sqlite3", embedding_provider=RankingProvider(),
                       semantic_enabled=True, event_scan_limit=10, fact_scan_limit=10,
                       lifecycle_policy=LifecyclePolicy(mode="off"))


def event(store, source, text, *, at=100, scope="private:one", **kwargs):
    store.record_event(conversation_id=scope, scope=scope, source_id=source,
                       text=text, created_at=at, **kwargs)


@pytest.mark.parametrize("noise_kind", ["event", "fact"])
def test_old_semantic_target_survives_unscaled_lexical_noise(tmp_path, noise_kind):
    """词法两词命中的裸分曾超过整路语义满分，把窗口外目标挤出前十。"""
    store = make_store(tmp_path)
    event(store, "old-target", "I prefer a quiet workspace", at=10)
    for i in range(24):
        if noise_kind == "event":
            event(store, f"noise-{i}", "alpha beta unrelated shipment", at=100+i)
        else:
            store.upsert_fact(conversation_id="one", scope="private:one", fact_key=f"noise-{i}",
                              source_id=f"noise-{i}", text="alpha beta unrelated shipment", now=100+i)
    store.backfill_embeddings(limit=100)
    rows = store.search(query="alpha beta", scope="private:one", limit=10, now=1000)
    assert rows[0].source_id == "old-target"
    assert rows[0].metadata["retrieval_method"] == "hybrid"
    assert sum(row.source_id == "old-target" for row in rows) == 1


@pytest.mark.parametrize("search_keys", [False, True])
def test_unindexed_lexical_match_is_retained_without_inventing_semantic_evidence(tmp_path, monkeypatch, search_keys):
    store = make_store(tmp_path)
    event(store, "semantic", "I prefer a quiet workspace")
    event(store, "unindexed", "alpha beta exact local identifier", at=101)
    store.backfill_embeddings(limit=100)
    with store._connect() as conn:
        conn.execute("DELETE FROM memory_embeddings WHERE record_id IN "
                     "(SELECT event_id FROM memory_events WHERE source_id='unindexed')")
    monkeypatch.setattr(store, "backfill_embeddings", lambda **kwargs: 0)
    store.configure_search_keys(enabled=search_keys)
    rows = store.search(query="alpha beta", scope="private:one", limit=2, now=1000)
    assert {row.source_id for row in rows} == {"semantic", "unindexed"}
    unindexed = next(row for row in rows if row.source_id == "unindexed")
    assert unindexed.metadata.get("retrieval_method") != "hybrid"
    assert "semantic_similarity" not in unindexed.metadata
    assert 0 <= unindexed.score <= 1
    assert unindexed.text == "alpha beta exact local identifier"
    if search_keys:
        assert unindexed.metadata["source_version"]


@pytest.mark.parametrize("mode", ["disabled", "no_matches", "provider_failure", "require_overlap"])
def test_lexical_fallback_remains_available_without_semantic_evidence(tmp_path, monkeypatch, mode):
    store = make_store(tmp_path)
    event(store, "exact", "alpha beta identifier", at=100)
    event(store, "partial", "alpha identifier", at=200)
    store.backfill_embeddings(limit=100)
    store.semantic_enabled = False
    before = store.search(query="alpha beta", scope="private:one", limit=5, now=1000)
    store.semantic_enabled = mode != "disabled"
    if mode == "provider_failure":
        def fail(text):
            raise RuntimeError("synthetic_encoder_failure")
        monkeypatch.setattr(store._embedding_provider, "embed_text", fail)
    after = store.search(query="alpha beta", scope="private:one", limit=5, now=1000,
                         require_term_overlap=mode == "require_overlap")
    assert [row.record_id for row in after] == [row.record_id for row in before]
    if mode in {"disabled", "require_overlap"}:
        assert [row.score for row in after] == [row.score for row in before]
    else:
        # 查询仍启用语义时，零命中/故障也需与其他 scope 的结果保持同一尺度。
        assert all(0 <= row.score <= 1 for row in after)
        assert all("semantic_similarity" not in row.metadata for row in after)


def test_scope_without_semantic_hits_cannot_reintroduce_raw_scores(tmp_path):
    store = make_store(tmp_path)
    event(store, "owner-target", "I prefer a quiet workspace", scope="private:owner")
    event(store, "group-noise", "alpha beta unrelated shipment", scope="group:one")
    store.backfill_embeddings(limit=100)
    owner = store.search(query="alpha beta", scope="private:owner", now=1000)
    group = store.search(query="alpha beta", scope="group:one", now=1000)
    rows = MemoryService._merge_by_source(owner, group, limit=1)
    assert rows[0].source_id == "owner-target"


def test_semantic_copy_deduplicates_without_losing_its_score_or_provenance(tmp_path):
    store = make_store(tmp_path)
    event(store, "both", "alpha beta quiet workspace", metadata={"sender_id": "one", "role": "user"})
    event(store, "other", "alpha beta unrelated shipment", at=101)
    rows = store.search(query="alpha beta", scope="private:one", limit=3, now=1000)
    assert [row.source_id for row in rows] == ["both", "other"]
    assert rows[0].metadata["sender_id"] == "one"
    assert rows[0].metadata["semantic_similarity"] == pytest.approx(0.9)


def test_calibrated_merge_keeps_scope_date_layer_and_forget_guards(tmp_path):
    store = make_store(tmp_path)
    event(store, "allowed", "quiet workspace", at=101)
    event(store, "other-scope", "quiet workspace", at=101, scope="private:two")
    event(store, "end-exclusive", "quiet workspace", at=102)
    event(store, "other-layer", "quiet workspace", at=101, layer="summary")
    event(store, "revoked", "quiet workspace withdrawn", at=101)
    event(store, "noise", "alpha beta unrelated shipment", at=101)
    store.backfill_embeddings(limit=100)
    store.forget(query="withdrawn", scope="private:one", now=1000)
    rows = store.search(query="alpha beta", scope="private:one", start_at=100, end_at=102,
                         layers={"episodic"}, limit=10, now=1000)
    assert [row.source_id for row in rows] == ["allowed", "noise"]
