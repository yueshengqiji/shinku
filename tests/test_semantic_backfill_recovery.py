"""Index recovery must reach older gaps and keep model collections separate."""
from __future__ import annotations

import sqlite3

from shinku.memory import MemoryStore


class Provider:
    name = "fixture"
    version = "v1"
    dimension = 2

    def collection_key(self):
        return f"{self.name}_{self.version}_{self.dimension}"

    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_text(self, text):
        return self.embed_texts([text])[0]


def make_store(path, provider=None):
    return MemoryStore(path, embedding_provider=provider or Provider(), semantic_enabled=True)


def test_backfill_reaches_old_gaps_after_indexed_newer_window_and_restart(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = make_store(path)
    for n in range(90):
        store.record_event(conversation_id="one", scope="private:one", text=f"记录 {n}",
                           source_id=f"item-{n}", created_at=100+n)
    # A one-row batch previously saw only the newest eight rows forever.
    assert sum(store.backfill_embeddings(limit=1) for _ in range(45)) == 45
    reopened = make_store(path)
    assert sum(reopened.backfill_embeddings(limit=1) for _ in range(60)) == 45
    assert reopened.semantic_status()["active_vectors"] == 90
    assert reopened.backfill_embeddings(limit=1) == 0


def test_stale_old_vector_is_repaired_without_rewriting_current_vectors(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = make_store(path)
    for n in range(20):
        store.record_event(conversation_id="one", scope="private:one", text=f"记录 {n}",
                           source_id=f"item-{n}", created_at=100+n)
    assert store.backfill_embeddings(limit=100) == 20
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE memory_events SET text='旧内容已纠正' WHERE source_id='item-0'")
        before = conn.execute("SELECT embedding_id,vector_blob,updated_at FROM memory_embeddings "
                              "WHERE record_id IN (SELECT event_id FROM memory_events WHERE source_id != 'item-0')").fetchall()
    assert make_store(path).backfill_embeddings(limit=1) == 1
    with sqlite3.connect(path) as conn:
        after = conn.execute("SELECT embedding_id,vector_blob,updated_at FROM memory_embeddings "
                             "WHERE record_id IN (SELECT event_id FROM memory_events WHERE source_id != 'item-0')").fetchall()
    assert before == after
    assert store.backfill_embeddings(limit=1) == 0


def test_unindexable_new_rows_cannot_hide_older_scope(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = make_store(path)
    store.record_event(conversation_id="one", scope="private:one", text="需要索引的旧记录",
                       source_id="older", created_at=1)
    for n in range(20):
        store.record_event(conversation_id="one", scope="private:one", text="[图片附件]",
                           source_id=f"image-{n}", created_at=100+n)
        store.record_event(conversation_id="two", scope="private:two", text="别人的记录",
                           source_id=f"other-{n}", created_at=100+n)
    assert store.backfill_embeddings(limit=1, scope="private:one") == 1
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT DISTINCT scope FROM memory_embeddings").fetchall() == [("private:one",)]


def test_new_model_recovery_keeps_old_collection_and_uses_new_identity(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = make_store(path)
    store.record_event(conversation_id="one", scope="private:one", text="保留索引",
                       source_id="source", created_at=100)
    assert store.backfill_embeddings() == 1
    provider = Provider()
    provider.version = "v2"
    newer = make_store(path, provider)
    assert newer.semantic_status()["vectors"] == 0
    assert newer.backfill_embeddings() == 1
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT model_key,count(*) FROM memory_embeddings GROUP BY model_key").fetchall() == [
            ("fixture_v1_2", 1), ("fixture_v2_2", 1)]


def test_backfill_reaches_old_facts_after_newer_facts_are_current(tmp_path):
    store = make_store(tmp_path / "memory.sqlite3")
    for n in range(20):
        store.upsert_fact(conversation_id="one", scope="private:one", text=f"事实 {n}",
                          fact_key=f"fact-{n}", now=100+n)
    assert sum(store.backfill_embeddings(limit=1) for _ in range(25)) == 20
    assert store.semantic_status()["active_vectors"] == 20


def test_encoding_failure_keeps_keyword_search_available_and_does_not_write_vectors(tmp_path):
    class BrokenProvider(Provider):
        def embed_texts(self, texts):
            raise RuntimeError("fixture_encoder_failed")

    store = make_store(tmp_path / "memory.sqlite3", BrokenProvider())
    store.record_event(conversation_id="one", scope="private:one", text="离线检索测试",
                       source_id="recoverable")
    assert store.backfill_embeddings() == 0
    assert store.semantic_status()["vectors"] == 0
    results = store.search(query="离线检索", scope="private:one")
    assert results[0].source_id == "recoverable"
    assert results[0].metadata.get("retrieval_method") != "hybrid"


def test_missing_model_degrades_to_distinct_hashed_collection(tmp_path, monkeypatch):
    from shinku.memory import embedding_factory
    from shinku.memory.policy import MemoryPolicy

    def missing_model(**kwargs):
        assert kwargs["local_files_only"] is True
        raise OSError("fixture_local_model_missing")

    monkeypatch.setattr(embedding_factory, "HuggingFaceEmbeddingProvider", missing_model)
    built = embedding_factory.build_embedding_provider(
        MemoryPolicy(semantic_enabled=True, embedding_local_files_only=True), default_cache_dir=tmp_path)
    assert built.status == "fallback_hashed"
    assert built.error == "fixture_local_model_missing"
    store = MemoryStore(tmp_path / "memory.sqlite3", embedding_provider=built.provider,
                        semantic_enabled=True, embedding_status=built.status, embedding_error=built.error)
    status = store.semantic_status()
    assert status["status"] == "fallback_hashed"
    assert status["model_key"] == "hashed_v1_128"
    assert status["dimension"] == 128
