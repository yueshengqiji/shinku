from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shinku.memory import MemoryStore


class _FakeSemanticProvider:
    name = "fake"
    version = "v1"
    dimension = 2

    def collection_key(self) -> str:
        return "fake_v1_2"

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts):
        values = []
        for text in texts:
            body = str(text)
            if any(piece in body for piece in ("安静", "宁静", "清静")):
                values.append([1.0, 0.0])
            elif any(piece in body for piece in ("吵", "喧闹")):
                values.append([0.0, 1.0])
            else:
                values.append([0.7, 0.7])
        return values


class MemorySemanticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(
            Path(self.tmp.name) / "memory.sqlite3",
            embedding_provider=_FakeSemanticProvider(),
            semantic_enabled=True,
        )

    def tearDown(self) -> None:
        self.store.stop_embedding_worker()
        self.tmp.cleanup()

    def test_semantic_search_finds_a_paraphrase_and_persists_vector(self) -> None:
        self.store.record_event(
            conversation_id="p-1",
            scope="private:p-1",
            text="我想要一个安静的环境",
            source_id="event-1",
            importance=0.8,
        )
        results = self.store.search(
            query="我想找个宁静的地方",
            scope="private:p-1",
            limit=3,
        )
        self.assertTrue(results)
        self.assertEqual(results[0].source_id, "event-1")
        self.assertEqual(results[0].metadata["retrieval_method"], "hybrid")
        self.assertGreater(results[0].metadata["semantic_similarity"], 0.99)
        self.assertEqual(self.store.semantic_status()["active_vectors"], 1)

    def test_semantic_search_respects_scope_and_date_guards(self) -> None:
        self.store.record_event(
            conversation_id="p-1",
            scope="private:p-1",
            text="安静的环境",
            source_id="event-1",
            created_at=100,
        )
        self.store.record_event(
            conversation_id="p-2",
            scope="private:p-2",
            text="安静的环境",
            source_id="event-2",
            created_at=100,
        )
        self.assertEqual(
            [item.source_id for item in self.store.search(query="宁静的地方", scope="private:p-1")],
            ["event-1"],
        )
        self.assertEqual(
            self.store.search(
                query="宁静的地方",
                scope="private:p-1",
                start_at=200,
                end_at=300,
            ),
            [],
        )

    def test_reset_removes_vectors_without_removing_schema(self) -> None:
        self.store.record_event(
            conversation_id="p-1",
            scope="private:p-1",
            text="安静",
            source_id="event-1",
        )
        self.store.search(query="宁静", scope="private:p-1")
        self.store.reset_all()
        self.assertEqual(self.store.semantic_status()["vectors"], 0)
        self.assertEqual(self.store.stats()["embedding_vectors"], 0)

    def test_image_only_placeholders_are_not_embedded(self) -> None:
        self.store.record_event(
            conversation_id="p-1",
            scope="private:p-1",
            text="[图片附件]",
            source_id="image-only",
        )
        self.store.record_event(
            conversation_id="p-1",
            scope="private:p-1",
            text="这张图讨论的是安静的房间",
            source_id="with-caption",
        )
        self.store.backfill_embeddings(scope="private:p-1")
        self.assertEqual(self.store.semantic_status()["active_vectors"], 1)


if __name__ == "__main__":
    unittest.main()
