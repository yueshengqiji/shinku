"""Build the optional local embedding provider for one runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shinku.providers.embedding import CachedEmbeddingProvider, HashedEmbeddingProvider
from shinku.providers.huggingface import HuggingFaceEmbeddingProvider

from .policy import MemoryPolicy

__all__ = ["EmbeddingBuild", "build_embedding_provider"]


@dataclass(frozen=True)
class EmbeddingBuild:
    provider: Any | None
    status: str
    error: str = ""


def build_embedding_provider(policy: MemoryPolicy, *, default_cache_dir: Path) -> EmbeddingBuild:
    """Build a provider without making the optional ML dependency mandatory.

    A missing local model/dependency falls back to the deterministic hashed
    provider.  The store exposes that degraded status, so the UI never claims
    that a real semantic model is active when it is not.
    """

    if not policy.semantic_enabled:
        return EmbeddingBuild(None, "disabled")
    provider_name = str(policy.semantic_provider or "huggingface").strip().lower()
    if provider_name in {"hash", "hashed", "fallback"}:
        return EmbeddingBuild(
            CachedEmbeddingProvider(HashedEmbeddingProvider()),
            "hashed",
        )
    if provider_name not in {"huggingface", "sentence-transformers", "bge"}:
        return EmbeddingBuild(None, "unsupported", f"unknown_embedding_provider:{provider_name}")
    cache_dir = Path(policy.embedding_cache_dir) if policy.embedding_cache_dir else default_cache_dir
    try:
        provider = HuggingFaceEmbeddingProvider(
            model_name=policy.embedding_model,
            device=policy.embedding_device or None,
            local_files_only=policy.embedding_local_files_only,
            cache_folder=str(cache_dir),
        )
    except Exception as exc:
        return EmbeddingBuild(
            CachedEmbeddingProvider(HashedEmbeddingProvider()),
            "fallback_hashed",
            str(exc)[:240],
        )
    return EmbeddingBuild(CachedEmbeddingProvider(provider), "huggingface")
