"""Memory construction and embedding-worker startup phase."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

@dataclass(frozen=True)
class MemoryAssemblyDependencies:
    MemoryService: Callable[..., Any]
    MemoryStore: Callable[..., Any]
    MemoryRouter: Callable[..., Any]
    LLMMemoryJudge: Callable[..., Any]
    LLMLifecycleReviewer: Callable[..., Any]
    build_embedding_provider: Callable[..., Any]


def build_memory_service(config, runtime, memory_policy, *, deps: MemoryAssemblyDependencies, character=None):
    memory_service = None
    if config.memory_enabled:
        db_path = Path(config.memory_db_path) if config.memory_db_path else Path("data") / "memory" / "shinku_memory.sqlite3"
        embedding_build = deps.build_embedding_provider(
            memory_policy,
            default_cache_dir=db_path.parent / "models" / "bge-m3",
        )
        judge = (
            deps.LLMMemoryJudge(runtime, recent_context_chars=memory_policy.recent_context_chars)
            if config.memory_judge_enabled
            else None
        )
        memory_service = deps.MemoryService(
            store=deps.MemoryStore(
                db_path,
                embedding_provider=embedding_build.provider,
                semantic_enabled=memory_policy.semantic_enabled,
                semantic_backfill_batch_size=memory_policy.semantic_backfill_batch_size,
                semantic_min_score=memory_policy.semantic_min_score,
                embedding_status=embedding_build.status,
                embedding_error=embedding_build.error,
            ),
            router=deps.MemoryRouter(
                judge=judge,
                deterministic_guards=memory_policy.deterministic_guards,
                max_query_chars=memory_policy.max_query_chars,
            ),
            policy=memory_policy,
            **({"character": character} if character is not None else {}),
            lifecycle_reviewer=(
                deps.LLMLifecycleReviewer(runtime)
                if memory_policy.lifecycle.semantic_review_enabled else None
            ),
        )
        if embedding_build.provider is not None and memory_service.store.semantic_enabled:
            memory_service.store.start_embedding_worker(
                interval_seconds=memory_policy.semantic_index_interval_seconds,
            )
        projection = getattr(memory_service, "projection", None)
        if projection is not None and memory_policy.projection.semantic_enabled:
            from shinku.memory.projection.semantics import ProjectionModel
            projection.semantic.model = ProjectionModel(runtime)
    return memory_service
