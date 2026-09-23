"""Download and verify the optional local embedding model for private Shinku."""

from __future__ import annotations

import argparse
from pathlib import Path

from shinku.providers.huggingface import (
    DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,
    HuggingFaceEmbeddingProvider,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_HUGGINGFACE_EMBEDDING_MODEL)
    parser.add_argument(
        "--cache-dir",
        default=str(Path("private") / "data" / "memory" / "models" / "bge-m3"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"model : {args.model}")
    print(f"cache : {cache_dir}")
    print(f"device: {args.device}")
    provider = HuggingFaceEmbeddingProvider(
        model_name=args.model,
        device=args.device or None,
        local_files_only=args.local_files_only,
        cache_folder=str(cache_dir),
        hf_endpoint=args.endpoint or None,
    )
    vectors = provider.embed_texts(["我想找一个安静的地方", "之前讨论过的记忆检索问题"])
    print(f"ready : dimension={provider.dimension}, vectors={len(vectors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
