"""Bounded vector arithmetic, with acceleration only when NumPy is already loaded.

The core still works without ML packages. Real embedding providers already load
NumPy; their scans use small float64 batches without retaining a corpus matrix.
"""
from __future__ import annotations

import math
import sys
from typing import Any, Sequence

from .semantic import cosine_similarity, unpack_vector


def cosine_scores(query: Sequence[float], rows: Sequence[Any]) -> list[float | None]:
    """Score one batch; malformed, nonfinite and zero vectors are ineligible."""
    scores: list[float | None] = [None] * len(rows)
    try:
        query = [float(value) for value in query]
        norm = math.sqrt(sum(value * value for value in query))
    except (TypeError, ValueError, OverflowError):
        return scores
    if not norm or not math.isfinite(norm):
        return scores
    positions, vectors = [], []
    for position, row in enumerate(rows):
        try:
            dimension = int(row["dimension"])
            blob = row["vector_blob"]
            if dimension != len(query) or len(blob) != dimension * 4:
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        positions.append(position)
        vectors.append(blob)
    np = sys.modules.get("numpy") if len(query) >= 32 else None
    if np is not None and vectors:
        matrix = np.stack([np.frombuffer(blob, dtype="<f4") for blob in vectors]).astype(np.float64)
        finite = np.isfinite(matrix).all(axis=1)
        matrix[~finite] = 0.0
        norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
        valid = finite & np.isfinite(norms) & (norms > 0)
        # einsum avoids starting a many-threaded BLAS job for each small batch.
        dots = np.einsum("ij,j->i", matrix, np.asarray(query, dtype=np.float64))
        for offset, position in enumerate(positions):
            if valid[offset]:
                scores[position] = float(np.clip(dots[offset] / (norms[offset] * norm), -1.0, 1.0))
    else:
        for position, blob in zip(positions, vectors):
            vector = unpack_vector(blob, dimension=len(query))
            if vector and all(math.isfinite(value) for value in vector) and any(vector):
                scores[position] = cosine_similarity(query, vector)
    return scores
