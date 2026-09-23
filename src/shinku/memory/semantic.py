"""Small, local-only primitives used by the memory semantic index.

The memory database remains the source of truth.  This module deliberately
does not depend on numpy, a vector database, or a model SDK: providers return
plain Python floats and vectors are stored as compact little-endian float32
blobs in SQLite.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable

__all__ = ["cosine_similarity", "pack_vector", "unpack_vector"]


def pack_vector(values: Iterable[float]) -> bytes:
    """Encode a vector without making numpy a core dependency."""

    numbers = [float(value) for value in values]
    if not numbers:
        return b""
    return struct.pack(f"<{len(numbers)}f", *numbers)


def unpack_vector(blob: bytes | bytearray | memoryview, *, dimension: int = 0) -> list[float]:
    """Decode a stored float32 vector, rejecting malformed blobs safely."""

    raw = bytes(blob or b"")
    if not raw or len(raw) % 4:
        return []
    width = len(raw) // 4
    if dimension and width != int(dimension):
        return []
    try:
        return list(struct.unpack(f"<{width}f", raw))
    except struct.error:
        return []


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    """Return a bounded cosine score; zero is used for incompatible vectors."""

    a = list(left)
    b = list(right)
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    left_norm = math.sqrt(sum(float(value) * float(value) for value in a))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in b))
    if not left_norm or not right_norm:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
