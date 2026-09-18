"""Character-pack assets: the visual and audio library a pack ships with.

Everything here is read-only with respect to the pack on disk.  The package
builds an index, answers questions about what a model asked for, and renders the
prompt sections that tell the model what exists.  Runtime additions stay in
memory and are passed per call.
"""

from __future__ import annotations

from .manifest import (
    AUDIO_EXTS,
    BACKGROUND_ALIASES,
    EMOTION_ALIASES,
    IMAGE_EXTS,
    PROMPT_NOTE_LIMIT,
    ResourceManifest,
)

__all__ = [
    "AUDIO_EXTS",
    "BACKGROUND_ALIASES",
    "EMOTION_ALIASES",
    "IMAGE_EXTS",
    "PROMPT_NOTE_LIMIT",
    "ResourceManifest",
]
