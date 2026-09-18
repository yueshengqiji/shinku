"""Index the visual and audio assets that ship with a character pack.

A character pack is a directory tree on disk: scene folders, outfit folders, a
background music tree, and loose sidecar notes that describe them.  This module
walks that tree once and turns it into a plain dictionary the rest of the
runtime can read, then answers two kinds of question against it - "what does
this model output actually mean", and "what should the model be told exists".

Two layers live here on purpose.  Building the index is a filesystem job with
no state worth keeping; answering questions is a cached lookup job.  Splitting
them means the walking code never has to know about normalisation, and the
lookup code never has to know about directories.

Nothing in here interprets character behaviour.  It maps names to paths, and it
keeps whatever the caller adds at runtime in memory only.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "AUDIO_EXTS",
    "BACKGROUND_ALIASES",
    "EMOTION_ALIASES",
    "IMAGE_EXTS",
    "PROMPT_NOTE_LIMIT",
    "ResourceManifest",
]


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_EXTS = {".mp3", ".ogg", ".wav", ".m4a", ".flac"}

# Loose notes a pack may drop next to its assets; first hit wins.
NOTE_FILENAMES = (
    "ai.md",
    "ai.txt",
    "prompt.md",
    "prompt.txt",
    "说明.md",
    "说明.txt",
    "readme.md",
    "readme.txt",
    "notes.md",
    "notes.txt",
)
SIDECAR_NOTE_SUFFIXES = (".md", ".txt", ".note.md", ".note.txt")
META_NOTE_KEYS = ("notes", "note", "prompt", "ai_hint", "ai_prompt", "usage")

# Notes are pasted into prompts, so they are clipped before they get there.
PROMPT_NOTE_LIMIT = 240

EMOTION_ALIASES = {"normal": "normal", "shy": "shy", "smug": "smug", "sumg": "smug", "cry": "cry"}

# Ordered fallback ladders: a model that asked for "happy" may be satisfied by
# any of these ids, in the order listed.
EMOTION_FALLBACK_CANDIDATES = {
    "normal": ["正常", "normal"],
    "idle": ["正常", "normal"],
    "quiet": ["正常", "normal"],
    "happy": ["开心", "卖萌", "得意", "smug", "normal"],
    "joy": ["开心", "卖萌", "得意", "smug", "normal"],
    "smile": ["开心", "卖萌", "得意", "smug", "normal"],
    "smug": ["得意", "smug", "开心", "normal"],
    "sumg": ["得意", "smug", "开心", "normal"],
    "shy": ["脸红", "求摸摸", "shy", "normal"],
    "embarrassed": ["脸红", "shy", "normal"],
    "cry": ["困困", "无语", "cry", "sad", "normal"],
    "sad": ["困困", "无语", "cry", "normal"],
    "angry": ["气鼓鼓", "angry", "normal"],
    "thinking": ["思考中", "困惑", "normal"],
    "confused": ["困惑", "思考中", "normal"],
    "listening": ["侧耳听", "正常", "normal"],
    "music": ["听歌中", "开心", "normal"],
    "sleepy": ["困困", "打哈欠", "normal"],
    "tired": ["困困", "打哈欠", "normal"],
    "pet": ["被摸头", "求摸摸", "开心", "normal"],
}

BACKGROUND_ALIASES = {
    "morning": "morning",
    "sunrise": "morning",
    "dawn": "morning",
    "day": "morning",
    "daytime": "morning",
    "清晨": "morning",
    "早晨": "morning",
    "早上": "morning",
    "白天": "morning",
    "afternoon": "afternoon",
    "noon": "afternoon",
    "午后": "afternoon",
    "下午": "afternoon",
    "evening": "evening",
    "dusk": "evening",
    "sunset": "evening",
    "黄昏": "evening",
    "傍晚": "evening",
    "晚上": "evening",
    "night": "night",
    "midnight": "night",
    "深夜": "night",
    "夜晚": "night",
    "夜里": "night",
}

BACKGROUND_LABELS = {"morning": "清晨", "afternoon": "午后", "evening": "黄昏", "night": "深夜"}
EMOTION_LABELS = {"normal": "平静", "shy": "害羞", "smug": "得意", "cry": "哭哭"}

# Which scene id a model most likely means when the background id is all it gave.
BACKGROUND_PRIORITY = {"evening": 0, "morning": 1, "afternoon": 2, "night": 3}

_RUNTIME_EXTRA_KEYS = ("extra_bgm_tracks", "extra_scene_groups", "extra_character_outfits")

# Lines of fixed guidance that open the desktop-pet prompt.
_CHARACTER_PROMPT_HEADER = 2

_FALLBACK_NAME = "(无)"
_DEFAULT_BACKGROUND = "default"
_DEFAULT_OUTFIT = "default"
_FALLBACK_PREFIX = "/assets"


# --------------------------------------------------------------------------- #
# small readers
# --------------------------------------------------------------------------- #


def _read_json_object(path: Path) -> dict[str, Any]:
    """Best-effort JSON read; anything unexpected yields an empty mapping."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_note_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    except (OSError, UnicodeError):
        return ""


def _slug(value: Any) -> str:
    """Fold a free-form name into a comparable key."""

    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _condense(value: Any, limit: int = PROMPT_NOTE_LIMIT) -> str:
    """Collapse whitespace, then clip with an ellipsis if still too long."""

    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _join_fragments(*parts: Any) -> str:
    """Join distinct non-empty fragments, each clipped on its own."""

    collected: list[str] = []
    for part in parts:
        text = _condense(part)
        if text and text not in collected:
            collected.append(text)
    return " ".join(collected)


def _is_ignored_dir(path: Path) -> bool:
    return path.name.lower() in {"backup", "backups", "备份", "__pycache__"}


def _canonical_emotion(value: Any) -> str:
    raw = str(value or "").strip()
    return EMOTION_ALIASES.get(_slug(raw), raw)


def _canonical_background(value: Any) -> str:
    raw = str(value or "").strip()
    return BACKGROUND_ALIASES.get(_slug(raw), raw)


def _declared_aliases(meta: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(_as_text_list(meta.get("aliases"))))


def _matches_entry(entry: dict[str, Any] | None, value: Any) -> bool:
    """Does ``value`` name this entry, by id, display name or alias?"""

    if not entry or value in (None, ""):
        return False
    wanted = str(value).strip().lower()
    candidates = [entry.get("id"), entry.get("name"), *entry.get("aliases", [])]
    return any(wanted == str(candidate or "").strip().lower() for candidate in candidates)


def _display_name(entry: dict[str, Any] | None) -> str:
    if not entry:
        return ""
    return str(entry.get("name") or entry.get("id") or "")


def _place_name(major: dict[str, Any], minor: dict[str, Any]) -> str:
    return f"{_display_name(major)}/{_display_name(minor)}"


def _ordered_dirs(root: Path) -> list[Path]:
    dirs = (item for item in root.iterdir() if item.is_dir() and not _is_ignored_dir(item))
    return sorted(dirs, key=lambda item: item.name)


def _first_directory_note(path: Path) -> str:
    for name in NOTE_FILENAMES:
        note = _read_note_text(path / name)
        if note:
            return _condense(note)
    return ""


def _first_sidecar_note(path: Path) -> str:
    for suffix in SIDECAR_NOTE_SUFFIXES:
        note = _read_note_text(path.with_suffix(suffix))
        if note:
            return note
    return ""


def _split_legacy_stem(stem: str) -> tuple[str, str]:
    """Read the ``场景_时段`` naming convention used by older packs."""

    if "_" not in stem:
        return _DEFAULT_BACKGROUND, stem
    minor, suffix = stem.rsplit("_", 1)
    if _slug(suffix) in BACKGROUND_ALIASES:
        return minor, _canonical_background(suffix)
    return _DEFAULT_BACKGROUND, stem


def _as_public_prefix(value: str) -> str:
    text = str(value or _FALLBACK_PREFIX).strip().replace("\\", "/")
    text = "/" + text.strip("/")
    return text or _FALLBACK_PREFIX


# --------------------------------------------------------------------------- #
# merging
# --------------------------------------------------------------------------- #


def _merge_by_id(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold entries sharing an id; the earliest one keeps its own fields."""

    merged: list[dict[str, Any]] = []
    for item in entries:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        existing = next((entry for entry in merged if str(entry["id"]) == str(item["id"])), None)
        if existing is None:
            merged.append(copy.deepcopy(item))
            continue
        existing.update({key: value for key, value in item.items() if key != "aliases"})
        existing["aliases"] = list(
            dict.fromkeys(existing.get("aliases", []) + item.get("aliases", []))
        )
    return merged


def _ordered_minors(minors: list[dict[str, Any]], default_id: str) -> list[dict[str, Any]]:
    return sorted(
        _merge_by_id(minors),
        key=lambda item: (0 if default_id and item["id"] == default_id else 1, str(item["id"])),
    )


def _absorb_scene_groups(manifest: dict[str, Any], groups: Iterable[Any]) -> None:
    existing = manifest["scenes"]["majors"]
    for item in groups:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        current = next((entry for entry in existing if _matches_entry(entry, item["id"])), None)
        if current is None:
            existing.append(copy.deepcopy(item))
            continue
        current.update({key: value for key, value in item.items() if key != "minors"})
        current["minors"] = _merge_by_id(current.get("minors", []) + item.get("minors", []))
    existing[:] = _merge_by_id(existing)


def _absorb_outfits(manifest: dict[str, Any], outfits: Iterable[Any]) -> None:
    existing = manifest["characters"]["outfits"]
    for item in outfits:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        current = next((entry for entry in existing if _matches_entry(entry, item["id"])), None)
        if current is None:
            existing.append(copy.deepcopy(item))
            continue
        current.update({key: value for key, value in item.items() if key != "emotions"})
        current["emotions"] = _merge_by_id(current.get("emotions", []) + item.get("emotions", []))
    existing[:] = _merge_by_id(existing)


def _recompute_defaults(manifest: dict[str, Any]) -> None:
    """Defaults always point at the first entry of each list, in order."""

    major = manifest["scenes"]["majors"][0]
    minor = major["minors"][0]
    outfit = manifest["characters"]["outfits"][0]
    manifest["defaults"] = {
        "major": major["id"],
        "minor": minor["id"],
        "background": minor["backgrounds"][0]["id"],
        "bgm": minor["bgm_tracks"][0]["id"] if minor["bgm_tracks"] else "",
        "outfit": outfit["id"],
        "emotion": outfit["emotions"][0]["id"],
    }


# --------------------------------------------------------------------------- #
# index building
# --------------------------------------------------------------------------- #


class _AssetIndex:
    """Walk an asset directory and produce the manifest dictionary.

    Kept separate from :class:`ResourceManifest` because it is pure filesystem
    work: it reads a directory tree once and hands back data, holding no cache
    and answering no questions about meaning.
    """

    def __init__(self, assets_dir: Path, public_prefix: str) -> None:
        self.assets_dir = assets_dir
        self.public_prefix = public_prefix

    def build(self) -> dict[str, Any]:
        scenes = self._major_scenes(self.assets_dir / "scenes")
        scenes.extend(self._flat_scene(self.assets_dir / "backgrounds"))
        scenes = _merge_by_id(scenes)
        if not scenes:
            scenes = [self._placeholder_scene()]

        outfits = self._outfits(self.assets_dir / "characters") or [self._placeholder_outfit()]

        manifest: dict[str, Any] = {
            "schema_version": 2,
            "scenes": {"majors": scenes},
            "characters": {"outfits": outfits},
            "defaults": {},
        }
        self._attach_tracks(manifest, self.assets_dir / "bgm")
        _recompute_defaults(manifest)
        return manifest

    # -- scenes ------------------------------------------------------------ #

    def _major_scenes(self, root: Path) -> list[dict[str, Any]]:
        if not root.is_dir():
            return []

        majors: list[dict[str, Any]] = []
        for major_dir in _ordered_dirs(root):
            meta = _read_json_object(major_dir / "meta.json")
            minors: list[dict[str, Any]] = []

            # Sub-folders are one minor each, as long as they hold an image.
            for child in _ordered_dirs(major_dir):
                backgrounds = [
                    self._background_record(path, _read_json_object(path.with_suffix(".meta.json")))
                    for path in child.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTS
                ]
                if backgrounds:
                    minors.append(self._minor_record(child.name, _read_json_object(child / "meta.json"), backgrounds))

            # Images sitting directly in the major folder group by legacy stem.
            grouped: dict[str, list[dict[str, Any]]] = {}
            for path in major_dir.iterdir():
                if not (path.is_file() and path.suffix.lower() in IMAGE_EXTS):
                    continue
                minor_name, background_id = _split_legacy_stem(path.stem)
                grouped.setdefault(minor_name, []).append(
                    self._background_record(
                        path,
                        _read_json_object(path.with_suffix(".meta.json")),
                        explicit_id=background_id,
                    )
                )
            for minor_name, backgrounds in grouped.items():
                minors.append(
                    self._minor_record(
                        minor_name,
                        _read_json_object(major_dir / f"{minor_name}.meta.json"),
                        backgrounds,
                    )
                )

            if not minors:
                continue
            major_id = str(meta.get("id") or major_dir.name)
            default_minor = str(meta.get("default_minor") or "")
            majors.append(
                {
                    "id": major_id,
                    "name": str(meta.get("name") or major_id),
                    "aliases": _declared_aliases(meta),
                    "description": _condense(meta.get("description")),
                    "notes": _first_directory_note(major_dir),
                    "minors": _ordered_minors(minors, default_minor),
                    "default_minor": default_minor,
                }
            )
        return majors

    def _flat_scene(self, root: Path) -> list[dict[str, Any]]:
        """Loose backgrounds become one implicit scene called ``default``."""

        if not root.is_dir():
            return []
        backgrounds = [
            self._background_record(path, _read_json_object(path.with_suffix(".meta.json")))
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ]
        if not backgrounds:
            return []
        return [
            {
                "id": _DEFAULT_BACKGROUND,
                "name": "默认",
                "aliases": [],
                "description": "",
                "notes": "",
                "minors": [self._minor_record(_DEFAULT_BACKGROUND, {}, backgrounds)],
            }
        ]

    def _minor_record(
        self, identifier: str, meta: dict[str, Any], backgrounds: list[dict[str, Any]]
    ) -> dict[str, Any]:
        ordered = sorted(
            _merge_by_id(backgrounds),
            key=lambda item: (BACKGROUND_PRIORITY.get(item["id"], 10), item["id"]),
        )
        return {
            "id": str(meta.get("id") or identifier),
            "name": str(meta.get("name") or meta.get("id") or identifier),
            "aliases": _declared_aliases(meta),
            "description": _condense(meta.get("description")),
            "notes": _join_fragments(*(meta.get(key) for key in META_NOTE_KEYS)),
            "default_background": _canonical_background(meta.get("default_background")),
            "default_bgm": str(meta.get("default_bgm") or ""),
            "backgrounds": ordered,
            "bgm_tracks": [],
        }

    # -- outfits ------------------------------------------------------------ #

    def _outfits(self, root: Path) -> list[dict[str, Any]]:
        if not root.is_dir():
            return []

        outfits: list[dict[str, Any]] = []
        for directory in _ordered_dirs(root):
            meta = _read_json_object(directory / "meta.json")
            emotions = [
                self._emotion_record(path, _read_json_object(path.with_suffix(".meta.json")))
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            ]
            if not emotions:
                continue

            allowed = _as_text_list(meta.get("allowed_emotions"))
            if allowed:
                kept = [
                    item
                    for item in emotions
                    if any(
                        _matches_entry(item, candidate) or _canonical_emotion(candidate) == item["id"]
                        for candidate in allowed
                    )
                ]
                # A whitelist that matches nothing is treated as "not set"
                # rather than as an empty outfit.
                emotions = kept or emotions

            default_emotion = _canonical_emotion(meta.get("default_emotion"))
            emotions = sorted(
                emotions,
                key=lambda item: (
                    0 if default_emotion and item["id"] == default_emotion else 1,
                    item["id"],
                ),
            )
            outfits.append(
                {
                    "id": str(meta.get("id") or directory.name),
                    "name": str(meta.get("name") or meta.get("id") or directory.name),
                    "aliases": _declared_aliases(meta),
                    "description": _condense(meta.get("description")),
                    "notes": _first_directory_note(directory),
                    "default_emotion": default_emotion or "",
                    "allowed_emotions": [item["id"] for item in emotions],
                    "emotions": emotions,
                }
            )
        return outfits

    # -- tracks ------------------------------------------------------------- #

    def _attach_tracks(self, manifest: dict[str, Any], root: Path) -> None:
        """Distribute the audio tree across scenes by relative folder depth."""

        if not root.is_dir():
            return

        buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for path in root.rglob("*"):
            if not (path.is_file() and path.suffix.lower() in AUDIO_EXTS):
                continue
            parts = path.relative_to(root).parts
            key = (parts[-3], parts[-2]) if len(parts) >= 3 else ("default", "default")
            buckets.setdefault(key, []).append(
                self._track_record(path, _read_json_object(path.with_suffix(".meta.json")))
            )

        shared = buckets.get(("default", "default"), [])
        for major in manifest["scenes"]["majors"]:
            for minor in major["minors"]:
                tracks = buckets.get((major["id"], minor["id"]), []) or list(shared)
                minor["bgm_tracks"] = sorted(
                    _merge_by_id(tracks),
                    key=lambda item: (BACKGROUND_PRIORITY.get(item["id"], 10), item["id"]),
                )

    # -- asset records ------------------------------------------------------ #

    def _background_record(
        self, path: Path, meta: dict[str, Any], *, explicit_id: str | None = None
    ) -> dict[str, Any]:
        identifier = str(meta.get("id") or explicit_id or _canonical_background(path.stem))
        return self._asset_record(identifier, path, meta, kind="background")

    def _emotion_record(self, path: Path, meta: dict[str, Any]) -> dict[str, Any]:
        identifier = _canonical_emotion(meta.get("id") or path.stem)
        return self._asset_record(identifier, path, meta, kind="emotion")

    def _track_record(self, path: Path, meta: dict[str, Any]) -> dict[str, Any]:
        return self._asset_record(str(meta.get("id") or path.stem), path, meta, kind="bgm")

    def _asset_record(
        self, identifier: str, path: Path, meta: dict[str, Any], *, kind: str
    ) -> dict[str, Any]:
        aliases = _declared_aliases(meta)
        if path.stem not in aliases and path.stem != identifier:
            aliases.append(path.stem)

        # Only backgrounds get the scene label table historically; every other
        # kind falls back to the emotion labels.  Preserved as-is.
        label_table = BACKGROUND_LABELS if kind == "background" else EMOTION_LABELS
        return {
            "id": identifier,
            "name": str(meta.get("name") or label_table.get(identifier) or identifier),
            "aliases": aliases,
            "description": _condense(meta.get("description")),
            "notes": _join_fragments(
                *(meta.get(key) for key in META_NOTE_KEYS),
                _first_sidecar_note(path),
            ),
            "path": self._public_path(path),
        }

    def _public_path(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.assets_dir.resolve()).as_posix()
        except (OSError, ValueError):
            return ""
        return f"{self.public_prefix}/{relative}"

    # -- placeholders ------------------------------------------------------- #

    def _placeholder_scene(self) -> dict[str, Any]:
        background = {
            "id": _DEFAULT_BACKGROUND,
            "name": "默认",
            "aliases": [],
            "description": "",
            "notes": "",
            "path": "",
        }
        return {
            "id": _DEFAULT_BACKGROUND,
            "name": "默认",
            "aliases": [],
            "description": "",
            "notes": "",
            "minors": [
                {
                    "id": _DEFAULT_BACKGROUND,
                    "name": "默认",
                    "aliases": [],
                    "description": "",
                    "notes": "",
                    "default_background": _DEFAULT_BACKGROUND,
                    "default_bgm": "",
                    "backgrounds": [background],
                    "bgm_tracks": [],
                }
            ],
        }

    def _placeholder_outfit(self) -> dict[str, Any]:
        emotion = {
            "id": "normal",
            "name": "平静",
            "aliases": [],
            "description": "",
            "notes": "",
            "path": "",
        }
        return {
            "id": _DEFAULT_OUTFIT,
            "name": "默认",
            "aliases": [],
            "description": "",
            "notes": "",
            "default_emotion": "normal",
            "allowed_emotions": ["normal"],
            "emotions": [emotion],
        }


# --------------------------------------------------------------------------- #
# the facade
# --------------------------------------------------------------------------- #


class ResourceManifest:
    """Cached view over a character pack's assets.

    Construct one per pack and call :meth:`refresh` whenever the directory may
    have changed.  Additions made in memory - extra outfits, scene groups or
    tracks the runtime discovered - are passed per call and never written back.
    """

    def __init__(
        self,
        assets_dir: Path,
        public_prefix: str = "/assets",
        *,
        emotion_aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self.assets_dir = Path(assets_dir)
        self.public_prefix = _as_public_prefix(public_prefix)
        self.emotion_aliases = {
            _slug(key): list(dict.fromkeys(_as_text_list(value)))
            for key, value in (emotion_aliases or {}).items()
            if _slug(key) and _as_text_list(value)
        }
        self._manifest: dict[str, Any] | None = None

    # -- cache -------------------------------------------------------------- #

    def refresh(self) -> dict[str, Any]:
        self._manifest = _AssetIndex(self.assets_dir, self.public_prefix).build()
        return self._manifest

    def get_manifest(self) -> dict[str, Any]:
        return self._manifest if self._manifest is not None else self.refresh()

    # -- runtime composition ------------------------------------------------ #

    def build_runtime_manifest(
        self,
        *,
        extra_bgm_tracks: list[dict[str, Any]] | None = None,
        extra_scene_groups: list[dict[str, Any]] | None = None,
        extra_character_outfits: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        manifest = copy.deepcopy(self.get_manifest())
        _absorb_scene_groups(manifest, extra_scene_groups or [])
        _absorb_outfits(manifest, extra_character_outfits or [])
        if extra_bgm_tracks:
            additions = [item for item in extra_bgm_tracks if isinstance(item, dict)]
            for major in manifest["scenes"]["majors"]:
                for minor in major["minors"]:
                    minor["bgm_tracks"] = _merge_by_id(
                        list(minor.get("bgm_tracks", [])) + additions
                    )
        _recompute_defaults(manifest)
        return manifest

    # -- prompt context ----------------------------------------------------- #

    def build_prompt_context(
        self,
        *,
        extra_bgm_tracks: list[dict[str, Any]] | None = None,
        extra_scene_groups: list[dict[str, Any]] | None = None,
        extra_character_outfits: list[dict[str, Any]] | None = None,
    ) -> str:
        manifest = self.build_runtime_manifest(
            extra_bgm_tracks=extra_bgm_tracks,
            extra_scene_groups=extra_scene_groups,
            extra_character_outfits=extra_character_outfits,
        )
        lines: list[str] = []

        notes = self._root_notes()
        if notes:
            lines.append("资源说明：")
            lines.extend(f"- {note}" for note in notes)

        scene_lines: list[str] = []
        for major in manifest["scenes"]["majors"]:
            for minor in major["minors"]:
                backgrounds = ", ".join(_display_name(item) for item in minor["backgrounds"]) or _FALLBACK_NAME
                scene_lines.append(f"- {_place_name(major, minor)} -> 背景: {backgrounds}")
        if scene_lines:
            lines.append(
                "场景输出规则：scene.major/minor/background 使用清单列出的名称或 id，"
                "不要把未列出的文件名自行拆成新场景。"
            )
            lines.append("可用场景与背景：")
            lines.extend(scene_lines)

        outfit_lines: list[str] = []
        for outfit in manifest["characters"]["outfits"]:
            emotions = ", ".join(_display_name(item) for item in outfit["emotions"]) or _FALLBACK_NAME
            outfit_lines.append(f"- {_display_name(outfit)} -> 表情: {emotions}")
        if outfit_lines:
            lines.append("可用服装与表情：")
            lines.extend(outfit_lines)

        track_lines: list[str] = []
        for major in manifest["scenes"]["majors"]:
            for minor in major["minors"]:
                if not minor["bgm_tracks"]:
                    continue
                titles = ", ".join(_display_name(item) for item in minor["bgm_tracks"])
                track_lines.append(f"- {_place_name(major, minor)} -> BGM: {titles}")
        if track_lines:
            lines.append("可用 BGM：")
            lines.extend(track_lines)

        return "\n".join(lines) if lines else "当前没有额外的视觉资源。"

    def build_character_prompt_context(
        self, *, extra_character_outfits: list[dict[str, Any]] | None = None
    ) -> str:
        manifest = self.build_runtime_manifest(extra_character_outfits=extra_character_outfits)
        lines = [
            "桌宠模式只渲染角色服装立绘和表情，不渲染场景、背景或 BGM。",
            "输出时优先沿用当前 character.outfit，只在同一套服装下选择可用 emotion；"
            "确实需要换衣服时才切换 character.outfit。",
        ]
        for outfit in manifest["characters"]["outfits"]:
            emotions = ", ".join(_display_name(item) for item in outfit["emotions"]) or _FALLBACK_NAME
            lines.append(f"- {_display_name(outfit)} -> 表情: {emotions}")
        if len(lines) <= _CHARACTER_PROMPT_HEADER:
            return "当前没有额外的角色视觉资源。"
        return "\n".join(lines)

    def build_emotion_prompt_context(self) -> str:
        manifest = self.get_manifest()
        lines = [
            "emotion 与桌宠共用当前角色包的表情图片变量。",
            "即使当前客户端不渲染立绘，也必须直接使用图片文件名去掉扩展名后的稳定 id；"
            "不要编造或改写 emotion。",
        ]
        for outfit in manifest["characters"]["outfits"]:
            ids = ", ".join(
                str(item.get("id") or "") for item in outfit.get("emotions", []) if item.get("id")
            )
            if ids:
                lines.append(f"- {outfit['id']}: {ids}")
        return "\n".join(lines)

    # -- normalisation ------------------------------------------------------ #

    def normalize_emotion_id(self, value: Any, *, preferred_outfit: str = "") -> str:
        manifest = self.get_manifest()
        outfits = manifest["characters"]["outfits"]
        outfit = (
            self._pick_outfit(manifest, preferred_outfit)
            or self._pick_outfit(manifest, manifest["defaults"]["outfit"])
            or outfits[0]
        )
        requested = _canonical_emotion(value) or manifest["defaults"]["emotion"]
        found = self._pick_emotion(outfit, requested)
        if found is None:
            for candidate in outfits:
                found = self._pick_emotion(candidate, requested)
                if found:
                    break
        return str((found or outfit["emotions"][0])["id"])

    def normalize_emotion_output(self, result: dict[str, Any]) -> dict[str, Any]:
        value = dict(result or {})
        character = value.get("character") if isinstance(value.get("character"), dict) else {}
        value["emotion"] = self.normalize_emotion_id(
            value.get("emotion"), preferred_outfit=str(character.get("outfit") or "")
        )
        return value

    def normalize_visual_output(
        self,
        result: dict[str, Any],
        *,
        extra_bgm_tracks: list[dict[str, Any]] | None = None,
        extra_scene_groups: list[dict[str, Any]] | None = None,
        extra_character_outfits: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        value = dict(result or {})
        manifest = self.build_runtime_manifest(
            extra_bgm_tracks=extra_bgm_tracks,
            extra_scene_groups=extra_scene_groups,
            extra_character_outfits=extra_character_outfits,
        )
        defaults = manifest["defaults"]
        scene = value.get("scene") if isinstance(value.get("scene"), dict) else {}
        character = value.get("character") if isinstance(value.get("character"), dict) else {}

        major = self._pick_major(manifest, scene.get("major")) or self._pick_major(
            manifest, defaults["major"]
        )
        minor = self._pick_minor(major, scene.get("minor"))
        requested_background = scene.get("background")
        if minor is None:
            # The model named a background but not the minor holding it; scan
            # the whole major before falling back to the default minor.
            located = self._background_elsewhere(major, requested_background)
            if located:
                minor, background = located
            else:
                minor, background = (
                    self._pick_minor(major, defaults["minor"]) or major["minors"][0],
                    None,
                )
        else:
            background = self._pick_background(minor, requested_background)
        background = (
            background
            or self._pick_background(minor, defaults["background"])
            or minor["backgrounds"][0]
        )

        outfit = (
            self._pick_outfit(manifest, character.get("outfit"))
            or self._pick_outfit(manifest, defaults["outfit"])
            or manifest["characters"]["outfits"][0]
        )
        emotion = (
            self._pick_emotion(outfit, value.get("emotion"))
            or self._pick_emotion(outfit, defaults["emotion"])
            or outfit["emotions"][0]
        )
        track = (
            self._pick_track(minor, scene.get("bgm"))
            or self._pick_track(minor, background["id"])
            or self._pick_track(minor, defaults["bgm"])
        )

        value["emotion"] = emotion["id"]
        value["character"] = {"outfit": outfit["id"]}
        value["scene"] = {
            "major": major["id"],
            "minor": minor["id"],
            "background": background["id"],
            "bgm": track["id"] if track else "",
        }
        return value

    def resolve_visual_bundle(self, result: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        snapshot = json.loads(json.dumps(result or {}, ensure_ascii=False))
        normalized = self.normalize_visual_output(snapshot, **kwargs)
        manifest = self.build_runtime_manifest(
            **{key: kwargs[key] for key in _RUNTIME_EXTRA_KEYS if key in kwargs}
        )
        major = self._pick_major(manifest, normalized["scene"]["major"])
        minor = self._pick_minor(major, normalized["scene"]["minor"]) if major else None
        outfit = self._pick_outfit(manifest, normalized["character"]["outfit"])
        return {
            "normalized": normalized,
            "manifest": manifest,
            "major": major,
            "minor": minor,
            "background": self._pick_background(minor, normalized["scene"]["background"])
            if minor
            else None,
            "outfit": outfit,
            "emotion": self._pick_emotion(outfit, normalized["emotion"]) if outfit else None,
            "bgm": self._pick_track(minor, normalized["scene"].get("bgm")) if minor else None,
        }

    # -- description -------------------------------------------------------- #

    def describe_visual_state(self, result: dict[str, Any], **kwargs: Any) -> str:
        bundle = self.resolve_visual_bundle(result, **kwargs)
        normalized = bundle["normalized"]
        outfit = bundle["outfit"]
        emotion = self._pick_emotion(outfit, normalized["emotion"]) if outfit else None
        place = (
            _place_name(bundle["major"], bundle["minor"])
            if bundle["major"] and bundle["minor"]
            else normalized["scene"]["major"]
        )
        return "；".join(
            [
                f"地点: {place}",
                f"背景: {_display_name(bundle['background']) if bundle['background'] else normalized['scene']['background']}",
                f"服装: {_display_name(outfit) if outfit else normalized['character']['outfit']}",
                f"表情: {_display_name(emotion) if emotion else normalized['emotion']}",
                f"BGM: {_display_name(bundle['bgm']) if bundle['bgm'] else (normalized['scene']['bgm'] or '未设置')}",
            ]
        )

    def describe_character_visual_state(
        self, result: dict[str, Any], *, extra_character_outfits: list[dict[str, Any]] | None = None
    ) -> str:
        bundle = self.resolve_visual_bundle(result, extra_character_outfits=extra_character_outfits)
        outfit = bundle["outfit"]
        emotion = self._pick_emotion(outfit, bundle["normalized"]["emotion"]) if outfit else None
        parts = [
            f"服装: {_display_name(outfit) if outfit else bundle['normalized']['character']['outfit']}",
            f"表情: {_display_name(emotion) if emotion else bundle['normalized']['emotion']}",
        ]
        if outfit and outfit.get("emotions"):
            parts.append("当前服装可用表情: " + ", ".join(_display_name(item) for item in outfit["emotions"]))
        return "；".join(parts)

    # -- lookups ------------------------------------------------------------ #

    def _pick_major(self, manifest: dict[str, Any], value: Any) -> dict[str, Any] | None:
        return next(
            (item for item in manifest["scenes"]["majors"] if _matches_entry(item, value)), None
        )

    def _pick_minor(self, major: dict[str, Any] | None, value: Any) -> dict[str, Any] | None:
        return next(
            (item for item in (major or {}).get("minors", []) if _matches_entry(item, value)), None
        )

    def _pick_background(self, minor: dict[str, Any] | None, value: Any) -> dict[str, Any] | None:
        canonical = _canonical_background(value)
        return next(
            (
                item
                for item in (minor or {}).get("backgrounds", [])
                if _matches_entry(item, value) or item["id"] == canonical
            ),
            None,
        )

    def _background_elsewhere(
        self, major: dict[str, Any] | None, value: Any
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        for minor in (major or {}).get("minors", []):
            background = self._pick_background(minor, value)
            if background:
                return minor, background
        return None

    def _pick_track(self, minor: dict[str, Any] | None, value: Any) -> dict[str, Any] | None:
        canonical = _canonical_background(value)
        return next(
            (
                item
                for item in (minor or {}).get("bgm_tracks", [])
                if _matches_entry(item, value) or item.get("id") == canonical
            ),
            None,
        )

    def _pick_outfit(self, manifest: dict[str, Any], value: Any) -> dict[str, Any] | None:
        return next(
            (item for item in manifest["characters"]["outfits"] if _matches_entry(item, value)),
            None,
        )

    def _emotion_ladder(self, value: Any) -> list[str]:
        raw = str(value or "").strip()
        ladder = [_canonical_emotion(raw)]
        ladder.extend(EMOTION_FALLBACK_CANDIDATES.get(_slug(raw), []))
        ladder.extend(self.emotion_aliases.get(_slug(raw), []))
        return list(dict.fromkeys(item for item in ladder if item))

    def _pick_emotion(self, outfit: dict[str, Any] | None, value: Any) -> dict[str, Any] | None:
        emotions = (outfit or {}).get("emotions", [])
        for candidate in self._emotion_ladder(value):
            found = next(
                (
                    item
                    for item in emotions
                    if _matches_entry(item, candidate) or item.get("id") == _canonical_emotion(candidate)
                ),
                None,
            )
            if found:
                return found
        return None

    # -- notes -------------------------------------------------------------- #

    def _root_notes(self) -> list[str]:
        notes: list[str] = []
        for name in NOTE_FILENAMES:
            value = _read_note_text(self.assets_dir / name)
            if value and _condense(value) not in notes:
                notes.append(_condense(value))
        return notes
