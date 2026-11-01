"""Versioned projection gaps; a digest does not certify upstream completeness."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import re

from shinku.agent.task_state import _is_secret_key, _SECRET_TEXT
from .contracts import AgentBoundaryError


_PRIVATE = {"reasoning", "reasoning_content", "analysis", "chain_of_thought", "raw_response",
            "system_prompt", "persona", "chat_history", "memory_context"}
_USERINFO = re.compile(r'(https?://)[^/@\s"<>]+@', re.I)
_QUERY_SECRET = re.compile(
    r'([?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|passwd|secret|sig|signature|authorization)=)'
    r'[^&#\s"\'<>\\]*', re.I)


def safe_text(value):
    value = _SECRET_TEXT.sub(lambda m: (m.group(1) or m.group(2)) + "[redacted]", value)
    value = _USERINFO.sub(r'\1[redacted]@', value)
    value = _QUERY_SECRET.sub(r'\1[redacted]', value)
    return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", value)


def safe_source(value, *, max_nodes=32768, max_depth=32):
    """Retain the entire permitted representation or reject, never silently cut.

    This excludes host/model secrets, not instructions embedded in documents.
    Such instructions stay in the data channel and grant no authority.
    """
    left = [max_nodes]
    def walk(item, depth):
        left[0] -= 1
        if depth > max_depth or left[0] < 0:
            raise AgentBoundaryError("material_source_oversized")
        if isinstance(item, Mapping):
            result = {}
            for key, child in item.items():
                name = str(key)
                if name in _PRIVATE or name.startswith("_native"):
                    continue
                result[name] = "[redacted]" if _is_secret_key(name, child) else walk(child, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return [walk(child, depth + 1) for child in item]
        if isinstance(item, str):
            return safe_text(item)
        if item is None or type(item) in (bool, int, float):
            if type(item) is float and not math.isfinite(item):
                raise AgentBoundaryError("material_source_type_unsupported")
            return item
        raise AgentBoundaryError("material_source_type_unsupported")
    return walk(value, 0)


def source_text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True,
                                                         separators=(",", ":"), allow_nan=False)


def content_hash(value):
    return hashlib.sha256(source_text(value).encode("utf-8")).hexdigest()


def pointer(path):
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in path) if path else ""


def bounded_projection(value, *, text_limit=2400, list_limit=128, mapping_limit=80,
                       depth_limit=8, excluded=(), protected_text_paths=(), max_gaps=128):
    gaps, gap_count = [], [0]
    excluded = set(excluded)
    def gap(path, kind, total, retained, reason):
        gap_count[0] += 1
        if len(gaps) < max_gaps:
            gaps.append({"path": pointer(path), "kind": kind, "total": total, "retained": retained,
                         "omitted": total - retained, "reason": reason, "retrieval": "unavailable"})
    def walk(item, depth, path):
        if depth > depth_limit:
            gap(path, "subtree", 1, 0, "depth_limit")
            return {"omitted": "depth_limit"}
        if isinstance(item, Mapping):
            rows = [(str(key), child) for key, child in item.items()
                    if str(key) not in excluded and not str(key).startswith("_native")]
            if len(rows) > mapping_limit:
                gap(path, "mapping", len(rows), mapping_limit, "mapping_limit")
            return {key: walk(child, depth + 1, (*path, key)) for key, child in rows[:mapping_limit]}
        if isinstance(item, (list, tuple)):
            if len(item) > list_limit:
                gap(path, "list", len(item), list_limit, "list_limit")
            return [walk(child, depth + 1, (*path, index)) for index, child in enumerate(item[:list_limit])]
        limit = 12000 if any(name in path for name in protected_text_paths) else text_limit
        if isinstance(item, str) and len(item) > limit:
            # Keep legacy excerpt shape, but count the actual retained prefix.
            retained = min(2400, limit)
            gap(path, "text", len(item), retained, "text_limit")
            return {"excerpt": item[:retained], "omitted": "text_limit"}
        return item
    projected = walk(value, 0, ())
    metadata = {"schema_version": 1, "status": "truncated" if gap_count[0] else "complete",
        "source_representation": "provided_value", "upstream_completeness": "unknown", "gaps": gaps,
        "gap_count": gap_count[0], "omitted_gap_records": max(0, gap_count[0] - len(gaps))}
    return projected, metadata


def information_for_result(result):
    value = result.get("information")
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        return {"schema_version": 1, "status": "unknown", "reason": "legacy_information_unavailable"}
    return value
