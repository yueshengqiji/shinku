"""Agent 任务 payload 的边界归一化。

模型和宿主传来的 JSON 经常混用字符串、列表、旧字段名和重复条目。这里把
它们压成执行器可以消费的稳定形状；模块本身保持无状态，不负责落库。
"""

from __future__ import annotations

from typing import Any

_STEP_STATUSES = {"queued", "running", "done", "failed", "waiting_user"}
_EMPTY = (None, "", [], {})


def coerce_positive_int(value: Any) -> int:
    try:
        number = int(value or 0)
    except Exception:
        return 0
    return number if number > 0 else 0


def normalize_text_list(value: Any, *, limit: int = 20, item_limit: int = 200) -> list[str]:
    """保留首次出现的文本，比较时忽略大小写，按出现顺序截断。"""

    values = value if isinstance(value, list) else [value]
    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text[:item_limit])
        if len(result) >= limit:
            break
    return result


def normalize_steps(value: Any, *, limit: int = 12, step_id_prefix: str = "worker_step") -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    prefix = str(step_id_prefix or "worker_step").strip() or "worker_step"
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            title = str(item.get("title") or item.get("name") or item.get("id") or "").strip()
            status = str(item.get("status") or "queued").strip().lower()
            note = str(item.get("note") or "").strip()
        else:
            title, status, note = str(item or "").strip(), "queued", ""
        if not title:
            continue
        result.append(
            {
                "id": f"{prefix}_{index}",
                "title": title[:120],
                "status": status if status in _STEP_STATUSES else "queued",
                "note": note[:200],
            }
        )
        if len(result) >= limit:
            break
    return result


def normalize_artifacts(value: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        artifact_id = str(item.get("id") or item.get("generated_handle") or item.get("attachment_handle") or "").strip()
        title = str(item.get("title") or "").strip()
        if not artifact_id and not title:
            continue
        identity = (artifact_id or title).casefold()
        if identity in seen:
            continue
        seen.add(identity)
        normalized: dict[str, Any] = {
            "id": artifact_id[:80],
            "kind": str(item.get("kind") or item.get("format") or "").strip()[:40],
            "title": title[:120],
            "source": str(item.get("source") or "task_worker").strip()[:60],
            "delivery_role": str(item.get("delivery_role") or item.get("role") or "workspace_material").strip()[:60],
        }
        if "deliverable" in item:
            normalized["deliverable"] = bool(item["deliverable"])
        result.append(normalized)
        if len(result) >= limit:
            break
    return result


def merge_artifacts(existing: list[dict[str, Any]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 id/title 合并 payload；新值只覆盖非空字段。"""

    merged: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for item in [*existing, *additions]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or item.get("title") or "").strip().casefold()
        if not key:
            continue
        position = positions.get(key)
        if position is None:
            positions[key] = len(merged)
            merged.append(dict(item))
            continue
        current = dict(merged[position])
        current.update({name: value for name, value in item.items() if value not in _EMPTY})
        merged[position] = current
    return merged


def _step_identity(item: dict[str, Any]) -> set[str]:
    identities: set[str] = set()
    if item.get("id"):
        identities.add(f"id:{str(item['id']).strip().casefold()}")
    if item.get("title"):
        identities.add(f"title:{str(item['title']).strip().casefold()}")
    return identities


def merge_steps(existing: list[dict[str, Any]], additions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = [dict(item) for item in existing if isinstance(item, dict)]
    for raw in additions:
        if not isinstance(raw, dict):
            continue
        candidate = dict(raw)
        keys = _step_identity(candidate)
        match = next((index for index, item in enumerate(merged) if keys & _step_identity(item)), None)
        if match is None:
            candidate.setdefault("id", f"worker_step_{len(merged) + 1}")
            merged.append(candidate)
            continue
        merged[match].update(
            {
                name: candidate[name]
                for name in ("id", "title", "status", "note")
                if candidate.get(name) not in (None, "")
            }
        )
    return merged[:20]


__all__ = [
    "coerce_positive_int",
    "merge_artifacts",
    "merge_steps",
    "normalize_artifacts",
    "normalize_steps",
    "normalize_text_list",
]
