"""Metadata-only request sizing and conservative, opt-in history trimming.

The estimate is deliberately not presented as provider token usage: image costs,
tokenizers and gateway wrappers vary. Actual usage remains in runtime metrics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping


def _positive_int(value: Any, default: int, *, maximum: int = 2_000_000) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(maximum, number))


def estimate_text_tokens(value: Any) -> int:
    text = str(value or "")
    ascii_count = 0
    unicode_cost = 0
    for char in text:
        if ord(char) < 128:
            ascii_count += 1
        else:
            # Kana, Hangul and emoji must not be priced like ASCII. This is
            # still a heuristic, never an exact provider tokenizer result.
            unicode_cost += 2 if ord(char) > 0xFFFF else 1
    return unicode_cost + (ascii_count + 3) // 4


@dataclass(frozen=True)
class ContextBudgetPolicy:
    mode: str = "observe"
    window_tokens: int = 0
    model_windows: Mapping[str, int] | None = None
    output_reserve_tokens: int = 2048
    safety_margin_tokens: int = 1024
    image_estimate_tokens: int = 1024
    tool_result_max_chars: int = 6000
    agent_feedback_max_chars: int = 12000

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "ContextBudgetPolicy":
        env = os.environ if environ is None else environ
        mode = str(env.get("SHINKU_LLM_CONTEXT_BUDGET_MODE", "observe") or "observe").strip().lower()
        if mode not in {"off", "observe", "enforce"}:
            mode = "observe"
        try:
            raw_windows = json.loads(env.get("SHINKU_LLM_MODEL_CONTEXT_WINDOWS_JSON", "{}") or "{}")
        except (TypeError, ValueError):
            raw_windows = {}
        windows = {
            str(name).strip().casefold(): _positive_int(value, 0)
            for name, value in (raw_windows.items() if isinstance(raw_windows, dict) else ())
            if str(name).strip()
        }
        return cls(
            mode=mode,
            window_tokens=_positive_int(env.get("SHINKU_LLM_CONTEXT_WINDOW_TOKENS"), 0),
            model_windows=windows,
            output_reserve_tokens=_positive_int(env.get("SHINKU_LLM_OUTPUT_RESERVE_TOKENS"), 2048),
            safety_margin_tokens=_positive_int(env.get("SHINKU_LLM_CONTEXT_MARGIN_TOKENS"), 1024),
            image_estimate_tokens=_positive_int(env.get("SHINKU_LLM_IMAGE_ESTIMATE_TOKENS"), 1024),
            tool_result_max_chars=_positive_int(env.get("SHINKU_AGENT_TOOL_RESULT_MAX_CHARS"), 6000),
            agent_feedback_max_chars=_positive_int(env.get("SHINKU_AGENT_FEEDBACK_MAX_CHARS"), 12000),
        )

    def input_limit(self, model: str) -> int:
        return max(0, self.context_window(model) - self.output_reserve_tokens - self.safety_margin_tokens)

    def context_window(self, model: str) -> int:
        return int((self.model_windows or {}).get(str(model or "").strip().casefold(), self.window_tokens))


def _content_cost(content: Any, image_estimate_tokens: int) -> tuple[int, int]:
    if isinstance(content, str):
        return estimate_text_tokens(content), 0
    if not isinstance(content, list):
        return estimate_text_tokens(content), 0
    text_tokens = 0
    image_count = 0
    for part in content:
        if not isinstance(part, dict):
            text_tokens += estimate_text_tokens(part)
            continue
        kind = str(part.get("type") or "").casefold()
        if kind in {"image", "image_url", "input_image"} or "image_url" in part:
            image_count += 1
        else:
            text_tokens += estimate_text_tokens(part.get("text") or part.get("content") or "")
    return text_tokens + image_count * image_estimate_tokens, image_count


def _has_agent_feedback_marker(content: Any) -> bool:
    marker = "【工具执行结果反馈与推进】"
    if isinstance(content, str):
        return marker in content
    if isinstance(content, list):
        return any(
            isinstance(part, dict) and marker in str(part.get("text") or "")
            for part in content
        )
    return False


def measure_payload(payload: Mapping[str, Any], policy: ContextBudgetPolicy) -> dict[str, int]:
    buckets = {"system": 0, "history": 0, "current_user": 0, "tools": 0, "images": 0, "overhead": 0}
    messages = payload.get("messages") or []
    last_user_index = max(
        (index for index, item in enumerate(messages) if item.get("role") == "user"),
        default=-1,
    )
    image_count = 0
    for index, message in enumerate(messages):
        cost, images = _content_cost(message.get("content"), policy.image_estimate_tokens)
        image_count += images
        image_tokens = images * policy.image_estimate_tokens
        buckets["images"] += image_tokens
        role = str(message.get("role") or "")
        bucket = "system" if role == "system" else "current_user" if index == last_user_index else "history"
        buckets[bucket] += cost - image_tokens
        buckets["overhead"] += 4
    for block in payload.get("system_extra_blocks") or []:
        buckets["system"] += estimate_text_tokens(block)
    tools = payload.get("tools") or []
    if tools:
        buckets["tools"] = estimate_text_tokens(json.dumps(tools, ensure_ascii=False, sort_keys=True))
        buckets["overhead"] += 4 * len(tools)
    buckets["image_count"] = image_count
    buckets["total"] = sum(value for name, value in buckets.items() if name not in {"total", "image_count"})
    return buckets


def assess_and_trim_payload(
    payload: dict[str, Any], policy: ContextBudgetPolicy
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Only removable history is eligible. Never change system, tools or current turn."""
    before = measure_payload(payload, policy)
    model = str(payload.get("model") or "")
    window = policy.context_window(model)
    limit = policy.input_limit(model)
    record: dict[str, Any] = {
        "mode": policy.mode,
        "model": model,
        "context_window_tokens": window,
        "input_limit_estimate": limit,
        "before": before,
        "after": before,
        "history_turns_removed": 0,
        "history_messages_removed": 0,
        "status": "window_unconfigured" if not window else "input_budget_exhausted" if not limit else "within_budget",
    }
    if policy.mode == "off" or not limit or before["total"] <= limit:
        return payload, record

    candidate = dict(payload)
    messages = list(payload.get("messages") or [])
    candidate["messages"] = messages
    last_user_index = max(
        (index for index, item in enumerate(messages) if item.get("role") == "user"), default=-1
    )
    protected_goal = -1
    if last_user_index >= 0 and _has_agent_feedback_marker(messages[last_user_index].get("content")):
        protected_goal = next(
            (index for index, item in enumerate(messages[:last_user_index]) if item.get("role") == "user"),
            -1,
        )
    # Remove complete old dialogue turns so no orphan assistant answer remains.
    groups: list[list[int]] = []
    current_group: list[int] = []
    for index, message in enumerate(messages[:last_user_index]):
        role = str(message.get("role") or "")
        if role == "user":
            if current_group:
                groups.append(current_group)
            current_group = [index]
        elif role == "assistant":
            current_group.append(index)
        elif current_group:
            groups.append(current_group)
            current_group = []
    if current_group:
        groups.append(current_group)

    removed_indices: set[int] = set()
    for group in groups:
        if measure_payload(candidate, policy)["total"] <= limit:
            break
        removable = [index for index in group if index != protected_goal]
        if not removable:
            continue
        removed_indices.update(removable)
        candidate["messages"] = [
            message for index, message in enumerate(messages) if index not in removed_indices
        ]
    after = measure_payload(candidate, policy)
    if after["total"] > limit:
        record["status"] = "unresolvable_preserved"
        record["projected_after"] = after
        return payload, record
    record["after"] = after
    record["history_turns_removed"] = sum(
        1 for group in groups if group and all(index in removed_indices for index in group)
    )
    record["history_messages_removed"] = len(removed_indices)
    record["status"] = "trimmed_history" if policy.mode == "enforce" else "would_trim_history"
    return (candidate if policy.mode == "enforce" else payload), record
