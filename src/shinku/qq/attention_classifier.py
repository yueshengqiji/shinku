"""Optional, data-trained local attention judge; never loads executable pickles."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .message import IncomingMessage

FEATURE_NAMES = (
    "owner", "question", "has_reply", "has_media", "text_chars",
    "recent_count", "recent_speakers", "same_last_sender", "other_mention_recent",
    "bot_exchange", "directed_followup", "quiet_seconds",
)


def attention_features(
    message: IncomingMessage, *, recent: tuple[dict[str, Any], ...],
    owner: bool, bot_exchange: bool, directed_followup: bool, now: float,
) -> dict[str, float]:
    """Use bounded local metadata, not chat contents or QQ names in the model."""

    previous = recent[-1] if recent else {}
    last_at = float(previous.get("timestamp") or 0.0)
    return {
        "owner": float(owner),
        "question": float(str(message.text_without_at_mentions or "").rstrip().endswith(("?", "？"))),
        "has_reply": float(bool(message.reply_message_ids)),
        "has_media": float(message.has_media),
        "text_chars": min(len(message.text), 500) / 500.0,
        "recent_count": min(len(recent), 8) / 8.0,
        "recent_speakers": min(len({str(item.get("sender_id") or "") for item in recent if item.get("sender_id")}), 8) / 8.0,
        "same_last_sender": float(bool(previous and str(previous.get("sender_id") or "") == message.sender_id)),
        "other_mention_recent": float(bool(previous.get("at_target_ids"))),
        "bot_exchange": float(bot_exchange),
        "directed_followup": float(directed_followup),
        "quiet_seconds": min(max(0.0, now - last_at), 300.0) / 300.0 if last_at else 1.0,
    }


@dataclass(frozen=True)
class LinearAttentionClassifier:
    coefficients: tuple[float, ...]
    intercept: float
    trained_examples: int

    @classmethod
    def load(cls, path: str | Path) -> "LinearAttentionClassifier":
        model_path = Path(path)
        if model_path.stat().st_size > 64_000:
            raise ValueError("attention_model_too_large")
        data = json.loads(model_path.read_text(encoding="utf-8"))
        if data.get("schema") != 1 or tuple(data.get("features") or ()) != FEATURE_NAMES:
            raise ValueError("attention_model_schema_mismatch")
        coefficients = tuple(float(value) for value in data.get("coefficients", ()))
        intercept = float(data.get("intercept"))
        trained_examples = int(data.get("trained_examples", 0))
        if len(coefficients) != len(FEATURE_NAMES) or not all(map(math.isfinite, (*coefficients, intercept))):
            raise ValueError("attention_model_invalid_weights")
        if trained_examples < 100:
            raise ValueError("attention_model_insufficient_examples")
        return cls(coefficients, intercept, trained_examples)

    def score(self, features: Mapping[str, float]) -> float:
        value = self.intercept + sum(
            weight * float(features[name]) for name, weight in zip(FEATURE_NAMES, self.coefficients)
        )
        if not math.isfinite(value):
            raise ValueError("attention_model_nonfinite_score")
        return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, value))))
