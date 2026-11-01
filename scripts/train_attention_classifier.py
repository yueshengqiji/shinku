"""Train an optional QQ turn-taking model from human-labeled trace features.

Input JSONL: {"label": 0 or 1, "features": {...}, "observed_at": number}.
Never use the old bot's reply/no-reply outcome as the label: that outcome was
selected by a random sampler and would reproduce its bias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shinku.qq.attention_classifier import FEATURE_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, help="Human-labeled JSONL, chronologically ordered")
    parser.add_argument("output", type=Path, help="JSON output model; not a pickle")
    args = parser.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score

    examples: list[tuple[list[float], int]] = []
    for line_no, line in enumerate(args.labels.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        label = item.get("label")
        features = item.get("features")
        if type(label) is not int or label not in (0, 1) or not isinstance(features, dict):
            raise ValueError(f"invalid_label_or_features_at_line_{line_no}")
        if set(features) != set(FEATURE_NAMES):
            raise ValueError(f"feature_schema_mismatch_at_line_{line_no}")
        examples.append(([float(features[name]) for name in FEATURE_NAMES], label))
    if len(examples) < 150 or min(sum(label == value for _, label in examples) for value in (0, 1)) < 30:
        raise ValueError("need_at_least_150_human_labels_and_30_per_class")
    split = int(len(examples) * 0.8)
    train, held_out = examples[:split], examples[split:]
    if len({label for _, label in train}) != 2 or len({label for _, label in held_out}) != 2:
        raise ValueError("chronological_holdout_needs_both_classes")
    model = LogisticRegression(max_iter=500)
    model.fit([row for row, _ in train], [label for _, label in train])
    truth = [label for _, label in held_out]
    scores = model.predict_proba([row for row, _ in held_out])[:, 1]
    guesses = [int(score >= 0.8) for score in scores]
    report = {
        "train_examples": len(train),
        "held_out_examples": len(held_out),
        "candidate_precision_at_0_8": float(precision_score(truth, guesses, zero_division=0)),
        "candidate_recall_at_0_8": float(recall_score(truth, guesses, zero_division=0)),
        "candidate_count": sum(guesses),
    }
    payload = {
        "schema": 1, "features": list(FEATURE_NAMES),
        "coefficients": [float(value) for value in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "trained_examples": len(train), "holdout_report": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
