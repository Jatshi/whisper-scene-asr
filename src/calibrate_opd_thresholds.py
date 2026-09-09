"""Select confidence/entropy fallback thresholds on validation predictions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.metrics import corpus_counts


def calibrate(report: Path, output: Path) -> dict:
    rows = pd.read_csv(report).to_dict("records")
    if not rows:
        raise ValueError("Validation report is empty")
    candidates = []
    for confidence in (0.45, 0.55, 0.65, 0.75, 0.85):
        for entropy in (0.6, 0.9, 1.2, 1.35, 1.5):
            hypotheses = []
            fallbacks = 0
            for row in rows:
                action_predictions = json.loads(row["action_predictions"])
                fallback = float(row["router_confidence"]) < confidence or float(row["router_entropy"]) > entropy
                fallbacks += int(fallback)
                hypotheses.append(action_predictions["base"] if fallback else action_predictions[row["policy_argmax"]])
            cer = corpus_counts([str(row["reference"]) for row in rows], hypotheses).rate
            candidates.append((cer, fallbacks / len(rows), confidence, entropy))
    cer, fallback_rate, confidence, entropy = min(candidates, key=lambda item: (item[0], item[1]))
    result = {
        "schema_version": "3.0",
        "selected_on": "validation_only",
        "confidence_threshold": confidence,
        "entropy_threshold": entropy,
        "validation_cer": cer,
        "fallback_rate": fallback_rate,
        "grid_size": len(candidates),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.validation_report, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
