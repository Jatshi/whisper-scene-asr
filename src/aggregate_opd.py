"""Aggregate three-seed OPD reports without turning missing runs into claims."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def aggregate(paths: list[Path], *, allow_single: bool = False, validation_only: bool = False) -> dict:
    if len(paths) < 2 and not allow_single:
        raise ValueError("Multi-seed aggregation requires at least two completed summaries")
    if not paths:
        raise ValueError("Aggregation requires at least one completed summary")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    systems = ("opd_hard", "opd_soft", "oracle")
    values = {
        system: [float(report["buckets"]["overall"]["metrics"][system]["cer"]) for report in reports]
        for system in systems
    }
    result = {
        "schema_version": "3.0",
        "status": "smoke_only" if len(reports) == 1 else ("validation_only" if validation_only else "complete"),
        "seeds": len(reports),
        "source_reports": [str(path) for path in paths],
        "metrics": {
            system: {
                "cer_mean": statistics.mean(items),
                "cer_std": statistics.stdev(items) if len(items) > 1 else None,
                "cer_by_seed": items,
            }
            for system, items in values.items()
        },
    }
    hard = result["metrics"]["opd_hard"]["cer_mean"]
    soft = result["metrics"]["opd_soft"]["cer_mean"]
    hard_probabilities = [
        float(report["paired_vs_v2"]["opd_hard_vs_v2_hard"]["probability_candidate_better"]) for report in reports
    ]
    best = min(hard, soft)
    claim_eligible = len(reports) > 1 and not validation_only
    reason = None
    if len(reports) == 1:
        reason = "single-seed smoke; not valid for formal claims"
    elif validation_only:
        reason = "stratified validation subset; not a full 5,000-utterance test claim"
    result["acceptance"] = {
        "claim_eligible": claim_eligible,
        "reason": reason,
        "opd_hard_not_worse_than_v2_hard": hard <= 0.17171,
        "opd_hard_bootstrap_probability_ge_0_95_all_seeds": all(value >= 0.95 for value in hard_probabilities),
        "opd_hard_bootstrap_probability_by_seed": hard_probabilities,
        "opd_soft_not_worse_than_v2_soft": soft <= 0.19353,
        "thresholds": {"v2_hard_cer": 0.17171, "v2_soft_cer": 0.19353, "v2_joint_cer": 0.15051},
        "hard_to_joint_gap_recovered_fraction": (0.17171 - best) / (0.17171 - 0.15051),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-single", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    result = aggregate(args.summaries, allow_single=args.allow_single, validation_only=args.validation_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
