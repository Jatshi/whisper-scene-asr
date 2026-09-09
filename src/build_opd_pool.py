"""Build the OPD train+validation pool while preserving the held-out test wall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.common import assert_disjoint_sources, manifest_fingerprint, read_jsonl, source_id, stable_id, write_jsonl


def build_pool(
    train: Path,
    validation: Path,
    test: Path,
    output: Path,
    report: Path,
    calibration_output: Path | None = None,
    calibration_fraction: float = 0.0,
) -> dict:
    if not 0 <= calibration_fraction < 1:
        raise ValueError("calibration_fraction must be in [0, 1)")
    train_rows, validation_rows, test_rows = read_jsonl(train), read_jsonl(validation), read_jsonl(test)
    assert_disjoint_sources(train_rows, validation_rows, test_rows)
    ranked_sources = sorted({source_id(row) for row in validation_rows}, key=stable_id)
    calibration_count = round(len(ranked_sources) * calibration_fraction)
    calibration_sources = set(ranked_sources[:calibration_count])
    calibration_rows = [row for row in validation_rows if source_id(row) in calibration_sources]
    pool_validation = [row for row in validation_rows if source_id(row) not in calibration_sources]
    pool = [*train_rows, *pool_validation]
    assert_disjoint_sources(pool, calibration_rows, test_rows)
    write_jsonl(pool, output)
    if calibration_output is not None:
        write_jsonl(calibration_rows, calibration_output)
    result = {
        "schema_version": "3.0",
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "validation_rows_in_pool": len(pool_validation),
        "calibration_rows": len(calibration_rows),
        "opd_pool_rows": len(pool),
        "held_out_test_rows": len(test_rows),
        "source_disjoint": True,
        "pool_fingerprint": manifest_fingerprint(pool),
        "test_fingerprint": manifest_fingerprint(test_rows),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path)
    parser.add_argument("--calibration-fraction", type=float, default=0.0)
    args = parser.parse_args()
    print(
        json.dumps(
            build_pool(
                args.train,
                args.validation,
                args.test,
                args.output,
                args.report,
                args.calibration_output,
                args.calibration_fraction,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
