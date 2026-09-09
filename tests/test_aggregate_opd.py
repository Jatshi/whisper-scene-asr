import json
from pathlib import Path

import pytest

from src.aggregate_opd import aggregate


def test_multiseed_aggregate_enforces_cer_and_bootstrap_gates(tmp_path: Path) -> None:
    paths = []
    for seed, hard in ((42, 0.16), (7, 0.17)):
        report = {
            "buckets": {
                "overall": {
                    "metrics": {
                        "opd_hard": {"cer": hard},
                        "opd_soft": {"cer": 0.18},
                        "oracle": {"cer": 0.1},
                    }
                }
            },
            "paired_vs_v2": {
                "opd_hard_vs_v2_hard": {"probability_candidate_better": 0.96},
            },
        }
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    result = aggregate(paths)
    assert result["metrics"]["opd_hard"]["cer_mean"] == 0.165
    assert result["acceptance"]["opd_hard_not_worse_than_v2_hard"]
    assert result["acceptance"]["opd_hard_bootstrap_probability_ge_0_95_all_seeds"]


def test_single_seed_requires_explicit_smoke_override(tmp_path: Path) -> None:
    report = {
        "buckets": {
            "overall": {
                "metrics": {
                    "opd_hard": {"cer": 0.2},
                    "opd_soft": {"cer": 0.19},
                    "oracle": {"cer": 0.1},
                }
            }
        },
        "paired_vs_v2": {
            "opd_hard_vs_v2_hard": {"probability_candidate_better": 0.4},
        },
    }
    path = tmp_path / "42.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="at least two"):
        aggregate([path])
    result = aggregate([path], allow_single=True)
    assert result["status"] == "smoke_only"
    assert result["metrics"]["opd_hard"]["cer_std"] is None
    assert not result["acceptance"]["claim_eligible"]


def test_multiseed_validation_is_not_claim_eligible(tmp_path: Path) -> None:
    paths = []
    for seed in (42, 7):
        report = {
            "buckets": {
                "overall": {
                    "metrics": {
                        "opd_hard": {"cer": 0.16},
                        "opd_soft": {"cer": 0.18},
                        "oracle": {"cer": 0.1},
                    }
                }
            },
            "paired_vs_v2": {"opd_hard_vs_v2_hard": {"probability_candidate_better": 0.96}},
        }
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    result = aggregate(paths, validation_only=True)
    assert result["status"] == "validation_only"
    assert not result["acceptance"]["claim_eligible"]
