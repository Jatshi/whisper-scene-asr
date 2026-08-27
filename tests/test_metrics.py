import pytest

from src.metrics import corpus_counts, error_counts, paired_bootstrap_difference


def test_error_counts_and_corpus_weighting() -> None:
    assert error_counts("你好", "你号").to_dict()["substitutions"] == 1
    counts = corpus_counts(["a", "abcdefghij"], ["", "abcdefghijx"])
    assert counts.errors == 2
    assert counts.reference_units == 11
    assert counts.rate == pytest.approx(2 / 11)


def test_paired_bootstrap_is_deterministic_for_equal_systems() -> None:
    report = paired_bootstrap_difference(["你好", "世界"], ["你号", "世界"], ["你号", "世界"], samples=50)
    assert report["candidate_minus_baseline"] == 0
    assert report["ci95_low"] == 0
    assert report["ci95_high"] == 0


def test_empty_bootstrap_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        paired_bootstrap_difference([], [], [])
