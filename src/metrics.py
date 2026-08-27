"""Corpus-level ASR metrics and paired bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.errors / max(self.reference_units, 1)

    def to_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "errors": self.errors, "cer": self.rate}


def normalize_for_cer(text: str) -> list[str]:
    """Use non-space characters as units, matching standard Mandarin CER."""
    return list("".join(text.split()))


def error_counts(reference: str, hypothesis: str) -> ErrorCounts:
    """Compute exact Levenshtein operation counts with deterministic tie-breaking."""
    ref, hyp = normalize_for_cer(reference), normalize_for_cer(hypothesis)
    # Each cell stores (total errors, substitutions, deletions, insertions).
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, ref_unit in enumerate(ref, start=1):
        current = [(i, 0, i, 0)]
        for j, hyp_unit in enumerate(hyp, start=1):
            if ref_unit == hyp_unit:
                current.append(previous[j - 1])
                continue
            sub = previous[j - 1]
            delete = previous[j]
            insert = current[j - 1]
            candidates = [
                (sub[0] + 1, sub[1] + 1, sub[2], sub[3]),
                (delete[0] + 1, delete[1], delete[2] + 1, delete[3]),
                (insert[0] + 1, insert[1], insert[2], insert[3] + 1),
            ]
            current.append(min(candidates))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return ErrorCounts(substitutions, deletions, insertions, len(ref))


def corpus_counts(references: Iterable[str], hypotheses: Iterable[str]) -> ErrorCounts:
    total = np.zeros(4, dtype=np.int64)
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        counts = error_counts(reference, hypothesis)
        total += [counts.substitutions, counts.deletions, counts.insertions, counts.reference_units]
    return ErrorCounts(*(int(value) for value in total))


def paired_bootstrap_difference(
    references: Sequence[str],
    baseline: Sequence[str],
    candidate: Sequence[str],
    *,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap candidate CER minus baseline CER by resampling utterances."""
    if not (len(references) == len(baseline) == len(candidate)):
        raise ValueError("references, baseline and candidate must have equal length")
    if not references:
        raise ValueError("bootstrap requires at least one utterance")
    base = np.asarray([[c.errors, c.reference_units] for c in map(error_counts, references, baseline)], dtype=np.int64)
    cand = np.asarray([[c.errors, c.reference_units] for c in map(error_counts, references, candidate)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(references), size=len(references))
        denominator = max(int(base[chosen, 1].sum()), 1)
        differences[index] = (cand[chosen, 0].sum() - base[chosen, 0].sum()) / denominator
    observed = corpus_counts(references, candidate).rate - corpus_counts(references, baseline).rate
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "candidate_minus_baseline": float(observed),
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "probability_candidate_better": float(np.mean(differences < 0)),
        "bootstrap_samples": samples,
        "seed": seed,
    }
