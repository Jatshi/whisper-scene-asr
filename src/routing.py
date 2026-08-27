"""Dependency-light routing policy shared by service and evaluation code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_weights(probabilities: dict[str, float], available: set[str]) -> dict[str, float]:
    if not available:
        return {}
    weights = {name: max(0.0, float(probabilities.get(name, 0.0))) for name in sorted(available)}
    total = sum(weights.values())
    return (
        {name: value / total for name, value in weights.items()}
        if total
        else {name: 1 / len(weights) for name in weights}
    )


def sparsify_weights(
    probabilities: dict[str, float],
    available: set[str],
    *,
    top_k: int = 2,
    quantization_step: float = 0.05,
) -> dict[str, float]:
    """Create bounded-cardinality soft routes for practical long-running inference."""
    weights = normalize_weights(probabilities, available)
    if not weights:
        return {}
    selected = sorted(weights, key=weights.get, reverse=True)[: max(1, top_k)]
    sparse = {name: weights[name] for name in selected}
    if quantization_step > 0:
        sparse = {name: round(value / quantization_step) * quantization_step for name, value in sparse.items()}
        if not any(sparse.values()):
            sparse[selected[0]] = quantization_step
    total = sum(sparse.values())
    return {name: value / total for name, value in sparse.items() if value > 0}


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    confidence: float
    entropy: float
    reason: str
    probabilities: dict[str, float]


def decide_route(
    probabilities: dict[str, float], confidence_threshold: float = 0.65, entropy_threshold: float = 1.35
) -> RoutingDecision:
    if not probabilities:
        return RoutingDecision("base", 0.0, 0.0, "router_unavailable", {})
    scene = max(probabilities, key=probabilities.get)
    confidence = float(probabilities[scene])
    values = np.asarray(list(probabilities.values()), dtype=np.float64)
    values = values / max(float(values.sum()), 1e-12)
    entropy = float(-(values * np.log(values.clip(min=1e-12))).sum())
    if confidence < confidence_threshold:
        return RoutingDecision("base", confidence, entropy, "low_confidence", probabilities)
    if entropy > entropy_threshold:
        return RoutingDecision("base", confidence, entropy, "high_entropy", probabilities)
    return RoutingDecision(scene, confidence, entropy, "accepted", probabilities)
