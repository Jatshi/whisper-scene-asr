import pytest

from src.routing import decide_route, normalize_weights, sparsify_weights


def test_normalize_weights_ignores_unavailable_and_normalizes() -> None:
    result = normalize_weights({"clean": 2, "noisy": 1, "unknown": 99}, {"clean", "noisy"})
    assert result == pytest.approx({"clean": 2 / 3, "noisy": 1 / 3})


def test_router_falls_back_on_low_confidence() -> None:
    probabilities = {"clean": 0.3, "noisy": 0.2, "reverb": 0.2, "fast_slow": 0.15, "noisy_reverb": 0.15}
    decision = decide_route(probabilities, confidence_threshold=0.65)
    assert decision.route == "base"
    assert decision.reason == "low_confidence"


def test_router_accepts_confident_scene() -> None:
    probabilities = {"clean": 0.9, "noisy": 0.025, "reverb": 0.025, "fast_slow": 0.025, "noisy_reverb": 0.025}
    decision = decide_route(probabilities)
    assert decision.route == "clean"
    assert decision.reason == "accepted"


def test_sparse_soft_route_is_top_two_quantized_and_normalized() -> None:
    probabilities = {"clean": 0.61, "noisy": 0.24, "reverb": 0.1, "fast_slow": 0.04, "noisy_reverb": 0.01}
    weights = sparsify_weights(probabilities, set(probabilities), top_k=2, quantization_step=0.05)
    assert set(weights) == {"clean", "noisy"}
    assert sum(weights.values()) == pytest.approx(1.0)
