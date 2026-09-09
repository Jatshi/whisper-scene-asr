import pytest

torch = pytest.importorskip("torch")

from src.opd_policy import DEFAULT_ACTION_NAMES, soft_weights_from_logits  # noqa: E402


def test_soft_weights_use_only_experts_top_k_and_normalize():
    logits = torch.tensor([[99.0, 1.0, 4.0, 2.0, 3.0, -1.0, 88.0]])
    weights = soft_weights_from_logits(logits, DEFAULT_ACTION_NAMES, top_k=2, temperature=1.0)
    assert weights.shape == (1, 5)
    assert torch.isclose(weights.sum(-1), torch.ones(1)).all()
    assert torch.count_nonzero(weights[0]).item() == 2
    # Expert logits are [1, 4, 2, 3, -1], so noisy and fast_slow survive.
    assert weights[0, 1] > weights[0, 3] > 0


def test_soft_weight_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature"):
        soft_weights_from_logits(torch.zeros(1, 7), DEFAULT_ACTION_NAMES, temperature=0.0)
