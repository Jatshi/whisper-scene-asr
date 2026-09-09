import pytest

torch = pytest.importorskip("torch")

from src.opd_trainer import OPDLossConfig, compute_opd_loss  # noqa: E402


def _batch(logits):
    return compute_opd_loss(
        logits=logits,
        sampled_actions=torch.tensor([0, 1]),
        advantages=torch.tensor([1.0, -0.5]),
        teacher_probs=torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0]]),
        teacher_mask=torch.tensor([[True, True, False], [True, True, False]]),
        fallback_mask=torch.tensor([False, True]),
        config=OPDLossConfig(entropy_weight=0.1, fallback_weight=2.0),
    )


def test_opd_loss_is_finite_and_backpropagates():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]], requires_grad=True)
    losses = _batch(logits)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_entropy_regularizer_rewards_higher_entropy_when_minimizing():
    uniform = torch.zeros(2, 3, requires_grad=True)
    peaked = torch.tensor([[8.0, -8.0, -8.0], [8.0, -8.0, -8.0]], requires_grad=True)
    assert _batch(uniform)["entropy_regularizer"] < _batch(peaked)["entropy_regularizer"]


def test_teacher_mask_requires_observed_action():
    with pytest.raises(ValueError, match="teacher support"):
        compute_opd_loss(
            logits=torch.zeros(1, 3),
            sampled_actions=torch.tensor([0]),
            advantages=torch.tensor([0.0]),
            teacher_probs=torch.zeros(1, 3),
            teacher_mask=torch.zeros(1, 3, dtype=torch.bool),
            fallback_mask=torch.tensor([False]),
            config=OPDLossConfig(),
        )


def test_single_observed_action_has_no_distillation_signal():
    losses = compute_opd_loss(
        logits=torch.tensor([[0.0, 3.0, -1.0]]),
        sampled_actions=torch.tensor([0]),
        advantages=torch.tensor([0.0]),
        teacher_probs=torch.tensor([[1.0, 0.0, 0.0]]),
        teacher_mask=torch.tensor([[True, False, False]]),
        fallback_mask=torch.tensor([False]),
        config=OPDLossConfig(policy_gradient_weight=0.0, entropy_weight=0.0),
    )
    assert losses["distillation"] == 0
    assert losses["total"] == 0
