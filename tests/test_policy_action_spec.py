import pytest

torch = pytest.importorskip("torch")

from src.opd_policy import DEFAULT_ACTION_NAMES, OPDRouterPolicy, action_names  # noqa: E402
from src.opd_trainer import initialize_policy  # noqa: E402
from src.scene_classifier import SceneClassifier  # noqa: E402


def test_action_order_is_stable_and_joint_is_opt_in():
    assert DEFAULT_ACTION_NAMES == (
        "base",
        "clean",
        "noisy",
        "reverb",
        "fast_slow",
        "noisy_reverb",
        "soft",
    )
    assert action_names() == DEFAULT_ACTION_NAMES
    assert action_names(include_joint=True) == (*DEFAULT_ACTION_NAMES, "joint")


def test_policy_rejects_checkpoint_action_order_mismatch():
    policy = OPDRouterPolicy(summary_dim=8, hidden_size=4)
    payload = policy.checkpoint_payload()
    payload["action_names"] = list(reversed(payload["action_names"]))
    with pytest.raises(ValueError, match="action order"):
        OPDRouterPolicy.from_checkpoint_payload(payload)


def test_v2_classifier_warm_start_maps_five_expert_rows(tmp_path):
    classifier = SceneClassifier(hidden_size=4)
    source = tmp_path / "scene_classifier.pt"
    torch.save({"state_dict": classifier.state_dict()}, source)
    destination = tmp_path / "policy.pt"
    initialize_policy(source, destination, include_joint=False, strategy_temperature=1.0, fusion_temperature=1.0)
    policy = OPDRouterPolicy.from_checkpoint_payload(torch.load(destination, weights_only=False))
    torch.testing.assert_close(policy.network[0].weight, classifier.net[0].weight)
    torch.testing.assert_close(policy.network[3].weight[1:6], classifier.net[3].weight)
