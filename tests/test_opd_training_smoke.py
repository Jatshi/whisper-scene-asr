import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.opd_policy import DEFAULT_ACTION_NAMES, OPDRouterPolicy  # noqa: E402
from src.opd_trainer import OPDLossConfig, train_annotations  # noqa: E402


def test_annotation_to_checkpoint_smoke(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    rows = []
    for index in range(8):
        sampled = "clean" if index % 2 else "base"
        rows.append(
            {
                "summary": [float(index), 1.0, 0.0, -1.0],
                "sampled_action": sampled,
                "base_cer": 0.4,
                "action_cer": 0.2 if sampled == "clean" else 0.4,
                "teacher_probs": {"base": 0.2, "clean": 0.8},
                "action_names": list(DEFAULT_ACTION_NAMES),
                "fallback": sampled == "base",
                "priority_weight": 0.6,
                "policy_probs": {name: 1 / len(DEFAULT_ACTION_NAMES) for name in DEFAULT_ACTION_NAMES},
            }
        )
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    initial = tmp_path / "initial.pt"
    torch.save(OPDRouterPolicy(summary_dim=4).checkpoint_payload(), initial)
    output = tmp_path / "trained"
    report = train_annotations(
        [annotations],
        output,
        seed=42,
        epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        loss_config=OPDLossConfig(),
        initial_checkpoint=initial,
    )
    assert report["status"] == "trained"
    payload = torch.load(output / "policy.pt", map_location="cpu", weights_only=False)
    restored = OPDRouterPolicy.from_checkpoint_payload(payload)
    assert restored(torch.zeros(2, 4)).shape == (2, len(DEFAULT_ACTION_NAMES))
