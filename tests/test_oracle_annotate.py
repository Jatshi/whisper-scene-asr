from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.opd_policy import DEFAULT_ACTION_NAMES  # noqa: E402
from src.oracle_router import (  # noqa: E402
    _decode_actions,
    annotate_predictions,
    annotation_request,
    validate_annotation_request,
)


def test_oracle_annotation_uses_deterministic_tie_break_and_rejects_empty_reference():
    row = {"audio_path": "a.wav", "text": "你好", "scene": "clean", "source_id": "s1"}
    predictions = {name: "你好" for name in DEFAULT_ACTION_NAMES}
    result = annotate_predictions(row, predictions, policy_hash="abc", teacher_temperature=0.2)
    assert result["oracle_route"] == "base"
    assert result["policy_hash"] == "abc"
    assert sum(result["teacher_probs"].values()) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="empty reference"):
        annotate_predictions({**row, "text": ""}, predictions, policy_hash="abc")


def test_dynamic_soft_cache_is_bound_to_policy_hash(tmp_path: Path):
    first = annotation_request("manifest", "policy-a", DEFAULT_ACTION_NAMES, 2, 0.7, 0.05)
    changed = annotation_request("manifest", "policy-b", DEFAULT_ACTION_NAMES, 2, 0.7, 0.05)
    request_path = tmp_path / "request.json"
    validate_annotation_request(request_path, first)
    with pytest.raises(RuntimeError, match="incompatible"):
        validate_annotation_request(request_path, changed)


def test_partial_action_decoder_only_calls_requested_paths():
    class FakeEngine:
        def transcribe_base_batch(self, paths):
            return [f"base:{path}" for path in paths]

        def transcribe_adapter_batch(self, paths, name):
            return [f"{name}:{path}" for path in paths]

        def transcribe_soft_weights_batch(self, paths, weights):
            return [f"soft:{path}:{sorted(weights)}" for path in paths]

    rows = [{"audio_path": "a.wav"}, {"audio_path": "b.wav"}]
    decoded = _decode_actions(
        FakeEngine(),
        rows,
        [{"base", "clean"}, {"base", "soft"}],
        [{"clean": 1.0}, {"noisy": 0.6, "reverb": 0.4}],
        include_joint=False,
    )
    assert set(decoded[0]) == {"base", "clean"}
    assert set(decoded[1]) == {"base", "soft"}
