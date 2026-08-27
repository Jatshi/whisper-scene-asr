import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.scene_classifier import SceneClassifier  # noqa: E402


def test_scene_classifier_accepts_hidden_states_and_cached_summaries() -> None:
    classifier = SceneClassifier(hidden_size=4).eval()
    hidden = torch.randn(3, 20, 4)
    direct = classifier(hidden)
    cached = classifier.forward_summary(SceneClassifier.summarize(hidden))
    assert direct.shape == (3, 5)
    torch.testing.assert_close(direct, cached)
