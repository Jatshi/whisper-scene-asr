import json

from src.common import SCENE_NAMES
from src.package_artifacts import package


def test_package_requires_complete_training_outputs(tmp_path) -> None:
    (tmp_path / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    (tmp_path / "src").mkdir()
    output = tmp_path / "output"; output.mkdir()
    (output / "scene_classifier.pt").write_bytes(b"classifier")
    (output / "evaluation.csv").write_text("reference\n", encoding="utf-8")
    (output / "evaluation.summary.json").write_text(json.dumps({}), encoding="utf-8")
    for scene in SCENE_NAMES:
        directory = output / "adapters" / scene; directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    archive = package(tmp_path, tmp_path / "bundle.tar.gz")
    assert archive.exists() and archive.stat().st_size > 0
