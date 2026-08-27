import json
import tarfile

from src.common import SCENE_NAMES
from src.package_artifacts import package, package_v2


def test_package_requires_complete_training_outputs(tmp_path) -> None:
    (tmp_path / "app.py").write_text("", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    (tmp_path / "README.md").write_text("", encoding="utf-8")
    (tmp_path / "src").mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "scene_classifier.pt").write_bytes(b"classifier")
    (output / "evaluation.csv").write_text("reference\n", encoding="utf-8")
    (output / "evaluation.summary.json").write_text(json.dumps({}), encoding="utf-8")
    for scene in SCENE_NAMES:
        directory = output / "adapters" / scene
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    archive = package(tmp_path, tmp_path / "bundle.tar.gz")
    assert archive.exists() and archive.stat().st_size > 0


def test_v2_package_excludes_itself_and_checkpoints(tmp_path) -> None:
    for name in ("src", "tests", "scripts", "docs"):
        (tmp_path / name).mkdir()
    for name in ("app.py", "README.md", "requirements.txt", "pyproject.toml"):
        (tmp_path / name).write_text("", encoding="utf-8")
    output = tmp_path / "output" / "v2"
    output.mkdir(parents=True)
    for name in (
        "scene_classifier.pt",
        "router_metrics.json",
        "evaluation.csv",
        "evaluation.summary.json",
        "run_manifest.json",
    ):
        (output / name).write_text("{}", encoding="utf-8")
    for scene in SCENE_NAMES:
        directory = output / "adapters" / scene
        directory.mkdir(parents=True)
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    joint = output / "joint" / "deploy" / "joint"
    joint.mkdir(parents=True)
    (joint / "adapter_config.json").write_text("{}", encoding="utf-8")
    checkpoint = output / "joint" / "checkpoint-1"
    checkpoint.mkdir()
    (checkpoint / "large.bin").write_bytes(b"x" * 100)
    archive = output / "bundle-v2.tar.gz"
    package_v2(tmp_path, output, archive)
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
    assert not any("checkpoint-1" in name for name in names)
    assert not any(name.endswith("bundle-v2.tar.gz") for name in names)
