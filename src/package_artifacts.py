"""Create a portable archive only when all trained ASR artifacts exist."""
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from src.common import SCENE_NAMES


def required_paths(project_dir: Path) -> list[Path]:
    output = project_dir / "output"
    paths = [project_dir / "app.py", project_dir / "requirements.txt", output / "scene_classifier.pt", output / "evaluation.csv", output / "evaluation.summary.json"]
    paths.extend(output / "adapters" / scene / "adapter_config.json" for scene in SCENE_NAMES)
    return paths


def package(project_dir: Path, archive: Path) -> Path:
    missing = [path for path in required_paths(project_dir) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path.relative_to(project_dir)}" for path in missing)
        raise FileNotFoundError(f"Refusing incomplete package; missing:\n{formatted}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    include = [project_dir / "src", project_dir / "app.py", project_dir / "requirements.txt", project_dir / "README.md", project_dir / "output"]
    with tarfile.open(archive, "w:gz") as bundle:
        for path in include:
            bundle.add(path, arcname=path.relative_to(project_dir))
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path.cwd()); parser.add_argument("--archive", type=Path, default=Path("whisper-scene-asr-project.tar.gz")); args = parser.parse_args()
    project_dir = args.project_dir.resolve(); destination = args.archive if args.archive.is_absolute() else project_dir / args.archive
    print(package(project_dir, destination))
