"""Create a portable archive only when all trained ASR artifacts exist."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from src.common import SCENE_NAMES


def required_paths(project_dir: Path) -> list[Path]:
    output = project_dir / "output"
    paths = [
        project_dir / "app.py",
        project_dir / "requirements.txt",
        output / "scene_classifier.pt",
        output / "evaluation.csv",
        output / "evaluation.summary.json",
    ]
    paths.extend(output / "adapters" / scene / "adapter_config.json" for scene in SCENE_NAMES)
    return paths


def package(project_dir: Path, archive: Path) -> Path:
    """Package the legacy v1 artifacts without changing their historical contract."""
    missing = [path for path in required_paths(project_dir) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path.relative_to(project_dir)}" for path in missing)
        raise FileNotFoundError(f"Refusing incomplete package; missing:\n{formatted}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    include = [
        project_dir / "src",
        project_dir / "app.py",
        project_dir / "requirements.txt",
        project_dir / "README.md",
        project_dir / "output",
    ]
    with tarfile.open(archive, "w:gz") as bundle:
        for path in include:
            bundle.add(path, arcname=path.relative_to(project_dir))
    return archive


def required_v2_paths(output_root: Path) -> list[Path]:
    paths = [
        output_root / "scene_classifier.pt",
        output_root / "router_metrics.json",
        output_root / "evaluation.csv",
        output_root / "evaluation.summary.json",
        output_root / "run_manifest.json",
        output_root / "joint" / "deploy" / "joint" / "adapter_config.json",
    ]
    paths.extend(output_root / "adapters" / scene / "adapter_config.json" for scene in SCENE_NAMES)
    return paths


def package_v2(project_dir: Path, output_root: Path, archive: Path) -> Path:
    missing = [path for path in required_v2_paths(output_root) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Refusing incomplete v2 package; missing:\n{formatted}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    source_items = [
        project_dir / name
        for name in (
            "src",
            "tests",
            "scripts",
            "docs",
            ".github",
            "app.py",
            "README.md",
            "requirements.txt",
            "requirements-ci.txt",
            "pyproject.toml",
        )
        if (project_dir / name).exists()
    ]

    def exclude_checkpoints(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        blocked = ("checkpoint-", "__pycache__", ".pytest_cache", "/logs/", ".partial.jsonl")
        return None if info.name.endswith(archive.name) or any(part in info.name for part in blocked) else info

    with tarfile.open(archive, "w:gz") as bundle:
        for path in source_items:
            bundle.add(path, arcname=path.relative_to(project_dir), filter=exclude_checkpoints)
        bundle.add(output_root, arcname=Path("output") / "v2", filter=exclude_checkpoints)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path, default=Path("whisper-scene-asr-project.tar.gz"))
    parser.add_argument("--version", choices=("legacy", "v2"), default="legacy")
    parser.add_argument("--output-root", type=Path, default=Path("output/v2"))
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    destination = args.archive if args.archive.is_absolute() else project_dir / args.archive
    if args.version == "v2":
        output_root = args.output_root if args.output_root.is_absolute() else project_dir / args.output_root
        print(package_v2(project_dir, output_root, destination))
    else:
        print(package(project_dir, destination))
