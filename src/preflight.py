"""Fail fast on the external requirements that cannot be proven before GPU access."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def verify(project_dir: Path, minimum_gb: int, require_gpu: bool) -> list[str]:
    messages: list[str] = []
    free_gb = shutil.disk_usage(project_dir).free / 1024**3
    if free_gb < minimum_gb:
        raise RuntimeError(f"Only {free_gb:.1f}GB free at {project_dir}; need at least {minimum_gb}GB")
    messages.append(f"disk: {free_gb:.1f}GB free")
    if require_gpu:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("NVIDIA GPU is required but nvidia-smi is unavailable") from exc
        messages.append("gpu: " + result.stdout.strip().replace("\n", "; "))
    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path.cwd()); parser.add_argument("--minimum-gb", type=int, default=100); parser.add_argument("--no-gpu", action="store_true"); args = parser.parse_args()
    for message in verify(args.project_dir, args.minimum_gb, not args.no_gpu): print(message)
