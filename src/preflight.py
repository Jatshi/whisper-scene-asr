"""Fail fast on the external requirements that cannot be proven before GPU access."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def verify(project_dir: Path, minimum_gb: int, require_gpu: bool) -> list[str]:
    messages: list[str] = []
    free_gb = shutil.disk_usage(project_dir).free / 1024**3
    if free_gb < minimum_gb:
        raise RuntimeError(f"Only {free_gb:.1f}GB free at {project_dir}; need at least {minimum_gb}GB")
    messages.append(f"disk: {free_gb:.1f}GB free")
    messages.append(f"python: {sys.version.split()[0]}")
    if require_gpu:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError("NVIDIA GPU is required but nvidia-smi is unavailable") from exc
        messages.append("gpu: " + result.stdout.strip().replace("\n", "; "))
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch must be supplied by the CUDA image before dependency install") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch is installed but CUDA is unavailable")
        messages.append(f"torch: {torch.__version__}; cuda={torch.version.cuda}")
    return messages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--minimum-gb", type=int, default=100)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    messages = verify(args.project_dir, args.minimum_gb, not args.no_gpu)
    for message in messages:
        print(message)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps({"checks": messages}, indent=2) + "\n", encoding="utf-8")
