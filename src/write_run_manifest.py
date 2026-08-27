"""Write an evidence-backed run manifest from artifacts that actually exist."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def build_manifest(data_root: Path, output_root: Path) -> dict:
    required = [
        data_root / "scenes-v2" / "dataset_report.json",
        output_root / "router_metrics.json",
        output_root / "evaluation.summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Cannot claim a complete run; missing:\n" + "\n".join(missing))
    artifacts = [
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and "checkpoint-" not in str(path)
        and "logs" not in path.relative_to(output_root).parts
        and "stages" not in path.relative_to(output_root).parts
        and not path.name.endswith(".partial.jsonl")
    ]
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "librosa", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": "2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": _command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "packages": packages,
        "dataset_report": json.loads(required[0].read_text(encoding="utf-8")),
        "router_metrics": json.loads(required[1].read_text(encoding="utf-8")),
        "evaluation": json.loads(required[2].read_text(encoding="utf-8")),
        "artifacts": [
            {"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in sorted(artifacts)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.data_root, args.output_root)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.destination)


if __name__ == "__main__":
    main()
