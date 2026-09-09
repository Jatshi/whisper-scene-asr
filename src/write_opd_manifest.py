"""Create an evidence manifest only after all requested OPD seed runs exist."""

from __future__ import annotations

import argparse
import hashlib
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


def build(output_root: Path, seeds: list[str], status: str = "complete") -> dict:
    required = [output_root / "pool_report.json", output_root / "multiseed_summary.json"]
    for seed in seeds:
        required.extend(
            [output_root / "seeds" / seed / "evaluation.csv", output_root / "seeds" / seed / "evaluation.summary.json"]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Cannot mark OPD complete; missing:\n" + "\n".join(missing))
    tracked = []
    for path in output_root.rglob("*"):
        if (
            path.is_file()
            and "logs" not in path.relative_to(output_root).parts
            and not path.name.endswith(".partial.jsonl")
        ):
            tracked.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return {
        "schema_version": "3.0",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_command(["git", "status", "--porcelain"])),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": _command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "pool_report": json.loads(required[0].read_text(encoding="utf-8")),
        "multiseed_summary": json.loads(required[1].read_text(encoding="utf-8")),
        "artifacts": sorted(tracked, key=lambda item: item["path"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--status", choices=("complete", "validation_only"), default="complete")
    args = parser.parse_args()
    manifest = build(args.output_root, args.seeds, args.status)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.destination)


if __name__ == "__main__":
    main()
