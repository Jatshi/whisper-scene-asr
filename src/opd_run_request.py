"""Bind resumable OPD stage markers to source and hyperparameter identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def source_fingerprint(project_dir: Path) -> str:
    digest = hashlib.sha256()
    paths = [*project_dir.joinpath("src").glob("*.py"), *project_dir.joinpath("scripts").glob("*opd*.sh")]
    for path in sorted(paths):
        digest.update(str(path.relative_to(project_dir)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate(path: Path, request: dict) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != request:
            raise RuntimeError("OPD run configuration/source changed; choose a new OUTPUT_ROOT")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--replay-window", type=int, required=True)
    parser.add_argument("--collection-size", type=int, required=True)
    parser.add_argument("--eval-limit", type=int, required=True)
    parser.add_argument("--oracle-fraction", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--policy-batch-size", type=int, required=True)
    parser.add_argument("--policy-epochs", type=int, required=True)
    parser.add_argument("--offline-epochs", type=int, required=True)
    parser.add_argument("--run-ablations", choices=("0", "1"), required=True)
    parser.add_argument("--kl-direction", choices=("forward", "reverse"), required=True)
    parser.add_argument("--include-joint", choices=("0", "1"), required=True)
    parser.add_argument("--allow-single-seed-aggregate", choices=("0", "1"), required=True)
    args = parser.parse_args()
    request = {
        "schema_version": "3.0",
        "source_fingerprint": source_fingerprint(args.project_dir),
        "seeds": args.seeds,
        "rounds": args.rounds,
        "replay_window": args.replay_window,
        "collection_size": args.collection_size,
        "eval_limit": args.eval_limit,
        "oracle_fraction": args.oracle_fraction,
        "batch_size": args.batch_size,
        "policy_batch_size": args.policy_batch_size,
        "policy_epochs": args.policy_epochs,
        "offline_epochs": args.offline_epochs,
        "run_ablations": args.run_ablations == "1",
        "kl_direction": args.kl_direction,
        "include_joint": args.include_joint == "1",
        "allow_single_seed_aggregate": args.allow_single_seed_aggregate == "1",
    }
    validate(args.path, request)
    print(json.dumps(request))


if __name__ == "__main__":
    main()
