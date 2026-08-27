from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

SCENE_NAMES = ("clean", "noisy", "reverb", "fast_slow", "noisy_reverb")
SCENE_TO_INDEX = {name: idx for idx, name in enumerate(SCENE_NAMES)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(*parts: object) -> str:
    """Return a short deterministic identifier without exposing local paths."""
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def source_id(row: dict) -> str:
    """Get the clean-source identity used to prevent split leakage."""
    return str(row.get("source_id") or row.get("utterance_id") or stable_id(row["audio_path"]))


def manifest_fingerprint(rows: Sequence[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(source_id(row).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(str(row.get("scene", "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def assert_disjoint_sources(*splits: Sequence[dict]) -> None:
    """Fail when augmented views of the same utterance cross data splits."""
    seen: set[str] = set()
    for split in splits:
        current = {source_id(row) for row in split}
        overlap = seen & current
        if overlap:
            sample = ", ".join(sorted(overlap)[:3])
            raise ValueError(f"Source leakage across splits ({len(overlap)} ids), e.g. {sample}")
        seen.update(current)
