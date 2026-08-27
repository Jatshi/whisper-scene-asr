"""Build leakage-safe, real-degradation acoustic-scene corpora."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from tqdm import tqdm

from src.audio import TARGET_SAMPLE_RATE, load_audio
from src.common import (
    SCENE_NAMES,
    SCENE_TO_INDEX,
    assert_disjoint_sources,
    read_jsonl,
    set_seed,
    source_id,
    stable_id,
    write_jsonl,
)


@dataclass(frozen=True)
class Assets:
    noise: list[dict]
    rir: list[dict]

    @classmethod
    def from_manifests(cls, noise_manifest: Path, rir_manifest: Path) -> Assets:
        noise, rir = read_jsonl(noise_manifest), read_jsonl(rir_manifest)
        if not noise or not rir:
            raise ValueError("Real-degradation manifests must both be non-empty")
        return cls(noise=noise, rir=rir)


def _fit_noise(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(noise) < length:
        noise = np.tile(noise, int(np.ceil(length / max(len(noise), 1))))
    start = int(rng.integers(0, len(noise) - length + 1))
    return noise[start : start + length]


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    clean_rms = np.sqrt(max(float(np.mean(clean**2)), 1e-12))
    noise_rms = np.sqrt(max(float(np.mean(noise**2)), 1e-12))
    return clean + noise * (clean_rms / (noise_rms * 10 ** (snr_db / 20)))


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    rir = rir.astype(np.float32)
    rir /= max(float(np.sqrt(np.sum(rir**2))), 1e-8)
    return fftconvolve(audio, rir, mode="full")[: len(audio)].astype(np.float32)


def peak_normalize(audio: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio.astype(np.float32) if peak <= ceiling or peak == 0 else (audio * ceiling / peak).astype(np.float32)


def degrade(clean: np.ndarray, scene: str, assets: Assets, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Apply a reproducible scene profile and return full provenance."""
    if scene not in SCENE_NAMES:
        raise ValueError(f"Unknown scene: {scene}")
    audio = clean.astype(np.float32, copy=True)
    metadata: dict[str, str | float] = {"degradation_version": "real-v2"}
    if scene in {"reverb", "noisy_reverb"}:
        item = assets.rir[int(rng.integers(len(assets.rir)))]
        rir, _ = load_audio(item["asset_path"])
        audio = apply_rir(audio, rir)
        metadata.update(rir_id=item["asset_id"], rir_source=item["source"])
    if scene in {"noisy", "noisy_reverb"}:
        item = assets.noise[int(rng.integers(len(assets.noise)))]
        noise, _ = load_audio(item["asset_path"])
        noise = _fit_noise(noise, len(audio), rng)
        snr_db = float(rng.uniform(0.0, 20.0))
        audio = mix_at_snr(audio, noise, snr_db)
        metadata.update(noise_id=item["asset_id"], noise_source=item["source"], snr_db=round(snr_db, 3))
    if scene == "fast_slow":
        rate = float(rng.choice([0.85, 0.90, 1.10, 1.15]))
        audio = librosa.effects.time_stretch(y=audio, rate=rate).astype(np.float32)
        metadata["speed_rate"] = rate
    return peak_normalize(audio), metadata


def build_split(
    rows: list[dict], output_dir: Path, split: str, per_scene: int, assets: Assets, seed: int
) -> list[dict]:
    if per_scene <= 0:
        raise ValueError("per_scene must be positive")
    if not rows:
        raise ValueError(f"Empty source split: {split}")
    rng = np.random.default_rng(seed)
    count = min(per_scene, len(rows))
    # Use the same clean utterances in every scene.  This removes content and
    # speaker identity as shortcuts for the scene classifier and yields paired
    # acoustic conditions for bucket analysis.
    chosen = rng.choice(len(rows), size=count, replace=False)
    output: list[dict] = []
    for scene in SCENE_NAMES:
        folder = output_dir / split / scene
        folder.mkdir(parents=True, exist_ok=True)
        for row_index in tqdm(chosen, desc=f"{split}/{scene}"):
            row = rows[int(row_index)]
            clean, sample_rate = load_audio(row["audio_path"])
            transformed, provenance = degrade(clean, scene, assets, rng)
            utterance_id = str(row.get("utterance_id") or Path(row["audio_path"]).stem)
            destination = folder / f"{stable_id(utterance_id, scene, seed)}.wav"
            sf.write(destination, transformed, sample_rate, subtype="PCM_16")
            output.append(
                {
                    **row,
                    "audio_path": str(destination.resolve()),
                    "source_audio_path": row["audio_path"],
                    "source_id": source_id(row),
                    "duration": len(transformed) / sample_rate,
                    "sample_rate": TARGET_SAMPLE_RATE,
                    "split": split,
                    "scene": scene,
                    "scene_label": SCENE_TO_INDEX[scene],
                    **provenance,
                }
            )
    write_jsonl(output, output_dir / f"{split}.jsonl")
    return output


def build_scene_corpus(
    train_manifest: Path,
    validation_manifest: Path,
    test_manifest: Path,
    noise_manifest: Path,
    rir_manifest: Path,
    output_dir: Path,
    counts: tuple[int, int, int],
    seed: int,
) -> dict[str, int]:
    source_splits = [read_jsonl(path) for path in (train_manifest, validation_manifest, test_manifest)]
    assert_disjoint_sources(*source_splits)
    assets = Assets.from_manifests(noise_manifest, rir_manifest)
    result: dict[str, int] = {}
    built: list[list[dict]] = []
    for offset, (name, rows, per_scene) in enumerate(
        zip(("train", "validation", "test"), source_splits, counts, strict=True)
    ):
        split_rows = build_split(rows, output_dir, name, per_scene, assets, seed + offset)
        built.append(split_rows)
        result[name] = len(split_rows)
    assert_disjoint_sources(*built)
    report = {
        "version": "2.0",
        "seed": seed,
        "counts": result,
        "scenes": list(SCENE_NAMES),
        "noise_sources": sorted({row["source"] for row in assets.noise}),
        "rir_sources": sorted({row["source"] for row in assets.rir}),
        "noise_licenses": sorted({row.get("source_license", "unknown") for row in assets.noise}),
        "rir_licenses": sorted({row.get("source_license", "unknown") for row in assets.rir}),
        "source_disjoint": True,
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--noise-manifest", type=Path, required=True)
    parser.add_argument("--rir-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-scene", type=int, default=5000)
    parser.add_argument("--validation-per-scene", type=int, default=500)
    parser.add_argument("--test-per-scene", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    print(
        build_scene_corpus(
            args.train_manifest,
            args.validation_manifest,
            args.test_manifest,
            args.noise_manifest,
            args.rir_manifest,
            args.output_dir,
            (args.train_per_scene, args.validation_per_scene, args.test_per_scene),
            args.seed,
        )
    )


if __name__ == "__main__":
    main()
