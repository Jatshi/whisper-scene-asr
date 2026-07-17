"""Deterministic, streaming waveform augmentation with explicit scene labels."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from tqdm import tqdm

from src.common import SCENE_TO_INDEX, read_jsonl, set_seed, write_jsonl
from src.audio import load_audio

CONFIGS = {
    "clean": {}, "noisy": {"snr": 10}, "reverb": {"rt60": 0.45},
    "fast_slow": {"speed": 1.15}, "noisy_reverb": {"snr": 10, "rt60": 0.35},
}


def augment(samples: np.ndarray, sr: int, config: dict, rng: np.random.Generator) -> np.ndarray:
    audio = samples.astype(np.float32, copy=True)
    if "snr" in config:
        power = max(float(np.mean(audio ** 2)), 1e-9)
        noise = rng.standard_normal(audio.shape).astype(np.float32)
        noise *= np.sqrt(power / (np.mean(noise ** 2) * 10 ** (config["snr"] / 10)))
        audio += noise
    if "rt60" in config:
        length = max(2, int(sr * config["rt60"]))
        impulse = np.exp(-6.91 * np.arange(length) / (sr * config["rt60"])).astype(np.float32)
        impulse[0] = 1.0
        audio = fftconvolve(audio, impulse, mode="full")[: len(audio)]
    if "speed" in config:
        points = np.arange(0, len(audio), config["speed"])
        audio = np.interp(points, np.arange(len(audio)), audio).astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    return audio / peak * min(peak, 0.98) if peak else audio


def build_augmented(input_manifest: Path, output_dir: Path, per_scene: int, seed: int) -> int:
    rows = read_jsonl(input_manifest)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(rows), size=min(per_scene, len(rows)), replace=False)
    output = []
    for scene, config in CONFIGS.items():
        folder = output_dir / scene
        folder.mkdir(parents=True, exist_ok=True)
        for index, row_index in enumerate(tqdm(chosen, desc=scene)):
            row = rows[int(row_index)]
            audio, sr = load_audio(row["audio_path"])
            transformed = augment(audio, sr, config, rng)
            destination = folder / f"{scene}_{index:05d}.wav"
            import soundfile as sf
            sf.write(destination, transformed, sr)
            output.append({**row, "audio_path": str(destination.resolve()), "duration": len(transformed) / sr,
                           "scene": scene, "scene_label": SCENE_TO_INDEX[scene]})
    write_jsonl(output, output_dir / "augmented_meta.jsonl")
    return len(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/processed/aishell1/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/augmented"))
    parser.add_argument("--per-scene", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); set_seed(args.seed)
    print(f"Created {build_augmented(args.input, args.output_dir, args.per_scene, args.seed)} samples")


if __name__ == "__main__":
    main()
