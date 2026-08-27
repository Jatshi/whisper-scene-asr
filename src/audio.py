"""Audio loading utilities with a single 16 kHz mono invariant for Whisper."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SAMPLE_RATE = 16_000


def load_audio(path: str | Path, target_sample_rate: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Return finite mono float32 samples at the requested sampling rate."""
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if not np.isfinite(samples).all():
        raise ValueError(f"Audio contains non-finite samples: {path}")
    if sample_rate != target_sample_rate:
        divisor = gcd(sample_rate, target_sample_rate)
        samples = resample_poly(samples, target_sample_rate // divisor, sample_rate // divisor).astype(np.float32)
    return samples, target_sample_rate
