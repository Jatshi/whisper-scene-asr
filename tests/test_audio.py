import numpy as np
import soundfile as sf

from src.audio import load_audio


def test_load_audio_mixes_channels_and_resamples(tmp_path) -> None:
    original = np.column_stack([np.ones(8000), -np.ones(8000)]).astype("float32")
    path = tmp_path / "stereo-8k.wav"
    sf.write(path, original, 8000)
    audio, sample_rate = load_audio(path)
    assert sample_rate == 16000
    assert audio.ndim == 1
    assert len(audio) == 16000
    assert abs(float(audio.mean())) < 1e-3
