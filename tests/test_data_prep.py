import json
import wave

import numpy as np

from src.data_prep import build_manifests, normalize_text, write_quality_report


def test_normalize_chinese_text_removes_punctuation_and_space() -> None:
    assert normalize_text("  你 好，世界！ A-1 ") == "你好世界A1"


def _write_wave(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(np.zeros(16000, dtype=np.int16).tobytes())


def test_build_manifests_and_quality_report(tmp_path) -> None:
    root = tmp_path / "data_aishell"
    transcript = root / "transcript" / "aishell_transcript_v0.8.txt"
    transcript.parent.mkdir(parents=True)
    transcript.write_text("UTT1 你 好！\nUTT2 测 试\nUTT3 开 发\n", encoding="utf-8")
    for split, utt in (("train", "UTT1"), ("dev", "UTT2"), ("test", "UTT3")):
        _write_wave(root / "wav" / split / "S0001" / f"{utt}.wav")
    output = tmp_path / "processed"
    assert build_manifests(root, output) == {"train": 1, "validation": 1, "test": 1}
    write_quality_report(output)
    report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    assert report["train"]["utterances"] == 1
    assert report["test"]["unique_speakers"] == 1
    assert (output / "data_card.md").exists()
