import numpy as np
import soundfile as sf

from src.common import read_jsonl, write_jsonl
from src.data_augmentation import Assets, build_scene_corpus, degrade, mix_at_snr


def test_real_noise_mixing_hits_requested_snr() -> None:
    clean = np.ones(16000, dtype=np.float32) * 0.1
    noise = np.linspace(-1, 1, 16000, dtype=np.float32)
    mixed = mix_at_snr(clean, noise, 10.0)
    added = mixed - clean
    measured = 20 * np.log10(np.sqrt(np.mean(clean**2)) / np.sqrt(np.mean(added**2)))
    np.testing.assert_allclose(measured, 10.0, atol=1e-4)


def test_degrade_records_real_asset_provenance(tmp_path) -> None:
    noise_path, rir_path = tmp_path / "noise.wav", tmp_path / "rir.wav"
    sf.write(noise_path, np.random.default_rng(0).normal(size=32000).astype(np.float32), 16000)
    rir = np.zeros(800, dtype=np.float32)
    rir[0] = 1.0
    rir[200] = 0.3
    sf.write(rir_path, rir, 16000)
    assets = Assets(
        noise=[{"asset_path": str(noise_path), "asset_id": "n1", "source": "MUSAN"}],
        rir=[{"asset_path": str(rir_path), "asset_id": "r1", "source": "OpenSLR"}],
    )
    audio, metadata = degrade(
        np.ones(16000, dtype=np.float32) * 0.05, "noisy_reverb", assets, np.random.default_rng(42)
    )
    assert len(audio) == 16000
    assert metadata["noise_source"] == "MUSAN"
    assert metadata["rir_source"] == "OpenSLR"
    assert 0 <= metadata["snr_db"] <= 20


def test_scene_corpus_builds_disjoint_five_bucket_splits(tmp_path) -> None:
    source_manifests = []
    for index, split in enumerate(("train", "validation", "test")):
        audio_path = tmp_path / f"{split}.wav"
        sf.write(audio_path, np.sin(np.linspace(0, 100, 8000)).astype(np.float32) * 0.05, 16000)
        manifest = tmp_path / f"source-{split}.jsonl"
        write_jsonl(
            [
                {
                    "audio_path": str(audio_path),
                    "text": "测试",
                    "source_id": f"u{index}",
                    "utterance_id": f"u{index}",
                    "speaker": f"s{index}",
                }
            ],
            manifest,
        )
        source_manifests.append(manifest)
    noise_path, rir_path = tmp_path / "noise-asset.wav", tmp_path / "rir-asset.wav"
    sf.write(noise_path, np.random.default_rng(1).normal(size=16000).astype(np.float32), 16000)
    rir = np.zeros(400, dtype=np.float32)
    rir[0] = 1
    rir[100] = 0.2
    sf.write(rir_path, rir, 16000)
    noise_manifest, rir_manifest = tmp_path / "noise.jsonl", tmp_path / "rir.jsonl"
    write_jsonl([{"asset_path": str(noise_path), "asset_id": "n", "source": "noise"}], noise_manifest)
    write_jsonl([{"asset_path": str(rir_path), "asset_id": "r", "source": "rir"}], rir_manifest)
    output = tmp_path / "scenes"
    counts = build_scene_corpus(*source_manifests, noise_manifest, rir_manifest, output, (1, 1, 1), 42)
    assert counts == {"train": 5, "validation": 5, "test": 5}
    assert {row["scene"] for row in read_jsonl(output / "test.jsonl")} == {
        "clean",
        "noisy",
        "reverb",
        "fast_slow",
        "noisy_reverb",
    }
