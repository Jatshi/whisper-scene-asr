import numpy as np
import soundfile as sf

from src.asset_manifest import build_asset_manifests_from_specs
from src.common import read_jsonl


def test_real_rir_list_is_resolved_and_provenance_preserved(tmp_path) -> None:
    raw = tmp_path / "raw"
    noise_root = raw / "musan" / "noise"
    rir_root = raw / "RIRS_NOISES" / "real_rirs_isotropic_noises"
    noise_root.mkdir(parents=True)
    rir_root.mkdir(parents=True)
    noise = noise_root / "noise.wav"
    rir = rir_root / "room_rir.wav"
    sf.write(noise, np.ones(1600, dtype=np.float32), 16000)
    sf.write(rir, np.ones(160, dtype=np.float32), 16000)
    rir_list = rir_root / "rir_list"
    rir_list.write_text("0 RIRS_NOISES/real_rirs_isotropic_noises/room_rir.wav\n", encoding="utf-8")
    output = tmp_path / "assets"
    counts = build_asset_manifests_from_specs(
        [("MUSAN", "CC BY 4.0", noise_root)],
        [("OpenSLR real RIR", "Apache-2.0", rir_list)],
        output,
    )
    assert counts == (1, 1)
    row = read_jsonl(output / "rir.jsonl")[0]
    assert row["asset_path"] == str(rir.resolve())
    assert row["source_license"] == "Apache-2.0"
