import tarfile

from src.download_assets import fetch_aishell


def test_fetch_aishell_extracts_nested_speaker_archives(tmp_path) -> None:
    source = tmp_path / "source"; dataset = source / "data_aishell"; (dataset / "transcript").mkdir(parents=True)
    (dataset / "transcript" / "aishell_transcript_v0.8.txt").write_text("UTT text\n", encoding="utf-8")
    inner_content = tmp_path / "inner" / "S0001"; inner_content.mkdir(parents=True); (inner_content / "UTT.wav").write_bytes(b"wav")
    (dataset / "wav").mkdir()
    with tarfile.open(dataset / "wav" / "S0001.tar.gz", "w:gz") as inner:
        inner.add(inner_content, arcname="S0001")
    outer = tmp_path / "data_aishell.gz"
    with tarfile.open(outer, "w:gz") as bundle:
        bundle.add(dataset, arcname="data_aishell")
    root = fetch_aishell(tmp_path / "destination", outer)
    assert (root / "wav" / "S0001" / "UTT.wav").exists()
    assert not (root / "wav" / "S0001.tar.gz").exists()
