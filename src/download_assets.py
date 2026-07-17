"""Download AISHELL-1 and cache Whisper assets; both operations are resumable."""
from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

AISHELL_URL = "https://www.openslr.org/resources/33/data_aishell.tgz"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        with urllib.request.urlopen(request) as response, temporary.open("ab" if offset and response.status == 206 else "wb") as out:
            shutil.copyfileobj(response, out)
    except Exception as exc:
        raise RuntimeError(f"Download failed; re-run to resume: {url}") from exc
    temporary.replace(destination)


def _safe_extract(bundle: tarfile.TarFile, destination: Path) -> None:
    resolved = destination.resolve()
    members = bundle.getmembers()
    if any(not (resolved / member.name).resolve().is_relative_to(resolved) for member in members):
        raise RuntimeError("Refusing archive containing paths outside the target directory")
    bundle.extractall(destination, members=members)


def _extract_nested_speaker_archives(dataset_root: Path) -> None:
    """AISHELL-1 stores WAVs in one `.tar.gz` per speaker in its outer archive."""
    wav_root = dataset_root / "wav"
    archives = sorted(wav_root.glob("S*.tar.gz"))
    for archive in archives:
        speaker_dir = wav_root / archive.name.removesuffix(".tar.gz")
        if not speaker_dir.exists():
            with tarfile.open(archive, "r:gz") as bundle:
                _safe_extract(bundle, wav_root)
        archive.unlink()


def fetch_aishell(data_dir: Path, source_archive: Path | None = None) -> Path:
    archive = source_archive or data_dir / "data_aishell.tgz"
    target = data_dir / "data_aishell"
    if not target.exists():
        if not source_archive and not archive.exists():
            download(AISHELL_URL, archive)
        with tarfile.open(archive, "r:gz") as bundle:
            _safe_extract(bundle, data_dir)
    _extract_nested_speaker_archives(target)
    return target


def fetch_model(model_name: str, cache_dir: Path) -> None:
    from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperProcessor
    cache_dir.mkdir(parents=True, exist_ok=True)
    WhisperProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    WhisperFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
    WhisperForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--aishell-archive", type=Path, help="Existing AISHELL outer archive; avoids downloading a second copy.")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    if not args.skip_data:
        print(f"AISHELL-1 ready: {fetch_aishell(args.data_dir, args.aishell_archive)}")
    if not args.skip_model:
        fetch_model(args.model, args.cache_dir)
        print(f"Model cached: {args.model}")


if __name__ == "__main__":
    main()
