"""Download AISHELL-1 and cache Whisper assets; both operations are resumable."""

from __future__ import annotations

import argparse
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

AISHELL_URL = "https://www.openslr.org/resources/33/data_aishell.tgz"
MUSAN_URLS = (
    "https://openslr.magicdatatech.com/resources/17/musan.tar.gz",
    "https://www.openslr.org/resources/17/musan.tar.gz",
    "https://openslr.elda.org/resources/17/musan.tar.gz",
)
RIRS_NOISES_URLS = (
    "https://openslr.magicdatatech.com/resources/28/rirs_noises.zip",
    "https://www.openslr.org/resources/28/rirs_noises.zip",
    "https://openslr.elda.org/resources/28/rirs_noises.zip",
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    offset = temporary.stat().st_size if temporary.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        with (
            urllib.request.urlopen(request) as response,
            temporary.open("ab" if offset and response.status == 206 else "wb") as out,
        ):
            started = time.monotonic()
            transferred = 0
            next_report = 256 * 1024**2
            while block := response.read(8 * 1024**2):
                out.write(block)
                transferred += len(block)
                if transferred >= next_report:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    speed = transferred / elapsed / 1024**2
                    print(f"downloaded={transferred / 1024**3:.2f}GB speed={speed:.1f}MB/s url={url}")
                    next_report += 256 * 1024**2
    except Exception as exc:
        raise RuntimeError(f"Download failed; re-run to resume: {url}") from exc
    temporary.replace(destination)


def download_from_mirrors(urls: tuple[str, ...], destination: Path) -> None:
    failures = []
    for url in urls:
        try:
            download(url, destination)
            return
        except RuntimeError as exc:
            failures.append(f"{url}: {exc}")
    raise RuntimeError("All download mirrors failed:\n" + "\n".join(failures))


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


def _safe_extract_zip(bundle: zipfile.ZipFile, destination: Path) -> None:
    resolved = destination.resolve()
    if any(not (resolved / name).resolve().is_relative_to(resolved) for name in bundle.namelist()):
        raise RuntimeError("Refusing zip containing paths outside the target directory")
    bundle.extractall(destination)


def fetch_real_degradation_assets(data_dir: Path) -> tuple[Path, Path]:
    """Fetch reusable MUSAN noise and OpenSLR RIR assets with resume support."""
    musan_root = data_dir / "musan"
    musan_archive = data_dir / "musan.tar.gz"
    musan_marker = data_dir / ".musan.complete"
    if not musan_marker.exists():
        if not musan_archive.exists():
            download_from_mirrors(MUSAN_URLS, musan_archive)
        with tarfile.open(musan_archive, "r:gz") as bundle:
            _safe_extract(bundle, data_dir)
        if not (musan_root / "noise").exists():
            raise RuntimeError("MUSAN extraction finished without the expected noise directory")
        musan_marker.write_text("complete\n", encoding="utf-8")
    rir_root = data_dir / "RIRS_NOISES"
    rir_archive = data_dir / "rirs_noises.zip"
    rir_marker = data_dir / ".rirs_noises.complete"
    if not rir_marker.exists():
        if not rir_archive.exists():
            download_from_mirrors(RIRS_NOISES_URLS, rir_archive)
        with zipfile.ZipFile(rir_archive) as bundle:
            _safe_extract_zip(bundle, data_dir)
        expected = rir_root / "real_rirs_isotropic_noises" / "rir_list"
        if not expected.exists():
            raise RuntimeError(f"RIR extraction finished without expected list: {expected}")
        rir_marker.write_text("complete\n", encoding="utf-8")
    return musan_root, rir_root


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
    parser.add_argument(
        "--aishell-archive", type=Path, help="Existing AISHELL outer archive; avoids downloading a second copy."
    )
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--cache-dir", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-real-assets", action="store_true")
    args = parser.parse_args()
    if not args.skip_data:
        print(f"AISHELL-1 ready: {fetch_aishell(args.data_dir, args.aishell_archive)}")
    if not args.skip_model:
        fetch_model(args.model, args.cache_dir)
        print(f"Model cached: {args.model}")
    if not args.skip_real_assets:
        musan, rir = fetch_real_degradation_assets(args.data_dir)
        print(f"Real degradation assets ready: {musan}; {rir}")


if __name__ == "__main__":
    main()
