"""Index real noise and room impulse response assets with provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from src.common import stable_id, write_jsonl

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg"}


def _audio_rows(roots: list[Path], kind: str, source: str, license_name: str) -> list[dict]:
    rows: list[dict] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"Missing {kind} root: {root}")
        if root.is_file():
            listed_paths = []
            for line in root.read_text(encoding="utf-8").splitlines():
                token = line.strip().split()[-1] if line.strip() else ""
                if not token:
                    continue
                candidate = Path(token)
                candidates = (
                    [candidate]
                    if candidate.is_absolute()
                    else [
                        root.parent / candidate,
                        root.parent.parent / candidate,
                        root.parent.parent.parent / candidate,
                    ]
                )
                match = next((path for path in candidates if path.exists()), None)
                if match is None:
                    raise FileNotFoundError(f"Asset list entry cannot be resolved: {token} (from {root})")
                listed_paths.append(match)
            paths = sorted(listed_paths)
        else:
            paths = sorted(item for item in root.rglob("*") if item.suffix.lower() in AUDIO_SUFFIXES)
        for path in paths:
            try:
                info = sf.info(path)
            except RuntimeError:
                continue
            if info.frames <= 0 or info.samplerate <= 0:
                continue
            rows.append(
                {
                    "asset_id": stable_id(source, kind, path),
                    "asset_path": str(path.resolve()),
                    "kind": kind,
                    "source": source,
                    "source_license": license_name,
                    "duration": info.frames / info.samplerate,
                    "sample_rate": info.samplerate,
                }
            )
    if not rows:
        raise ValueError(f"No readable audio found under {roots}")
    return rows


def build_asset_manifests(
    noise_roots: list[Path],
    rir_roots: list[Path],
    output_dir: Path,
    *,
    noise_source: str,
    noise_license: str,
    rir_source: str,
    rir_license: str,
) -> tuple[int, int]:
    noise = _audio_rows(noise_roots, "noise", noise_source, noise_license)
    rir = _audio_rows(rir_roots, "rir", rir_source, rir_license)
    write_jsonl(noise, output_dir / "noise.jsonl")
    write_jsonl(rir, output_dir / "rir.jsonl")
    return len(noise), len(rir)


def parse_spec(value: str) -> tuple[str, str, Path]:
    """Parse SOURCE::LICENSE::PATH while allowing spaces in all fields."""
    parts = value.split("::", maxsplit=2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("asset spec must be SOURCE::LICENSE::PATH")
    return parts[0], parts[1], Path(parts[2])


def build_asset_manifests_from_specs(
    noise_specs: list[tuple[str, str, Path]],
    rir_specs: list[tuple[str, str, Path]],
    output_dir: Path,
) -> tuple[int, int]:
    noise = [
        row for source, license_name, root in noise_specs for row in _audio_rows([root], "noise", source, license_name)
    ]
    rir = [row for source, license_name, root in rir_specs for row in _audio_rows([root], "rir", source, license_name)]
    write_jsonl(noise, output_dir / "noise.jsonl")
    write_jsonl(rir, output_dir / "rir.jsonl")
    return len(noise), len(rir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-root", type=Path, action="append")
    parser.add_argument("--rir-root", type=Path, action="append")
    parser.add_argument(
        "--noise-spec",
        type=parse_spec,
        action="append",
        help="Repeatable SOURCE::LICENSE::PATH; use for MUSAN + DNS + WHAM mixtures",
    )
    parser.add_argument("--rir-spec", type=parse_spec, action="append", help="Repeatable SOURCE::LICENSE::PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--noise-source", default="MUSAN")
    parser.add_argument("--noise-license", default="CC BY 4.0")
    parser.add_argument("--rir-source", default="OpenSLR RIRS_NOISES")
    parser.add_argument("--rir-license", default="Apache-2.0")
    args = parser.parse_args()
    if args.noise_spec or args.rir_spec:
        if not args.noise_spec or not args.rir_spec:
            parser.error("--noise-spec and --rir-spec must both be supplied")
        noise, rir = build_asset_manifests_from_specs(args.noise_spec, args.rir_spec, args.output_dir)
    else:
        if not args.noise_root or not args.rir_root:
            parser.error("provide noise/rir roots or repeatable noise/rir specs")
        noise, rir = build_asset_manifests(
            args.noise_root,
            args.rir_root,
            args.output_dir,
            noise_source=args.noise_source,
            noise_license=args.noise_license,
            rir_source=args.rir_source,
            rir_license=args.rir_license,
        )
    print(f"Indexed {noise} noise files and {rir} RIR files")


if __name__ == "__main__":
    main()
