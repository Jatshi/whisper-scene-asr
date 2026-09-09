"""CER-oracle labels with strict cache provenance for OPD routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.common import SCENE_NAMES, manifest_fingerprint, read_jsonl, stable_id
from src.metrics import error_counts
from src.opd_policy import DEFAULT_ACTION_NAMES, OPDRouterPolicy, soft_weights_from_logits
from src.routing import sparsify_weights


@dataclass(frozen=True)
class OracleEvidence:
    """Minimal auditable oracle schema shared by collectors and reports."""

    eval_id: str
    policy_hash: str
    action_names: tuple[str, ...]
    cer_by_action: dict[str, float]
    oracle_route: str
    teacher_probs: dict[str, float]
    base_cer: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_request(
    manifest_hash: str,
    policy_hash: str,
    actions: Sequence[str] = DEFAULT_ACTION_NAMES,
    fusion_top_k: int = 2,
    fusion_temperature: float = 1.0,
    fusion_quantization: float = 0.05,
) -> dict[str, Any]:
    """All values that can change dynamic soft-action labels belong in the key."""
    return {
        "schema_version": "3.0",
        "manifest_hash": manifest_hash,
        "policy_hash": policy_hash,
        "action_names": list(actions),
        "fusion_top_k": fusion_top_k,
        "fusion_temperature": fusion_temperature,
        "fusion_quantization": fusion_quantization,
    }


def validate_annotation_request(path: Path, request: Mapping[str, Any]) -> None:
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != dict(request):
            raise RuntimeError("Existing oracle cache has an incompatible request; use a new output directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(request), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def teacher_distribution(cer_by_action: Mapping[str, float], temperature: float = 0.2) -> dict[str, float]:
    if temperature <= 0:
        raise ValueError("Teacher temperature must be positive")
    if not cer_by_action:
        raise ValueError("No observed actions")
    logits = {key: -float(value) / temperature for key, value in cer_by_action.items()}
    maximum = max(logits.values())
    scores = {key: math.exp(value - maximum) for key, value in logits.items()}
    total = sum(scores.values())
    return {key: value / total for key, value in scores.items()}


def annotate_predictions(
    row: Mapping[str, Any],
    predictions: Mapping[str, str],
    *,
    policy_hash: str,
    teacher_temperature: float = 0.2,
    actions: Sequence[str] = DEFAULT_ACTION_NAMES,
) -> dict[str, Any]:
    reference = str(row.get("text", "")).strip()
    if not reference:
        raise ValueError("Cannot compute CER oracle from an empty reference")
    observed = [name for name in actions if name in predictions]
    if not observed:
        raise ValueError("No requested action has a prediction")
    counts = {name: error_counts(reference, str(predictions[name])) for name in observed}
    cer = {name: count.rate for name, count in counts.items()}
    # Python min is stable, so the versioned action order is the tie-break rule.
    oracle_route = min(observed, key=lambda name: cer[name])
    return {
        **dict(row),
        "eval_id": row.get("eval_id") or stable_id(row.get("source_id", row["audio_path"]), row.get("scene", "")),
        "action_names": list(actions),
        "policy_hash": policy_hash,
        "predictions": {name: str(predictions[name]) for name in observed},
        "cer_by_action": cer,
        "errors_by_action": {name: counts[name].errors for name in observed},
        "reference_characters": counts[observed[0]].reference_units,
        "oracle_route": oracle_route,
        "oracle_cer": cer[oracle_route],
        "base_cer": cer.get("base"),
        "teacher_probs": teacher_distribution(cer, teacher_temperature),
    }


def stratified_oracle_ids(rows: Sequence[Mapping[str, Any]], fraction: float, seed: int) -> set[str]:
    """Choose a deterministic per-scene oracle subset; 0 means no full oracle."""
    if not 0 <= fraction <= 1:
        raise ValueError("oracle fraction must be in [0, 1]")
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for row in rows:
        key = str(row.get("scene", "unknown"))
        groups.setdefault(key, []).append(
            stable_id(row.get("source_id", row["audio_path"]), row.get("scene", "unknown"))
        )
    chosen: set[str] = set()
    for identifiers in groups.values():
        count = min(len(identifiers), math.ceil(len(identifiers) * fraction)) if fraction else 0
        chosen.update(rng.sample(sorted(identifiers), count))
    return chosen


def _weight_dict(
    logits: torch.Tensor, names: Sequence[str], top_k: int, temperature: float, step: float
) -> dict[str, float]:
    dense = soft_weights_from_logits(logits[None], names, top_k=top_k, temperature=temperature)[0].tolist()
    raw = dict(zip(SCENE_NAMES, dense, strict=True))
    return sparsify_weights(raw, SCENE_NAMES, top_k=top_k, quantization_step=step)


def _decode_actions(
    engine: Any,
    rows: list[dict[str, Any]],
    requested: list[set[str]],
    soft_weights: list[dict[str, float]],
    include_joint: bool,
) -> list[dict[str, str]]:
    """Decode requested actions, batching identical experts/fusion signatures."""
    paths = [row["audio_path"] for row in rows]
    output = [{} for _ in rows]
    base_indices = [i for i, actions in enumerate(requested) if "base" in actions]
    for index, value in zip(base_indices, engine.transcribe_base_batch([paths[i] for i in base_indices]), strict=True):
        output[index]["base"] = value
    for expert in SCENE_NAMES:
        indices = [i for i, actions in enumerate(requested) if expert in actions]
        values = engine.transcribe_adapter_batch([paths[i] for i in indices], expert)
        for index, value in zip(indices, values, strict=True):
            output[index][expert] = value
    if include_joint:
        indices = [i for i, actions in enumerate(requested) if "joint" in actions]
        values = engine.transcribe_adapter_batch([paths[i] for i in indices], "joint")
        for index, value in zip(indices, values, strict=True):
            output[index]["joint"] = value
    groups: dict[str, list[int]] = {}
    for index, actions in enumerate(requested):
        if "soft" in actions:
            signature = json.dumps(soft_weights[index], sort_keys=True)
            groups.setdefault(signature, []).append(index)
    for signature, indices in groups.items():
        weights = json.loads(signature)
        values = engine.transcribe_soft_weights_batch([paths[i] for i in indices], weights)
        for index, value in zip(indices, values, strict=True):
            output[index]["soft"] = value
    return output


def collect_annotations(
    *,
    manifest: Path,
    policy_checkpoint: Path,
    classifier_checkpoint: Path,
    adapter_dir: Path,
    output: Path,
    model_name: str,
    batch_size: int,
    seed: int,
    oracle_fraction: float,
    teacher_temperature: float,
    fusion_top_k: int,
    fusion_temperature: float,
    fusion_weight_step: float,
    joint_adapter: Path | None = None,
    collection_size: int = 5000,
    summary_cache: Path | None = None,
    confidence_threshold: float = 0.65,
    entropy_threshold: float = 1.35,
) -> None:
    """Collect one on-policy bandit round with resumable, provenance-bound labels."""
    from src.evaluate_asr import _adapter_path
    from src.fused_inference import FusedWhisperInference, SceneRouter

    if batch_size < 1 or collection_size < 0:
        raise ValueError("batch_size must be positive and collection_size must be non-negative")
    if not 0 <= confidence_threshold <= 1 or entropy_threshold < 0:
        raise ValueError("Invalid fallback thresholds")
    pool_rows = read_jsonl(manifest)
    if not pool_rows:
        raise ValueError("Manifest is empty")
    checkpoint = torch.load(policy_checkpoint, map_location="cpu", weights_only=False)
    expected_actions = tuple(checkpoint["action_names"])
    policy = OPDRouterPolicy.from_checkpoint_payload(checkpoint, expected_actions).eval()
    policy_hash = file_sha256(policy_checkpoint)
    request = annotation_request(
        manifest_fingerprint(pool_rows),
        policy_hash,
        expected_actions,
        fusion_top_k,
        fusion_temperature,
        fusion_weight_step,
    ) | {
        "seed": seed,
        "collection_size": collection_size,
        "oracle_fraction": oracle_fraction,
        "teacher_temperature": teacher_temperature,
        "confidence_threshold": confidence_threshold,
        "entropy_threshold": entropy_threshold,
    }
    request_path = output.with_suffix(".request.json")
    validate_annotation_request(request_path, request)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_jsonl(output) if output.exists() else []
    completed_ids = {row["eval_id"] for row in completed}
    engine = FusedWhisperInference(model_name, fusion_top_k=fusion_top_k, fusion_weight_step=fusion_weight_step)
    for scene in SCENE_NAMES:
        engine.load_adapter(scene, _adapter_path(adapter_dir, scene))
    if "joint" in expected_actions:
        if joint_adapter is None:
            raise ValueError("--joint-adapter is required for the optional joint action")
        engine.load_adapter("joint", _adapter_path(joint_adapter, "joint"))
    router = SceneRouter(classifier_checkpoint, model_name, engine.device)
    generator = torch.Generator().manual_seed(seed)
    pool_ids = [stable_id(row.get("source_id", row["audio_path"]), row.get("scene", "unknown")) for row in pool_rows]
    cache_path = summary_cache or output.parent.parent.parent / "summary_cache.pt"
    pool_fingerprint = manifest_fingerprint(pool_rows)
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            cache.get("manifest_fingerprint") != pool_fingerprint
            or cache.get("eval_ids") != pool_ids
            or cache.get("model_name") != model_name
        ):
            raise RuntimeError("Summary cache is incompatible with the current OPD pool")
        pool_summaries = cache["summaries"]
    else:
        summary_parts = []
        for start in range(0, len(pool_rows), batch_size):
            paths = [row["audio_path"] for row in pool_rows[start : start + batch_size]]
            summary_parts.append(router.summarize_batch(paths))
            print(f"summarized={min(start + batch_size, len(pool_rows))}/{len(pool_rows)}")
        pool_summaries = torch.cat(summary_parts)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "manifest_fingerprint": pool_fingerprint,
                "eval_ids": pool_ids,
                "model_name": model_name,
                "summaries": pool_summaries,
            },
            cache_path,
        )
    with torch.inference_mode():
        pool_logits = policy(pool_summaries)
        pool_probabilities = torch.softmax(pool_logits / policy.strategy_temperature, -1)
    if collection_size > 0 and collection_size < len(pool_rows):
        from src.opd_trainer import difficulty_sampling_weights

        priority = difficulty_sampling_weights(pool_probabilities)
        selected_indices = torch.multinomial(priority, collection_size, replacement=False, generator=generator).tolist()
    else:
        selected_indices = list(range(len(pool_rows)))
    selected_probabilities = pool_probabilities[selected_indices]
    selected_actions = torch.multinomial(selected_probabilities, 1, generator=generator).squeeze(1)
    selected = [
        (
            pool_rows[index],
            pool_summaries[index],
            pool_logits[index],
            pool_probabilities[index],
            selected_actions[position],
        )
        for position, index in enumerate(selected_indices)
    ]
    oracle_ids = stratified_oracle_ids([item[0] for item in selected], oracle_fraction, seed)
    pending = [
        item
        for item in selected
        if stable_id(item[0].get("source_id", item[0]["audio_path"]), item[0].get("scene", "unknown"))
        not in completed_ids
    ]
    for start in range(0, len(pending), batch_size):
        items = pending[start : start + batch_size]
        batch = [item[0] for item in items]
        summaries = torch.stack([item[1] for item in items])
        logits = torch.stack([item[2] for item in items])
        probabilities = torch.stack([item[3] for item in items])
        sampled = torch.stack([item[4] for item in items])
        weights = [
            _weight_dict(logits[i], expected_actions, fusion_top_k, fusion_temperature, fusion_weight_step)
            for i in range(len(batch))
        ]
        requests = []
        for row, action_index in zip(batch, sampled.tolist(), strict=True):
            sampled_action = expected_actions[action_index]
            actions = {"base", sampled_action}
            identifier = stable_id(row.get("source_id", row["audio_path"]), row.get("scene", "unknown"))
            if identifier in oracle_ids:
                actions.update(expected_actions)
            requests.append(actions)
        predictions = _decode_actions(engine, batch, requests, weights, "joint" in expected_actions)
        records = []
        for i, row in enumerate(batch):
            action = expected_actions[int(sampled[i])]
            confidence = float(probabilities[i].max())
            entropy = float(
                -(probabilities[i] * probabilities[i].clamp_min(torch.finfo(probabilities.dtype).eps).log()).sum()
            )
            record = annotate_predictions(
                row,
                predictions[i],
                policy_hash=policy_hash,
                teacher_temperature=teacher_temperature,
                actions=expected_actions,
            )
            record.update(
                {
                    "summary": summaries[i].tolist(),
                    "features_summary_hash": hashlib.sha256(summaries[i].contiguous().numpy().tobytes()).hexdigest(),
                    "sampled_action": action,
                    "action_cer": record["cer_by_action"][action],
                    "fallback": confidence < confidence_threshold or entropy > entropy_threshold,
                    "policy_probs": dict(zip(expected_actions, probabilities[i].tolist(), strict=True)),
                    "priority_weight": 1.0 / (1.0 + float(probabilities[i, int(sampled[i])])),
                    "soft_weights": weights[i],
                    "full_oracle": record["eval_id"] in oracle_ids,
                }
            )
            records.append(record)
        with output.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"collected={len(completed) + start + len(records)}/{len(selected)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect provenance-safe OPD bandit rewards and CER-oracle labels")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--joint-adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oracle-fraction", type=float, default=0.25)
    parser.add_argument("--teacher-temperature", type=float, default=0.2)
    parser.add_argument("--fusion-top-k", type=int, default=2)
    parser.add_argument("--fusion-temperature", type=float, default=1.0)
    parser.add_argument("--fusion-weight-step", type=float, default=0.05)
    parser.add_argument("--collection-size", type=int, default=5000)
    parser.add_argument("--summary-cache", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--entropy-threshold", type=float, default=1.35)
    args = parser.parse_args()
    collect_annotations(
        manifest=args.manifest,
        policy_checkpoint=args.policy,
        classifier_checkpoint=args.classifier,
        adapter_dir=args.adapter_dir,
        output=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        seed=args.seed,
        oracle_fraction=args.oracle_fraction,
        teacher_temperature=args.teacher_temperature,
        fusion_top_k=args.fusion_top_k,
        fusion_temperature=args.fusion_temperature,
        fusion_weight_step=args.fusion_weight_step,
        joint_adapter=args.joint_adapter,
        collection_size=args.collection_size,
        summary_cache=args.summary_cache,
        confidence_threshold=args.confidence_threshold,
        entropy_threshold=args.entropy_threshold,
    )


if __name__ == "__main__":
    main()
