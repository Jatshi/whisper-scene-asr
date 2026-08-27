"""Resumable bucketed evaluation of isolated base, hard, soft and joint ASR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common import SCENE_NAMES, read_jsonl, stable_id
from src.fused_inference import FusedWhisperInference, SceneRouter
from src.metrics import corpus_counts, error_counts, paired_bootstrap_difference
from src.routing import decide_route


def _adapter_path(root: Path, name: str) -> Path:
    candidates = [root / name, root / name / "final", root]
    for candidate in candidates:
        if (candidate / "adapter_config.json").exists():
            return candidate
    matches = [path.parent for path in root.rglob("adapter_config.json") if path.parent.name == name]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not uniquely locate adapter '{name}' below {root}")


def _prediction_fields(reference: str, name: str, prediction: str) -> dict[str, str | int | float]:
    counts = error_counts(reference, prediction)
    return {
        f"{name}_prediction": prediction,
        f"{name}_errors": counts.errors,
        f"{name}_reference_chars": counts.reference_units,
        f"{name}_cer": counts.rate,
    }


def _evaluate_chunk(
    rows: list[dict],
    engine: FusedWhisperInference,
    router: SceneRouter | None,
    systems: set[str],
    confidence_threshold: float,
    entropy_threshold: float,
) -> list[dict]:
    paths = [row["audio_path"] for row in rows]
    base_predictions = engine.transcribe_base_batch(paths)
    probabilities = router.predict_batch(paths) if router else [{} for _ in paths]
    decisions = [decide_route(item, confidence_threshold, entropy_threshold) for item in probabilities]
    predictions: dict[str, list[str]] = {"base": base_predictions}
    if "hard" in systems:
        hard = list(base_predictions)
        for scene in SCENE_NAMES:
            indices = [index for index, decision in enumerate(decisions) if decision.route == scene]
            values = engine.transcribe_adapter_batch([paths[index] for index in indices], scene)
            for index, value in zip(indices, values, strict=True):
                hard[index] = value
        predictions["hard"] = hard
    if "soft" in systems:
        predictions["soft"] = [
            base_predictions[index]
            if decision.route == "base"
            else engine.transcribe_soft(paths[index], probabilities[index])
            for index, decision in enumerate(decisions)
        ]
    if "joint" in systems:
        predictions["joint"] = engine.transcribe_adapter_batch(paths, "joint")
    output = []
    for index, row in enumerate(rows):
        decision = decisions[index]
        item = {
            "eval_id": stable_id(row.get("source_id", row["audio_path"]), row.get("scene", "unknown")),
            "source_id": row.get("source_id", ""),
            "reference": row["text"],
            "true_scene": row.get("scene", "unknown"),
            "router_route": decision.route,
            "router_confidence": decision.confidence,
            "router_entropy": decision.entropy,
            "router_reason": decision.reason,
            "scene_probabilities": json.dumps(probabilities[index], ensure_ascii=False, sort_keys=True),
            "soft_applied_weights": json.dumps(
                engine.soft_weights(probabilities[index]), ensure_ascii=False, sort_keys=True
            ),
        }
        for system, values in predictions.items():
            item.update(_prediction_fields(row["text"], system, values[index]))
        output.append(item)
    return output


def summarize(results: list[dict], systems: list[str], bootstrap_samples: int, seed: int) -> dict:
    summary: dict = {
        "utterances": len(results),
        "systems": systems,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "buckets": {},
    }
    for bucket in ("overall", *SCENE_NAMES):
        selected = results if bucket == "overall" else [row for row in results if row["true_scene"] == bucket]
        if not selected:
            continue
        references = [row["reference"] for row in selected]
        reason_counts = {
            reason: sum(row["router_reason"] == reason for row in selected)
            for reason in sorted({row["router_reason"] for row in selected})
        }
        route_counts = {
            route: sum(row["router_route"] == route for row in selected)
            for route in sorted({row["router_route"] for row in selected})
        }
        bucket_result: dict = {
            "utterances": len(selected),
            "routing": {
                "reason_counts": reason_counts,
                "route_counts": route_counts,
                "fallback_rate": sum(row["router_route"] == "base" for row in selected) / len(selected),
                "mean_confidence": sum(float(row["router_confidence"]) for row in selected) / len(selected),
                "mean_entropy": sum(float(row["router_entropy"]) for row in selected) / len(selected),
            },
            "metrics": {},
            "paired_vs_base": {},
        }
        for system in systems:
            hypotheses = [row[f"{system}_prediction"] for row in selected]
            bucket_result["metrics"][system] = corpus_counts(references, hypotheses).to_dict()
            if system != "base":
                bucket_result["paired_vs_base"][system] = paired_bootstrap_difference(
                    references,
                    [row["base_prediction"] for row in selected],
                    hypotheses,
                    samples=bootstrap_samples,
                    seed=seed,
                )
        summary["buckets"][bucket] = bucket_result
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--joint-adapter", type=Path)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument(
        "--systems", nargs="+", choices=("base", "hard", "soft", "joint"), default=["base", "hard", "soft", "joint"]
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--entropy-threshold", type=float, default=1.35)
    parser.add_argument("--fusion-top-k", type=int, default=2)
    parser.add_argument("--fusion-weight-step", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("output/v2/evaluation.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    systems = list(dict.fromkeys(["base", *args.systems]))
    if any(system in systems for system in ("hard", "soft")) and not args.classifier.exists():
        raise FileNotFoundError(f"Routing systems require classifier: {args.classifier}")
    engine = FusedWhisperInference(
        args.model, fusion_top_k=args.fusion_top_k, fusion_weight_step=args.fusion_weight_step
    )
    for scene in SCENE_NAMES:
        engine.load_adapter(scene, _adapter_path(args.adapter_dir, scene))
    if "joint" in systems:
        if args.joint_adapter is None:
            raise ValueError("--joint-adapter is required when evaluating the joint system")
        engine.load_adapter("joint", _adapter_path(args.joint_adapter, "joint"))
    router = SceneRouter(args.classifier, args.model, engine.device) if args.classifier.exists() else None
    rows = read_jsonl(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    partial = args.report.with_suffix(".partial.jsonl")
    request_path = args.report.with_suffix(".request.json")
    request = {
        "manifest": str(args.manifest.resolve()),
        "systems": systems,
        "model": args.model,
        "confidence_threshold": args.confidence_threshold,
        "entropy_threshold": args.entropy_threshold,
        "fusion_top_k": args.fusion_top_k,
        "fusion_weight_step": args.fusion_weight_step,
        "limit": args.limit,
    }
    if request_path.exists():
        previous_request = json.loads(request_path.read_text(encoding="utf-8"))
        if previous_request != request:
            raise RuntimeError(
                f"Evaluation request changed; move {partial} and {request_path} aside before starting a different run"
            )
    else:
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    completed = read_jsonl(partial) if partial.exists() else []
    completed_ids = {row["eval_id"] for row in completed}
    pending = [
        row
        for row in rows
        if stable_id(row.get("source_id", row["audio_path"]), row.get("scene", "unknown")) not in completed_ids
    ]
    for start in range(0, len(pending), args.batch_size):
        batch = _evaluate_chunk(
            pending[start : start + args.batch_size],
            engine,
            router,
            set(systems),
            args.confidence_threshold,
            args.entropy_threshold,
        )
        with partial.open("a", encoding="utf-8") as handle:
            for row in batch:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"evaluated={len(completed) + start + len(batch)}/{len(rows)}")
    order = {
        stable_id(row.get("source_id", row["audio_path"]), row.get("scene", "unknown")): i for i, row in enumerate(rows)
    }
    results = [row for row in read_jsonl(partial) if row["eval_id"] in order]
    results.sort(key=lambda row: order[row["eval_id"]])
    pd.DataFrame(results).to_csv(args.report, index=False, encoding="utf-8-sig")
    summary = summarize(results, systems, args.bootstrap_samples, args.seed)
    summary_path = args.report.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
