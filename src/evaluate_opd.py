"""Evaluate OPD hard/meta-soft routing and its per-utterance CER oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.common import SCENE_NAMES, manifest_fingerprint, read_jsonl, stable_id, stratified_limit
from src.evaluate_asr import _adapter_path, _prediction_fields, summarize
from src.fused_inference import FusedWhisperInference, SceneRouter
from src.metrics import paired_bootstrap_difference
from src.opd_policy import OPDRouterPolicy
from src.oracle_router import _decode_actions, _weight_dict, file_sha256, validate_annotation_request


def evaluate(
    *,
    manifest: Path,
    policy_checkpoint: Path,
    classifier_checkpoint: Path,
    adapter_dir: Path,
    report: Path,
    model_name: str,
    batch_size: int,
    confidence_threshold: float,
    entropy_threshold: float,
    fusion_top_k: int,
    fusion_weight_step: float,
    bootstrap_samples: int,
    seed: int,
    joint_adapter: Path | None,
    limit: int,
    v2_evaluation: Path | None,
) -> dict:
    rows = read_jsonl(manifest)
    rows = stratified_limit(rows, limit)
    checkpoint = torch.load(policy_checkpoint, map_location="cpu", weights_only=False)
    names = tuple(checkpoint["action_names"])
    policy = OPDRouterPolicy.from_checkpoint_payload(checkpoint, names).eval()
    request = {
        "schema_version": "3.0",
        "manifest_hash": manifest_fingerprint(rows),
        "policy_hash": file_sha256(policy_checkpoint),
        "classifier_hash": file_sha256(classifier_checkpoint),
        "action_names": list(names),
        "model": model_name,
        "confidence_threshold": confidence_threshold,
        "entropy_threshold": entropy_threshold,
        "fusion_top_k": fusion_top_k,
        "fusion_weight_step": fusion_weight_step,
        "limit": limit,
        "v2_evaluation_hash": file_sha256(v2_evaluation) if v2_evaluation else None,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    validate_annotation_request(report.with_suffix(".request.json"), request)
    partial = report.with_suffix(".partial.jsonl")
    completed = read_jsonl(partial) if partial.exists() else []
    completed_ids = {row["eval_id"] for row in completed}
    engine = FusedWhisperInference(model_name, fusion_top_k=fusion_top_k, fusion_weight_step=fusion_weight_step)
    for scene in SCENE_NAMES:
        engine.load_adapter(scene, _adapter_path(adapter_dir, scene))
    if "joint" in names:
        if joint_adapter is None:
            raise ValueError("--joint-adapter is required by this policy action spec")
        engine.load_adapter("joint", _adapter_path(joint_adapter, "joint"))
    router = SceneRouter(classifier_checkpoint, model_name, engine.device)
    pending = [
        row
        for row in rows
        if stable_id(row.get("source_id", row["audio_path"]), row.get("scene", "unknown")) not in completed_ids
    ]
    all_actions = [set(names) for _ in range(batch_size)]
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        summaries = router.summarize_batch([row["audio_path"] for row in batch])
        with torch.inference_mode():
            logits = policy(summaries)
            probabilities = torch.softmax(logits / policy.strategy_temperature, -1)
        weights = [
            _weight_dict(logits[i], names, fusion_top_k, policy.fusion_temperature, fusion_weight_step)
            for i in range(len(batch))
        ]
        action_predictions = _decode_actions(
            engine, batch, all_actions[: len(batch)], weights, include_joint="joint" in names
        )
        records = []
        for index, row in enumerate(batch):
            probs = probabilities[index]
            chosen_index = int(probs.argmax())
            chosen_action = names[chosen_index]
            confidence = float(probs[chosen_index])
            entropy = float(-(probs * probs.clamp_min(torch.finfo(probs.dtype).eps).log()).sum())
            fallback = confidence < confidence_threshold or entropy > entropy_threshold
            hard_action = "base" if fallback else chosen_action
            hard_prediction = action_predictions[index][hard_action]
            if fallback or chosen_action == "base":
                soft_prediction = action_predictions[index]["base"]
            elif chosen_action == "joint":
                soft_prediction = action_predictions[index]["joint"]
            else:
                soft_prediction = action_predictions[index]["soft"]
            reference = row["text"]
            oracle_action = min(
                names,
                key=lambda action: _prediction_fields(reference, action, action_predictions[index][action])[
                    f"{action}_cer"
                ],
            )
            record = {
                "eval_id": stable_id(row.get("source_id", row["audio_path"]), row.get("scene", "unknown")),
                "source_id": row.get("source_id", ""),
                "reference": reference,
                "true_scene": row.get("scene", "unknown"),
                "router_route": hard_action,
                "router_confidence": confidence,
                "router_entropy": entropy,
                "router_reason": "uncertainty_fallback" if fallback else "opd_policy",
                "policy_argmax": chosen_action,
                "policy_probabilities": json.dumps(dict(zip(names, probs.tolist(), strict=True)), sort_keys=True),
                "soft_applied_weights": json.dumps(weights[index], sort_keys=True),
                "oracle_route": oracle_action,
                "action_predictions": json.dumps(action_predictions[index], ensure_ascii=False, sort_keys=True),
            }
            record.update(_prediction_fields(reference, "base", action_predictions[index]["base"]))
            record.update(_prediction_fields(reference, "opd_hard", hard_prediction))
            record.update(_prediction_fields(reference, "opd_soft", soft_prediction))
            record.update(_prediction_fields(reference, "oracle", action_predictions[index][oracle_action]))
            records.append(record)
        with partial.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"evaluated={len(completed) + start + len(records)}/{len(rows)}")
    order = {
        stable_id(row.get("source_id", row["audio_path"]), row.get("scene", "unknown")): i for i, row in enumerate(rows)
    }
    results = [row for row in read_jsonl(partial) if row["eval_id"] in order]
    results.sort(key=lambda row: order[row["eval_id"]])
    pd.DataFrame(results).to_csv(report, index=False, encoding="utf-8-sig")
    systems = ["base", "opd_hard", "opd_soft", "oracle"]
    summary = summarize(results, systems, bootstrap_samples, seed)
    overall = summary["buckets"]["overall"]["metrics"]
    summary["oracle_gap"] = {
        system: overall[system]["cer"] - overall["oracle"]["cer"] for system in ("opd_hard", "opd_soft")
    }
    summary["policy_hash"] = request["policy_hash"]
    if v2_evaluation:
        v2_rows = pd.read_csv(v2_evaluation).set_index("eval_id")
        aligned = [row for row in results if row["eval_id"] in v2_rows.index]
        if len(aligned) != len(results):
            raise RuntimeError("v2 evaluation does not contain the exact OPD test set")
        references = [row["reference"] for row in aligned]
        summary["paired_vs_v2"] = {
            "opd_hard_vs_v2_hard": paired_bootstrap_difference(
                references,
                [str(v2_rows.loc[row["eval_id"], "hard_prediction"]) for row in aligned],
                [row["opd_hard_prediction"] for row in aligned],
                samples=bootstrap_samples,
                seed=seed,
            ),
            "opd_soft_vs_v2_soft": paired_bootstrap_difference(
                references,
                [str(v2_rows.loc[row["eval_id"], "soft_prediction"]) for row in aligned],
                [row["opd_soft_prediction"] for row in aligned],
                samples=bootstrap_samples,
                seed=seed,
            ),
        }
    report.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--joint-adapter", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--entropy-threshold", type=float, default=1.35)
    parser.add_argument("--fusion-top-k", type=int, default=2)
    parser.add_argument("--fusion-weight-step", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--v2-evaluation", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                manifest=args.manifest,
                policy_checkpoint=args.policy,
                classifier_checkpoint=args.classifier,
                adapter_dir=args.adapter_dir,
                report=args.report,
                model_name=args.model,
                batch_size=args.batch_size,
                confidence_threshold=args.confidence_threshold,
                entropy_threshold=args.entropy_threshold,
                fusion_top_k=args.fusion_top_k,
                fusion_weight_step=args.fusion_weight_step,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                joint_adapter=args.joint_adapter,
                limit=args.limit,
                v2_evaluation=args.v2_evaluation,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
