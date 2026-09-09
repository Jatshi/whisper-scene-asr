"""Losses, replay utilities and CPU-trainable core for OPD router post-training.

The routing decision is a one-step contextual bandit: an utterance summary is the
context, one ASR path is the action, and negative CER is the observed reward.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.common import read_jsonl, set_seed
from src.opd_policy import DEFAULT_ACTION_NAMES, OPDRouterPolicy, action_names
from src.scene_classifier import SceneClassifier


@dataclass(frozen=True)
class OPDLossConfig:
    distillation_weight: float = 1.0
    policy_gradient_weight: float = 0.25
    entropy_weight: float = 0.01
    fallback_weight: float = 0.5
    kl_direction: str = "forward"


def _masked_teacher(teacher_probs: torch.Tensor, teacher_mask: torch.Tensor) -> torch.Tensor:
    if teacher_probs.shape != teacher_mask.shape:
        raise ValueError("teacher_probs and teacher_mask shapes differ")
    support = teacher_mask.sum(-1)
    if torch.any(support == 0):
        raise ValueError("Every example needs at least one observed teacher support action")
    masked = teacher_probs.clamp_min(0) * teacher_mask
    totals = masked.sum(-1, keepdim=True)
    if torch.any(totals <= 0):
        raise ValueError("Teacher probability mass on observed support must be positive")
    return masked / totals


def compute_opd_loss(
    *,
    logits: torch.Tensor,
    sampled_actions: torch.Tensor,
    advantages: torch.Tensor,
    teacher_probs: torch.Tensor,
    teacher_mask: torch.Tensor,
    fallback_mask: torch.Tensor,
    config: OPDLossConfig,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute masked distillation + REINFORCE + anti-collapse entropy loss."""
    teacher = _masked_teacher(teacher_probs, teacher_mask)
    if config.kl_direction not in {"forward", "reverse"}:
        raise ValueError("kl_direction must be 'forward' or 'reverse'")
    log_policy = torch.log_softmax(logits, dim=-1)
    policy = log_policy.exp()
    eps = torch.finfo(logits.dtype).eps
    if config.kl_direction == "forward":
        per_example_kl = (teacher * (teacher.clamp_min(eps).log() - log_policy) * teacher_mask).sum(-1)
    else:
        masked_logits = logits.masked_fill(~teacher_mask, torch.finfo(logits.dtype).min)
        support_log_policy = torch.log_softmax(masked_logits, dim=-1)
        support_policy = support_log_policy.exp() * teacher_mask
        per_example_kl = (support_policy * (support_log_policy - teacher.clamp_min(eps).log()) * teacher_mask).sum(-1)
    # A base-only observation contains no pairwise preference. Treating its
    # degenerate teacher as a target would silently train collapse to base.
    per_example_kl = per_example_kl * teacher_mask.sum(-1).gt(1)
    chosen_log_probability = log_policy.gather(1, sampled_actions[:, None]).squeeze(1)
    policy_gradient = -(advantages.detach() * chosen_log_probability)
    entropy = -(policy * log_policy).sum(-1)
    row_weight = torch.where(fallback_mask, torch.full_like(entropy, config.fallback_weight), torch.ones_like(entropy))
    if sample_weights is not None:
        if sample_weights.shape != row_weight.shape or torch.any(sample_weights <= 0):
            raise ValueError("sample_weights must be positive and match the batch")
        row_weight = row_weight * sample_weights
    # We minimize the objective, hence -lambda*H encourages exploration.
    entropy_regularizer = -config.entropy_weight * entropy
    supervised = config.distillation_weight * per_example_kl + config.policy_gradient_weight * policy_gradient
    total = (row_weight * supervised).sum() / row_weight.sum().clamp_min(eps) + entropy_regularizer.mean()
    return {
        "total": total,
        "distillation": per_example_kl.mean(),
        "policy_gradient": policy_gradient.mean(),
        "entropy": entropy.mean(),
        "entropy_regularizer": entropy_regularizer.mean(),
    }


def difficulty_sampling_weights(
    probabilities: torch.Tensor, action_indices: torch.Tensor | None = None
) -> torch.Tensor:
    """Priority 1/(1+p_old(a|s)); before sampling, a is the greedy action."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be rank two")
    if action_indices is None:
        confidence = probabilities.max(-1).values
    else:
        confidence = probabilities.gather(1, action_indices[:, None]).squeeze(1)
    weights = 1.0 / (1.0 + confidence.clamp(0, 1))
    return weights / weights.sum()


def adapt_temperature(current: float, mean_entropy: float, action_count: int, rate: float = 0.05) -> float:
    if current <= 0 or action_count < 2:
        raise ValueError("Invalid temperature adaptation inputs")
    target = 0.65 * math.log(action_count)
    direction = 1.0 if mean_entropy < target else -1.0
    return min(4.0, max(0.25, current * math.exp(direction * rate)))


class RoundReplay:
    def __init__(self, max_rounds: int = 4) -> None:
        self.max_rounds = max(1, int(max_rounds))
        self.rounds: list[list[dict[str, Any]]] = []

    def append(self, rows: list[dict[str, Any]]) -> None:
        self.rounds.append(rows)
        self.rounds = self.rounds[-self.max_rounds :]

    def rows(self) -> list[dict[str, Any]]:
        return [row for round_rows in self.rounds for row in round_rows]


def _tensorize(rows: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("No OPD annotations supplied")
    summaries = torch.tensor([row["summary"] for row in rows], dtype=torch.float32)
    sampled = torch.tensor([names.index(row["sampled_action"]) for row in rows], dtype=torch.long)
    advantages = torch.tensor([row["base_cer"] - row["action_cer"] for row in rows], dtype=torch.float32)
    teacher = torch.zeros(len(rows), len(names), dtype=torch.float32)
    mask = torch.zeros_like(teacher, dtype=torch.bool)
    for index, row in enumerate(rows):
        for action, probability in row["teacher_probs"].items():
            if action in names:
                teacher[index, names.index(action)] = float(probability)
                mask[index, names.index(action)] = True
    fallback = torch.tensor([bool(row.get("fallback", False)) for row in rows])
    priority = torch.tensor([float(row.get("priority_weight", 1.0)) for row in rows])
    return {
        "summaries": summaries,
        "sampled": sampled,
        "advantages": advantages,
        "teacher": teacher,
        "mask": mask,
        "fallback": fallback,
        "priority": priority,
    }


def initialize_policy(
    classifier_checkpoint: Path,
    output: Path,
    *,
    include_joint: bool,
    strategy_temperature: float,
    fusion_temperature: float,
) -> None:
    saved = torch.load(classifier_checkpoint, map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    summary_dim = int(state["net.0.weight"].shape[1])
    classifier = SceneClassifier(summary_dim // 2)
    classifier.load_state_dict(state)
    policy = OPDRouterPolicy(
        summary_dim=summary_dim,
        names=action_names(include_joint),
        strategy_temperature=strategy_temperature,
        fusion_temperature=fusion_temperature,
    )
    policy.initialize_from_scene_classifier(classifier)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        policy.checkpoint_payload(
            initialized_from=str(classifier_checkpoint),
            initialization="v2_scene_classifier_shared_and_expert_rows",
            experimental_joint_action=include_joint,
        ),
        output,
    )


def train_annotations(
    annotations: list[Path],
    output: Path,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    loss_config: OPDLossConfig,
    initial_checkpoint: Path | None = None,
    mode: str = "onpolicy",
) -> dict[str, Any]:
    """Train from collected on-policy/offline annotations; no ASR model gradients."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    set_seed(seed)
    rows = [row for path in annotations for row in read_jsonl(path)]
    names = tuple(rows[0].get("action_names", DEFAULT_ACTION_NAMES)) if rows else DEFAULT_ACTION_NAMES
    if initial_checkpoint:
        policy = OPDRouterPolicy.from_checkpoint_payload(
            torch.load(initial_checkpoint, map_location="cpu", weights_only=False), names
        )
    else:
        policy = OPDRouterPolicy(summary_dim=len(rows[0]["summary"]), names=names)
    if mode not in {"onpolicy", "offline-kd"}:
        raise ValueError("mode must be onpolicy or offline-kd")
    tensors = _tensorize(rows, names)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    history = []
    policy.train()
    for epoch in range(epochs):
        order = torch.randperm(len(rows), generator=generator)
        totals = []
        for offset in range(0, len(rows), batch_size):
            index = order[offset : offset + batch_size]
            losses = compute_opd_loss(
                logits=policy(tensors["summaries"][index]) / policy.strategy_temperature,
                sampled_actions=tensors["sampled"][index],
                advantages=tensors["advantages"][index],
                teacher_probs=tensors["teacher"][index],
                teacher_mask=tensors["mask"][index],
                fallback_mask=tensors["fallback"][index],
                config=loss_config,
                sample_weights=tensors["priority"][index] if mode == "onpolicy" else None,
            )
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            totals.append(float(losses["total"].detach()))
        history.append({"epoch": epoch + 1, "loss": sum(totals) / len(totals)})
    previous_temperature = policy.strategy_temperature
    if mode == "onpolicy":
        entropies = []
        for row in rows:
            values = torch.tensor(list(row.get("policy_probs", {}).values()), dtype=torch.float32)
            if len(values):
                entropies.append(float(-(values * values.clamp_min(1e-8).log()).sum()))
        if entropies:
            policy.strategy_temperature = adapt_temperature(
                policy.strategy_temperature, sum(entropies) / len(entropies), len(names)
            )
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        policy.checkpoint_payload(seed=seed, mode=mode, loss_config=asdict(loss_config), history=history),
        output / "policy.pt",
    )
    report = {
        "status": "trained",
        "seed": seed,
        "examples": len(rows),
        "annotations": [str(path) for path in annotations],
        "mode": mode,
        "strategy_temperature_before": previous_temperature,
        "strategy_temperature_after": policy.strategy_temperature,
        "history": history,
    }
    (output / "training_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize/train the lightweight OPD routing policy")
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--classifier", type=Path, required=True)
    init_parser.add_argument("--output", type=Path, required=True)
    init_parser.add_argument("--include-joint", action="store_true")
    init_parser.add_argument("--strategy-temperature", type=float, default=1.0)
    init_parser.add_argument("--fusion-temperature", type=float, default=1.0)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--initial-checkpoint", type=Path)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--kl-direction", choices=("forward", "reverse"), default="forward")
    train_parser.add_argument("--mode", choices=("onpolicy", "offline-kd"), default="onpolicy")
    args = parser.parse_args()
    if args.command == "init":
        initialize_policy(
            args.classifier,
            args.output,
            include_joint=args.include_joint,
            strategy_temperature=args.strategy_temperature,
            fusion_temperature=args.fusion_temperature,
        )
        print(args.output)
        return
    report = train_annotations(
        args.annotations,
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss_config=OPDLossConfig(
            kl_direction=args.kl_direction,
            policy_gradient_weight=0.0 if args.mode == "offline-kd" else 0.25,
        ),
        initial_checkpoint=args.initial_checkpoint,
        mode=args.mode,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
