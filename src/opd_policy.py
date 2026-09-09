"""Policy network and action contract for one-step OPD Whisper routing."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from src.common import SCENE_NAMES

BASE_ACTION = "base"
SOFT_ACTION = "soft"
JOINT_ACTION = "joint"
DEFAULT_ACTION_NAMES = (BASE_ACTION, *SCENE_NAMES, SOFT_ACTION)


def action_names(include_joint: bool = False) -> tuple[str, ...]:
    """Return the versioned action order. Never infer this order from a set/dict."""
    return (*DEFAULT_ACTION_NAMES, JOINT_ACTION) if include_joint else DEFAULT_ACTION_NAMES


def _expert_indices(names: Sequence[str]) -> list[int]:
    missing = set(SCENE_NAMES) - set(names)
    if missing:
        raise ValueError(f"Action specification misses scene experts: {sorted(missing)}")
    return [names.index(scene) for scene in SCENE_NAMES]


def soft_weights_from_logits(
    logits: torch.Tensor,
    names: Sequence[str] = DEFAULT_ACTION_NAMES,
    *,
    top_k: int = 2,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Turn only the five expert logits into normalized top-k fusion weights."""
    if logits.ndim != 2 or logits.shape[-1] != len(names):
        raise ValueError(f"Expected [batch, {len(names)}] logits, got {tuple(logits.shape)}")
    if temperature <= 0:
        raise ValueError("Fusion temperature must be positive")
    if top_k < 1 or top_k > len(SCENE_NAMES):
        raise ValueError(f"top_k must be in [1, {len(SCENE_NAMES)}]")
    expert_logits = logits[:, _expert_indices(list(names))] / temperature
    probabilities = torch.softmax(expert_logits, dim=-1)
    if top_k == len(SCENE_NAMES):
        return probabilities
    values, indices = probabilities.topk(top_k, dim=-1)
    sparse = torch.zeros_like(probabilities).scatter(-1, indices, values)
    return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(sparse.dtype).eps)


class OPDRouterPolicy(nn.Module):
    """Small policy head over mean+std Whisper encoder summaries."""

    def __init__(
        self,
        summary_dim: int = 1536,
        hidden_size: int = 256,
        names: Sequence[str] = DEFAULT_ACTION_NAMES,
        dropout: float = 0.1,
        strategy_temperature: float = 1.0,
        fusion_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if strategy_temperature <= 0 or fusion_temperature <= 0:
            raise ValueError("Policy temperatures must be positive")
        self.summary_dim = int(summary_dim)
        self.hidden_size = int(hidden_size)
        self.action_names = tuple(names)
        self.dropout = float(dropout)
        self.strategy_temperature = float(strategy_temperature)
        self.fusion_temperature = float(fusion_temperature)
        self.network = nn.Sequential(
            nn.Linear(self.summary_dim, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, len(self.action_names)),
        )

    def forward(self, summaries: torch.Tensor) -> torch.Tensor:
        if summaries.ndim != 2 or summaries.shape[-1] != self.summary_dim:
            raise ValueError(f"Expected [batch, {self.summary_dim}] summaries, got {tuple(summaries.shape)}")
        return self.network(summaries)

    def probabilities(self, summaries: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(summaries) / self.strategy_temperature, dim=-1)

    def fusion_weights(self, summaries: torch.Tensor, top_k: int = 2) -> torch.Tensor:
        return soft_weights_from_logits(
            self(summaries), self.action_names, top_k=top_k, temperature=self.fusion_temperature
        )

    def checkpoint_payload(self, **metadata: Any) -> dict[str, Any]:
        return {
            "schema_version": "3.0",
            "model_type": "opd_router_policy",
            "summary_dim": self.summary_dim,
            "hidden_size": self.hidden_size,
            "action_names": list(self.action_names),
            "dropout": self.dropout,
            "strategy_temperature": self.strategy_temperature,
            "fusion_temperature": self.fusion_temperature,
            "state_dict": self.state_dict(),
            **metadata,
        }

    @classmethod
    def from_checkpoint_payload(
        cls, payload: dict[str, Any], expected_actions: Sequence[str] = DEFAULT_ACTION_NAMES
    ) -> OPDRouterPolicy:
        saved_actions = tuple(payload["action_names"])
        if saved_actions != tuple(expected_actions):
            raise ValueError(f"OPD action order mismatch: saved={saved_actions}, expected={tuple(expected_actions)}")
        policy = cls(
            summary_dim=int(payload["summary_dim"]),
            hidden_size=int(payload["hidden_size"]),
            names=saved_actions,
            dropout=float(payload.get("dropout", 0.1)),
            strategy_temperature=float(payload.get("strategy_temperature", 1.0)),
            fusion_temperature=float(payload.get("fusion_temperature", 1.0)),
        )
        policy.load_state_dict(payload["state_dict"])
        return policy

    def initialize_from_scene_classifier(self, classifier: nn.Module) -> None:
        """Warm-start shared layers and five expert rows from the v2 classifier."""
        source = getattr(classifier, "net", None)
        if source is None or len(source) != 4:
            raise ValueError("Unsupported scene-classifier architecture")
        with torch.no_grad():
            self.network[0].weight.copy_(source[0].weight)
            self.network[0].bias.copy_(source[0].bias)
            expert_rows = _expert_indices(list(self.action_names))
            indices = torch.tensor(expert_rows, device=self.network[3].weight.device)
            self.network[3].weight.index_copy_(0, indices, source[3].weight)
            self.network[3].bias.index_copy_(0, indices, source[3].bias)
