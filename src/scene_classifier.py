"""Train a calibrated scene router on source-disjoint real degradations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import WhisperFeatureExtractor, WhisperModel

from src.audio import load_audio
from src.common import SCENE_NAMES, assert_disjoint_sources, read_jsonl, set_seed


class SceneClassifier(nn.Module):
    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(SCENE_NAMES)),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.forward_summary(self.summarize(hidden))

    @staticmethod
    def summarize(hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat([hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)], dim=-1)

    def forward_summary(self, summary: torch.Tensor) -> torch.Tensor:
        return self.net(summary)


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict], extractor: WhisperFeatureExtractor) -> None:
        self.rows, self.extractor = rows, extractor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        audio, sample_rate = load_audio(self.rows[index]["audio_path"])
        feature = self.extractor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features[0]
        return feature, int(self.rows[index]["scene_label"])


def expected_calibration_error(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels)
    error = torch.zeros((), device=probabilities.device)
    boundaries = torch.linspace(0, 1, bins + 1, device=probabilities.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.float().mean() * (correct[selected].float().mean() - confidence[selected].mean()).abs()
    return float(error.cpu())


def calibrate_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    log_temperature = nn.Parameter(torch.zeros((), device=logits.device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=50)
    loss_fn = nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = loss_fn(logits / log_temperature.exp().clamp(min=0.05, max=20.0), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(min=0.05, max=20.0).cpu())


def _encode_summaries(encoder: nn.Module, loader: DataLoader, device: str) -> TensorDataset:
    summaries, labels = [], []
    use_amp = device == "cuda"
    with torch.inference_mode():
        for features, batch_labels in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                hidden = encoder(features.to(device)).last_hidden_state
            summaries.append(SceneClassifier.summarize(hidden.float()).cpu())
            labels.append(batch_labels)
    return TensorDataset(torch.cat(summaries), torch.cat(labels))


def _validation_logits(
    classifier: SceneClassifier, loader: DataLoader, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, labels = [], []
    classifier.eval()
    with torch.no_grad():
        for summaries, batch_labels in loader:
            logits.append(classifier.forward_summary(summaries.to(device)))
            labels.append(batch_labels.to(device))
    return torch.cat(logits), torch.cat(labels)


def train(
    train_manifest: Path,
    validation_manifest: Path,
    output: Path,
    report_path: Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    workers: int,
    seed: int,
) -> None:
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows, validation_rows = read_jsonl(train_manifest), read_jsonl(validation_manifest)
    assert_disjoint_sources(train_rows, validation_rows)
    extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    raw_train_loader = DataLoader(
        ManifestDataset(train_rows, extractor),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device == "cuda",
    )
    raw_validation_loader = DataLoader(
        ManifestDataset(validation_rows, extractor),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device == "cuda",
    )
    encoder = WhisperModel.from_pretrained(model_name).get_encoder().to(device).eval()
    hidden_size = encoder.config.d_model
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    # The frozen Whisper encoder is expensive.  Compute compact mean/std
    # summaries once, then train/calibrate only the small head for all epochs.
    train_features = _encode_summaries(encoder, raw_train_loader, device)
    validation_features = _encode_summaries(encoder, raw_validation_loader, device)
    train_loader = DataLoader(train_features, batch_size=batch_size * 4, shuffle=True)
    validation_loader = DataLoader(validation_features, batch_size=batch_size * 4)
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()
    classifier = SceneClassifier(hidden_size).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_accuracy, best_state = -1.0, None
    for epoch in range(epochs):
        classifier.train()
        for summaries, labels in train_loader:
            loss = loss_fn(classifier.forward_summary(summaries.to(device)), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        logits, labels = _validation_logits(classifier, validation_loader, device)
        accuracy = float(logits.argmax(-1).eq(labels).float().mean().cpu())
        print(f"epoch={epoch + 1} validation_accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in classifier.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Scene router produced no checkpoint")
    classifier.load_state_dict(best_state)
    logits, labels = _validation_logits(classifier, validation_loader, device)
    temperature = calibrate_temperature(logits, labels)
    probabilities = torch.softmax(logits / temperature, dim=-1)
    confusion = torch.zeros((len(SCENE_NAMES), len(SCENE_NAMES)), dtype=torch.int64, device=device)
    for truth, prediction in zip(labels, probabilities.argmax(-1), strict=True):
        confusion[truth, prediction] += 1
    report = {
        "validation_accuracy": best_accuracy,
        "temperature": temperature,
        "ece": expected_calibration_error(probabilities, labels),
        "confusion_matrix": confusion.cpu().tolist(),
        "scenes": list(SCENE_NAMES),
        "source_disjoint": True,
        "architecture": "whisper_mean_std_mlp_v2",
        "encoder_passes_per_utterance": 1,
        "seed": seed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "accuracy": best_accuracy,
            "temperature": temperature,
            "scenes": SCENE_NAMES,
            "model_name": model_name,
            "architecture": "whisper_mean_std_mlp_v2",
        },
        output,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("output/v2/scene_classifier.pt"))
    parser.add_argument("--report", type=Path, default=Path("output/v2/router_metrics.json"))
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        args.train_manifest,
        args.validation_manifest,
        args.output,
        args.report,
        args.model,
        args.epochs,
        args.batch_size,
        args.workers,
        args.seed,
    )


if __name__ == "__main__":
    main()
