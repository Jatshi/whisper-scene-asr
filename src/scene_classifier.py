"""Train and apply an acoustic-scene classifier atop frozen Whisper encoder features."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import WhisperFeatureExtractor, WhisperModel

from src.common import SCENE_NAMES, read_jsonl, set_seed
from src.audio import load_audio


class SceneClassifier(nn.Module):
    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Conv1d(hidden_size, 256, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(256, len(SCENE_NAMES)))
    def forward(self, hidden: torch.Tensor) -> torch.Tensor: return self.net(hidden.transpose(1, 2))


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, extractor: WhisperFeatureExtractor) -> None: self.rows = read_jsonl(manifest); self.extractor = extractor
    def __len__(self) -> int: return len(self.rows)
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        audio, sr = load_audio(self.rows[index]["audio_path"])
        feature = self.extractor(audio, sampling_rate=sr, return_tensors="pt").input_features[0]
        return feature, self.rows[index]["scene_label"]


def train(manifest: Path, output: Path, model_name: str, epochs: int, batch_size: int, seed: int) -> None:
    set_seed(seed); device = "cuda" if torch.cuda.is_available() else "cpu"; extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    data = ManifestDataset(manifest, extractor); n_train = int(len(data) * .9); train_data, val_data = random_split(data, [n_train, len(data)-n_train], generator=torch.Generator().manual_seed(seed))
    encoder = WhisperModel.from_pretrained(model_name).get_encoder().to(device).eval()
    for param in encoder.parameters(): param.requires_grad = False
    classifier = SceneClassifier(encoder.config.d_model).to(device); opt = torch.optim.AdamW(classifier.parameters(), lr=1e-3); loss_fn = nn.CrossEntropyLoss(); best = 0.0
    for _ in range(epochs):
        classifier.train()
        for features, labels in DataLoader(train_data, batch_size=batch_size, shuffle=True):
            with torch.no_grad(): hidden = encoder(features.to(device)).last_hidden_state
            loss = loss_fn(classifier(hidden), labels.to(device)); opt.zero_grad(); loss.backward(); opt.step()
        classifier.eval(); correct = total = 0
        with torch.no_grad():
            for features, labels in DataLoader(val_data, batch_size=batch_size):
                prediction = classifier(encoder(features.to(device)).last_hidden_state).argmax(-1).cpu(); correct += int((prediction == labels).sum()); total += len(labels)
        accuracy = correct / max(total, 1)
        if accuracy > best: best = accuracy; output.parent.mkdir(parents=True, exist_ok=True); torch.save({"state_dict": classifier.state_dict(), "accuracy": accuracy, "scenes": SCENE_NAMES}, output)
    print(f"best validation accuracy: {best:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, default=Path("data/augmented/augmented_meta.jsonl")); parser.add_argument("--output", type=Path, default=Path("output/scene_classifier.pt")); parser.add_argument("--model", default="openai/whisper-small"); parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args(); train(args.manifest, args.output, args.model, args.epochs, args.batch_size, args.seed)
