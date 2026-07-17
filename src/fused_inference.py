"""Scene-probability-weighted LoRA composition for Whisper inference."""
from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.common import SCENE_NAMES
from src.audio import load_audio
from src.scene_classifier import SceneClassifier


def normalize_weights(probabilities: dict[str, float], available: set[str]) -> dict[str, float]:
    weights = {name: max(0.0, float(probabilities.get(name, 0.0))) for name in available}
    total = sum(weights.values())
    return ({name: value / total for name, value in weights.items()} if total else {name: 1 / len(weights) for name in weights})


class SceneRouter:
    def __init__(self, checkpoint: str | Path, model_name: str, device: str) -> None:
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.device = device; self.extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.encoder = WhisperModel.from_pretrained(model_name).get_encoder().to(device).eval()
        saved = torch.load(checkpoint, map_location=device); self.classifier = SceneClassifier(self.encoder.config.d_model).to(device).eval(); self.classifier.load_state_dict(saved["state_dict"])
    def predict(self, audio_path: str) -> dict[str, float]:
        audio, sr = load_audio(audio_path)
        features = self.extractor(audio, sampling_rate=sr, return_tensors="pt").input_features.to(self.device)
        with torch.no_grad(): probabilities = torch.softmax(self.classifier(self.encoder(features).last_hidden_state), -1)[0].cpu().tolist()
        return dict(zip(SCENE_NAMES, probabilities, strict=True))


class FusedWhisperInference:
    """Uses PEFT weighted-adapter composition, not one-step logit averaging."""
    def __init__(self, model_name: str = "openai/whisper-small", device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_name, language="zh", task="transcribe")
        base = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        base.generation_config.language = "zh"; base.generation_config.task = "transcribe"; base.config.forced_decoder_ids = None
        self.model: PeftModel | None = None; self.base = base; self.adapter_names: set[str] = set()
    def load_adapter(self, scene: str, path: str | Path) -> None:
        if self.model is None: self.model = PeftModel.from_pretrained(self.base, path, adapter_name=scene)
        else: self.model.load_adapter(path, adapter_name=scene)
        self.adapter_names.add(scene)
    def _features(self, audio_path: str) -> torch.Tensor:
        audio, sr = load_audio(audio_path)
        features = self.processor(audio, sampling_rate=sr, return_tensors="pt").input_features
        # The inference base is FP16 on CUDA; align feature dtype with it.
        # CPU inference remains FP32.
        return features.to(device=self.device, dtype=self.base.dtype)
    def transcribe_base(self, audio_path: str) -> str:
        with torch.no_grad():
            token_ids = self.base.generate(self._features(audio_path))
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
    def transcribe(self, audio_path: str, probabilities: dict[str, float] | None = None) -> str:
        if not self.model or not probabilities:
            return self.transcribe_base(audio_path)
        model = self.model
        if probabilities:
            weights = normalize_weights(probabilities, self.adapter_names)
            signature = "_".join(f"{key}{value:.2f}" for key, value in sorted(weights.items()))
            name = f"fusion_{signature}".replace(".", "p")
            if name not in self.model.peft_config:
                self.model.add_weighted_adapter(list(weights), list(weights.values()), name, combination_type="linear")
            self.model.set_adapter(name)
        with torch.no_grad(): token_ids = model.generate(self._features(audio_path))
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
