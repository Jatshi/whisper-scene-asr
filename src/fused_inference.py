"""Calibrated routing and bounded-cache PEFT inference for Whisper."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.audio import load_audio
from src.common import SCENE_NAMES
from src.routing import RoutingDecision, decide_route, sparsify_weights
from src.scene_classifier import SceneClassifier


class SceneRouter:
    def __init__(self, checkpoint: str | Path, model_name: str, device: str) -> None:
        from transformers import WhisperFeatureExtractor, WhisperModel

        self.device = device
        self.extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.encoder = WhisperModel.from_pretrained(model_name).get_encoder().to(device).eval()
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        checkpoint_scenes = tuple(saved.get("scenes", SCENE_NAMES))
        if checkpoint_scenes != SCENE_NAMES:
            raise ValueError(f"Router scene order mismatch: {checkpoint_scenes}")
        self.temperature = max(float(saved.get("temperature", 1.0)), 0.05)
        self.classifier = SceneClassifier(self.encoder.config.d_model).to(device).eval()
        self.classifier.load_state_dict(saved["state_dict"])

    def predict(self, audio_path: str) -> dict[str, float]:
        return self.predict_batch([audio_path])[0]

    def summarize_batch(self, audio_paths: list[str]) -> torch.Tensor:
        """Return the exact mean+std encoder state consumed by v2/v3 routers."""
        if not audio_paths:
            return torch.empty((0, self.classifier.net[0].in_features), dtype=torch.float32)
        waveforms = [load_audio(path)[0] for path in audio_paths]
        features = self.extractor(waveforms, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"),
        ):
            hidden = self.encoder(features).last_hidden_state
            summaries = self.classifier.summarize(hidden)
        return summaries.float().cpu()

    def predict_batch(self, audio_paths: list[str]) -> list[dict[str, float]]:
        summaries = self.summarize_batch(audio_paths).to(self.device)
        with torch.inference_mode():
            logits = self.classifier.forward_summary(summaries) / self.temperature
            probabilities = torch.softmax(logits, -1).float().cpu().tolist()
        return [dict(zip(SCENE_NAMES, row, strict=True)) for row in probabilities]


class FusedWhisperInference:
    """Provide isolated base, hard-route, soft-fusion and joint-adapter paths."""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: str | None = None,
        fusion_cache_size: int = 32,
        fusion_top_k: int = 2,
        fusion_weight_step: float = 0.05,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = WhisperProcessor.from_pretrained(model_name, language="zh", task="transcribe")
        self.base = WhisperForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).to(self.device)
        self.base.generation_config.language = "zh"
        self.base.generation_config.task = "transcribe"
        self.base.generation_config.forced_decoder_ids = None
        self.base.config.forced_decoder_ids = None
        self.base.config.suppress_tokens = []
        self.model: PeftModel | None = None
        self.adapter_names: set[str] = set()
        self.fusion_cache_size = max(1, fusion_cache_size)
        self.fusion_top_k = max(1, fusion_top_k)
        self.fusion_weight_step = max(0.0, fusion_weight_step)
        self._fusion_cache: OrderedDict[str, None] = OrderedDict()

    def load_adapter(self, name: str, path: str | Path) -> None:
        if self.model is None:
            self.model = PeftModel.from_pretrained(self.base, path, adapter_name=name, is_trainable=False)
        else:
            self.model.load_adapter(path, adapter_name=name, is_trainable=False)
        self.adapter_names.add(name)

    def _model_inputs(self, audio_path: str) -> dict[str, torch.Tensor]:
        audio, sample_rate = load_audio(audio_path)
        encoded = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "input_features": encoded.input_features.to(device=self.device, dtype=self.base.dtype),
            "attention_mask": encoded.attention_mask.to(device=self.device),
        }

    def _model_inputs_batch(self, audio_paths: list[str]) -> dict[str, torch.Tensor]:
        waveforms = [load_audio(path)[0] for path in audio_paths]
        encoded = self.processor(
            waveforms,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        return {
            "input_features": encoded.input_features.to(device=self.device, dtype=self.base.dtype),
            "attention_mask": encoded.attention_mask.to(device=self.device),
        }

    def _generate(self, audio_path: str, disable_adapters: bool = False) -> str:
        target = self.model or self.base
        context = target.disable_adapter() if disable_adapters and self.model is not None else nullcontext()
        with context, torch.inference_mode():
            token_ids = target.generate(**self._model_inputs(audio_path))
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()

    def transcribe_base(self, audio_path: str) -> str:
        # PeftModel mutates the base module when adapters are injected.  The
        # explicit context is required for a genuinely adapter-free baseline.
        return self._generate(audio_path, disable_adapters=True)

    def transcribe_base_batch(self, audio_paths: list[str]) -> list[str]:
        if not audio_paths:
            return []
        target = self.model or self.base
        context = target.disable_adapter() if self.model is not None else nullcontext()
        with context, torch.inference_mode():
            token_ids = target.generate(**self._model_inputs_batch(audio_paths))
        return [text.strip() for text in self.processor.batch_decode(token_ids, skip_special_tokens=True)]

    def transcribe_adapter(self, audio_path: str, name: str) -> str:
        if self.model is None or name not in self.adapter_names:
            raise KeyError(f"Adapter not loaded: {name}")
        self.model.set_adapter(name)
        return self._generate(audio_path)

    def transcribe_adapter_batch(self, audio_paths: list[str], name: str) -> list[str]:
        if self.model is None or name not in self.adapter_names:
            raise KeyError(f"Adapter not loaded: {name}")
        if not audio_paths:
            return []
        self.model.set_adapter(name)
        with torch.inference_mode():
            token_ids = self.model.generate(**self._model_inputs_batch(audio_paths))
        return [text.strip() for text in self.processor.batch_decode(token_ids, skip_special_tokens=True)]

    def _fusion_name(self, weights: dict[str, float]) -> str:
        signature = "_".join(f"{key}-{value:.4f}" for key, value in sorted(weights.items()))
        return "fusion_" + signature.replace(".", "p")

    def _activate_soft_weights(self, weights: dict[str, float]) -> str:
        if self.model is None:
            raise RuntimeError("No adapters loaded")
        unknown = set(weights) - self.adapter_names
        if unknown:
            raise KeyError(f"Fusion weights reference unloaded adapters: {sorted(unknown)}")
        weights = {name: float(value) for name, value in weights.items() if float(value) > 0}
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Fusion weights must contain positive mass")
        weights = {name: value / total for name, value in weights.items()}
        if not weights:
            raise RuntimeError("No scene adapters available for soft fusion")
        name = self._fusion_name(weights)
        if name in self.model.peft_config:
            self._fusion_cache.move_to_end(name)
        else:
            while len(self._fusion_cache) >= self.fusion_cache_size:
                evicted, _ = self._fusion_cache.popitem(last=False)
                self.model.delete_adapter(evicted)
            self.model.add_weighted_adapter(list(weights), list(weights.values()), name, combination_type="linear")
            self._fusion_cache[name] = None
        self.model.set_adapter(name)
        return name

    def _activate_soft_fusion(self, probabilities: dict[str, float]) -> str:
        return self._activate_soft_weights(self.soft_weights(probabilities))

    def soft_weights(self, probabilities: dict[str, float]) -> dict[str, float]:
        experts = self.adapter_names & set(SCENE_NAMES)
        return sparsify_weights(
            probabilities, experts, top_k=self.fusion_top_k, quantization_step=self.fusion_weight_step
        )

    def transcribe_soft(self, audio_path: str, probabilities: dict[str, float]) -> str:
        self._activate_soft_fusion(probabilities)
        return self._generate(audio_path)

    def transcribe_soft_weights(self, audio_path: str, weights: dict[str, float]) -> str:
        """Decode with policy-provided expert weights instead of v2 scene probabilities."""
        self._activate_soft_weights(weights)
        return self._generate(audio_path)

    def transcribe_soft_weights_batch(self, audio_paths: list[str], weights: dict[str, float]) -> list[str]:
        if not audio_paths:
            return []
        if self.model is None:
            raise RuntimeError("No adapters loaded")
        self._activate_soft_weights(weights)
        with torch.inference_mode():
            token_ids = self.model.generate(**self._model_inputs_batch(audio_paths))
        return [text.strip() for text in self.processor.batch_decode(token_ids, skip_special_tokens=True)]

    def transcribe_routed(
        self,
        audio_path: str,
        probabilities: dict[str, float],
        *,
        mode: str,
        confidence_threshold: float = 0.65,
        entropy_threshold: float = 1.35,
    ) -> tuple[str, RoutingDecision]:
        decision = decide_route(probabilities, confidence_threshold, entropy_threshold)
        if decision.route == "base":
            return self.transcribe_base(audio_path), decision
        if mode == "hard":
            return self.transcribe_adapter(audio_path, decision.route), decision
        if mode == "soft":
            return self.transcribe_soft(audio_path, probabilities), decision
        raise ValueError(f"Unknown routing mode: {mode}")

    @property
    def fusion_cache_entries(self) -> tuple[str, ...]:
        return tuple(self._fusion_cache)
