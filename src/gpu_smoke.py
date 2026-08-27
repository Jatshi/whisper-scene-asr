"""Fast real-model compatibility check before expensive data generation/training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def run(model_name: str) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU smoke test")
    processor = WhisperProcessor.from_pretrained(model_name, language="zh", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(model_name)
    base.config.use_cache = False
    # Whisper consumes ``input_features`` rather than the ``input_ids`` used by
    # text encoder-decoder models.  Leaving ``task_type`` unset makes PEFT use
    # its generic wrapper, which forwards Whisper's native arguments unchanged.
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], bias="none")
    model = get_peft_model(base, config, adapter_name="scene_a")
    model.add_adapter("scene_b", config)
    model.add_weighted_adapter(["scene_a", "scene_b"], [0.5, 0.5], "joint", combination_type="linear")
    model.set_adapter("joint")
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".joint." in name
    model.to("cuda").train()
    # Match the real trainer: frozen-backbone LoRA needs non-reentrant
    # checkpointing or the loss graph is detached before reaching adapters.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    inputs = torch.zeros((1, 80, 3000), device="cuda")
    labels = processor.tokenizer("兼容性测试", return_tensors="pt").input_ids.to("cuda")
    loss = model(input_features=inputs, labels=labels).loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke loss: {loss}")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if not gradients or not all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Joint adapter gradients are missing or non-finite")
    return {
        "status": "passed",
        "model": model_name,
        "loss": float(loss.detach().cpu()),
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
