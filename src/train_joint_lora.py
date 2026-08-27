"""Initialize from adapter fusion, then jointly fine-tune on all real scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.common import SCENE_NAMES, read_jsonl, set_seed
from src.train_lora import train_model


def normalized_weights(values: list[float]) -> list[float]:
    if len(values) != len(SCENE_NAMES):
        raise ValueError(f"Expected {len(SCENE_NAMES)} weights, received {len(values)}")
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("Adapter weights must be non-negative with a positive sum")
    total = sum(values)
    return [value / total for value in values]


def initialize_joint_model(
    model_name: str,
    adapter_dir: Path,
    weights: list[float],
    combination_type: str = "linear",
    density: float = 0.5,
) -> PeftModel:
    base = WhisperForConditionalGeneration.from_pretrained(model_name)
    base.generation_config.language = "zh"
    base.generation_config.task = "transcribe"
    base.config.forced_decoder_ids = None
    base.config.suppress_tokens = []
    base.config.use_cache = False
    first = SCENE_NAMES[0]
    model = PeftModel.from_pretrained(base, adapter_dir / first, adapter_name=first, is_trainable=False)
    for scene in SCENE_NAMES[1:]:
        model.load_adapter(adapter_dir / scene, adapter_name=scene, is_trainable=False)
    values = normalized_weights(weights)
    model.add_weighted_adapter(
        list(SCENE_NAMES),
        values,
        "joint",
        combination_type=combination_type,
        density=density if combination_type in {"ties", "dare_linear"} else None,
    )
    model.set_adapter("joint")
    model.peft_config["joint"].inference_mode = False
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".joint." in name
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable == 0:
        raise RuntimeError("PEFT did not expose trainable parameters for the joint adapter")
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/v2/joint"))
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument(
        "--weights",
        type=float,
        nargs=5,
        default=[0.2] * 5,
        metavar=("CLEAN", "NOISY", "REVERB", "FAST_SLOW", "NOISY_REVERB"),
    )
    parser.add_argument("--combination-type", choices=("linear", "ties", "dare_linear"), default="linear")
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Joint LoRA training requires CUDA")
    train_rows, eval_rows = read_jsonl(args.train_manifest), read_jsonl(args.eval_manifest)
    processor = WhisperProcessor.from_pretrained(args.model, language="zh", task="transcribe")
    model = initialize_joint_model(args.model, args.adapter_dir, args.weights, args.combination_type, args.density)
    train_model(
        model,
        processor,
        train_rows,
        eval_rows,
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.gradient_accumulation,
        args.workers,
        args.seed,
    )
    # Save a minimal deployment copy containing only the jointly trained adapter.
    model.save_pretrained(args.output_dir / "deploy", selected_adapters=["joint"])
    processor.save_pretrained(args.output_dir / "deploy")


if __name__ == "__main__":
    main()
