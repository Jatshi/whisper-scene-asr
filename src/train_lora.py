"""Leakage-safe Whisper LoRA training with resumable, on-demand audio loading."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from src.audio import load_audio
from src.common import SCENE_NAMES, assert_disjoint_sources, manifest_fingerprint, read_jsonl, set_seed
from src.metrics import corpus_counts


class ManifestDataset(Dataset):
    """Extract Whisper features lazily so large corpora do not inflate Arrow caches."""

    def __init__(self, rows: list[dict], processor: WhisperProcessor) -> None:
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        audio, sample_rate = load_audio(row["audio_path"])
        features = self.processor.feature_extractor(audio, sampling_rate=sample_rate).input_features[0]
        labels = self.processor.tokenizer(row["text"]).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class Collator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        inputs = self.processor.feature_extractor.pad(
            [{"input_features": item["input_features"]} for item in features], return_tensors="pt"
        )
        labels = self.processor.tokenizer.pad([{"input_ids": item["labels"]} for item in features], return_tensors="pt")
        label_ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        if (label_ids[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            label_ids = label_ids[:, 1:]
        inputs["labels"] = label_ids
        return inputs


def _latest_checkpoint(output: Path) -> str | None:
    checkpoints = [path for path in output.glob("checkpoint-*") if path.is_dir()]
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda path: int(path.name.rsplit("-", 1)[-1])))


def build_lora_model(model_name: str) -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = "zh"
    model.generation_config.task = "transcribe"
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            # PEFT's SEQ_2_SEQ_LM wrapper injects text-model ``input_ids``.
            # Whisper's native forward instead expects ``input_features``, so
            # use the generic wrapper and preserve Whisper's API verbatim.
            task_type=None,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        ),
    )
    model.print_trainable_parameters()
    return model


def train_model(
    model: Any,
    processor: WhisperProcessor,
    train_rows: list[dict],
    eval_rows: list[dict],
    output: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation: int,
    workers: int,
    seed: int,
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires CUDA; local tests intentionally stop before this stage")
    if not train_rows or not eval_rows:
        raise ValueError("Training and validation rows must both be non-empty")
    assert_disjoint_sources(train_rows, eval_rows)
    use_bf16 = torch.cuda.is_bf16_supported()
    request = {
        "train_fingerprint": manifest_fingerprint(train_rows),
        "eval_fingerprint": manifest_fingerprint(eval_rows),
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation": gradient_accumulation,
        "learning_rate": learning_rate,
        "seed": seed,
    }
    request_path = output / "training_request.json"
    if request_path.exists() and json.loads(request_path.read_text(encoding="utf-8")) != request:
        raise RuntimeError(
            f"Training request changed; move {output} aside instead of resuming an incompatible checkpoint"
        )
    output.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")

    def metrics(prediction: Any) -> dict[str, float]:
        predictions = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
        labels = np.where(prediction.label_ids == -100, processor.tokenizer.pad_token_id, prediction.label_ids)
        hypotheses = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        references = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"cer": corpus_counts(references, hypotheses).rate}

    arguments = Seq2SeqTrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size // 2),
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        bf16=use_bf16,
        fp16=not use_bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=25,
        predict_with_generate=True,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=workers,
        gradient_checkpointing=True,
        # The Whisper backbone is frozen during LoRA tuning, so checkpoint
        # inputs do not require gradients.  Re-entrant checkpointing would
        # therefore detach the whole loss graph; the non-reentrant variant
        # correctly records the LoRA operations inside each checkpoint.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=seed,
        data_seed=seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=ManifestDataset(train_rows, processor),
        eval_dataset=ManifestDataset(eval_rows, processor),
        data_collator=Collator(processor),
        compute_metrics=metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train(resume_from_checkpoint=_latest_checkpoint(output))
    trainer.save_model(str(output / "final"))
    processor.save_pretrained(output / "final")
    model.save_pretrained(output)
    processor.save_pretrained(output)


def train(
    train_manifest: Path,
    eval_manifest: Path,
    output: Path,
    model_name: str,
    scene: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation: int,
    workers: int,
    seed: int,
) -> None:
    set_seed(seed)
    train_rows, eval_rows = read_jsonl(train_manifest), read_jsonl(eval_manifest)
    if scene:
        train_rows = [row for row in train_rows if row["scene"] == scene]
        eval_rows = [row for row in eval_rows if row["scene"] == scene]
    processor = WhisperProcessor.from_pretrained(model_name, language="zh", task="transcribe")
    train_model(
        build_lora_model(model_name),
        processor,
        train_rows,
        eval_rows,
        output,
        epochs,
        batch_size,
        learning_rate,
        gradient_accumulation,
        workers,
        seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/v2/adapters"))
    parser.add_argument("--mode", choices=("mixed", "per-scene"), default="per-scene")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.mode == "mixed":
        train(
            args.train_manifest,
            args.eval_manifest,
            args.output_dir / "mixed",
            args.model,
            None,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.gradient_accumulation,
            args.workers,
            args.seed,
        )
        return
    for scene in SCENE_NAMES:
        train(
            args.train_manifest,
            args.eval_manifest,
            args.output_dir / scene,
            args.model,
            scene,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.gradient_accumulation,
            args.workers,
            args.seed,
        )


if __name__ == "__main__":
    main()
