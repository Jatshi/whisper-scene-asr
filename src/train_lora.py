"""Fine-tune Whisper with one mixed adapter or one adapter per acoustic scene."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import Audio, Dataset
from peft import LoraConfig, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from src.common import read_jsonl, set_seed


@dataclass
class Collator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict]) -> dict:
        inputs = self.processor.feature_extractor.pad(
            [{"input_features": x["input_features"]} for x in features], return_tensors="pt"
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": x["labels"]} for x in features], return_tensors="pt"
        )
        inputs["labels"] = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        return inputs


def load_dataset(manifest: Path, processor: WhisperProcessor) -> Dataset:
    dataset = Dataset.from_list(read_jsonl(manifest)).cast_column("audio_path", Audio(sampling_rate=16000))
    def encode(row: dict) -> dict:
        audio = row["audio_path"]
        return {"input_features": processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0],
                "labels": processor.tokenizer(row["text"]).input_ids}
    return dataset.map(encode, remove_columns=dataset.column_names, num_proc=1)


def train(manifest: Path, output: Path, model_name: str, epochs: int, batch_size: int, learning_rate: float, seed: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training requires a CUDA GPU; run this stage on the provisioned GPU instance.")
    set_seed(seed)
    processor = WhisperProcessor.from_pretrained(model_name, language="zh", task="transcribe")
    # Keep trainable LoRA parameters in FP32.  Seq2SeqTrainer's ``fp16``
    # setting handles autocast and gradient scaling; pre-casting the whole
    # model to FP16 makes Accelerate attempt to unscale FP16 gradients.
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = "zh"; model.generation_config.task = "transcribe"; model.config.forced_decoder_ids = None; model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]))
    data = load_dataset(manifest, processor)
    split = data.train_test_split(test_size=0.05, seed=seed)
    cer = evaluate.load("cer")
    def metrics(pred):
        predictions = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        labels = np.where(pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids)
        return {"cer": cer.compute(predictions=processor.tokenizer.batch_decode(predictions, skip_special_tokens=True), references=processor.tokenizer.batch_decode(labels, skip_special_tokens=True))}
    args = Seq2SeqTrainingArguments(output_dir=str(output), per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2, learning_rate=learning_rate, num_train_epochs=epochs, fp16=True, evaluation_strategy="epoch", save_strategy="epoch",
        logging_steps=25, predict_with_generate=True, generation_max_length=225, load_best_model_at_end=True, metric_for_best_model="cer", greater_is_better=False,
        save_total_limit=2, report_to=["tensorboard"], remove_unused_columns=False)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=split["train"], eval_dataset=split["test"], data_collator=Collator(processor), compute_metrics=metrics, tokenizer=processor.feature_extractor)
    trainer.train(); trainer.save_model(str(output / "final")); processor.save_pretrained(output / "final"); model.save_pretrained(output); processor.save_pretrained(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/augmented/augmented_meta.jsonl")); parser.add_argument("--output-dir", type=Path, default=Path("output/adapters"))
    parser.add_argument("--mode", choices=("mixed", "per-scene"), default="per-scene"); parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--batch-size", type=int, default=8); parser.add_argument("--learning-rate", type=float, default=3e-4); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); rows = read_jsonl(args.manifest)
    if args.mode == "mixed": train(args.manifest, args.output_dir / "mixed", args.model, args.epochs, args.batch_size, args.learning_rate, args.seed); return
    for scene in sorted({row["scene"] for row in rows}):
        scene_manifest = args.output_dir / f"{scene}.jsonl"; scene_manifest.parent.mkdir(parents=True, exist_ok=True)
        from src.common import write_jsonl
        write_jsonl((row for row in rows if row["scene"] == scene), scene_manifest)
        train(scene_manifest, args.output_dir / scene, args.model, args.epochs, args.batch_size, args.learning_rate, args.seed)


if __name__ == "__main__": main()
