#!/usr/bin/env bash
set -euo pipefail

# Run after connecting to the user-created AutoDL RTX 4090 instance.
PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/whisper-scene-asr}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/whisper-scene-asr-data}
AISHELL_ARCHIVE=${AISHELL_ARCHIVE:-/root/autodl-pub/Aishell/data_aishell.gz}
MINIMUM_DATA_GB=${MINIMUM_DATA_GB:-25}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}
cd "$PROJECT_DIR"
mkdir -p "$DATA_ROOT"
"$PYTHON_BIN" -m src.preflight --project-dir "$DATA_ROOT" --minimum-gb "$MINIMUM_DATA_GB"
export HF_HOME=${HF_HOME:-"$PROJECT_DIR/models/huggingface"}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m src.download_assets --data-dir "$DATA_ROOT/raw" --aishell-archive "$AISHELL_ARCHIVE" --cache-dir "$HF_HOME"
"$PYTHON_BIN" -m src.data_prep --aishell-root "$DATA_ROOT/raw/data_aishell" --output-dir "$DATA_ROOT/processed/aishell1"
"$PYTHON_BIN" -m src.data_augmentation --input "$DATA_ROOT/processed/aishell1/train.jsonl" --output-dir "$DATA_ROOT/augmented" --per-scene 5000
"$PYTHON_BIN" -m accelerate.commands.launch -m src.train_lora --manifest "$DATA_ROOT/augmented/augmented_meta.jsonl" --mode per-scene
"$PYTHON_BIN" -m src.scene_classifier --manifest "$DATA_ROOT/augmented/augmented_meta.jsonl"
"$PYTHON_BIN" -m src.evaluate_asr --manifest "$DATA_ROOT/processed/aishell1/test.jsonl"
"$PYTHON_BIN" -m src.package_artifacts
