#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/whisper-scene-asr}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/whisper-scene-asr-data}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_DIR/output/v3-opd}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}
AISHELL_ARCHIVE=${AISHELL_ARCHIVE:-/root/autodl-pub/Aishell/data_aishell.gz}
HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
TRAIN_PER_SCENE=${TRAIN_PER_SCENE:-5000}
VALIDATION_PER_SCENE=${VALIDATION_PER_SCENE:-500}
TEST_PER_SCENE=${TEST_PER_SCENE:-1000}
DATA_SEED=${DATA_SEED:-42}

cd "$PROJECT_DIR"
mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT/data-stages"
export HF_HOME PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

data_stage() {
  local name=$1
  shift
  local marker="$OUTPUT_ROOT/data-stages/$name.done"
  if [[ -f "$marker" ]]; then
    echo "[skip:data] $name"
    return
  fi
  echo "[run:data] $name"
  "$@"
  date --iso-8601=seconds > "$marker"
}

if [[ -f "$DATA_ROOT/scenes-v2/train.jsonl" && -f "$DATA_ROOT/scenes-v2/validation.jsonl" && -f "$DATA_ROOT/scenes-v2/test.jsonl" ]]; then
  echo "Existing v2 scene manifests found; data rebuild is unnecessary."
  exit 0
fi

if [[ -f "$AISHELL_ARCHIVE" ]]; then
  data_stage 01_download "$PYTHON_BIN" -m src.download_assets --data-dir "$DATA_ROOT/raw" \
    --aishell-archive "$AISHELL_ARCHIVE" --cache-dir "$HF_HOME"
else
  data_stage 01_download "$PYTHON_BIN" -m src.download_assets --data-dir "$DATA_ROOT/raw" --cache-dir "$HF_HOME"
fi
data_stage 02_aishell_manifest "$PYTHON_BIN" -m src.data_prep \
  --aishell-root "$DATA_ROOT/raw/data_aishell" --output-dir "$DATA_ROOT/processed/aishell1"
data_stage 03_asset_manifest "$PYTHON_BIN" -m src.asset_manifest \
  --noise-spec "MUSAN::CC BY 4.0::$DATA_ROOT/raw/musan/noise" \
  --rir-spec "OpenSLR RIRS_NOISES real RIR::Apache-2.0::$DATA_ROOT/raw/RIRS_NOISES/real_rirs_isotropic_noises/rir_list" \
  --output-dir "$DATA_ROOT/assets"
data_stage 04_scene_corpus "$PYTHON_BIN" -m src.data_augmentation \
  --train-manifest "$DATA_ROOT/processed/aishell1/train.jsonl" \
  --validation-manifest "$DATA_ROOT/processed/aishell1/validation.jsonl" \
  --test-manifest "$DATA_ROOT/processed/aishell1/test.jsonl" \
  --noise-manifest "$DATA_ROOT/assets/noise.jsonl" --rir-manifest "$DATA_ROOT/assets/rir.jsonl" \
  --output-dir "$DATA_ROOT/scenes-v2" --train-per-scene "$TRAIN_PER_SCENE" \
  --validation-per-scene "$VALIDATION_PER_SCENE" --test-per-scene "$TEST_PER_SCENE" --seed "$DATA_SEED"
