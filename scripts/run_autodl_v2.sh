#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/whisper-scene-asr}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/whisper-scene-asr-data}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_DIR/output/v2}
AISHELL_ARCHIVE=${AISHELL_ARCHIVE:-/root/autodl-pub/Aishell/data_aishell.gz}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}
MINIMUM_DATA_GB=${MINIMUM_DATA_GB:-80}
TRAIN_PER_SCENE=${TRAIN_PER_SCENE:-5000}
VALIDATION_PER_SCENE=${VALIDATION_PER_SCENE:-500}
TEST_PER_SCENE=${TEST_PER_SCENE:-1000}
BATCH_SIZE=${BATCH_SIZE:-8}
SEED=${SEED:-42}

cd "$PROJECT_DIR"
mkdir -p "$DATA_ROOT" "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/stages"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false
exec > >(tee -a "$OUTPUT_ROOT/logs/pipeline.log") 2>&1

stage() {
  local name=$1
  shift
  local marker="$OUTPUT_ROOT/stages/$name.done"
  if [[ -f "$marker" ]]; then
    echo "[skip] $name"
    return
  fi
  echo "[run] $name"
  "$@"
  date --iso-8601=seconds > "$marker"
}

stage 00_preflight "$PYTHON_BIN" -m src.preflight --project-dir "$DATA_ROOT" \
  --minimum-gb "$MINIMUM_DATA_GB" --json-output "$OUTPUT_ROOT/preflight.json"
stage 01_dependencies "$PYTHON_BIN" -m pip install --disable-pip-version-check -r requirements.txt

if [[ -f "$AISHELL_ARCHIVE" ]]; then
  stage 02_download "$PYTHON_BIN" -m src.download_assets --data-dir "$DATA_ROOT/raw" \
    --aishell-archive "$AISHELL_ARCHIVE" --cache-dir "$HF_HOME"
else
  stage 02_download "$PYTHON_BIN" -m src.download_assets --data-dir "$DATA_ROOT/raw" --cache-dir "$HF_HOME"
fi
stage 02_gpu_smoke "$PYTHON_BIN" -m src.gpu_smoke --output "$OUTPUT_ROOT/gpu_smoke.json"

stage 03_aishell_manifest "$PYTHON_BIN" -m src.data_prep \
  --aishell-root "$DATA_ROOT/raw/data_aishell" --output-dir "$DATA_ROOT/processed/aishell1"
stage 04_asset_manifest "$PYTHON_BIN" -m src.asset_manifest \
  --noise-spec "MUSAN::CC BY 4.0::$DATA_ROOT/raw/musan/noise" \
  --rir-spec "OpenSLR RIRS_NOISES real RIR::Apache-2.0::$DATA_ROOT/raw/RIRS_NOISES/real_rirs_isotropic_noises/rir_list" \
  --output-dir "$DATA_ROOT/assets"
stage 05_scene_corpus "$PYTHON_BIN" -m src.data_augmentation \
  --train-manifest "$DATA_ROOT/processed/aishell1/train.jsonl" \
  --validation-manifest "$DATA_ROOT/processed/aishell1/validation.jsonl" \
  --test-manifest "$DATA_ROOT/processed/aishell1/test.jsonl" \
  --noise-manifest "$DATA_ROOT/assets/noise.jsonl" --rir-manifest "$DATA_ROOT/assets/rir.jsonl" \
  --output-dir "$DATA_ROOT/scenes-v2" --train-per-scene "$TRAIN_PER_SCENE" \
  --validation-per-scene "$VALIDATION_PER_SCENE" --test-per-scene "$TEST_PER_SCENE" --seed "$SEED"
stage 06_scene_adapters "$PYTHON_BIN" -m src.train_lora \
  --train-manifest "$DATA_ROOT/scenes-v2/train.jsonl" --eval-manifest "$DATA_ROOT/scenes-v2/validation.jsonl" \
  --output-dir "$OUTPUT_ROOT/adapters" --mode per-scene --batch-size "$BATCH_SIZE" --seed "$SEED"
stage 07_router "$PYTHON_BIN" -m src.scene_classifier \
  --train-manifest "$DATA_ROOT/scenes-v2/train.jsonl" \
  --validation-manifest "$DATA_ROOT/scenes-v2/validation.jsonl" \
  --output "$OUTPUT_ROOT/scene_classifier.pt" --report "$OUTPUT_ROOT/router_metrics.json" --seed "$SEED"
stage 08_joint_adapter "$PYTHON_BIN" -m src.train_joint_lora \
  --train-manifest "$DATA_ROOT/scenes-v2/train.jsonl" --eval-manifest "$DATA_ROOT/scenes-v2/validation.jsonl" \
  --adapter-dir "$OUTPUT_ROOT/adapters" --output-dir "$OUTPUT_ROOT/joint" \
  --batch-size "$BATCH_SIZE" --seed "$SEED"
stage 09_bucketed_evaluation "$PYTHON_BIN" -m src.evaluate_asr \
  --manifest "$DATA_ROOT/scenes-v2/test.jsonl" --adapter-dir "$OUTPUT_ROOT/adapters" \
  --classifier "$OUTPUT_ROOT/scene_classifier.pt" --joint-adapter "$OUTPUT_ROOT/joint/deploy" \
  --report "$OUTPUT_ROOT/evaluation.csv" --batch-size "$BATCH_SIZE" \
  --bootstrap-samples 2000 --seed "$SEED"
stage 10_manifest "$PYTHON_BIN" -m src.write_run_manifest \
  --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --destination "$OUTPUT_ROOT/run_manifest.json"
stage 11_package "$PYTHON_BIN" -m src.package_artifacts --version v2 \
  --output-root "$OUTPUT_ROOT" --archive "$OUTPUT_ROOT/whisper-scene-asr-v2.tar.gz"

echo "Pipeline complete: $OUTPUT_ROOT"
