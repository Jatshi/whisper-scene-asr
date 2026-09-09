#!/usr/bin/env bash
# OPD post-training layered on the completed v2 experts. Outputs are v3-opd.
set -Eeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/whisper-scene-asr}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/whisper-scene-asr-data}
V2_ROOT=${V2_ROOT:-$PROJECT_DIR/output/v2}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_DIR/output/v3-opd}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/bin/python}
BATCH_SIZE=${BATCH_SIZE:-8}
POLICY_BATCH_SIZE=${POLICY_BATCH_SIZE:-128}
POLICY_EPOCHS=${POLICY_EPOCHS:-10}
OFFLINE_EPOCHS=${OFFLINE_EPOCHS:-140}
OPD_ROUNDS=${OPD_ROUNDS:-4}
REPLAY_WINDOW=${REPLAY_WINDOW:-4}
ORACLE_FRACTION=${ORACLE_FRACTION:-0.25}
COLLECTION_SIZE=${COLLECTION_SIZE:-5000}
EVAL_LIMIT=${EVAL_LIMIT:-0}
SEEDS=${SEEDS:-42,7,2026}
RUN_ABLATIONS=${RUN_ABLATIONS:-1}
INCLUDE_JOINT=${INCLUDE_JOINT:-0}
ALLOW_SINGLE_SEED_AGGREGATE=${ALLOW_SINGLE_SEED_AGGREGATE:-0}
KL_DIRECTION=${KL_DIRECTION:-forward}

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/stages"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$DATA_ROOT/huggingface}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TOKENIZERS_PARALLELISM=false
exec > >(tee -a "$OUTPUT_ROOT/logs/pipeline.log") 2>&1

"$PYTHON_BIN" -m src.opd_run_request --project-dir "$PROJECT_DIR" \
  --path "$OUTPUT_ROOT/run_request.json" --seeds "$SEEDS" --rounds "$OPD_ROUNDS" \
  --replay-window "$REPLAY_WINDOW" --collection-size "$COLLECTION_SIZE" \
  --eval-limit "$EVAL_LIMIT" \
  --oracle-fraction "$ORACLE_FRACTION" --batch-size "$BATCH_SIZE" \
  --policy-batch-size "$POLICY_BATCH_SIZE" --policy-epochs "$POLICY_EPOCHS" \
  --offline-epochs "$OFFLINE_EPOCHS" --run-ablations "$RUN_ABLATIONS" \
  --kl-direction "$KL_DIRECTION" --include-joint "$INCLUDE_JOINT" \
  --allow-single-seed-aggregate "$ALLOW_SINGLE_SEED_AGGREGATE"

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

verify_v2() {
  [[ -f "$V2_ROOT/scene_classifier.pt" ]] || { echo "Missing v2 scene classifier"; return 1; }
  [[ -f "$V2_ROOT/evaluation.csv" ]] || { echo "Missing v2 evaluation.csv"; return 1; }
  for scene in clean noisy reverb fast_slow noisy_reverb; do
    find "$V2_ROOT/adapters/$scene" -name adapter_config.json -print -quit | grep -q . || return 1
  done
}

stage 00_dependencies "$PYTHON_BIN" -m pip install --disable-pip-version-check -r requirements.txt
stage 01_data bash scripts/prepare_opd_data.sh
stage 02_verify_v2 verify_v2
stage 03_pool "$PYTHON_BIN" -m src.build_opd_pool \
  --train "$DATA_ROOT/scenes-v2/train.jsonl" --validation "$DATA_ROOT/scenes-v2/validation.jsonl" \
  --test "$DATA_ROOT/scenes-v2/test.jsonl" --output "$OUTPUT_ROOT/opd_pool.jsonl" \
  --report "$OUTPUT_ROOT/pool_report.json" --calibration-output "$OUTPUT_ROOT/calibration.jsonl" \
  --calibration-fraction 0.1

IFS=',' read -r -a seed_values <<< "$SEEDS"
summary_paths=()
for seed in "${seed_values[@]}"; do
  seed_root="$OUTPUT_ROOT/seeds/$seed"
  mkdir -p "$seed_root"
  joint_init=()
  joint_decode=()
  if [[ "$INCLUDE_JOINT" == "1" ]]; then
    joint_init=(--include-joint)
    joint_decode=(--joint-adapter "$V2_ROOT/joint/deploy")
  fi
  stage "10_init_$seed" "$PYTHON_BIN" -m src.opd_trainer init \
    --classifier "$V2_ROOT/scene_classifier.pt" --output "$seed_root/policy_init.pt" "${joint_init[@]}"
  current_policy="$seed_root/policy_init.pt"
  annotation_paths=()
  for ((round=0; round<=OPD_ROUNDS; round++)); do
    annotation="$seed_root/round-$round/annotations.jsonl"
    stage "20_collect_${seed}_$round" "$PYTHON_BIN" -m src.oracle_router \
      --manifest "$OUTPUT_ROOT/opd_pool.jsonl" --policy "$current_policy" \
      --classifier "$V2_ROOT/scene_classifier.pt" --adapter-dir "$V2_ROOT/adapters" \
      --output "$annotation" --seed "$((seed + round))" --oracle-fraction "$ORACLE_FRACTION" \
      --collection-size "$COLLECTION_SIZE" --summary-cache "$OUTPUT_ROOT/summary_cache.pt" \
      --batch-size "$BATCH_SIZE" "${joint_decode[@]}"
    annotation_paths+=("$annotation")
    first=$(( ${#annotation_paths[@]} > REPLAY_WINDOW ? ${#annotation_paths[@]} - REPLAY_WINDOW : 0 ))
    replay=("${annotation_paths[@]:first}")
    train_root="$seed_root/round-$round/train"
    stage "30_train_${seed}_$round" "$PYTHON_BIN" -m src.opd_trainer train \
      --annotations "${replay[@]}" --initial-checkpoint "$current_policy" \
      --output "$train_root" --seed "$seed" --epochs "$POLICY_EPOCHS" \
      --batch-size "$POLICY_BATCH_SIZE" --kl-direction "$KL_DIRECTION"
    current_policy="$train_root/policy.pt"
  done
  stage "40_eval_$seed" "$PYTHON_BIN" -m src.evaluate_opd \
    --manifest "$DATA_ROOT/scenes-v2/test.jsonl" --policy "$current_policy" \
    --classifier "$V2_ROOT/scene_classifier.pt" --adapter-dir "$V2_ROOT/adapters" \
    --report "$seed_root/evaluation.csv" --v2-evaluation "$V2_ROOT/evaluation.csv" \
    --batch-size "$BATCH_SIZE" --seed "$seed" --limit "$EVAL_LIMIT" "${joint_decode[@]}"
  summary_paths+=("$seed_root/evaluation.summary.json")

  if [[ "$RUN_ABLATIONS" == "1" ]]; then
    stage "50_offline_fkl_$seed" "$PYTHON_BIN" -m src.opd_trainer train \
      --annotations "${annotation_paths[0]}" --initial-checkpoint "$seed_root/policy_init.pt" \
      --output "$seed_root/ablations/offline-fkl" --seed "$seed" --epochs "$OFFLINE_EPOCHS" \
      --batch-size "$POLICY_BATCH_SIZE" --mode offline-kd
    stage "51_offline_rkl_$seed" "$PYTHON_BIN" -m src.opd_trainer train \
      --annotations "${annotation_paths[0]}" --initial-checkpoint "$seed_root/policy_init.pt" \
      --output "$seed_root/ablations/offline-rkl" --seed "$seed" --epochs "$OFFLINE_EPOCHS" \
      --batch-size "$POLICY_BATCH_SIZE" \
      --mode offline-kd --kl-direction reverse
    for kd in offline-fkl offline-rkl; do
      stage "51_eval_${kd}_$seed" "$PYTHON_BIN" -m src.evaluate_opd \
        --manifest "$DATA_ROOT/scenes-v2/test.jsonl" --policy "$seed_root/ablations/$kd/policy.pt" \
        --classifier "$V2_ROOT/scene_classifier.pt" --adapter-dir "$V2_ROOT/adapters" \
        --report "$seed_root/ablations/$kd-evaluation.csv" --batch-size "$BATCH_SIZE" --seed "$seed" \
        --limit "$EVAL_LIMIT" "${joint_decode[@]}"
    done
    for top_k in 1 3; do
      stage "52_topk_${top_k}_$seed" "$PYTHON_BIN" -m src.evaluate_opd \
        --manifest "$DATA_ROOT/scenes-v2/test.jsonl" --policy "$current_policy" \
        --classifier "$V2_ROOT/scene_classifier.pt" --adapter-dir "$V2_ROOT/adapters" \
        --report "$seed_root/ablations/topk-$top_k.csv" --fusion-top-k "$top_k" \
        --batch-size "$BATCH_SIZE" --seed "$seed" --limit "$EVAL_LIMIT" "${joint_decode[@]}"
    done
    stage "53_validation_$seed" "$PYTHON_BIN" -m src.evaluate_opd \
      --manifest "$OUTPUT_ROOT/calibration.jsonl" --policy "$current_policy" \
      --classifier "$V2_ROOT/scene_classifier.pt" --adapter-dir "$V2_ROOT/adapters" \
      --report "$seed_root/ablations/validation.csv" --batch-size "$BATCH_SIZE" --seed "$seed" \
      --limit "$EVAL_LIMIT" "${joint_decode[@]}"
    stage "54_threshold_calibration_$seed" "$PYTHON_BIN" -m src.calibrate_opd_thresholds \
      --validation-report "$seed_root/ablations/validation.csv" \
      --output "$seed_root/ablations/adaptive_thresholds.json"
    read -r adaptive_confidence adaptive_entropy < <("$PYTHON_BIN" -c \
      'import json,sys; d=json.load(open(sys.argv[1])); print(d["confidence_threshold"], d["entropy_threshold"])' \
      "$seed_root/ablations/adaptive_thresholds.json")
    stage "55_adaptive_eval_$seed" "$PYTHON_BIN" -m src.evaluate_opd \
      --manifest "$DATA_ROOT/scenes-v2/test.jsonl" --policy "$current_policy" \
      --classifier "$V2_ROOT/scene_classifier.pt" --adapter-dir "$V2_ROOT/adapters" \
      --report "$seed_root/ablations/adaptive-evaluation.csv" --batch-size "$BATCH_SIZE" --seed "$seed" \
      --confidence-threshold "$adaptive_confidence" --entropy-threshold "$adaptive_entropy" \
      --limit "$EVAL_LIMIT" "${joint_decode[@]}"
  fi
done

aggregate_flags=()
if [[ "$ALLOW_SINGLE_SEED_AGGREGATE" == "1" ]]; then
  aggregate_flags=(--allow-single)
fi
manifest_status=complete
if [[ "$EVAL_LIMIT" -gt 0 ]]; then
  aggregate_flags+=(--validation-only)
  manifest_status=validation_only
fi
stage 60_aggregate "$PYTHON_BIN" -m src.aggregate_opd --summaries "${summary_paths[@]}" \
  --output "$OUTPUT_ROOT/multiseed_summary.json" "${aggregate_flags[@]}"
stage 61_manifest "$PYTHON_BIN" -m src.write_opd_manifest --output-root "$OUTPUT_ROOT" \
  --seeds "${seed_values[@]}" --destination "$OUTPUT_ROOT/run_manifest.json" --status "$manifest_status"
echo "OPD pipeline complete: $OUTPUT_ROOT"
