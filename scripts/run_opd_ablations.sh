#!/usr/bin/env bash
# Expensive full-loop ablations. The default main run already covers offline KD,
# top-k inference and validation-only threshold calibration.
set -Eeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-/root/autodl-tmp/whisper-scene-asr}

OUTPUT_ROOT="$PROJECT_DIR/output/v3-opd-rkl" KL_DIRECTION=reverse RUN_ABLATIONS=0 \
  bash "$PROJECT_DIR/scripts/run_opd_v2.sh"

if [[ "${RUN_JOINT_ACTION:-0}" == "1" ]]; then
  OUTPUT_ROOT="$PROJECT_DIR/output/v3-opd-joint-action" INCLUDE_JOINT=1 RUN_ABLATIONS=0 \
    bash "$PROJECT_DIR/scripts/run_opd_v2.sh"
fi
