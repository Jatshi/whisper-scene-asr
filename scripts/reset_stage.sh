#!/usr/bin/env bash
set -euo pipefail
OUTPUT_ROOT=${OUTPUT_ROOT:-output/v2}
if [[ $# -ne 1 || ! $1 =~ ^[0-9]{2}_[a-z_]+$ ]]; then
  echo "usage: OUTPUT_ROOT=output/v2 bash scripts/reset_stage.sh 08_joint_adapter" >&2
  exit 2
fi
marker="$OUTPUT_ROOT/stages/$1.done"
if [[ -f "$marker" ]]; then
  rm -- "$marker"
  echo "Removed only the stage marker: $marker"
else
  echo "No marker exists: $marker"
fi
