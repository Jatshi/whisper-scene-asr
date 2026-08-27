#!/usr/bin/env bash
set -euo pipefail

echo "scripts/run_gpu.sh is retained as a compatibility entrypoint; running the v2 staged pipeline."
exec bash "$(dirname "$0")/run_autodl_v2.sh"
