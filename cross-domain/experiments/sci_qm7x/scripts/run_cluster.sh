#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
CONFIG="${CONFIG:-cross-domain/experiments/sci_qm7x/config/cluster_50k.json}"
PYTHON="${PYTHON:-python}"
if [[ -n "${CONDA_ENV:-}" ]]; then
  source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  conda activate "$CONDA_ENV"
fi
read -r -a GPUS <<< "${GPU_POOL:-0}"
exec "$PYTHON" cross-domain/experiments/sci_qm7x/scripts/run_cluster.py --config "$CONFIG" --gpus "${GPUS[@]}" --retries "${RETRIES:-1}" "$@"
