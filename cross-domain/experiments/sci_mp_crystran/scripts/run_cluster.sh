#!/usr/bin/env bash
# MP single-node pool; directly runs the full configured matrix.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$task_root"
if [[ -n "${CONDA_ENV:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
task_config=${CONFIG:-cross-domain/experiments/sci_mp_crystran/config/cluster_20k.json}
task_python=${PYTHON:-python}
read -r -a task_gpus <<< "${GPU_POOL:-0}"
exec "$task_python" cross-domain/experiments/sci_mp_crystran/scripts/run_cluster.py \
    --config "$task_config" --gpus "${task_gpus[@]}" --retries "${RETRIES:-1}" "$@"