#!/usr/bin/env bash
# Single-node pool: one independent upstream seed lifecycle per selected GPU.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$task_root"
if [[ -n "${CONDA_ENV:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
task_config=${CONFIG:-cross-domain/config/cv_nyu_mtan/cluster_full.json}
task_python=${PYTHON:-python}
read -r -a task_gpus <<< "${GPU_POOL:-0}"
exec "$task_python" cross-domain/scripts/cv_nyu_mtan/run_cluster.py \
    --config "$task_config" --gpus "${task_gpus[@]}" --retries "${RETRIES:-1}" "$@"
