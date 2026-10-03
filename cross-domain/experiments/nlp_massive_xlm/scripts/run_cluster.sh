#!/usr/bin/env bash
# Single-node pool: one independent upstream seed lifecycle per selected GPU.
set -euo pipefail
task_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$task_root"
if [[ -n "${CONDA_ENV:-}" ]]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi
task_config=${CONFIG:-cross-domain/experiments/nlp_massive_xlm/config/cluster_full.json}
task_python=${PYTHON:-python}
task_allocation=${CUDA_VISIBLE_DEVICES:-}
task_default_gpus=${task_allocation//,/ }
read -r -a task_gpus <<< "${GPU_POOL:-${task_default_gpus:-0}}"
exec "$task_python" cross-domain/experiments/nlp_massive_xlm/scripts/run_cluster.py \
    --config "$task_config" --gpus "${task_gpus[@]}" --retries "${RETRIES:-1}" "$@"
