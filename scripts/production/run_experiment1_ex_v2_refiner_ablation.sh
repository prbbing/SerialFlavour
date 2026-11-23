#!/usr/bin/env bash
# Supplement the 1M and 5M EX V2 allocations with four refiner recipes.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
read -r -a total_m <<< "${TOTAL_M:-1 5}"
read -r -a gpu_pool <<< "${GPU_POOL:-0}"
exec "${PYTHON:-python}" scripts/production/run_experiment1_ex_v2_refine.py \
    --total-m "${total_m[@]}" --gpus "${gpu_pool[@]}" \
    --suffix refiner_ablation_v1 --downstream-workers 5 --stage all --prune-after-aggregate \
    --recipe F1_double --recipe F2_origin --recipe F2_vertex --recipe FG1_48d "$@"
