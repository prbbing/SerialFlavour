#!/usr/bin/env bash
# Single-node multi-GPU launcher for the QM9 gap + charge/bond experiment.
#
# Runs the pipeline pool: prepare once, then one (variant, seed) lifecycle per
# free GPU across GPU_POOL, then evaluate and analyze. Completed seeds are
# skipped, so the script resumes.
#
# Usage:
#   GPU_POOL="0 1 2 3" bash cross-domain/experiments/qm9/scripts/run_full.sh
#
# Environment overrides:
#   CONFIG      config path (default cross-domain/experiments/qm9/config/qm9_gap_charge_bond_full_100k.json)
#   GPU_POOL    space-separated physical GPU ids (default 0)
#   RETRIES     per-seed retries inside run_seed (default 1)
#   PYTHON      interpreter (default python; use the cluster env interpreter)
#   CONDA_ENV   optional conda env name to activate before launching

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$root"

config=${CONFIG:-cross-domain/experiments/qm9/config/qm9_gap_charge_bond_full_100k.json}
gpu_pool=${GPU_POOL:-0}
retries=${RETRIES:-1}
python_bin=${PYTHON:-python}

if [ -n "${CONDA_ENV:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        python_bin=${PYTHON:-python}
    else
        echo "WARNING: CONDA_ENV set but conda not found; using $python_bin" >&2
    fi
fi

if [ ! -f "$config" ]; then
    echo "ABORT: missing config $config" >&2
    exit 2
fi

echo "launcher: config=$config gpus='$gpu_pool' retries=$retries python=$python_bin"

# shellcheck disable=SC2086
"$python_bin" cross-domain/scripts/run_pool.py \
    --config "$config" \
    --gpus $gpu_pool \
    --retries "$retries" \
    --python "$python_bin"
