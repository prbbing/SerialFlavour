#!/usr/bin/env bash
# Experiment 1 EX unified per-seed pipeline.
#
# Phase 1 runs stage 1 (processed caches) for every config in ``order``,
# sequentially in that order.  Phase 2 is a dependency-aware, one-job-per-GPU
# work queue over (config, Parallel seed, stage) units, where a recipe unit
# trains that recipe's downstream seeds in parallel; a freed GPU immediately
# receives the next dependency-ready unit.  When every seed of a config is
# pruned the scheduler aggregates that config's rejection curves on the fly;
# phase 3 re-runs only configs whose aggregate marker is still missing.
#
# Usage: bash scripts/production/run_experiment1_ex_full.sh [--dry-run] [--force]
#
# --force   ignore completed-seed markers and recompute this run's stage
#           artifacts (Parallel checkpoints, B/Y caches, refiners, evaluation)
#           instead of skipping already-finished work.
#
# A failing unit is retried RETRIES times by the seed runner; if it still
# fails, only that Parallel seed is skipped (its later stages are dropped) and
# the run continues.  The script still aggregates the finished seeds, then
# exits non-zero if any seed was skipped.
#
# Environment overrides:
#   GPU_POOL="0 1 2 3"    physical GPUs used for the per-seed work-queue
#   RETRIES=1             per-stage retries inside the seed runner
#   MIN_FREE_GB=200       abort below this free space on /data
#   PATIENCE=100          Parallel-training early-stopping patience (epochs)
#   REFINE_GPUS=GPUs-1    GPUs allowed to run refine chains at once; the rest
#                         keep training Parallel models (>=1)
#   PYTHON=python         interpreter for stage scripts
#   LOG_BASE=...          base directory for logs and state

set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"

python_bin=${PYTHON:-python}
gpu_pool=(${GPU_POOL:-0 1 2 3})
retries=${RETRIES:-1}
min_free_gb=${MIN_FREE_GB:-200}
log_base=${LOG_BASE:-logs/parallel_refine/experiment1_ex/full}
# Parallel-training early-stopping patience (epochs) passed to train_parallel.py.
patience=${PATIENCE:-100}
# Cap on GPUs running refine chains; keep at least one GPU on Parallel training.
refine_gpus=${REFINE_GPUS:-$(( ${#gpu_pool[@]} - 1 ))}
if [ "$refine_gpus" -lt 1 ]; then
    refine_gpus=1
fi

dry_run=0
force=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        --force) force=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# Ordered by A/B allocation then dataset size.  Add or remove names here.
order=(
    n1m_a100_b000
    n2m_a100_b000
    n3m_a100_b000
    n1m_a095_b005
    n2m_a095_b005
    n3m_a095_b005
    n1m_a090_b010
    n2m_a090_b010
    n3m_a090_b010
    n1m_a085_b015
    n2m_a085_b015
    n3m_a085_b015
    n1m_a080_b020
    n2m_a080_b020
    n3m_a080_b020
)
config_prefix=configs/parallel_refine/experiments/experiment1_ex/experiment1_ex_p122k_

config_path() {
    printf '%s%s.json' "$config_prefix" "$1"
}

state_dir="$log_base/state"
stage1_dir="$log_base/stage1"
units_dir="$log_base/units"
scheduler_log="$log_base/scheduler.log"
failed_log="$log_base/failed_jobs.log"
mkdir -p "$state_dir" "$stage1_dir" "$units_dir"
: >"$scheduler_log"
: >"$failed_log"

free_gb() {
    df -BG --output=avail /data | tail -1 | tr -dc '0-9'
}

check_space() {
    local avail
    avail=$(free_gb)
    if [ "$avail" -lt "$min_free_gb" ]; then
        echo "ABORT: only ${avail}G free on /data (< ${min_free_gb}G)" >&2
        exit 1
    fi
}

echo "per-seed pipeline: ${#order[@]} configs, gpus=${gpu_pool[*]}"
echo "log base: $log_base"

# --------------------------------------------------------------------------
# Phase 1: stage 1 processed caches, sequential in order.
# --------------------------------------------------------------------------
echo
echo "=== PHASE 1: stage 1 processed caches ==="
for name in "${order[@]}"; do
    config=$(config_path "$name")
    log="$stage1_dir/$name.log"
    if [ ! -f "$config" ]; then
        echo "ABORT: missing config $config" >&2
        exit 1
    fi
    if [ "$dry_run" -eq 1 ]; then
        echo "  $name: $python_bin scripts/prepare_data.py --config $config \\"
        echo "      --build-processed-caches --processed-split a_train \\"
        echo "      --processed-split a_val --processed-split b_train \\"
        echo "      --processed-split b_val --processed-split y_test"
        continue
    fi
    check_space
    echo "stage1 $name"
    printf '%s stage1-start %s\n' "$(date -Is)" "$name" >>"$scheduler_log"
    if ! "$python_bin" scripts/prepare_data.py --config "$config" \
        --build-processed-caches \
        --processed-split a_train --processed-split a_val \
        --processed-split b_train --processed-split b_val \
        --processed-split y_test >"$log" 2>&1; then
        echo "FAILED stage1 $name log=$log" >&2
        printf '%s stage1 job=%s rc=%s log=%s\n' \
            "$(date -Is)" "$name" "1" "$log" >>"$failed_log"
        exit 1
    fi
    printf '%s stage1-done %s\n' "$(date -Is)" "$name" >>"$scheduler_log"
done

# --------------------------------------------------------------------------
# Phase 2: dependency-aware work queue.  A GPU receives exactly one task;
# recipe tasks are serial per Parallel seed.  A seed's refine chain is
# prioritised once its Parallel finishes, capped at REFINE_GPUS concurrent
# refine chains so at least one GPU keeps training Parallel models.
# --------------------------------------------------------------------------
echo
echo "=== PHASE 2: dependency-aware refine-priority queue (max-refine=$refine_gpus) ==="

queue_args=()
for name in "${order[@]}"; do
    config=$(config_path "$name")
    if [ ! -f "$config" ]; then
        echo "ABORT: missing config $config" >&2
        exit 1
    fi
    queue_args+=(--config "$config")
done

if [ "$dry_run" -eq 1 ]; then
    echo "dry-run: dependency-ready Parallel, cache, recipe, Y, evaluation, and prune units"
    echo "  GPUs: ${gpu_pool[*]}"
    echo "DRY RUN COMPLETE"
    exit 0
fi

check_space
scheduler_args=()
if [ "$force" -eq 1 ]; then
    scheduler_args+=(--force)
fi
set +e
"$python_bin" scripts/production/run_experiment1_ex_recipe_queue.py "${queue_args[@]}" \
    --gpus "${gpu_pool[@]}" --log-base "$log_base" --retries "$retries" \
    --patience "$patience" --max-refine "$refine_gpus" \
    --python "$python_bin" --state-dir "$state_dir" \
    "${scheduler_args[@]}"
scheduler_rc=$?
set -e
if [ "$scheduler_rc" -ne 0 ]; then
    echo "WARNING: scheduler skipped one or more seeds after retries; see $failed_log" >&2
fi

echo
echo "=== PHASE 3: cross-seed rejection aggregation (fallback) ==="
aggregate_dir="$log_base/aggregate"
mkdir -p "$aggregate_dir"
for name in "${order[@]}"; do
    config=$(config_path "$name")
    log="$aggregate_dir/$name.log"
    if [ -f "$aggregate_dir/$name.done" ]; then
        echo "aggregate $name: already done by scheduler, skipping"
        continue
    fi
    echo "aggregate $name"
    printf '%s aggregate-start %s\n' "$(date -Is)" "$name" >>"$scheduler_log"
    if ! "$python_bin" scripts/evaluate.py --config "$config" \
        --model parallel_dnn --aggregate-parallel-seeds >"$log" 2>&1; then
        echo "FAILED aggregate $name log=$log" >&2
        printf '%s aggregate job=%s rc=%s log=%s\n' \
            "$(date -Is)" "$name" "1" "$log" >>"$failed_log"
        exit 1
    fi
    touch "$aggregate_dir/$name.done"
    printf '%s aggregate-done %s\n' "$(date -Is)" "$name" >>"$scheduler_log"
done

echo
echo "PER-SEED PIPELINE COMPLETE"
if [ "$scheduler_rc" -ne 0 ]; then
    echo "PIPELINE FINISHED WITH SKIPPED SEEDS (scheduler rc=$scheduler_rc)"
    exit "$scheduler_rc"
fi
