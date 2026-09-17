#!/usr/bin/env bash
# Background queue for the Experiment 2 A-train scaling matrix.
#
# The shared A-val/B/Y split must be materialized before training. This queue
# runs that preparation once (without --force), then runs the 11 outstanding
# configurations sequentially. The 56k/A=1M entry remains in the matrix but
# is omitted here because its prior run is recorded in the Experiment 2 README.
# Inside each configuration, the generic runner trains its configured seeds
# in parallel on GPUs 0,1,1,2,2.

set -uo pipefail

# Keep running independent configurations after a failed configuration. Set
# to 0 to stop at the first failure.
CONTINUE_ON_ERROR=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE_SCRIPT="$ROOT/scripts/run_experiment2_matrix_queue.sh"
EXPERIMENT_RUNNER="$ROOT/scripts/run_parallel_refine_experiment.sh"
SHARED_SPLIT_SCRIPT="$ROOT/scripts/prepare_experiment2_shared_splits.py"
QUEUE_DIR="$ROOT/logs/parallel_refine/experiment2_matrix_queue"
MASTER_LOG="$QUEUE_DIR/matrix_queue.log"
STATE_LOG="$QUEUE_DIR/matrix_queue_state.tsv"
PID_FILE="$QUEUE_DIR/matrix_queue.pid"
DONE_DIR="$QUEUE_DIR/completed"
SPLIT_LOCK_DIR="$ROOT/logs/parallel_refine/experiment2_shared_split.lock"

CONFIGS=(
    # A-train = 600k.
    "configs/parallel_refine/experiments/experiment2/experiment2_p056k_a600k.json"
    "configs/parallel_refine/experiments/experiment2/experiment2_p122k_a600k.json"

    # A-train = 800k.
    "configs/parallel_refine/experiments/experiment2/experiment2_p056k_a800k.json"
    "configs/parallel_refine/experiments/experiment2/experiment2_p122k_a800k.json"

    # A-train = 1M. The p056k entry is already represented by
    # parallel_refine_a1m_6layers and is intentionally not retrained here.
    "configs/parallel_refine/experiments/experiment2/experiment2_p122k_a1m.json"

    # A-train = 1.5M.
    "configs/parallel_refine/experiments/experiment2/experiment2_p056k_a1500k.json"
    "configs/parallel_refine/experiments/experiment2/experiment2_p122k_a1500k.json"

    # A-train = 2M.
    "configs/parallel_refine/experiments/experiment2/experiment2_p056k_a2m.json"
    "configs/parallel_refine/experiments/experiment2/experiment2_p122k_a2m.json"

    # A-train = 3M.
    "configs/parallel_refine/experiments/experiment2/experiment2_p056k_a3m.json"
    "configs/parallel_refine/experiments/experiment2/experiment2_p122k_a3m.json"
)

mkdir -p "$QUEUE_DIR" "$DONE_DIR"
cd "$ROOT"

SPLIT_LOCK_HELD=0
cleanup_split_lock() {
    if ((SPLIT_LOCK_HELD == 1)); then
        rmdir "$SPLIT_LOCK_DIR" 2>/dev/null || true
        SPLIT_LOCK_HELD=0
    fi
}

is_running() {
    if [[ ! -f "$PID_FILE" ]]; then
        return 1
    fi
    local pid
    pid="$(<"$PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

config_hash() {
    python - "$1" <<'PY'
import sys
from src.parallel_refine.config import load_study_config
print(load_study_config(sys.argv[1]).source_sha256)
PY
}

prepare_shared_splits() {
    local waited=0
    while ! mkdir "$SPLIT_LOCK_DIR" 2>/dev/null; do
        if ((waited >= 3600)); then
            echo "ERROR: timed out waiting for shared-split preparation lock" >&2
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done
    SPLIT_LOCK_HELD=1

    echo "$(date -Is) SHARED_SPLIT_PREPARE start"
    if python -u "$SHARED_SPLIT_SCRIPT" 2>&1 | tee "$QUEUE_DIR/shared_split_prepare.log"; then
        echo "$(date -Is) SHARED_SPLIT_PREPARE complete"
        cleanup_split_lock
        return 0
    fi
    echo "$(date -Is) SHARED_SPLIT_PREPARE failed" >&2
    cleanup_split_lock
    return 1
}

run_queue() {
    if is_running && [[ "$(<"$PID_FILE")" != "$$" ]]; then
        echo "ERROR: matrix queue is already running with PID $(<"$PID_FILE")" >&2
        return 2
    fi

    echo "$$" >"$PID_FILE"
    trap 'cleanup_split_lock; rm -f "$PID_FILE"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    local total="${#CONFIGS[@]}"
    local position=0
    local config
    local name
    local hash
    local marker
    local marker_hash
    local started
    local failures=()

    echo "$(date -Is) QUEUE_START total_configs=$total"
    printf "timestamp\tstatus\tconfig\n" >>"$STATE_LOG"

    if ! prepare_shared_splits; then
        echo "$(date -Is) QUEUE_ABORT shared split preparation failed"
        return 1
    fi
    printf "%s\tSHARED_SPLITS_READY\t%s\n" "$(date -Is)" "$SHARED_SPLIT_SCRIPT" >>"$STATE_LOG"

    for config in "${CONFIGS[@]}"; do
        position=$((position + 1))
        name="$(basename "$config" .json)"
        marker="$DONE_DIR/${name}.done"

        if [[ ! -f "$config" ]]; then
            echo "$(date -Is) MISSING [$position/$total] $config"
            printf "%s\tMISSING\t%s\n" "$(date -Is)" "$config" >>"$STATE_LOG"
            failures+=("$config")
            if ((CONTINUE_ON_ERROR == 0)); then
                break
            fi
            continue
        fi

        hash="$(config_hash "$config")"
        marker_hash=""
        if [[ -f "$marker" ]]; then
            marker_hash="$(<"$marker")"
        fi
        if [[ "$marker_hash" == "$hash" ]]; then
            echo "$(date -Is) SKIP_COMPLETE [$position/$total] $config"
            printf "%s\tSKIP_COMPLETE\t%s\n" "$(date -Is)" "$config" >>"$STATE_LOG"
            continue
        fi

        started="$(date -Is)"
        echo "$started RUNNING [$position/$total] $config"
        printf "%s\tRUNNING\t%s\n" "$started" "$config" >>"$STATE_LOG"

        if PARALLEL_REFINE_CONFIG="$config" bash "$EXPERIMENT_RUNNER"; then
            printf "%s\n" "$hash" >"$marker"
            echo "$(date -Is) COMPLETE [$position/$total] $config"
            printf "%s\tCOMPLETE\t%s\n" "$(date -Is)" "$config" >>"$STATE_LOG"
        else
            echo "$(date -Is) FAILED [$position/$total] $config"
            printf "%s\tFAILED\t%s\n" "$(date -Is)" "$config" >>"$STATE_LOG"
            failures+=("$config")
            if ((CONTINUE_ON_ERROR == 0)); then
                break
            fi
        fi
    done

    if ((${#failures[@]})); then
        echo "$(date -Is) QUEUE_FINISHED_WITH_FAILURES count=${#failures[@]}"
        printf "  %s\n" "${failures[@]}"
        return 1
    fi

    echo "$(date -Is) QUEUE_COMPLETE total_configs=$total"
}

show_status() {
    if is_running; then
        echo "RUNNING pid=$(<"$PID_FILE")"
    else
        echo "NOT RUNNING"
    fi
    echo "Master log: $MASTER_LOG"
    echo "State log:  $STATE_LOG"
    echo
    if [[ -f "$MASTER_LOG" ]]; then
        tail -n 25 "$MASTER_LOG"
    fi
}

MODE="${1:-help}"
case "$MODE" in
    start)
        if is_running; then
            echo "Matrix queue is already running with PID $(<"$PID_FILE")"
            exit 2
        fi
        nohup bash "$QUEUE_SCRIPT" run >>"$MASTER_LOG" 2>&1 &
        queue_pid="$!"
        echo "$queue_pid" >"$PID_FILE"
        echo "Started Experiment 2 matrix queue in background."
        echo "PID: $queue_pid"
        echo "Log: $MASTER_LOG"
        ;;
    run)
        run_queue
        ;;
    prepare)
        prepare_shared_splits
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: bash scripts/run_experiment2_matrix_queue.sh {start|status|run|prepare}"
        echo
        echo "  start    launch the 11-config queue with nohup"
        echo "  status   show PID status and the latest master-log lines"
        echo "  run      run in the foreground (mainly for debugging)"
        echo "  prepare  generate/verify shared A-val/B/Y splits only"
        ;;
esac
