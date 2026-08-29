#!/usr/bin/env bash
# Background queue for the complete Experiment 1 matrix.
#
# The 20 experiment configs run sequentially. Inside each experiment, the
# generic runner still trains its configured seeds in parallel on GPUs
# 0,1,1,2,2. Successful configs receive hash-bound completion markers, so a
# restarted queue skips only configs whose resolved configuration is unchanged.

set -uo pipefail

# Keep running independent configs after a failed config. Set to 0 to stop at
# the first failure. No data-processing or experiment config is changed here.
CONTINUE_ON_ERROR=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUEUE_SCRIPT="$ROOT/scripts/run_experiment1_matrix_queue.sh"
EXPERIMENT_RUNNER="$ROOT/scripts/run_parallel_refine_experiment.sh"
QUEUE_DIR="$ROOT/logs/parallel_refine/experiment1_matrix_queue"
MASTER_LOG="$QUEUE_DIR/matrix_queue.log"
STATE_LOG="$QUEUE_DIR/matrix_queue_state.tsv"
PID_FILE="$QUEUE_DIR/matrix_queue.pid"
DONE_DIR="$QUEUE_DIR/completed"

CONFIGS=(
    # A/B = 100/0: Transformer-only baselines.
    "configs/parallel_refine/experiments/experiment1/experiment1_p023k_a100_b000.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p056k_a100_b000.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p086k_a100_b000.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p122k_a100_b000.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p160k_a100_b000.json"

    # A/B = 90/10.
    "configs/parallel_refine/experiments/experiment1/experiment1_p023k_a090_b010.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p056k_a090_b010.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p086k_a090_b010.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p122k_a090_b010.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p160k_a090_b010.json"

    # A/B = 80/20.
    "configs/parallel_refine/experiments/experiment1/experiment1_p023k_a080_b020.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p056k_a080_b020.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p086k_a080_b020.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p122k_a080_b020.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json"

    # A/B = 70/30.
    "configs/parallel_refine/experiments/experiment1/experiment1_p023k_a070_b030.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p056k_a070_b030.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p086k_a070_b030.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p122k_a070_b030.json"
    "configs/parallel_refine/experiments/experiment1/experiment1_p160k_a070_b030.json"
)

mkdir -p "$QUEUE_DIR" "$DONE_DIR"
cd "$ROOT"

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

run_queue() {
    if is_running && [[ "$(<"$PID_FILE")" != "$$" ]]; then
        echo "ERROR: matrix queue is already running with PID $(<"$PID_FILE")" >&2
        return 2
    fi

    echo "$$" >"$PID_FILE"
    trap 'rm -f "$PID_FILE"' EXIT
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
        echo "Started Experiment 1 matrix queue in background."
        echo "PID: $queue_pid"
        echo "Log: $MASTER_LOG"
        ;;
    run)
        run_queue
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: bash scripts/run_experiment1_matrix_queue.sh {start|status|run}"
        echo
        echo "  start   launch the 20-config queue with nohup"
        echo "  status  show PID status and the latest master-log lines"
        echo "  run     run in the foreground (mainly for debugging)"
        ;;
esac
