#!/usr/bin/env bash
# Generic end-to-end Parallel Refine runner.
#
# Usage:
#   1. Activate the SerialFlavour/GN2 environment.
#   2. Change only CONFIG below to select an experiment JSON, or set the
#      PARALLEL_REFINE_CONFIG environment variable from a queue script.
#   3. Run: bash scripts/run_parallel_refine_experiment.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT THIS LINE TO SELECT THE EXPERIMENT
# ---------------------------------------------------------------------------
CONFIG="${PARALLEL_REFINE_CONFIG:-configs/parallel_refine/experiments/experiment1/experiment1_p122k_a080_b020.json}"

# Physical GPU allocation slots. Seeds are assigned to these entries in order;
# if a config contains more seeds than slots, the list is reused cyclically.
GPU_SLOTS=(0 1 1 2 2)

cd "$(dirname "$0")/.."

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: experiment config not found: $CONFIG" >&2
    exit 1
fi

mapfile -t CONFIG_METADATA < <(
    python - "$CONFIG" <<'PY'
import sys

from src.parallel_refine.config import load_study_config

study = load_study_config(sys.argv[1])
print(study.study_name)
print(study.data["sizes"]["b_train"])
print(" ".join(str(run.seed) for run in study.seeds))
print(" ".join(study.refiners["recipes"]))
PY
)

EXPERIMENT_NAME="${CONFIG_METADATA[0]}"
B_TRAIN="${CONFIG_METADATA[1]}"
read -r -a SEEDS <<< "${CONFIG_METADATA[2]}"
read -r -a RECIPES <<< "${CONFIG_METADATA[3]}"

LOG_DIR="logs/parallel_refine/${EXPERIMENT_NAME}"
mkdir -p "$LOG_DIR"

PIDS=()
JOB_NAMES=()

gpu_for_index() {
    local index="$1"
    echo "${GPU_SLOTS[$((index % ${#GPU_SLOTS[@]}))]}"
}

launch_job() {
    local gpu="$1"
    local job_name="$2"
    local log_file="$3"
    shift 3
    echo "  launch ${job_name} on GPU ${gpu}; log=${log_file}"
    CUDA_VISIBLE_DEVICES="$gpu" "$@" >"$log_file" 2>&1 &
    PIDS+=("$!")
    JOB_NAMES+=("$job_name")
}

wait_for_jobs() {
    local failed=0
    local index
    for index in "${!PIDS[@]}"; do
        if wait "${PIDS[$index]}"; then
            echo "  complete ${JOB_NAMES[$index]}"
        else
            echo "  FAILED ${JOB_NAMES[$index]}" >&2
            failed=1
        fi
    done
    PIDS=()
    JOB_NAMES=()
    if ((failed)); then
        return 1
    fi
}

echo "CONFIG: $CONFIG"
echo "EXPERIMENT: $EXPERIMENT_NAME"
echo "SEEDS: ${SEEDS[*]}"
echo "B_TRAIN: $B_TRAIN"
echo "RECIPES: ${RECIPES[*]}"
echo "LOG_DIR: $LOG_DIR"

echo "STAGE 1: prepare event-disjoint splits and processed caches"
PREPARE_ARGS=(
    python scripts/prepare_data.py
    --config "$CONFIG"
    --build-processed-caches
    --processed-split a_train
    --processed-split a_val
    --processed-split y_test
)
if ((B_TRAIN > 0)); then
    PREPARE_ARGS+=(--processed-split b_train --processed-split b_val)
fi
"${PREPARE_ARGS[@]}" 2>&1 | tee "$LOG_DIR/prepare_data.log"

echo "STAGE 2: train Parallel models"
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    gpu="$(gpu_for_index "$index")"
    launch_job "$gpu" "parallel_seed${seed}" \
        "$LOG_DIR/parallel_seed${seed}_gpu${gpu}.log" \
        python scripts/train_parallel.py --config "$CONFIG" \
        --seed "$seed" --skip-complete
done
wait_for_jobs

if ((B_TRAIN > 0)); then
    echo "STAGE 3: generate B-train/B-val frozen feature caches"
    for index in "${!SEEDS[@]}"; do
        seed="${SEEDS[$index]}"
        gpu="$(gpu_for_index "$index")"
        launch_job "$gpu" "b_cache_seed${seed}" \
            "$LOG_DIR/b_cache_seed${seed}_gpu${gpu}.log" \
            python scripts/generate_cache.py --config "$CONFIG" \
            --seed "$seed" --split b_train --split b_val
    done
    wait_for_jobs

    echo "STAGE 4: train configured DNN recipes"
    for recipe in "${RECIPES[@]}"; do
        echo "  recipe: $recipe"
        for index in "${!SEEDS[@]}"; do
            seed="${SEEDS[$index]}"
            gpu="$(gpu_for_index "$index")"
            launch_job "$gpu" "dnn_seed${seed}_${recipe}" \
                "$LOG_DIR/dnn_seed${seed}_${recipe}_gpu${gpu}.log" \
                python scripts/train_dnn.py --config "$CONFIG" \
                --seed "$seed" --recipe "$recipe" --skip-complete
        done
        wait_for_jobs
    done
else
    echo "STAGES 3-4: skipped because b_train=0 (Transformer-only route)"
fi

echo "STAGE 5: generate locked Y-test frozen feature caches"
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    gpu="$(gpu_for_index "$index")"
    launch_job "$gpu" "y_cache_seed${seed}" \
        "$LOG_DIR/y_cache_seed${seed}_gpu${gpu}.log" \
        python scripts/generate_cache.py --config "$CONFIG" \
        --seed "$seed" --split y_test
done
wait_for_jobs

if ((B_TRAIN > 0)); then
    EVALUATION_MODEL="parallel_dnn"
else
    EVALUATION_MODEL="parallel"
fi

echo "STAGE 6: locked Y-test evaluation (${EVALUATION_MODEL})"
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    gpu="$(gpu_for_index "$index")"
    launch_job "$gpu" "y_evaluate_seed${seed}" \
        "$LOG_DIR/y_evaluate_seed${seed}_gpu${gpu}.log" \
        python scripts/evaluate.py --config "$CONFIG" \
        --seed "$seed" --model "$EVALUATION_MODEL"
done
wait_for_jobs

echo "ALL STAGES COMPLETE. Logs: $LOG_DIR"
