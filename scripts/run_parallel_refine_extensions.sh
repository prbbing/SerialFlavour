#!/usr/bin/env bash
# Run only added F1O/F1V/FG refiners against already-complete Parallel seeds.
# It intentionally never prepares data or retrains the upstream checkpoint.

set -euo pipefail

CONFIG="${PARALLEL_REFINE_CONFIG:-configs/parallel_refine/experiments/experiment1/experiment1_p122k_a080_b020_graph_extensions.json}"
GPU_SLOTS=(0 1 1 2 2)

cd "$(dirname "$0")/.."
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: extension config not found: $CONFIG" >&2
    exit 1
fi

mapfile -t CONFIG_METADATA < <(
    python - "$CONFIG" <<'PY'
import sys
from src.parallel_refine.config import load_study_config

study = load_study_config(sys.argv[1])
missing = [str(study.checkpoint(run)) for run in study.seeds
           if not study.checkpoint(run).is_file()]
if missing:
    raise SystemExit(
        "missing required upstream Parallel checkpoint(s):\n" + "\n".join(missing))
print(study.study_name)
print(study.upstream_experiment_name)
print(" ".join(str(run.seed) for run in study.seeds))
print(" ".join(study.refiners["recipes"]))
PY
)

EXPERIMENT_NAME="${CONFIG_METADATA[0]}"
UPSTREAM_NAME="${CONFIG_METADATA[1]}"
read -r -a SEEDS <<< "${CONFIG_METADATA[2]}"
read -r -a RECIPES <<< "${CONFIG_METADATA[3]}"
LOG_DIR="logs/parallel_refine/${EXPERIMENT_NAME}"
mkdir -p "$LOG_DIR"

PIDS=()
JOB_NAMES=()

gpu_for_index() {
    echo "${GPU_SLOTS[$(($1 % ${#GPU_SLOTS[@]}))]}"
}

launch_job() {
    local gpu="$1" job_name="$2" log_file="$3"
    shift 3
    echo "  launch ${job_name} on GPU ${gpu}; log=${log_file}"
    CUDA_VISIBLE_DEVICES="$gpu" "$@" >"$log_file" 2>&1 &
    PIDS+=("$!")
    JOB_NAMES+=("$job_name")
}

wait_for_jobs() {
    local failed=0 index
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
    ((failed == 0))
}

echo "CONFIG: $CONFIG"
echo "REFINER OUTPUT: $EXPERIMENT_NAME"
echo "UPSTREAM CHECKPOINT OWNER: $UPSTREAM_NAME"
echo "SEEDS: ${SEEDS[*]}"
echo "RECIPES: ${RECIPES[*]}"
echo "LOG_DIR: $LOG_DIR"

echo "STAGE 1: reuse/build B-train and B-val frozen caches"
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    gpu="$(gpu_for_index "$index")"
    launch_job "$gpu" "b_cache_seed${seed}" "$LOG_DIR/b_cache_seed${seed}_gpu${gpu}.log" \
        python scripts/generate_cache.py --config "$CONFIG" --seed "$seed" \
        --split b_train --split b_val
done
wait_for_jobs

echo "STAGE 2: train only configured extension refiners"
for recipe in "${RECIPES[@]}"; do
    echo "  recipe: $recipe"
    for index in "${!SEEDS[@]}"; do
        seed="${SEEDS[$index]}"
        gpu="$(gpu_for_index "$index")"
        case "$recipe" in
            FG0|FG1|FG2)
                launch_job "$gpu" "graph_seed${seed}_${recipe}" \
                    "$LOG_DIR/graph_seed${seed}_${recipe}_gpu${gpu}.log" \
                    python scripts/train_graph_refiner.py --config "$CONFIG" \
                    --seed "$seed" --recipe "$recipe" --skip-complete
                ;;
            *)
                launch_job "$gpu" "dnn_seed${seed}_${recipe}" \
                    "$LOG_DIR/dnn_seed${seed}_${recipe}_gpu${gpu}.log" \
                    python scripts/train_dnn.py --config "$CONFIG" \
                    --seed "$seed" --recipe "$recipe" --skip-complete
                ;;
        esac
    done
    wait_for_jobs
done

echo "STAGE 3: reuse/build locked Y-test frozen caches"
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    gpu="$(gpu_for_index "$index")"
    launch_job "$gpu" "y_cache_seed${seed}" "$LOG_DIR/y_cache_seed${seed}_gpu${gpu}.log" \
        python scripts/generate_cache.py --config "$CONFIG" --seed "$seed" --split y_test
done
wait_for_jobs

echo "STAGE 4: locked Y-test evaluation of extension refiners"
for index in "${!SEEDS[@]}"; do
    seed="${SEEDS[$index]}"
    gpu="$(gpu_for_index "$index")"
    launch_job "$gpu" "y_evaluate_seed${seed}" "$LOG_DIR/y_evaluate_seed${seed}_gpu${gpu}.log" \
        python scripts/evaluate.py --config "$CONFIG" --seed "$seed" --model parallel_dnn
done
wait_for_jobs

echo "EXTENSION STAGES COMPLETE. Logs: $LOG_DIR"
