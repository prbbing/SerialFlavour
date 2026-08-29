#!/usr/bin/env bash
# Parallel Refine end-to-end runner.
# Activate the SerialFlavour/GN2 environment before running this script.

set -euo pipefail

cd "$(dirname "$0")/.."

# =============================================================================
# User configuration
# =============================================================================

PYTHON=python
CONFIG="parallel_refine/configs/experiments/default.json"

# Hardware scheduling belongs here, not in the experiment config. SEEDS and
# GPU_IDS are positional pairs and must have the same length.
SEEDS=(1 2 3 4 5)
GPU_IDS=(0 1 1 2 2)

# Empty means all recipes configured in CONFIG.
# Example: RECIPES=(F2_jet_aux F3_embed_aux F4_all).
RECIPES=()

# Maximum number of simultaneous downstream Stage-4 jobs.
MAX_JOBS=12

RUN_PREPARE_DATA=true
# Build the seed-independent A/B/Y processed caches once before multi-seed jobs
# start. Building Y here is preprocessing only; model inference waits until the
# locked-Y stages at the end of the script.
BUILD_PROCESSED_CACHES=true
PROCESSED_SPLITS=(a_train a_val b_train b_val y_test)
RUN_PARALLEL_TRAINING=true
RUN_B_CACHE=true
RUN_DNN=true
# Retained only for reproducing earlier BDT studies; disabled by default.
RUN_XGBOOST=false

# The default is the complete locked workflow, including Y cache generation and
# final evaluation. For pilot/tuning runs, set both Y switches to false before
# launching; turn them on only after every A/B choice is frozen.
RUN_Y_CACHE=true
RUN_Y_EVALUATION=true
EVALUATION_MODEL=direct_dnn  # direct, dnn, direct_dnn, bdt, or all

LOG_ROOT="parallel_refine/logs"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}"

# =============================================================================
# Helpers
# =============================================================================

section() {
    echo
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
}

wait_for_jobs() {
    local failures=0
    local index

    for index in "${!pids[@]}"; do
        if wait "${pids[$index]}"; then
            echo "[DONE] ${labels[$index]}"
        else
            echo "[FAILED] ${labels[$index]} -- ${logs[$index]}" >&2
            failures=$((failures + 1))
        fi
    done

    pids=()
    labels=()
    logs=()

    if ((failures > 0)); then
        echo "ERROR: ${failures} background job(s) failed." >&2
        exit 1
    fi
}

wait_if_job_batch_full() {
    if ((${#pids[@]} >= MAX_JOBS)); then
        echo "[WAIT] Stage-4 batch reached MAX_JOBS=${MAX_JOBS}"
        wait_for_jobs
    fi
}

run_on_visible_gpu() {
    local gpu_id="$1"
    shift

    if [[ "${gpu_id}" == "-1" ]]; then
        "$@"
    else
        CUDA_VISIBLE_DEVICES="${gpu_id}" "$@"
    fi
}

# Validate that every scheduled seed exists in the experiment config.
"${PYTHON}" - "${CONFIG}" "${SEEDS[@]}" <<'PY'
import sys

from src.parallel_refine.config import load_study_config

study = load_study_config(sys.argv[1])
study.selected_seeds(int(value) for value in sys.argv[2:])
PY

if ((${#SEEDS[@]} == 0)); then
    echo "ERROR: SEEDS cannot be empty." >&2
    exit 1
fi
if ((${#SEEDS[@]} != ${#GPU_IDS[@]})); then
    echo "ERROR: SEEDS and GPU_IDS must have the same length." >&2
    exit 1
fi
if [[ ! "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_JOBS must be a positive integer." >&2
    exit 1
fi

configured_runs=()
for index in "${!SEEDS[@]}"; do
    configured_runs+=("${SEEDS[$index]}"$'\t'"${GPU_IDS[$index]}")
done

selected_recipes=("${RECIPES[@]}")
if ((${#selected_recipes[@]} == 0)); then
    mapfile -t selected_recipes < <(
        "${PYTHON}" - "${CONFIG}" <<'PY'
import sys

from src.parallel_refine.config import load_study_config

for recipe in load_study_config(sys.argv[1]).refiners["recipes"]:
    print(recipe)
PY
    )
fi
if ((${#selected_recipes[@]} == 0)); then
    echo "ERROR: no downstream recipes were selected." >&2
    exit 1
fi

recipe_args=()
for recipe in "${selected_recipes[@]}"; do
    recipe_args+=(--recipe "${recipe}")
done

prepare_args=()
if [[ "${BUILD_PROCESSED_CACHES}" == true ]]; then
    prepare_args+=(--build-processed-caches)
    for split in "${PROCESSED_SPLITS[@]}"; do
        prepare_args+=(--processed-split "${split}")
    done
fi

case "${EVALUATION_MODEL}" in
    direct|dnn|direct_dnn|bdt|all) ;;
    *)
        echo "ERROR: EVALUATION_MODEL must be direct, dnn, direct_dnn, bdt, or all." >&2
        exit 1
        ;;
esac

mkdir -p "${LOG_DIR}"

echo "CONFIG=${CONFIG}"
echo "LOG_DIR=${LOG_DIR}"
echo "BUILD_PROCESSED_CACHES=${BUILD_PROCESSED_CACHES}"
echo "PROCESSED_SPLITS=${PROCESSED_SPLITS[*]}"
echo "RUN_Y_CACHE=${RUN_Y_CACHE}"
echo "RUN_Y_EVALUATION=${RUN_Y_EVALUATION} model=${EVALUATION_MODEL}"
echo "DOWNSTREAM_RECIPES=${selected_recipes[*]} MAX_JOBS=${MAX_JOBS}"
echo "Selected seed/GPU pairs:"
for entry in "${configured_runs[@]}"; do
    IFS=$'\t' read -r seed gpu_id <<< "${entry}"
    echo "  seed=${seed} gpu=${gpu_id}"
done

# =============================================================================
# Stage 1: event-disjoint A/B/Y indices
# =============================================================================

if [[ "${RUN_PREPARE_DATA}" == true ]]; then
    section "STAGE 1: PREPARE OR VALIDATE A/B/Y SPLITS"
    "${PYTHON}" scripts/prepare_data.py \
        --config "${CONFIG}" \
        "${prepare_args[@]}" \
        2>&1 | tee "${LOG_DIR}/prepare_data.log"
fi

# =============================================================================
# Stage 2: one Parallel training process per configured seed/GPU
# =============================================================================

if [[ "${RUN_PARALLEL_TRAINING}" == true ]]; then
    section "STAGE 2: TRAIN PARALLEL SEEDS IN PARALLEL"
    pids=()
    labels=()
    logs=()

    for entry in "${configured_runs[@]}"; do
        IFS=$'\t' read -r seed gpu_id <<< "${entry}"
        log="${LOG_DIR}/parallel_seed${seed}_gpu${gpu_id}.log"
        echo "[START] Parallel seed=${seed} gpu=${gpu_id} -- ${log}"
        run_on_visible_gpu "${gpu_id}" \
            "${PYTHON}" scripts/train_parallel.py \
            --config "${CONFIG}" \
            --seed "${seed}" \
            --skip-complete \
            >"${log}" 2>&1 &
        pids+=("$!")
        labels+=("Parallel seed=${seed} gpu=${gpu_id}")
        logs+=("${log}")
    done

    wait_for_jobs
fi

# =============================================================================
# Stage 3: B-only frozen feature caches, one process per seed/GPU
# =============================================================================

if [[ "${RUN_B_CACHE}" == true ]]; then
    section "STAGE 3: GENERATE B-TRAIN/B-VAL CACHES IN PARALLEL"
    pids=()
    labels=()
    logs=()

    for entry in "${configured_runs[@]}"; do
        IFS=$'\t' read -r seed gpu_id <<< "${entry}"
        log="${LOG_DIR}/b_cache_seed${seed}_gpu${gpu_id}.log"
        echo "[START] B cache seed=${seed} gpu=${gpu_id} -- ${log}"
        run_on_visible_gpu "${gpu_id}" \
            "${PYTHON}" scripts/generate_cache.py \
            --config "${CONFIG}" \
            --seed "${seed}" \
            --split b_train \
            --split b_val \
            >"${log}" 2>&1 &
        pids+=("$!")
        labels+=("B cache seed=${seed} gpu=${gpu_id}")
        logs+=("${log}")
    done

    wait_for_jobs
fi

# =============================================================================
# Stage 4: one independent job per (seed, recipe, model type), with at most
# MAX_JOBS processes per batch. The default workflow trains DNN only.
# =============================================================================

if [[ "${RUN_DNN}" == true || "${RUN_XGBOOST}" == true ]]; then
    section "STAGE 4: TRAIN ENABLED DOWNSTREAM REFINERS (MAX_JOBS=${MAX_JOBS})"
    pids=()
    labels=()
    logs=()

    for recipe in "${selected_recipes[@]}"; do
        for entry in "${configured_runs[@]}"; do
            IFS=$'\t' read -r seed gpu_id <<< "${entry}"

            if [[ "${RUN_DNN}" == true ]]; then
                log="${LOG_DIR}/dnn_seed${seed}_${recipe}_gpu${gpu_id}.log"
                echo "[START] DNN seed=${seed} recipe=${recipe} gpu=${gpu_id} -- ${log}"
                run_on_visible_gpu "${gpu_id}" \
                    "${PYTHON}" scripts/train_dnn.py \
                    --config "${CONFIG}" \
                    --seed "${seed}" \
                    --recipe "${recipe}" \
                    --skip-complete \
                    >"${log}" 2>&1 &
                pids+=("$!")
                labels+=("DNN seed=${seed} recipe=${recipe} gpu=${gpu_id}")
                logs+=("${log}")
                wait_if_job_batch_full
            fi

            if [[ "${RUN_XGBOOST}" == true ]]; then
                log="${LOG_DIR}/xgboost_seed${seed}_${recipe}_gpu${gpu_id}.log"
                echo "[START] XGBoost seed=${seed} recipe=${recipe} gpu=${gpu_id} -- ${log}"
                run_on_visible_gpu "${gpu_id}" \
                    "${PYTHON}" scripts/train_bdt.py \
                    --config "${CONFIG}" \
                    --seed "${seed}" \
                    --recipe "${recipe}" \
                    --skip-complete \
                    >"${log}" 2>&1 &
                pids+=("$!")
                labels+=("XGBoost seed=${seed} recipe=${recipe} gpu=${gpu_id}")
                logs+=("${log}")
                wait_if_job_batch_full
            fi
        done
    done

    if ((${#pids[@]} > 0)); then
        wait_for_jobs
    fi
fi

# =============================================================================
# Stage 5: checkpoint-bound locked-Y feature caches. This is the first stage
# that runs trained models on Y; it starts only after all A/B choices are fixed.
# =============================================================================

if [[ "${RUN_Y_CACHE}" == true ]]; then
    section "STAGE 5: GENERATE CHECKPOINT-BOUND LOCKED Y FEATURE CACHES"
    pids=()
    labels=()
    logs=()

    for entry in "${configured_runs[@]}"; do
        IFS=$'\t' read -r seed gpu_id <<< "${entry}"
        log="${LOG_DIR}/y_cache_seed${seed}_gpu${gpu_id}.log"
        echo "[START] Y cache seed=${seed} gpu=${gpu_id} -- ${log}"
        run_on_visible_gpu "${gpu_id}" \
            "${PYTHON}" scripts/generate_cache.py \
            --config "${CONFIG}" \
            --seed "${seed}" \
            --split y_test \
            >"${log}" 2>&1 &
        pids+=("$!")
        labels+=("Y cache seed=${seed} gpu=${gpu_id}")
        logs+=("${log}")
    done

    wait_for_jobs
fi

# =============================================================================
# Stage 6: final locked-Y evaluation. The default direct_dnn mode evaluates
# direct Parallel (including origin/pair diagnostics and plots) and every DNN.
# =============================================================================

if [[ "${RUN_Y_EVALUATION}" == true ]]; then
    section "STAGE 6: EVALUATE LOCKED Y IN PARALLEL OVER SEEDS"
    pids=()
    labels=()
    logs=()

    for entry in "${configured_runs[@]}"; do
        IFS=$'\t' read -r seed gpu_id <<< "${entry}"
        log="${LOG_DIR}/y_evaluate_seed${seed}_gpu${gpu_id}.log"
        echo "[START] Y evaluation seed=${seed} gpu=${gpu_id} -- ${log}"
        run_on_visible_gpu "${gpu_id}" \
            "${PYTHON}" scripts/evaluate.py \
            --config "${CONFIG}" \
            --seed "${seed}" \
            --model "${EVALUATION_MODEL}" \
            "${recipe_args[@]}" \
            >"${log}" 2>&1 &
        pids+=("$!")
        labels+=("Y evaluation seed=${seed} gpu=${gpu_id}")
        logs+=("${log}")
    done

    wait_for_jobs
fi

if [[ "${RUN_Y_CACHE}" != true && "${RUN_Y_EVALUATION}" != true ]]; then
    section "B-STAGE WORKFLOW COMPLETE"
    echo "RUN_Y_CACHE=false and RUN_Y_EVALUATION=false: no Y model inference or evaluation was run."
fi

section "ALL REQUESTED STAGES COMPLETE"
echo "Logs: ${LOG_DIR}"
