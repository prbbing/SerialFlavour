#!/usr/bin/env bash
# Plain, manually staged Parallel Refine runner.
#
# Change only CONFIG to select another experiment. Commands for all five seeds
# are intentionally written out explicitly so that individual lines or blocks
# can also be copied and run by hand.
#
# Examples:
#   bash scripts/run_parallel_refine_experiment_plain.sh prepare
#   bash scripts/run_parallel_refine_experiment_plain.sh parallel
#   bash scripts/run_parallel_refine_experiment_plain.sh b-cache
#   bash scripts/run_parallel_refine_experiment_plain.sh dnn-f1
#   bash scripts/run_parallel_refine_experiment_plain.sh y-cache
#   bash scripts/run_parallel_refine_experiment_plain.sh evaluate
#   bash scripts/run_parallel_refine_experiment_plain.sh all

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT THIS LINE TO SELECT THE EXPERIMENT
# ---------------------------------------------------------------------------
CONFIG="configs/parallel_refine/experiments/experiment1/experiment1_p122k_a080_b020.json"

STAGE="${1:-help}"

cd "$(dirname "$0")/.."

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: experiment config not found: $CONFIG" >&2
    exit 1
fi

EXPERIMENT_NAME="$(
    python - "$CONFIG" <<'PY'
import sys
from src.parallel_refine.config import load_study_config
print(load_study_config(sys.argv[1]).study_name)
PY
)"

B_TRAIN="$(
    python - "$CONFIG" <<'PY'
import sys
from src.parallel_refine.config import load_study_config
print(load_study_config(sys.argv[1]).data["sizes"]["b_train"])
PY
)"

LOG_DIR="logs/parallel_refine/${EXPERIMENT_NAME}/plain"
mkdir -p "$LOG_DIR"

wait_for_five_jobs() {
    local failed=0
    local completed
    for completed in 1 2 3 4 5; do
        if ! wait -n; then
            failed=1
        fi
    done
    if ((failed)); then
        echo "ERROR: one or more background jobs failed; inspect $LOG_DIR" >&2
        return 1
    fi
}

if [[ "$STAGE" == "help" ]]; then
    echo "Usage: bash scripts/run_parallel_refine_experiment_plain.sh STAGE"
    echo
    echo "Available stages:"
    echo "  prepare"
    echo "  parallel"
    echo "  b-cache"
    echo "  dnn-f0"
    echo "  dnn-f1"
    echo "  dnn-f2"
    echo "  dnn-f3"
    echo "  dnn-f4"
    echo "  y-cache"
    echo "  evaluate"
    echo "  all"
    echo
    echo "CONFIG: $CONFIG"
    echo "EXPERIMENT: $EXPERIMENT_NAME"
    echo "B_TRAIN: $B_TRAIN"
    exit 0
fi

echo "CONFIG: $CONFIG"
echo "EXPERIMENT: $EXPERIMENT_NAME"
echo "SELECTED STAGE: $STAGE"
echo "LOG_DIR: $LOG_DIR"

if [[ "$STAGE" == "prepare" || "$STAGE" == "all" ]]; then
    echo "STAGE: prepare A/B/Y processed caches"
    if ((B_TRAIN > 0)); then
        python scripts/prepare_data.py \
            --config "$CONFIG" \
            --build-processed-caches \
            --processed-split a_train \
            --processed-split a_val \
            --processed-split b_train \
            --processed-split b_val \
            --processed-split y_test \
            2>&1 | tee "$LOG_DIR/prepare_data.log"
    else
        python scripts/prepare_data.py \
            --config "$CONFIG" \
            --build-processed-caches \
            --processed-split a_train \
            --processed-split a_val \
            --processed-split y_test \
            2>&1 | tee "$LOG_DIR/prepare_data.log"
    fi
fi

if [[ "$STAGE" == "parallel" || "$STAGE" == "all" ]]; then
    echo "STAGE: train five Parallel models"
    CUDA_VISIBLE_DEVICES=0 python scripts/train_parallel.py --config "$CONFIG" --seed 1 --skip-complete >"$LOG_DIR/parallel_seed1_gpu0.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python scripts/train_parallel.py --config "$CONFIG" --seed 2 --skip-complete >"$LOG_DIR/parallel_seed2_gpu1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python scripts/train_parallel.py --config "$CONFIG" --seed 3 --skip-complete >"$LOG_DIR/parallel_seed3_gpu1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python scripts/train_parallel.py --config "$CONFIG" --seed 4 --skip-complete >"$LOG_DIR/parallel_seed4_gpu2.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python scripts/train_parallel.py --config "$CONFIG" --seed 5 --skip-complete >"$LOG_DIR/parallel_seed5_gpu2.log" 2>&1 &
    wait_for_five_jobs
fi

if [[ "$STAGE" == "b-cache" || "$STAGE" == "all" ]]; then
    if ((B_TRAIN == 0)); then
        echo "STAGE: B cache skipped because b_train=0"
    else
        echo "STAGE: generate B-train/B-val frozen feature caches"
        CUDA_VISIBLE_DEVICES=0 python scripts/generate_cache.py --config "$CONFIG" --seed 1 --split b_train --split b_val >"$LOG_DIR/b_cache_seed1_gpu0.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config "$CONFIG" --seed 2 --split b_train --split b_val >"$LOG_DIR/b_cache_seed2_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config "$CONFIG" --seed 3 --split b_train --split b_val >"$LOG_DIR/b_cache_seed3_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config "$CONFIG" --seed 4 --split b_train --split b_val >"$LOG_DIR/b_cache_seed4_gpu2.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config "$CONFIG" --seed 5 --split b_train --split b_val >"$LOG_DIR/b_cache_seed5_gpu2.log" 2>&1 &
        wait_for_five_jobs
    fi
fi

if [[ "$STAGE" == "dnn-f0" || "$STAGE" == "all" ]]; then
    if ((B_TRAIN > 0)); then
        echo "STAGE: train F0_aux DNN"
        CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config "$CONFIG" --seed 1 --recipe F0_aux --skip-complete >"$LOG_DIR/dnn_seed1_F0_aux_gpu0.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 2 --recipe F0_aux --skip-complete >"$LOG_DIR/dnn_seed2_F0_aux_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 3 --recipe F0_aux --skip-complete >"$LOG_DIR/dnn_seed3_F0_aux_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 4 --recipe F0_aux --skip-complete >"$LOG_DIR/dnn_seed4_F0_aux_gpu2.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 5 --recipe F0_aux --skip-complete >"$LOG_DIR/dnn_seed5_F0_aux_gpu2.log" 2>&1 &
        wait_for_five_jobs
    fi
fi

if [[ "$STAGE" == "dnn-f1" || "$STAGE" == "all" ]]; then
    if ((B_TRAIN > 0)); then
        echo "STAGE: train F1_embed DNN"
        CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config "$CONFIG" --seed 1 --recipe F1_embed --skip-complete >"$LOG_DIR/dnn_seed1_F1_embed_gpu0.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 2 --recipe F1_embed --skip-complete >"$LOG_DIR/dnn_seed2_F1_embed_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 3 --recipe F1_embed --skip-complete >"$LOG_DIR/dnn_seed3_F1_embed_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 4 --recipe F1_embed --skip-complete >"$LOG_DIR/dnn_seed4_F1_embed_gpu2.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 5 --recipe F1_embed --skip-complete >"$LOG_DIR/dnn_seed5_F1_embed_gpu2.log" 2>&1 &
        wait_for_five_jobs
    fi
fi

if [[ "$STAGE" == "dnn-f2" || "$STAGE" == "all" ]]; then
    if ((B_TRAIN > 0)); then
        echo "STAGE: train F2_jet_aux DNN"
        CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config "$CONFIG" --seed 1 --recipe F2_jet_aux --skip-complete >"$LOG_DIR/dnn_seed1_F2_jet_aux_gpu0.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 2 --recipe F2_jet_aux --skip-complete >"$LOG_DIR/dnn_seed2_F2_jet_aux_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 3 --recipe F2_jet_aux --skip-complete >"$LOG_DIR/dnn_seed3_F2_jet_aux_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 4 --recipe F2_jet_aux --skip-complete >"$LOG_DIR/dnn_seed4_F2_jet_aux_gpu2.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 5 --recipe F2_jet_aux --skip-complete >"$LOG_DIR/dnn_seed5_F2_jet_aux_gpu2.log" 2>&1 &
        wait_for_five_jobs
    fi
fi

if [[ "$STAGE" == "dnn-f3" || "$STAGE" == "all" ]]; then
    if ((B_TRAIN > 0)); then
        echo "STAGE: train F3_embed_aux DNN"
        CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config "$CONFIG" --seed 1 --recipe F3_embed_aux --skip-complete >"$LOG_DIR/dnn_seed1_F3_embed_aux_gpu0.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 2 --recipe F3_embed_aux --skip-complete >"$LOG_DIR/dnn_seed2_F3_embed_aux_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 3 --recipe F3_embed_aux --skip-complete >"$LOG_DIR/dnn_seed3_F3_embed_aux_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 4 --recipe F3_embed_aux --skip-complete >"$LOG_DIR/dnn_seed4_F3_embed_aux_gpu2.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 5 --recipe F3_embed_aux --skip-complete >"$LOG_DIR/dnn_seed5_F3_embed_aux_gpu2.log" 2>&1 &
        wait_for_five_jobs
    fi
fi

if [[ "$STAGE" == "dnn-f4" || "$STAGE" == "all" ]]; then
    if ((B_TRAIN > 0)); then
        echo "STAGE: train F4_all DNN"
        CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config "$CONFIG" --seed 1 --recipe F4_all --skip-complete >"$LOG_DIR/dnn_seed1_F4_all_gpu0.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 2 --recipe F4_all --skip-complete >"$LOG_DIR/dnn_seed2_F4_all_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 3 --recipe F4_all --skip-complete >"$LOG_DIR/dnn_seed3_F4_all_gpu1.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 4 --recipe F4_all --skip-complete >"$LOG_DIR/dnn_seed4_F4_all_gpu2.log" 2>&1 &
        CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 5 --recipe F4_all --skip-complete >"$LOG_DIR/dnn_seed5_F4_all_gpu2.log" 2>&1 &
        wait_for_five_jobs
    fi
fi

if [[ "$STAGE" == "y-cache" || "$STAGE" == "all" ]]; then
    echo "STAGE: generate locked Y-test frozen feature caches"
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_cache.py --config "$CONFIG" --seed 1 --split y_test >"$LOG_DIR/y_cache_seed1_gpu0.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config "$CONFIG" --seed 2 --split y_test >"$LOG_DIR/y_cache_seed2_gpu1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config "$CONFIG" --seed 3 --split y_test >"$LOG_DIR/y_cache_seed3_gpu1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config "$CONFIG" --seed 4 --split y_test >"$LOG_DIR/y_cache_seed4_gpu2.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config "$CONFIG" --seed 5 --split y_test >"$LOG_DIR/y_cache_seed5_gpu2.log" 2>&1 &
    wait_for_five_jobs
fi

if [[ "$STAGE" == "evaluate" || "$STAGE" == "all" ]]; then
    if ((B_TRAIN > 0)); then
        EVALUATION_MODEL="parallel_dnn"
    else
        EVALUATION_MODEL="parallel"
    fi
    echo "STAGE: locked Y-test evaluation (${EVALUATION_MODEL})"
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config "$CONFIG" --seed 1 --model "$EVALUATION_MODEL" >"$LOG_DIR/y_evaluate_seed1_gpu0.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --config "$CONFIG" --seed 2 --model "$EVALUATION_MODEL" >"$LOG_DIR/y_evaluate_seed2_gpu1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --config "$CONFIG" --seed 3 --model "$EVALUATION_MODEL" >"$LOG_DIR/y_evaluate_seed3_gpu1.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py --config "$CONFIG" --seed 4 --model "$EVALUATION_MODEL" >"$LOG_DIR/y_evaluate_seed4_gpu2.log" 2>&1 &
    CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py --config "$CONFIG" --seed 5 --model "$EVALUATION_MODEL" >"$LOG_DIR/y_evaluate_seed5_gpu2.log" 2>&1 &
    wait_for_five_jobs
fi

case "$STAGE" in
    prepare|parallel|b-cache|dnn-f0|dnn-f1|dnn-f2|dnn-f3|dnn-f4|y-cache|evaluate|all)
        ;;
    *)
        echo "ERROR: unknown stage: $STAGE" >&2
        exit 2
        ;;
esac

echo "SELECTED STAGE COMPLETE: $STAGE"
echo "Logs: $LOG_DIR"
