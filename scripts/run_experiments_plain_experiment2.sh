#!/usr/bin/env bash
# Fully expanded single-configuration runner for Experiment 2.
# Change only CONFIG below, then execute stages manually by commenting
# out later sections when desired. Activate the SerialFlavour/GN2
# environment before running this script.

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT ONLY THIS LINE TO SELECT THE EXPERIMENT
# ---------------------------------------------------------------------------
CONFIG="configs/parallel_refine/experiments/experiment2/experiment2_p122k_a800k.json"

cd "$(dirname "$0")/.."
EXPERIMENT_NAME="$(basename "$CONFIG" .json)"
LOG_DIR="logs/parallel_refine/${EXPERIMENT_NAME}/plain"
mkdir -p "$LOG_DIR"

echo "CONFIG: $CONFIG"
echo "LOG_DIR: $LOG_DIR"

echo "STAGE 0: prepare or verify shared Experiment 2 split anchor"
python -u scripts/prepare_experiment2_shared_splits.py \
    2>&1 | tee "$LOG_DIR/shared_split_prepare.log"

# CACHE_VERSION creates a new field-wise mmap generation automatically; do not
# pass --force here. Old processed .npz files remain untouched for rollback.
echo "STAGE 1: build versioned mmap A/B/Y processed caches before parallel training"
python scripts/prepare_data.py \
    --config "$CONFIG" \
    --build-processed-caches \
    --processed-split a_train \
    --processed-split a_val \
    --processed-split b_train \
    --processed-split b_val \
    --processed-split y_test \
    2>&1 | tee "$LOG_DIR/prepare_data.log"

echo "STAGE 2: train five Parallel models"
CUDA_VISIBLE_DEVICES=0 python scripts/train_parallel.py --config "$CONFIG" --seed 1 --skip-complete >"$LOG_DIR/parallel_seed1_gpu0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_parallel.py --config "$CONFIG" --seed 2 --skip-complete >"$LOG_DIR/parallel_seed2_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_parallel.py --config "$CONFIG" --seed 3 --skip-complete >"$LOG_DIR/parallel_seed3_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_parallel.py --config "$CONFIG" --seed 4 --skip-complete >"$LOG_DIR/parallel_seed4_gpu2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_parallel.py --config "$CONFIG" --seed 5 --skip-complete >"$LOG_DIR/parallel_seed5_gpu2.log" 2>&1 &
wait

echo "STAGE 3: generate B-train/B-val frozen feature caches"
CUDA_VISIBLE_DEVICES=0 python scripts/generate_cache.py --config "$CONFIG" --seed 1 --split b_train --split b_val >"$LOG_DIR/b_cache_seed1_gpu0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config "$CONFIG" --seed 2 --split b_train --split b_val >"$LOG_DIR/b_cache_seed2_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config "$CONFIG" --seed 3 --split b_train --split b_val >"$LOG_DIR/b_cache_seed3_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config "$CONFIG" --seed 4 --split b_train --split b_val >"$LOG_DIR/b_cache_seed4_gpu2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config "$CONFIG" --seed 5 --split b_train --split b_val >"$LOG_DIR/b_cache_seed5_gpu2.log" 2>&1 &
wait

echo "STAGE 4: train F0_aux DNN"
CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config "$CONFIG" --seed 1 --recipe F0_aux --skip-complete >"$LOG_DIR/dnn_seed1_F0_aux_gpu0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 2 --recipe F0_aux --skip-complete >"$LOG_DIR/dnn_seed2_F0_aux_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 3 --recipe F0_aux --skip-complete >"$LOG_DIR/dnn_seed3_F0_aux_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 4 --recipe F0_aux --skip-complete >"$LOG_DIR/dnn_seed4_F0_aux_gpu2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 5 --recipe F0_aux --skip-complete >"$LOG_DIR/dnn_seed5_F0_aux_gpu2.log" 2>&1 &


echo "STAGE 4: train F1_embed DNN"
CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config "$CONFIG" --seed 1 --recipe F1_embed --skip-complete >"$LOG_DIR/dnn_seed1_F1_embed_gpu0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 2 --recipe F1_embed --skip-complete >"$LOG_DIR/dnn_seed2_F1_embed_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 3 --recipe F1_embed --skip-complete >"$LOG_DIR/dnn_seed3_F1_embed_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 4 --recipe F1_embed --skip-complete >"$LOG_DIR/dnn_seed4_F1_embed_gpu2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 5 --recipe F1_embed --skip-complete >"$LOG_DIR/dnn_seed5_F1_embed_gpu2.log" 2>&1 &
wait

echo "STAGE 4: train F2_jet_aux DNN"
CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config "$CONFIG" --seed 1 --recipe F2_jet_aux --skip-complete >"$LOG_DIR/dnn_seed1_F2_jet_aux_gpu0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 2 --recipe F2_jet_aux --skip-complete >"$LOG_DIR/dnn_seed2_F2_jet_aux_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 3 --recipe F2_jet_aux --skip-complete >"$LOG_DIR/dnn_seed3_F2_jet_aux_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 4 --recipe F2_jet_aux --skip-complete >"$LOG_DIR/dnn_seed4_F2_jet_aux_gpu2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 5 --recipe F2_jet_aux --skip-complete >"$LOG_DIR/dnn_seed5_F2_jet_aux_gpu2.log" 2>&1 &


echo "STAGE 4: train F3_embed_aux DNN"
CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config "$CONFIG" --seed 1 --recipe F3_embed_aux --skip-complete >"$LOG_DIR/dnn_seed1_F3_embed_aux_gpu0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 2 --recipe F3_embed_aux --skip-complete >"$LOG_DIR/dnn_seed2_F3_embed_aux_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 3 --recipe F3_embed_aux --skip-complete >"$LOG_DIR/dnn_seed3_F3_embed_aux_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 4 --recipe F3_embed_aux --skip-complete >"$LOG_DIR/dnn_seed4_F3_embed_aux_gpu2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 5 --recipe F3_embed_aux --skip-complete >"$LOG_DIR/dnn_seed5_F3_embed_aux_gpu2.log" 2>&1 &
wait

echo "STAGE 4: train F4_all DNN"
CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config "$CONFIG" --seed 1 --recipe F4_all --skip-complete >"$LOG_DIR/dnn_seed1_F4_all_gpu0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 2 --recipe F4_all --skip-complete >"$LOG_DIR/dnn_seed2_F4_all_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config "$CONFIG" --seed 3 --recipe F4_all --skip-complete >"$LOG_DIR/dnn_seed3_F4_all_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 4 --recipe F4_all --skip-complete >"$LOG_DIR/dnn_seed4_F4_all_gpu2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config "$CONFIG" --seed 5 --recipe F4_all --skip-complete >"$LOG_DIR/dnn_seed5_F4_all_gpu2.log" 2>&1 &
wait

echo "STAGE 5: generate locked Y-test frozen feature caches"
CUDA_VISIBLE_DEVICES=0 python scripts/generate_cache.py --config "$CONFIG" --seed 1 --split y_test >"$LOG_DIR/y_cache_seed1_gpu0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config "$CONFIG" --seed 2 --split y_test >"$LOG_DIR/y_cache_seed2_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config "$CONFIG" --seed 3 --split y_test >"$LOG_DIR/y_cache_seed3_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config "$CONFIG" --seed 4 --split y_test >"$LOG_DIR/y_cache_seed4_gpu2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config "$CONFIG" --seed 5 --split y_test >"$LOG_DIR/y_cache_seed5_gpu2.log" 2>&1 &
wait

echo "STAGE 6: final locked Y-test evaluation, including origin/pair plots"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config "$CONFIG" --seed 1 --model parallel_dnn >"$LOG_DIR/y_evaluate_seed1_gpu0.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --config "$CONFIG" --seed 2 --model parallel_dnn >"$LOG_DIR/y_evaluate_seed2_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --config "$CONFIG" --seed 3 --model parallel_dnn >"$LOG_DIR/y_evaluate_seed3_gpu1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py --config "$CONFIG" --seed 4 --model parallel_dnn >"$LOG_DIR/y_evaluate_seed4_gpu2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py --config "$CONFIG" --seed 5 --model parallel_dnn >"$LOG_DIR/y_evaluate_seed5_gpu2.log" 2>&1 &
wait

echo "ALL STAGES COMPLETE. Logs: $LOG_DIR"
