#!/usr/bin/env bash
# Fully expanded end-to-end runner for Experiment 1, Parallel p122k, A/B=80/20.
# Activate the SerialFlavour/GN2 environment before running this script.
# Every command is written explicitly to make manual stage-by-stage execution easy.

set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p logs/parallel_refine/experiment1_p160k_a080_b020/plain

echo "STAGE 1: prepare A/B/Y processed caches"
python scripts/prepare_data.py \
    --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json \
    --build-processed-caches \
    --processed-split a_train \
    --processed-split a_val \
    --processed-split b_train \
    --processed-split b_val \
    --processed-split y_test \
    2>&1 | tee logs/parallel_refine/experiment1_p160k_a080_b020/plain/prepare_data.log

echo "STAGE 2: train five Parallel models"
CUDA_VISIBLE_DEVICES=0 python scripts/train_parallel.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 1 --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/parallel_seed1_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_parallel.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 2 --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/parallel_seed2_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_parallel.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 3 --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/parallel_seed3_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_parallel.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 4 --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/parallel_seed4_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_parallel.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 5 --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/parallel_seed5_gpu2.log 2>&1 &
wait

echo "STAGE 3: generate B-train/B-val frozen feature caches"
CUDA_VISIBLE_DEVICES=0 python scripts/generate_cache.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 1 --split b_train --split b_val >logs/parallel_refine/experiment1_p160k_a080_b020/plain/b_cache_seed1_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 2 --split b_train --split b_val >logs/parallel_refine/experiment1_p160k_a080_b020/plain/b_cache_seed2_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 3 --split b_train --split b_val >logs/parallel_refine/experiment1_p160k_a080_b020/plain/b_cache_seed3_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 4 --split b_train --split b_val >logs/parallel_refine/experiment1_p160k_a080_b020/plain/b_cache_seed4_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 5 --split b_train --split b_val >logs/parallel_refine/experiment1_p160k_a080_b020/plain/b_cache_seed5_gpu2.log 2>&1 &
wait

echo "STAGE 4A: train F0_aux DNN"
CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 1 --recipe F0_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed1_F0_aux_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 2 --recipe F0_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed2_F0_aux_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 3 --recipe F0_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed3_F0_aux_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 4 --recipe F0_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed4_F0_aux_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 5 --recipe F0_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed5_F0_aux_gpu2.log 2>&1 &
wait

echo "STAGE 4B: train F1_embed DNN"
CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 1 --recipe F1_embed --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed1_F1_embed_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 2 --recipe F1_embed --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed2_F1_embed_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 3 --recipe F1_embed --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed3_F1_embed_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 4 --recipe F1_embed --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed4_F1_embed_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 5 --recipe F1_embed --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed5_F1_embed_gpu2.log 2>&1 &
wait

echo "STAGE 4C: train F2_jet_aux DNN"
CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 1 --recipe F2_jet_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed1_F2_jet_aux_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 2 --recipe F2_jet_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed2_F2_jet_aux_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 3 --recipe F2_jet_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed3_F2_jet_aux_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 4 --recipe F2_jet_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed4_F2_jet_aux_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 5 --recipe F2_jet_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed5_F2_jet_aux_gpu2.log 2>&1 &
wait

echo "STAGE 4D: train F3_embed_aux DNN"
CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 1 --recipe F3_embed_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed1_F3_embed_aux_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 2 --recipe F3_embed_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed2_F3_embed_aux_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 3 --recipe F3_embed_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed3_F3_embed_aux_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 4 --recipe F3_embed_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed4_F3_embed_aux_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 5 --recipe F3_embed_aux --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed5_F3_embed_aux_gpu2.log 2>&1 &
wait

echo "STAGE 4E: train F4_all DNN"
CUDA_VISIBLE_DEVICES=0 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 1 --recipe F4_all --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed1_F4_all_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 2 --recipe F4_all --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed2_F4_all_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 3 --recipe F4_all --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed3_F4_all_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 4 --recipe F4_all --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed4_F4_all_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/train_dnn.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 5 --recipe F4_all --skip-complete >logs/parallel_refine/experiment1_p160k_a080_b020/plain/dnn_seed5_F4_all_gpu2.log 2>&1 &
wait

echo "STAGE 5: generate locked Y-test frozen feature caches"
CUDA_VISIBLE_DEVICES=0 python scripts/generate_cache.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 1 --split y_test >logs/parallel_refine/experiment1_p160k_a080_b020/plain/y_cache_seed1_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 2 --split y_test >logs/parallel_refine/experiment1_p160k_a080_b020/plain/y_cache_seed2_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/generate_cache.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 3 --split y_test >logs/parallel_refine/experiment1_p160k_a080_b020/plain/y_cache_seed3_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 4 --split y_test >logs/parallel_refine/experiment1_p160k_a080_b020/plain/y_cache_seed4_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/generate_cache.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 5 --split y_test >logs/parallel_refine/experiment1_p160k_a080_b020/plain/y_cache_seed5_gpu2.log 2>&1 &
wait

echo "STAGE 6: final locked Y-test evaluation, including origin/pair plots"
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 1 --model parallel_dnn >logs/parallel_refine/experiment1_p160k_a080_b020/plain/y_evaluate_seed1_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 2 --model parallel_dnn >logs/parallel_refine/experiment1_p160k_a080_b020/plain/y_evaluate_seed2_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 3 --model parallel_dnn >logs/parallel_refine/experiment1_p160k_a080_b020/plain/y_evaluate_seed3_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 4 --model parallel_dnn >logs/parallel_refine/experiment1_p160k_a080_b020/plain/y_evaluate_seed4_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python scripts/evaluate.py --config configs/parallel_refine/experiments/experiment1/experiment1_p160k_a080_b020.json --seed 5 --model parallel_dnn >logs/parallel_refine/experiment1_p160k_a080_b020/plain/y_evaluate_seed5_gpu2.log 2>&1 &
wait

echo "ALL STAGES COMPLETE. Logs: logs/parallel_refine/experiment1_p160k_a080_b020/plain"
