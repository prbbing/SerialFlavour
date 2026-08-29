#!/usr/bin/env bash
# Deliberately plain end-to-end runner for the A=500k, four-layer experiment.
# Activate the SerialFlavour/GN2 environment before running this script.
# Every command is written explicitly; there are no configurable shell variables.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p parallel_refine/logs/plain_a500k_4layers

echo "STAGE 1: prepare A/B/Y processed caches"
python -m parallel_refine.training.prepare_data \
    --config parallel_refine/configs/experiments/a500k_4layers.json \
    --build-processed-caches \
    --processed-split a_train \
    --processed-split a_val \
    --processed-split b_train \
    --processed-split b_val \
    --processed-split y_test \
    2>&1 | tee parallel_refine/logs/plain_a500k_4layers/prepare_data.log

echo "STAGE 2: train five Parallel models"
CUDA_VISIBLE_DEVICES=0 python -m parallel_refine.training.train_parallel --config parallel_refine/configs/experiments/a500k_4layers.json --seed 1 --skip-complete >parallel_refine/logs/plain_a500k_4layers/parallel_seed1_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.train_parallel --config parallel_refine/configs/experiments/a500k_4layers.json --seed 2 --skip-complete >parallel_refine/logs/plain_a500k_4layers/parallel_seed2_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.train_parallel --config parallel_refine/configs/experiments/a500k_4layers.json --seed 3 --skip-complete >parallel_refine/logs/plain_a500k_4layers/parallel_seed3_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.train_parallel --config parallel_refine/configs/experiments/a500k_4layers.json --seed 4 --skip-complete >parallel_refine/logs/plain_a500k_4layers/parallel_seed4_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.train_parallel --config parallel_refine/configs/experiments/a500k_4layers.json --seed 5 --skip-complete >parallel_refine/logs/plain_a500k_4layers/parallel_seed5_gpu2.log 2>&1 &
wait

echo "STAGE 3: generate B frozen feature caches"
CUDA_VISIBLE_DEVICES=0 python -m parallel_refine.training.generate_cache --config parallel_refine/configs/experiments/a500k_4layers.json --seed 1 --split b_train --split b_val >parallel_refine/logs/plain_a500k_4layers/b_cache_seed1_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.generate_cache --config parallel_refine/configs/experiments/a500k_4layers.json --seed 2 --split b_train --split b_val >parallel_refine/logs/plain_a500k_4layers/b_cache_seed2_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.generate_cache --config parallel_refine/configs/experiments/a500k_4layers.json --seed 3 --split b_train --split b_val >parallel_refine/logs/plain_a500k_4layers/b_cache_seed3_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.generate_cache --config parallel_refine/configs/experiments/a500k_4layers.json --seed 4 --split b_train --split b_val >parallel_refine/logs/plain_a500k_4layers/b_cache_seed4_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.generate_cache --config parallel_refine/configs/experiments/a500k_4layers.json --seed 5 --split b_train --split b_val >parallel_refine/logs/plain_a500k_4layers/b_cache_seed5_gpu2.log 2>&1 &
wait

# Each downstream batch launches five DNN jobs, below max_job=12.
echo "STAGE 4A: train F1_embed DNN"
CUDA_VISIBLE_DEVICES=0 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 1 --recipe F1_embed --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed1_F1_embed_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 2 --recipe F1_embed --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed2_F1_embed_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 3 --recipe F1_embed --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed3_F1_embed_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 4 --recipe F1_embed --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed4_F1_embed_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 5 --recipe F1_embed --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed5_F1_embed_gpu2.log 2>&1 &

echo "STAGE 4B: train F2_jet_aux DNN"
CUDA_VISIBLE_DEVICES=0 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 1 --recipe F2_jet_aux --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed1_F2_jet_aux_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 2 --recipe F2_jet_aux --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed2_F2_jet_aux_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 3 --recipe F2_jet_aux --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed3_F2_jet_aux_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 4 --recipe F2_jet_aux --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed4_F2_jet_aux_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 5 --recipe F2_jet_aux --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed5_F2_jet_aux_gpu2.log 2>&1 &
wait

echo "STAGE 4C: train F3_embed_aux DNN"
CUDA_VISIBLE_DEVICES=0 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 1 --recipe F3_embed_aux --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed1_F3_embed_aux_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 2 --recipe F3_embed_aux --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed2_F3_embed_aux_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 3 --recipe F3_embed_aux --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed3_F3_embed_aux_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 4 --recipe F3_embed_aux --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed4_F3_embed_aux_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 5 --recipe F3_embed_aux --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed5_F3_embed_aux_gpu2.log 2>&1 &

echo "STAGE 4D: train F4_all DNN"
CUDA_VISIBLE_DEVICES=0 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 1 --recipe F4_all --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed1_F4_all_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 2 --recipe F4_all --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed2_F4_all_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 3 --recipe F4_all --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed3_F4_all_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 4 --recipe F4_all --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed4_F4_all_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.train_dnn --config parallel_refine/configs/experiments/a500k_4layers.json --seed 5 --recipe F4_all --skip-complete >parallel_refine/logs/plain_a500k_4layers/dnn_seed5_F4_all_gpu2.log 2>&1 &
wait

echo "STAGE 5: generate locked Y frozen feature caches"
CUDA_VISIBLE_DEVICES=0 python -m parallel_refine.training.generate_cache --config parallel_refine/configs/experiments/a500k_4layers.json --seed 1 --split y_test >parallel_refine/logs/plain_a500k_4layers/y_cache_seed1_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.generate_cache --config parallel_refine/configs/experiments/a500k_4layers.json --seed 2 --split y_test >parallel_refine/logs/plain_a500k_4layers/y_cache_seed2_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.generate_cache --config parallel_refine/configs/experiments/a500k_4layers.json --seed 3 --split y_test >parallel_refine/logs/plain_a500k_4layers/y_cache_seed3_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.generate_cache --config parallel_refine/configs/experiments/a500k_4layers.json --seed 4 --split y_test >parallel_refine/logs/plain_a500k_4layers/y_cache_seed4_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.generate_cache --config parallel_refine/configs/experiments/a500k_4layers.json --seed 5 --split y_test >parallel_refine/logs/plain_a500k_4layers/y_cache_seed5_gpu2.log 2>&1 &
wait

echo "STAGE 6: final locked Y evaluation, including origin/pair plots"
CUDA_VISIBLE_DEVICES=0 python -m parallel_refine.training.evaluate --config parallel_refine/configs/experiments/a500k_4layers.json --seed 1 --model direct_dnn >parallel_refine/logs/plain_a500k_4layers/y_evaluate_seed1_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.evaluate --config parallel_refine/configs/experiments/a500k_4layers.json --seed 2 --model direct_dnn >parallel_refine/logs/plain_a500k_4layers/y_evaluate_seed2_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m parallel_refine.training.evaluate --config parallel_refine/configs/experiments/a500k_4layers.json --seed 3 --model direct_dnn >parallel_refine/logs/plain_a500k_4layers/y_evaluate_seed3_gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.evaluate --config parallel_refine/configs/experiments/a500k_4layers.json --seed 4 --model direct_dnn >parallel_refine/logs/plain_a500k_4layers/y_evaluate_seed4_gpu2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m parallel_refine.training.evaluate --config parallel_refine/configs/experiments/a500k_4layers.json --seed 5 --model direct_dnn >parallel_refine/logs/plain_a500k_4layers/y_evaluate_seed5_gpu2.log 2>&1 &
wait

echo "ALL STAGES COMPLETE. Logs: parallel_refine/logs/plain_a500k_4layers"
