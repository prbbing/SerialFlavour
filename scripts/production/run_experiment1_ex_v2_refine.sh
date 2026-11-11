#!/usr/bin/env bash
# 完整流程运行示例（在 GPU 服务器的仓库根目录执行）：
#
# cd /data/yuyang/SerialFlavour
#
# 1. 检查 3M 全部非零 B allocation 的完整流程，不启动任务：
# TOTAL_M=3 GPU_POOL="0 1 2 3" bash scripts/production/run_experiment1_ex_v2_refine.sh --stage all --dry-run
#
# 2. 后台运行：B 缓存 -> 5 个新增配方训练 -> Y 缓存 -> 评估 -> 跨 seed 聚合。
# mkdir -p logs/parallel_refine/experiment1_ex_v2/refine_inputs
# nohup env TOTAL_M=3 GPU_POOL="0 1 2 3" PYTHON=python \
#   bash scripts/production/run_experiment1_ex_v2_refine.sh --stage all \
#   > logs/parallel_refine/experiment1_ex_v2/refine_inputs/launcher_n3m.log 2>&1 &
#
# 3. 查看进度：
# tail -f logs/parallel_refine/experiment1_ex_v2/refine_inputs/launcher_n3m.log
#
# 仅运行 3M、A/B=75/25 的完整流程：
# TOTAL_M=3 GPU_POOL="0 1 2 3" bash scripts/production/run_experiment1_ex_v2_refine.sh --allocation 75 --stage all
#
# TOTAL_M 可选 1、3、5；默认选择该总量的全部 10 个非零 B allocation。
# 每个 allocation：5 upstream seeds x 5 downstream seeds x 5 个新增配方。
# GPU_POOL 指定的 GPU 直接并行使用，不检查其他训练进程。
# 原 checkpoint 和 split 文件须已存在；结果写入独立的 *_refine_inputs_v1 目录。
# 共享缓存保留；请勿同时运行会清理相同缓存的旧 full 队列。
# 默认仅训练 B；上述 --stage all 显式开启训练后的 Y 评估。
# Defaults to B-only supplementary training. Use --stage evaluate for locked Y.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
read -r -a gpu_pool <<< "${GPU_POOL:-0}"
exec "${PYTHON:-python}" scripts/production/run_experiment1_ex_v2_refine.py \
    --total-m "${TOTAL_M:-3}" --gpus "${gpu_pool[@]}" "$@"
