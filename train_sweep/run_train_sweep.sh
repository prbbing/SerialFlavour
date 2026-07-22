#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

python -m train_sweep \
  --config configs/1M_training_jets/vertex_ablation/a5_no_vertex_loss.json \
  --config configs/1M_training_jets/vertex_ablation/a6_vertex_loss_only.json \
  --config configs/1M_training_jets/vertex_ablation/a7_no_vertex_task_path.json \
  --config configs/1M_training_jets/residual_refine/geometry_only.json \
  --config configs/1M_training_jets/residual_refine/geometry_origin_probs.json \
  --max-concurrent 5 \
  --seed 42 \
  "$@"
