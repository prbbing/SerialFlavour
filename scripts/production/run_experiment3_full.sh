#!/usr/bin/env bash
# Experiment 3: 16 loss-weight configurations, dependency-aware GPU queue.
# Reuses the existing production queue and seed worker without changing them.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/production/run_experiment3_full.sh [--dry-run | --prepare-only]

GPU_POOL="0 1 2 3"   GPUs reserved for this run; one queue unit per GPU.
                     Defaults to numeric CUDA_VISIBLE_DEVICES, else 0 1 2 3.
NODE_COUNT=1          Number of independent nodes sharing data/result storage.
NODE_RANK=0           This node's zero-based configuration shard (rank < count).
PYTHON=python         Python executable from the activated training environment.
PREPARE_WORKERS=8     CPU processes used for shared processed-cache preparation.
RETRIES=1             Additional retries per stage; partial outputs are rebuilt.
PATIENCE=100          Upstream early-stopping patience, matching the current run.
REFINE_GPUS=GPUs-1    Refinement concurrency while upstream work remains queued.
LOG_BASE=logs/parallel_refine/experiment3/full
MIN_FREE_GB=200       Minimum free GiB before preparation and queue launch.
SPACE_PATH=/data      Filesystem on which to check available space.

--dry-run            Validate all configs and print this shard's commands.
                     Does not create logs, caches, manifests or train models.
--prepare-only       Prepare this shard's manifests and shared processed caches.
--help               Print this help.

Nodes run independent jobs, not DDP. Each recipe unit launches the configured
five downstream seeds on its assigned GPU, as in the existing production queue.
Re-running skips completed seeds; incomplete training restarts its stage.
After successful evaluation the worker prunes only that seed's frozen/graph
caches. Checkpoints, processed caches and evaluation results are retained.
HELP
}

dry_run=0
prepare_only=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        --prepare-only) prepare_only=1 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done
if ((dry_run && prepare_only)); then
    echo 'ERROR: --dry-run and --prepare-only are mutually exclusive' >&2
    exit 2
fi

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
python_bin=${PYTHON:-python}
node_count=${NODE_COUNT:-1}
node_rank=${NODE_RANK:-0}
workers=${PREPARE_WORKERS:-8}
retries=${RETRIES:-1}
patience=${PATIENCE:-100}
min_free_gb=${MIN_FREE_GB:-200}
space_path=${SPACE_PATH:-/data}
log_base=${LOG_BASE:-logs/parallel_refine/experiment3/full}
node_log="$log_base/node${node_rank}"
visible_gpus=${CUDA_VISIBLE_DEVICES:-}
gpu_spec=${GPU_POOL:-${visible_gpus//,/ }}
gpu_spec=${gpu_spec:-0 1 2 3}
read -r -a gpu_pool <<< "$gpu_spec"
refine_gpus=${REFINE_GPUS:-$(( ${#gpu_pool[@]} > 1 ? ${#gpu_pool[@]} - 1 : 1 ))}

# Validate numeric shell options before arithmetic or any filesystem changes.
for value in "$node_count" "$node_rank" "$workers" "$retries" "$patience" \
             "$refine_gpus" "$min_free_gb"; do
    if [[ ! "$value" =~ ^(0|[1-9][0-9]*)$ ]]; then
        printf 'ERROR: expected a non-negative decimal integer, got %s\n' "$value" >&2
        exit 2
    fi
done
if ((node_count < 1 || node_count > 16 || node_rank >= node_count ||
     workers < 1 || patience < 1 || refine_gpus < 1 ||
     refine_gpus > ${#gpu_pool[@]})); then
    echo 'ERROR: invalid node rank/count, workers, patience or REFINE_GPUS' >&2
    exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpu_pool[@]}"; do
    if [[ ! "$gpu" =~ ^(0|[1-9][0-9]*)$ || -n "${seen_gpus[$gpu]:-}" ]]; then
        echo 'ERROR: GPU_POOL must contain distinct non-negative numeric GPU IDs' >&2
        exit 2
    fi
    seen_gpus[$gpu]=1
done

all_configs=(
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin0p2_pair0p2.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin0p2_pair0p5.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin0p2_pair1p0.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin0p2_pair1p5.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin0p5_pair0p2.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin0p5_pair0p5.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin0p5_pair1p0.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin0p5_pair1p5.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin1p0_pair0p2.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin1p0_pair0p5.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin1p0_pair1p0.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin1p0_pair1p5.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin1p5_pair0p2.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin1p5_pair0p5.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin1p5_pair1p0.json
    configs/parallel_refine/experiments/experiment3/experiment3_p122k_n3m_a075_b025_origin1p5_pair1p5.json
)
configs=()
for index in "${!all_configs[@]}"; do
    if ((index % node_count == node_rank)); then
        configs+=("${all_configs[$index]}")
    fi
done

# Resolve the entire matrix before preparing data or launching any workers.
# Command substitution preserves the validator's failure status.
metadata=$("$python_bin" - "${all_configs[@]}" <<'PY'
from itertools import product
from pathlib import Path
import sys
from src.parallel_refine.config import load_study_config

studies = [load_study_config(path) for path in sys.argv[1:]]
assert len(studies) == 16
reference = studies[0]
assert reference.data['sizes'] == dict(a_train=1575000, a_val=300000,
                                     b_train=525000, b_val=300000, y_test=600000)
assert reference.data['shared_validation'] is True
seen = set()
for study in studies:
    assert study.study_name == study.path.stem
    assert study.data == reference.data, 'all weights must use identical data'
    assert study.refiners == reference.refiners
    assert study.values['feature_cache'] == reference.values['feature_cache']
    assert study.seeds == reference.seeds
    assert [run.seed for run in study.seeds] == [1, 2, 3, 4, 5]
    assert list(study.downstream_seeds) == [1, 2, 3, 4, 5]
    weights = study.parallel['loss_weights']
    assert weights['jet'] == 1
    seen.add((weights['origin'], weights['pair']))
    assert {k: v for k, v in study.parallel.items() if k != 'loss_weights'} == {
        k: v for k, v in reference.parallel.items() if k != 'loss_weights'}
assert seen == set(product((0.2, 0.5, 1.0, 1.5), repeat=2))
for helper in ('run_experiment1_ex_recipe_queue.py', 'run_experiment1_ex_seed.py'):
    assert (Path('scripts/production') / helper).is_file()
print(reference.data['split_dir'])
print(len(reference.seeds))
print(len(reference.seeds) * len(reference.downstream_seeds) * len(reference.refiners['recipes']))
PY
)
# Native Windows Python emits CRLF when this plan is checked through Git Bash.
metadata=${metadata//$'\r'/}
mapfile -t metadata_lines <<< "$metadata"
split_dir=${metadata_lines[0]}
upstream_per_config=${metadata_lines[1]}
downstream_per_config=${metadata_lines[2]}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}
prepare_args=(--build-processed-caches --workers "$workers")
for split in a_train a_val b_train b_val y_test; do
    prepare_args+=(--processed-split "$split")
done
queue_args=("$python_bin" scripts/production/run_experiment1_ex_recipe_queue.py)
for config in "${configs[@]}"; do
    queue_args+=(--config "$config")
done
queue_args+=(--gpus "${gpu_pool[@]}" --log-base "$node_log" --state-dir "$node_log/state"
            --retries "$retries" --patience "$patience" --max-refine "$refine_gpus"
            --python "$python_bin")

printf 'Experiment 3: node %s/%s, %s of 16 configurations, GPUs: %s\n' \
       "$node_rank" "$node_count" "${#configs[@]}" "${gpu_pool[*]}"
printf 'This shard: %s upstream and %s downstream trainings; logs: %s\n' \
       "$(( ${#configs[@]} * upstream_per_config ))" \
       "$(( ${#configs[@]} * downstream_per_config ))" "$node_log"
printf 'Shared split: %s\n' "$split_dir"
if ((dry_run)); then
    for config in "${configs[@]}"; do
        print_command "$python_bin" scripts/prepare_data.py --config "$config" "${prepare_args[@]}"
    done
    print_command "${queue_args[@]}"
    echo 'DRY RUN COMPLETE: no preparation, training or evaluation executed'
    exit 0
fi

command -v flock >/dev/null || { echo 'ERROR: Linux flock is required' >&2; exit 1; }
check_space() {
    "$python_bin" - "$space_path" "$min_free_gb" <<'PY'
import shutil
import sys
free = shutil.disk_usage(sys.argv[1]).free / 1024 ** 3
minimum = int(sys.argv[2])
if free < minimum:
    raise SystemExit(f'ERROR: {free:.1f} GiB free on {sys.argv[1]}, requires {minimum}')
print(f'Disk space: {free:.1f} GiB free on {sys.argv[1]}')
PY
}
if ((!prepare_only)); then
    gpu_csv=$(IFS=,; printf '%s' "${gpu_pool[*]}")
    CUDA_VISIBLE_DEVICES="$gpu_csv" "$python_bin" - "${#gpu_pool[@]}" <<'PY'
import sys
import torch
expected = int(sys.argv[1])
if not torch.cuda.is_available() or torch.cuda.device_count() != expected:
    raise SystemExit('ERROR: CUDA is unavailable or GPU_POOL includes an unavailable device')
print(f'CUDA preflight: {expected} visible GPUs')
PY
fi

mkdir -p "$node_log/stage1" "$node_log/state" "$split_dir"
# Locks must be honoured by the shared filesystem when running on many nodes.
exec 9>"$split_dir/.experiment3.node${node_rank}.run.lock"
flock -n 9 || { echo 'ERROR: this node shard already has a running launcher' >&2; exit 1; }
# Hold the preparation lock across all configs. Existing processed caches are
# reused, while prepare_data writes each experiment's own provenance manifest.
exec 8>"$split_dir/.experiment3.prepare.lock"
echo 'Waiting for shared data-preparation lock'
flock 8
check_space
for config in "${configs[@]}"; do
    name=${config##*/}
    name=${name%.json}
    log="$node_log/stage1/$name.log"
    printf 'Prepare %s; log=%s\n' "$name" "$log"
    "$python_bin" scripts/prepare_data.py --config "$config" "${prepare_args[@]}" >"$log" 2>&1 || {
        printf 'FAILED preparation: %s; inspect %s\n' "$name" "$log" >&2
        exit 1
    }
done
flock -u 8
exec 8>&-
if ((prepare_only)); then
    echo 'PREPARATION COMPLETE'
    exit 0
fi

check_space
export PYTHONUNBUFFERED=1
echo 'Launch dependency-aware queue: Parallel -> B cache -> recipes -> Y cache -> evaluation -> cache pruning'
# The queue aggregates only after all five upstream seeds of a config finish.
# It returns nonzero if retries are exhausted; never report a partial run as complete.
"${queue_args[@]}"
echo 'EXPERIMENT 3 SHARD COMPLETE'
