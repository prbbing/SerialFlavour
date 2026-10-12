#!/usr/bin/env bash
# Experiment 1 EX V2: 11 A/B allocations at 3M, dependency-aware GPU queue.
# Reuses the existing production queue and seed worker without changing them.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: bash scripts/production/run_experiment1_ex_v2_full.sh [--dry-run | --prepare-only]

GPU_POOL="0 1 2 3"   GPUs reserved for this run; one queue unit per GPU.
                     Defaults to numeric CUDA_VISIBLE_DEVICES, else 0 1 2 3.
NODE_COUNT=1          Number of independent nodes sharing data/result storage.
NODE_RANK=0           This node's zero-based configuration shard (rank < count).
PYTHON=python         Python executable from the activated training environment.
PREPARE_WORKERS=8     CPU processes used for shared processed-cache preparation.
RETRIES=1             Additional retries per stage; partial outputs are rebuilt.
PATIENCE=100          Upstream early-stopping patience, matching the current run.
REFINE_GPUS=GPUs-1    Refinement concurrency while upstream work remains queued.
LOG_BASE=logs/parallel_refine/experiment1_ex_v2/full
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
log_base=${LOG_BASE:-logs/parallel_refine/experiment1_ex_v2/full}
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
if ((node_count < 1 || node_count > 11 || node_rank >= node_count ||
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
    configs/parallel_refine/experiments/experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a100_b000.json
    configs/parallel_refine/experiments/experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a095_b005.json
    configs/parallel_refine/experiments/experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a090_b010.json
    configs/parallel_refine/experiments/experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a085_b015.json
    configs/parallel_refine/experiments/experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a080_b020.json
    configs/parallel_refine/experiments/experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a075_b025.json
    configs/parallel_refine/experiments/experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a070_b030.json
    configs/parallel_refine/experiments/experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a065_b035.json
    configs/parallel_refine/experiments/experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a060_b040.json
    configs/parallel_refine/experiments/experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a055_b045.json
    configs/parallel_refine/experiments/experiment1_ex_v2/experiment1_ex_v2_p122k_n3m_a050_b050.json
)
configs=()
for index in "${!all_configs[@]}"; do
    if ((index % node_count == node_rank)); then
        configs+=("${all_configs[$index]}")
    fi
done

# Resolve the entire matrix before preparing data or launching any workers.
# Command substitution preserves the validator's failure status.
metadata=$("$python_bin" - "$node_count" "$node_rank" "${all_configs[@]}" <<'PY'
import copy
from pathlib import Path, PurePosixPath
import sys
from src.parallel_refine.config import load_study_config

node_count, node_rank = map(int, sys.argv[1:3])
studies = [load_study_config(path) for path in sys.argv[3:]]
expected = {(100,0),(95,5),(90,10),(85,15),(80,20),(75,25),
            (70,30),(65,35),(60,40),(55,45),(50,50)}
if len(studies) != 11:
    raise ValueError('the matrix must contain exactly 11 configs')
reference = studies[0]
seen, splits, outputs, cache_ids = set(), set(), set(), set()
for study in studies:
    variables = study.experiment_markers['variables']
    a, b = variables['a_fraction_percent'], variables['b_fraction_percent']
    if (a, b) in seen or (a, b) not in expected:
        raise ValueError(f'duplicate or unexpected allocation: {a}/{b}')
    seen.add((a, b))
    name = f'experiment1_ex_v2_p122k_n3m_a{a:03d}_b{b:03d}'
    if study.study_name != name or study.path.stem != name:
        raise ValueError(f'incorrect config identity: {study.path}')
    sizes = dict(a_train=2100000*a//100, a_val=300000,
                 b_train=2100000*b//100, b_val=300000, y_test=600000)
    if study.data['sizes'] != sizes or study.data['shared_validation'] is not True:
        raise ValueError(f'incorrect 3M data allocation: {name}')
    if (variables['a_train_jets'], variables['b_train_jets'], variables['dnn_enabled']) != (
            sizes['a_train'], sizes['b_train'], b > 0):
        raise ValueError(f'incorrect allocation markers: {name}')
    data = copy.deepcopy(study.data)
    data['sizes'] = reference.data['sizes']
    data['split_dir'] = reference.data['split_dir']
    if data != reference.data:
        raise ValueError('non-allocation data settings must match across configs')
    if study.parallel != reference.parallel or study.parallel['loss_weights'] != {
            'jet': 1, 'origin': 0.5, 'pair': 1.5}:
        raise ValueError('model/training settings and loss weights must match')
    if (study.refiners != reference.refiners
            or study.values['feature_cache'] != reference.values['feature_cache']):
        raise ValueError('downstream settings must match across configs')
    if [run.seed for run in study.seeds] != [1,2,3,4,5] or list(study.downstream_seeds) != [1,2,3,4,5]:
        raise ValueError('expected five upstream and five downstream seeds')
    split_suffix = f'experiment1_ex_v2_n3m_a{a:03d}_b{b:03d}'
    if not study.data['split_dir'].endswith('/' + split_suffix):
        raise ValueError(f'incorrect v2 split directory: {name}')
    splits.add(study.data['split_dir'])
    outputs.add(study.output_directory)
    cache_ids.add(study.cache_identity_name)
if seen != expected or len(splits) != 11 or len(outputs) != 11 or len(cache_ids) != 11:
    raise ValueError('matrix must contain 11 distinct allocations and output identities')
for helper in ('run_experiment1_ex_recipe_queue.py', 'run_experiment1_ex_seed.py'):
    if not (Path('scripts/production') / helper).is_file():
        raise FileNotFoundError(helper)
selected = [study for index, study in enumerate(studies) if index % node_count == node_rank]
lock_dir = PurePosixPath(reference.data['processed_cache_dir']) / '.locks' / 'experiment1_ex_v2'
print(lock_dir)
print(sum(len(study.seeds) for study in selected))
print(sum(len(study.seeds) * len(study.downstream_seeds) * len(study.refiners['recipes'])
          for study in selected if study.data['sizes']['b_train'] > 0))
print(' '.join(str(study.data['sizes']['b_train']) for study in selected))
PY
)
# Native Windows Python emits CRLF when this plan is checked through Git Bash.
metadata=${metadata//$'\r'/}
mapfile -t metadata_lines <<< "$metadata"
lock_dir=${metadata_lines[0]}
upstream_count=${metadata_lines[1]}
downstream_count=${metadata_lines[2]}
read -r -a b_train_sizes <<< "${metadata_lines[3]}"

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}
# The 100/0 baseline has no B data to cache or downstream model to train.
build_prepare_command() {
    local index=$1
    prepare_command=("$python_bin" scripts/prepare_data.py --config "${configs[$index]}"
                     --build-processed-caches --workers "$workers"
                     --processed-split a_train --processed-split a_val --processed-split y_test)
    if ((b_train_sizes[index] > 0)); then
        prepare_command+=(--processed-split b_train --processed-split b_val)
    fi
}
queue_args=("$python_bin" scripts/production/run_experiment1_ex_recipe_queue.py)
for config in "${configs[@]}"; do
    queue_args+=(--config "$config")
done
queue_args+=(--gpus "${gpu_pool[@]}" --log-base "$node_log" --state-dir "$node_log/state"
            --retries "$retries" --patience "$patience" --max-refine "$refine_gpus"
            --python "$python_bin")

printf 'Experiment 1 EX V2: node %s/%s, %s of 11 configurations, GPUs: %s\n' \
       "$node_rank" "$node_count" "${#configs[@]}" "${gpu_pool[*]}"
printf 'This shard: %s upstream and %s downstream trainings; logs: %s\n' \
       "$upstream_count" \
       "$downstream_count" "$node_log"
printf 'Data-preparation lock directory: %s\n' "$lock_dir"
if ((dry_run)); then
    for index in "${!configs[@]}"; do
        build_prepare_command "$index"
        print_command "${prepare_command[@]}"
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

mkdir -p "$node_log/stage1" "$node_log/state" "$lock_dir"
# Locks must be honoured by the shared filesystem when running on many nodes.
exec 9>"$lock_dir/node${node_rank}.run.lock"
flock -n 9 || { echo 'ERROR: this node shard already has a running launcher' >&2; exit 1; }
# Hold the preparation lock across all configs. Existing processed caches are
# reused, while prepare_data writes each experiment's own provenance manifest.
exec 8>"$lock_dir/prepare.lock"
echo 'Waiting for matrix data-preparation lock'
flock 8
check_space
for index in "${!configs[@]}"; do
    config=${configs[$index]}
    build_prepare_command "$index"
    name=${config##*/}
    name=${name%.json}
    log="$node_log/stage1/$name.log"
    printf 'Prepare %s; log=%s\n' "$name" "$log"
    "${prepare_command[@]}" >"$log" 2>&1 || {
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
echo 'EXPERIMENT 1 EX V2 SHARD COMPLETE'
