#!/usr/bin/env bash
# Complete the F3_embed_aux_104 allocation and loss-weight studies before DNN-pair-99.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
read -r -a total_m <<< "${TOTAL_M:-1 3 5}"
read -r -a gpu_pool <<< "${GPU_POOL:-0}"

stage=all
suffix=pair_statistics_ablation_v1
allocation_args=()
shared_args=()
prepare_only=false
while (($#)); do
    case "$1" in
        --stage) stage=${2:?--stage requires a value}; shift 2 ;;
        --stage=*) stage=${1#*=}; shift ;;
        --suffix) suffix=${2:?--suffix requires a value}; shift 2 ;;
        --suffix=*) suffix=${1#*=}; shift ;;
        --allocation|--config|--output-root) allocation_args+=("$1" "${2:?missing selection value}"); shift 2 ;;
        --allocation=*|--config=*|--output-root=*) allocation_args+=("$1"); shift ;;
        --prepare-configs-only) prepare_only=true; shared_args+=("$1"); shift ;;
        --recipe|--recipe=*) printf '%s\n' 'This launcher sets the recipe for each phase.' >&2; exit 2 ;;
        --help|-h)
            cat <<'EOF'
Usage: run_experiment1_ex_v2_pair_statistics_ablation.sh [runner options]
Default order: train F3_embed_aux_104 on allocations, train and evaluate it on
the Experiment 3 loss grid, evaluate it on allocations, then train and evaluate
DNN-pair-99 on allocations. Use --dry-run to print the full plan.
TOTAL_M (default: 1 3 5), --allocation and --config select allocation runs.
--stage all|train|evaluate selects full, training-only or evaluation-only runs.
Set GPU_POOL (default: 0) and PYTHON (default: python) for the training environment.
LOSS_OUTPUT_ROOT selects the Experiment 3 results root (default: parallel_refine_e3).
EOF
            exit 0 ;;
        *) shared_args+=("$1"); shift ;;
    esac
done
case "$stage" in
    all|train|evaluate) ;;
    *) printf 'Invalid --stage: %s\n' "$stage" >&2; exit 2 ;;
esac

loss_configs=()
loss_dir=configs/parallel_refine/experiments/experiment3
loss_output_root=${LOSS_OUTPUT_ROOT:-/data/yuyang/SerialFlavour/results/parallel_refine_e3}
for origin in 0p2 0p5 1p0 1p5; do
    for pair in 0p2 0p5 1p0 1p5; do
        loss_configs+=(--config "$loss_dir/experiment3_p122k_n3m_a075_b025_origin${origin}_pair${pair}.json")
    done
done

runner=("${PYTHON:-python}" scripts/production/run_experiment1_ex_v2_refine.py
    --gpus "${gpu_pool[@]}" --downstream-workers 5 --prune-after-aggregate)
failed_phases=()
run_phase() {
    local label=$1
    shift
    printf '\n%s\n' "$label"
    if ! "${runner[@]}" "$@" "${shared_args[@]}"; then
        failed_phases+=("$label")
    fi
}

allocation_args=(--total-m "${total_m[@]}" "${allocation_args[@]}")
f3_args=(--recipe F3_embed_aux_104 --suffix "${suffix}_f3_104")
if [[ $stage != evaluate ]]; then
    run_phase 'F3_embed_aux_104: allocation training' --stage train "${f3_args[@]}" "${allocation_args[@]}"
fi
run_phase 'F3_embed_aux_104: Experiment 3 loss grid' --stage "$stage" "${f3_args[@]}" \
    --output-root "$loss_output_root" "${loss_configs[@]}"
if [[ $stage == evaluate || ( $stage == all && $prepare_only == false ) ]]; then
    run_phase 'F3_embed_aux_104: allocation evaluation' --stage evaluate "${f3_args[@]}" "${allocation_args[@]}"
fi

if ((${#failed_phases[@]} == 0)); then
    run_phase 'DNN-pair-99: allocations' --stage "$stage" --recipe DNN-pair-99 \
        --suffix "${suffix}_pair_99" "${allocation_args[@]}"
else
    printf '%s\n' 'DNN-pair-99 is pending successful completion of the F3_embed_aux_104 phases.' >&2
fi
if ((${#failed_phases[@]})); then
    printf 'Failed phase: %s\n' "${failed_phases[@]}" >&2
    exit 1
fi
