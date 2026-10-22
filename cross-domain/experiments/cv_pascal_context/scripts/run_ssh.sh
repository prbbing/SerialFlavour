#!/usr/bin/env bash
# USTC_ATLAS_GPU: one GPU, complete seed matrix, bounded cache lifecycle.
set -euo pipefail
mode=${1:---plan}
if [[ $# -gt 1 || ! "$mode" =~ ^--(plan|check|run)$ ]]; then
  echo "Usage: bash ${BASH_SOURCE[0]} [--plan|--check|--run]" >&2; exit 2
fi
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
cd "$repo"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export PYTHON=${PYTHON:-/data/yuyang/miniconda3/envs/gn2_study_cross/bin/python}
export DEPLOY_ROOT=${DEPLOY_ROOT:-/data/yuyang/cross-domain}
export GPU_PHYSICAL=${GPU_PHYSICAL:-0}
# An existing allocation wins; GPU_POOL indexes that allocation logically.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-$GPU_PHYSICAL}
read -r -a gpus <<< "${GPU_POOL:-0}"
if [[ -z "$CUDA_VISIBLE_DEVICES" || ${#gpus[@]} -eq 0 ]]; then
  echo "A nonempty GPU allocation is required" >&2; exit 2
fi
if [[ ! -x "$PYTHON" ]]; then echo "Missing dedicated Python: $PYTHON" >&2; exit 2; fi
export TEMPLATE=${CONFIG:-cross-domain/experiments/cv_pascal_context/config/cluster_full.json}
temporary=$(mktemp "${TMPDIR:-/tmp}/cv_pascal_context_ssh.XXXXXXXX.json")
trap 'rm -f -- "$temporary"' EXIT
# Only deployment paths/checksum/experiment suffix change; matrix and budgets stay intact.
"$PYTHON" -B - "$temporary" <<'PY'
import json, os, re, sys
from pathlib import Path
base = Path(os.environ['DEPLOY_ROOT']).expanduser()
if not base.is_absolute():
    raise SystemExit('DEPLOY_ROOT must be absolute')
config = json.loads(Path(os.environ['TEMPLATE']).read_text())
if config['dataset'] != 'cv_pascal_context' or config['workflow']['mode'] != 'seed_lifecycle':
    raise SystemExit('wrong dataset or workflow')
suffix = os.environ.get('RUN_SUFFIX', 'ssh_a10')
if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', suffix):
    raise SystemExit('RUN_SUFFIX must be a simple identifier starting with a letter')
config['experiment'] += '_' + suffix
config['data_root'] = str(Path(os.environ.get('DATA_ROOT', str(base / 'inputs' / 'cv_pascal_context'))).expanduser())
if not Path(config['data_root']).is_absolute():
    raise SystemExit('DATA_ROOT must be absolute')
config['output_root'] = str(base / 'results')
config['log_root'] = str(base / 'logs')
config['data']['raw_root'] = str(Path(config['data_root']) / 'PASCAL_MT')
config['data']['archive_path'] = str(Path(config['data_root']) / 'PASCAL_MT.tgz')
config['data']['extract_root'] = config['data_root']
config['model']['pretrained']['path'] = os.environ.get('PRETRAINED_PATH', str(Path(config['data_root']) / 'hrnet_w18_small_model_v2.pth'))
config['model']['pretrained']['sha256'] = os.environ.get('PRETRAINED_SHA256') or config['model']['pretrained']['sha256']
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY
entry=cross-domain/experiments/cv_pascal_context/scripts/run_cluster.py
echo "mode=$mode python=$PYTHON CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES logical_gpus=${gpus[*]}"
"$PYTHON" -B "$entry" --config "$temporary" --gpus "${gpus[@]}" --dry-run
if [[ "$mode" == --plan ]]; then exit 0; fi
# Import/data/CUDA availability checks; no model creation, forward or training.
"$PYTHON" -B - "$temporary" <<'PY'
import hashlib, importlib, json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
for module in ('numpy', 'PIL', 'scipy', 'yaml', 'cv2', 'skimage', 'six'):
    importlib.import_module(module)
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in the dedicated environment')
print('torch', torch.__version__, 'CUDA', torch.version.cuda,
      'visible GPUs', torch.cuda.device_count())
raw = Path(config['data']['raw_root'])
for relative in ('ImageSets/Context/train.txt', 'ImageSets/Context/val.txt'):
    if not (raw / relative).is_file():
        raise SystemExit('Missing extracted PASCAL input: ' + str(raw / relative))
weights = Path(config['model']['pretrained']['path'])
expected = config['model']['pretrained']['sha256']
if not weights.is_absolute() or not weights.is_file():
    raise SystemExit('Provide an absolute PRETRAINED_PATH to the local HRNet ImageNet weights')
if not isinstance(expected, str) or len(expected) != 64:
    raise SystemExit('Set PRETRAINED_SHA256 from a reviewed weights source')
with weights.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
if digest != expected:
    raise SystemExit('Pretrained weights SHA256 mismatch')
# Shared scheduler performs the estimate + reserve check, including existing cache.
sys.path.insert(0, str(Path('cross-domain').resolve()))
from pipeline.context import load_context
from pipeline.lifecycle import check_output_space
check_output_space(load_context(sys.argv[1], Path('cross-domain').resolve()))
print('Preflight passed; full schema/provenance checks still run during prepare.')
PY
if [[ "$mode" == --check ]]; then exit 0; fi
# Keep the resolved config immutable: re-running the same command resumes it.
resolved=$("$PYTHON" -B - "$temporary" <<'PY'
import json, os, sys
from pathlib import Path
source = Path(sys.argv[1]).read_bytes()
c = json.loads(source)
path = Path(os.environ['DEPLOY_ROOT']) / 'deploy' / c['dataset'] / (c['experiment'] + '.json')
path.parent.mkdir(parents=True, exist_ok=True)
try:
    with path.open('xb') as stream:
        stream.write(source)
except FileExistsError:
    if path.read_bytes() != source:
        raise SystemExit('Resolved config differs; choose a new RUN_SUFFIX: ' + str(path))
print(path)
PY
)
echo "Resolved config: $resolved"
echo "Resume with exactly the same environment, code and command."
# Keep the shell so EXIT removes the temporary file when the scheduler exits.
"$PYTHON" -B "$entry" --config "$resolved" --gpus "${gpus[@]}" --retries "${RETRIES:-1}"
