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
export GPU_PHYSICAL=${GPU_PHYSICAL:-1}
# An existing allocation wins; GPU_POOL indexes that allocation logically.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-$GPU_PHYSICAL}
read -r -a gpus <<< "${GPU_POOL:-0}"
if [[ -z "$CUDA_VISIBLE_DEVICES" || ${#gpus[@]} -eq 0 ]]; then
  echo "A nonempty GPU allocation is required" >&2; exit 2
fi
if [[ ! -x "$PYTHON" ]]; then echo "Missing dedicated Python: $PYTHON" >&2; exit 2; fi
export TEMPLATE=${CONFIG:-cross-domain/experiments/sci_qm7x/config/cluster_50k.json}
temporary=$(mktemp "${TMPDIR:-/tmp}/sci_qm7x_ssh.XXXXXXXX.json")
trap 'rm -f -- "$temporary"' EXIT
# Only deployment paths/checksum/experiment suffix change; matrix and budgets stay intact.
"$PYTHON" -B - "$temporary" <<'PY'
import json, os, re, sys
from pathlib import Path
base = Path(os.environ['DEPLOY_ROOT']).expanduser()
if not base.is_absolute():
    raise SystemExit('DEPLOY_ROOT must be absolute')
config = json.loads(Path(os.environ['TEMPLATE']).read_text())
if config['dataset'] != 'sci_qm7x' or config['workflow']['mode'] != 'seed_lifecycle':
    raise SystemExit('wrong dataset or workflow')
suffix = os.environ.get('RUN_SUFFIX', 'ssh_a10')
if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', suffix):
    raise SystemExit('RUN_SUFFIX must be a simple identifier starting with a letter')
config['experiment'] += '_' + suffix
config['data_root'] = str(Path(os.environ.get('DATA_ROOT', str(base / 'inputs' / 'sci_qm7x'))).expanduser())
if not Path(config['data_root']).is_absolute():
    raise SystemExit('DATA_ROOT must be absolute')
config['output_root'] = str(base / 'results')
config['log_root'] = str(base / 'logs')
config['data']['allow_download'] = False
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + '\n')
PY
entry=cross-domain/experiments/sci_qm7x/scripts/run_cluster.py
echo "mode=$mode python=$PYTHON CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES logical_gpus=${gpus[*]}"
"$PYTHON" -B "$entry" --config "$temporary" --gpus "${gpus[@]}" --dry-run
if [[ "$mode" == --plan ]]; then exit 0; fi
# Import/data/CUDA availability checks; no model creation, forward or training.
"$PYTHON" -B - "$temporary" <<'PY'
import hashlib, importlib, json, sys
from pathlib import Path
config = json.loads(Path(sys.argv[1]).read_text())
for module in ('numpy', 'h5py'):
    importlib.import_module(module)
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in the dedicated environment')
print('torch', torch.__version__, 'CUDA', torch.version.cuda,
      'visible GPUs', torch.cuda.device_count())
raw = Path(config['data_root'])
for source in config['data']['sources']:
    path = Path(source['path'])
    path = path if path.is_absolute() else raw / path
    if not path.is_file():
        raise SystemExit('Missing extracted QM7-X shard: ' + str(path))
duplicates = Path(config['data']['duplicates_file'])
duplicates = duplicates if duplicates.is_absolute() else raw / duplicates
if not duplicates.is_file():
    raise SystemExit('Missing DupMols.dat: ' + str(duplicates))
with duplicates.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'md5').hexdigest()
if digest != config['data']['duplicates_md5']:
    raise SystemExit('DupMols.dat MD5 mismatch')
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
