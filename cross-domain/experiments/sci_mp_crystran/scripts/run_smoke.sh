#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON="${PYTHON:-/home/yuyang/miniconda3/envs/gn2_study_cross/bin/python}"
cd "$REPO"
"$PYTHON" -m pytest cross-domain/experiments/sci_mp_crystran/tests -q
"$PYTHON" cross-domain/scripts/run.py --config cross-domain/experiments/sci_mp_crystran/config/smoke.json --stage all
"$PYTHON" cross-domain/experiments/sci_mp_crystran/scripts/verify_run.py --config cross-domain/experiments/sci_mp_crystran/config/smoke.json