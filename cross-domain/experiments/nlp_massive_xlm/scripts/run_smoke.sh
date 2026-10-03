#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
if [[ -n "${PYTHON:-}" ]]; then
  "$PYTHON" cross-domain/scripts/run.py --config cross-domain/experiments/nlp_massive_xlm/config/smoke.json --stage all
else
  conda run --no-capture-output -n gn2_study_cross python cross-domain/scripts/run.py --config cross-domain/experiments/nlp_massive_xlm/config/smoke.json --stage all
fi
