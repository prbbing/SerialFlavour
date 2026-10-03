"""Experiment loading and code identity stay independent across domains."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.context import Context, load_context
from pipeline.units import enumerate_units


@pytest.mark.parametrize('dataset', ['qm9', 'cv_nyu_mtan'])
def test_code_identity_dependency_scope(tmp_path, dataset):
    other = 'cv_nyu_mtan' if dataset == 'qm9' else 'qm9'
    files = ['pipeline/worker.py', 'scripts/run_unit.py', 'experiments/__init__.py',
             f'experiments/{dataset}/__init__.py', f'experiments/{dataset}/model.py',
             f'experiments/{dataset}/scripts/run.sh', f'experiments/{other}/model.py',
             f'experiments/{other}/scripts/run.py', f'experiments/{dataset}/tests/test_model.py']
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# original\n')
    context = Context({'dataset': dataset, 'experiment': 'layout_test'}, tmp_path, tmp_path / 'config.json')
    before = context.code_hash
    for name in [files[6], files[7], files[8]]:
        (tmp_path / name).write_text('# unrelated change\n')
        assert context.code_hash == before
    for name in files[:6]:
        path = tmp_path / name
        original = path.read_text()
        path.write_text('# dependency change\n')
        assert context.code_hash != before
        path.write_text(original)


@pytest.mark.parametrize('dataset', ['qm9', 'cv_nyu_mtan'])
def test_all_adapters_load_from_experiment_package(dataset):
    context = Context({'dataset': dataset}, ROOT, ROOT / 'unused.json')
    for kind in ('data', 'model', 'training', 'refine', 'evaluate', 'analysis'):
        module = context.module(kind)
        assert Path(module.__file__).resolve() == ROOT / 'experiments' / dataset / f'{kind}.py'


def test_nyu_full_matrix_is_preserved():
    context = load_context(ROOT / 'experiments/cv_nyu_mtan/config/cluster_full.json', ROOT)
    units = enumerate_units(context)
    assert len(units) == 173
    assert sum(unit.startswith('refine:') for unit in units) == 150
