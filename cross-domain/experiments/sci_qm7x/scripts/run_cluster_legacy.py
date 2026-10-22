"""Read-only plan or audited launch of the shared single-node run_pool.py."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

DOMAIN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DOMAIN_ROOT))
from pipeline.context import load_context
from pipeline.io import read_json, sha256_file
from pipeline.units import enumerate_units
from pipeline.worker import marker_path
from experiments.sci_qm7x.common import validate_config, verify_unit_markers, assert_prepared


def plan(context):
    validate_config(context.config)
    config = context.config
    if len(config['upstream']['seeds']) != 5 or len(config['refiner']['seeds']) != 5:
        raise ValueError('formal cluster plan requires complete 5 x 5 seeds')
    units = enumerate_units(context)
    if len(units) != len(set(units)):
        raise ValueError('duplicate matrix units')
    w, h = config['model']['width'], config['model']['head_hidden']
    radial, blocks = config['model']['num_gaussians'], config['model']['num_interactions']
    backbone = 18 * w + blocks * (4 * w * w + (radial + 4) * w)
    upstream = backbone + (w + 1) * h + h + 1 + (2 * w + 1) * h + 2 * (h + 1)
    hidden = config['refiner']['hidden']
    first = hidden[0]
    readout = (w + h + config['refiner']['geometry_bins'] + 1) * first + (first + 1) * first + first + 1
    previous = 3 * first + w + 2
    for output in hidden:
        readout += (previous + 1) * output
        previous = output
    readout += previous + 1
    downstream_n = sum(config['data']['sizes'][s] for s in ('b_train', 'b_val', 'y_test'))
    cache_bytes = downstream_n * (config['data']['max_atoms'] * ((w + h + 5) * 4 + 1) + (w + 2) * 4)
    return {'identity': context.identity, 'config': str(context.config_path), 'data': str(context.data_dir),
            'output': str(context.output_dir), 'structure_budget': sum(config['data']['sizes'].values()),
            'structure_splits': config['data']['sizes'], 'group_counts': 'reported after local HDF5 prepare, never guessed',
            'upstream_seeds': config['upstream']['seeds'], 'downstream_seeds': config['refiner']['seeds'],
            'upstream_jobs': len(config['upstream']['variants']) * len(config['upstream']['seeds']),
            'refine_jobs': sum(u.startswith('refine:') for u in units), 'total_units': len(units),
            'parameters_static': {'single_task': upstream, 'multi_task': upstream, 'every_readout': readout},
            'cache_tensor_bytes_per_upstream_estimate': cache_bytes,
            'cache_tensor_bytes_total_estimate': cache_bytes * len(config['upstream']['variants']) * len(config['upstream']['seeds']),
            'units': units, 'execution': 'read-only plan; no model, CUDA, data reads, mkdir, subprocess or download'}


def gpu_mapping(indices):
    if not indices or len(set(indices)) != len(indices) or any(not value.isdigit() for value in indices):
        raise ValueError('distinct nonnegative GPU indices required')
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    if inherited is None:
        return list(indices)
    available = [part.strip() for part in inherited.split(',') if part.strip()]
    if not available or '-1' in available or any(int(index) >= len(available) for index in indices):
        raise ValueError('GPU indices must address the inherited CUDA_VISIBLE_DEVICES allocation')
    return [available[int(index)] for index in indices]


def preflight(context):
    run = context.output_dir / 'run_manifest.json'
    if run.exists() and read_json(run).get('identity') != context.identity:
        raise ValueError('run manifest identity changed; choose a new experiment')
    verify_unit_markers(context)
    units = enumerate_units(context)
    prepare_marker = marker_path(context, 'prepare')
    prepare_complete = prepare_marker.exists() and read_json(prepare_marker).get('status') == 'complete'
    if any(marker_path(context, unit).exists() for unit in units if unit not in ('prepare', 'analyze', 'evaluate')) and not prepare_complete:
        raise ValueError('downstream evidence exists without verified complete preparation')
    if prepare_complete:
        assert_prepared(context)
        source = read_json(context.output_dir / 'source_manifest.json')
        downloaded = read_json(context.output_dir / 'download_manifest.json')
        if downloaded['identity'] != context.identity or downloaded['sources'] != source['source_files'] or downloaded['duplicates'] != source['duplicates']:
            raise ValueError('source/download manifest disagreement')
        for entry in [*source['source_files'], source['duplicates']]:
            if sha256_file(entry['path']) != entry['sha256']:
                raise ValueError('local raw source or official duplicate list changed')
    for variant in context.config['upstream']['variants']:
        for seed in context.config['upstream']['seeds']:
            up = marker_path(context, f'upstream:{variant}:{seed}')
            cached = marker_path(context, f'cache:{variant}:{seed}')
            refiners = [marker_path(context, unit) for unit in units if unit.startswith(f'refine:{variant}:{seed}:')]
            complete = lambda p: p.exists() and read_json(p).get('status') == 'complete'
            if (cached.exists() or any(p.exists() for p in refiners)) and not complete(up):
                raise ValueError('dependent evidence exists without complete upstream')
            if any(p.exists() for p in refiners) and not complete(cached):
                raise ValueError('refiner evidence exists without complete cache')


@contextmanager
def exclusive_pool(context):
    import fcntl
    path = context.resolve(context.config.get('log_root', 'cross-domain/logs')) / 'sci_qm7x' / context.config['experiment'] / 'qm7x_pool.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another QM7-X pool already owns this experiment') from None
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpus', nargs='+', default=['0'])
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.retries < 0:
        parser.error('retries must be nonnegative')
    context = load_context(args.config, DOMAIN_ROOT)
    description = plan(context)
    devices = gpu_mapping(args.gpus)
    if args.dry_run:
        print(json.dumps({**description, 'gpu_indices': args.gpus, 'cuda_device_mapping': devices}, indent=2))
        return
    if context.config['runtime']['device'] != 'cuda':
        parser.error('formal cluster launch requires runtime.device=cuda')
    # Verification precedes directory creation, locking and the shared pool.
    preflight(context)
    with exclusive_pool(context):
        preflight(context)
        subprocess.run([sys.executable, str(DOMAIN_ROOT / 'scripts' / 'run_pool.py'), '--config', str(context.config_path),
                        '--gpus', *devices, '--retries', str(args.retries)], cwd=DOMAIN_ROOT.parent, check=True)


if __name__ == '__main__':
    main()
