"""MP single-node GPU pool using generic artifact-backed run_unit.py.

--dry-run only reads config/source and prints the complete planned matrix.
One worker per selected GPU runs a (variant, upstream seed) lifecycle.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

DOMAIN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DOMAIN_ROOT))
from pipeline.context import load_context
from pipeline.io import artifacts_match, read_json, write_json
from pipeline.units import enumerate_units
from pipeline.worker import marker_path


def validate(context):
    c = context.config
    if c['dataset'] != 'sci_mp_crystran' or c['runtime']['device'] != 'cuda':
        raise ValueError('requires a sci_mp_crystran CUDA config')
    if c['upstream']['variants'] != ['single_task', 'mt_main_only', 'multi_task']:
        raise ValueError('requires the declared ST, matched main-only and MT controls')
    for settings in (c['upstream'], c['refiner']):
        seeds = settings['seeds']
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError('seeds must be distinct nonnegative integers')
        if settings['epochs'] <= 0 or settings['batch_size'] <= 0 or settings['learning_rate'] <= 0:
            raise ValueError('invalid training budget')
        scheduler = settings.get('scheduler')
        if scheduler and scheduler['patience'] >= settings['early_stopping_patience']:
            raise ValueError('scheduler patience must be shorter than early stopping')
    if not c['refiner'].get('include_initial_checkpoint'):
        raise ValueError('native epoch 0 must be retained')
    if set(c['data']['sizes']) != {'a_train', 'a_val', 'b_train', 'b_val', 'y_test'}:
        raise ValueError('requires complete A/B/Y split')
    if any(type(n) is not int or n <= 0 for n in c['data']['sizes'].values()):
        raise ValueError('split sizes must be positive integers')
    m = c['model']
    if m['feature_size'] % 2 or m['feature_size'] % m['num_heads']:
        raise ValueError('width must divide evenly into atom/coordinate halves and attention heads')
    required = {'embedding', 'embedding_capacity', 'embedding_aux', 'embedding_aux_shuffle', 'embedding_hidden'}
    if set(c['refiner']['recipes']) != required:
        raise ValueError('cluster matrix must retain all configured readout controls')
    units = enumerate_units(context)
    if len(set(units)) != len(units):
        raise ValueError('duplicate unit identifiers')
    # Parameter counts from layer shapes only; no model/CUDA initialization.
    hidden = c['refiner']['hidden']
    if not hidden or any(type(value) is not int or value <= 0 for value in hidden):
        raise ValueError('readout hidden widths must be positive integers')
    width, ffn = m['feature_size'], m['dim_feedforward']
    backbone = 105*(width//2) + m['num_layers']*(4*width*width + 2*width*ffn + 9*width + ffn)
    st_parameters = backbone + (width+1)*m['st_head_hidden'] + m['st_head_hidden']+1
    mt_head_parameters = (width+1)*width + width+1
    readout_parameters = (width+1)*hidden[0]
    input_width = width + hidden[0] + 2
    for output_width in hidden:
        readout_parameters += (input_width+1)*output_width
        input_width = output_width
    readout_parameters += input_width+1
    # Static FP32 H-only lower bounds; tensor/checkpoint/container overhead is additional.
    downstream_n = sum(c['data']['sizes'][s] for s in ('b_train', 'b_val', 'y_test'))
    h_bytes = downstream_n * c['data']['max_atoms'] * m['feature_size'] * 4
    return {'identity': context.identity, 'config': str(context.config_path),
            'data': str(context.data_dir), 'output': str(context.output_dir),
            'materials': sum(c['data']['sizes'].values()), 'split': c['data']['sizes'],
            'model': m, 'refiner_hidden': hidden,
            'parameter_counts_static': {'single_task': st_parameters, 'mt_main_only': backbone+2*mt_head_parameters,
                                        'multi_task': backbone+2*mt_head_parameters, 'readout': readout_parameters},
            'upstream_seeds': c['upstream']['seeds'], 'downstream_seeds': c['refiner']['seeds'],
            'upstream_jobs': len(c['upstream']['variants'])*len(c['upstream']['seeds']),
            'refine_jobs': sum(u.startswith('refine:') for u in units), 'total_units': len(units),
            'H_cache_bytes_per_upstream': h_bytes,
            'H_cache_bytes_total': h_bytes*len(c['upstream']['variants'])*len(c['upstream']['seeds']),
            'units': units}


def gpu_devices(indices):
    """Respect an existing allocation's CUDA_VISIBLE_DEVICES (including UUIDs)."""
    inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
    if inherited is None:
        return {index: index for index in indices}
    devices = [value.strip() for value in inherited.split(',') if value.strip()]
    if not devices or '-1' in devices or any(int(index) >= len(devices) for index in indices):
        raise ValueError('GPU_POOL must index the devices allocated by CUDA_VISIBLE_DEVICES')
    return {index: devices[int(index)] for index in indices}

def unit_complete(context, unit):
    try:
        record = read_json(marker_path(context, unit))
        return (record['status'] == 'complete' and record['identity'] == context.identity
                and artifacts_match(record['artifacts']))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def seed_units(context, variant, seed):
    return [f'upstream:{variant}:{seed}', f'cache:{variant}:{seed}',
            *[f'refine:{variant}:{seed}:{recipe}:{ds}'
              for recipe in context.module('data').applicable_recipes(context, variant)
              for ds in context.config['refiner']['seeds']]]


def run_unit(context, config, unit, retries):
    if unit_complete(context, unit):
        print(f'{unit}: SHA256 verified; skip', flush=True)
        return
    if unit.startswith(('upstream:', 'cache:')):
        _, variant, seed = unit.split(':')
        downstream = [u for u in seed_units(context, variant, int(seed)) if u.startswith('refine:')]
        if unit.startswith('upstream:'):
            downstream.append(f'cache:{variant}:{seed}')
        if any(marker_path(context, u).exists() for u in downstream):
            raise RuntimeError(f'{unit}: invalid prerequisite with dependent evidence; use a new experiment')
    log_dir = marker_path(context, unit).parent.parent / 'unit_output'
    log_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        log = log_dir / f'{unit.replace(":", "__")}.attempt{attempt}.log'
        with log.open('a', encoding='utf-8') as handle:
            result = subprocess.run([sys.executable, str(DOMAIN_ROOT / 'scripts' / 'run_unit.py'),
                                     '--config', str(config), '--unit', unit], cwd=DOMAIN_ROOT.parent,
                                    stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode == 0 and unit_complete(context, unit):
            print(f'{unit}: complete; {log}', flush=True)
            return
        print(f'{unit}: attempt {attempt+1} failed; {log}', flush=True)
    raise RuntimeError(f'{unit} failed after {retries+1} attempts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpus', nargs='+', default=['0'])
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--variant', choices=['single_task', 'mt_main_only', 'multi_task'], help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.retries < 0 or len(set(args.gpus)) != len(args.gpus) or any(not g.isdigit() for g in args.gpus):
        parser.error('nonnegative retries and distinct nonnegative GPU indices required')
    if (args.variant is None) != (args.seed is None):
        parser.error('internal worker requires both variant and seed')
    context = load_context(args.config, DOMAIN_ROOT)
    plan = validate(context)
    devices = gpu_devices(args.gpus)
    config = context.config_path
    if args.dry_run:
        print(json.dumps({**plan, 'GPU_indices': args.gpus, 'CUDA_device_mapping': devices, 'execution': 'read-only plan; no initialization, download, subprocess or training'}, indent=2))
        return
    if args.variant is not None:
        if args.seed not in context.config['upstream']['seeds'] or not unit_complete(context, 'prepare'):
            parser.error('worker seed not configured or preparation not verified')
        for unit in seed_units(context, args.variant, args.seed):
            run_unit(context, config, unit, args.retries)
        return
    log_dir = marker_path(context, 'prepare').parent.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    running = {}
    def stop_pool(signum, frame):
        raise KeyboardInterrupt(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, stop_pool)
    with (log_dir / 'cluster.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('another pool owns this experiment')
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[args.gpus[0]]
        context.initialize()
        units = plan['units']
        if not unit_complete(context, 'prepare') and any(marker_path(context, u).exists() for u in units if u != 'prepare'):
            raise RuntimeError('preparation invalid but downstream evidence exists; use a new experiment')
        run_unit(context, config, 'prepare', args.retries)
        prerequisites = [u for u in units if u not in ('evaluate', 'analyze')]
        evaluation = marker_path(context, 'evaluate')
        if evaluation.exists() and read_json(evaluation).get('status') == 'complete':
            if not unit_complete(context, 'evaluate') or not all(unit_complete(context, u) for u in prerequisites):
                raise RuntimeError('completed Y evaluation has changed prerequisites; use a new experiment')
            run_unit(context, config, 'analyze', args.retries)
            print('completed evaluation verified; no Y re-evaluation', flush=True)
            return
        queue = [(variant, seed) for variant in context.config['upstream']['variants']
                 for seed in context.config['upstream']['seeds']]
        failed = []
        try:
            while queue or running:
                for gpu in args.gpus:
                    if gpu in running or not queue:
                        continue
                    variant, seed = queue.pop(0)
                    if all(unit_complete(context, u) for u in seed_units(context, variant, seed)):
                        print(f'{variant}:{seed}: complete lifecycle verified; skip', flush=True)
                        continue
                    log = log_dir / f'{variant}__seed{seed}.pool.log'
                    handle = log.open('a', encoding='utf-8')
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = devices[gpu]
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--config', str(config),
                                                '--variant', variant, '--seed', str(seed), '--retries', str(args.retries)],
                                               cwd=DOMAIN_ROOT.parent, env=env, stdout=handle,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    running[gpu] = (variant, seed, process, handle)
                    print(f'{variant}:{seed} -> GPU index {gpu}; {log}', flush=True)
                time.sleep(2)
                for gpu, (variant, seed, process, handle) in list(running.items()):
                    rc = process.poll()
                    if rc is None:
                        continue
                    handle.close()
                    del running[gpu]
                    if rc or not all(unit_complete(context, u) for u in seed_units(context, variant, seed)):
                        failed.append(f'{variant}:{seed}')
                    print(f'{variant}:{seed}: rc={rc}', flush=True)
        finally:
            for _, _, process, handle in running.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                handle.close()
        if failed or not all(unit_complete(context, u) for u in prerequisites):
            raise SystemExit(f'incomplete seed lifecycles: {failed}; Y evaluation not run')
        run_unit(context, config, 'evaluate', args.retries)
        run_unit(context, config, 'analyze', args.retries)
        complete = log_dir / 'cluster_complete.json'
        write_json(complete, {'status': 'complete', 'identity': context.identity, 'units': len(units),
                              'evaluation_artifacts': read_json(evaluation)['artifacts'],
                              'resume': 'verified unit-level completion; no optimizer-state resume'})
        print(f'cluster pool complete; {complete}', flush=True)


if __name__ == '__main__':
    main()