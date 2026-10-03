"""NYUv2 single-node GPU pool over the generic run_unit.py interface.

Dry-run only reads configuration and source. Real runs serialize preparation,
dispatch complete (variant, upstream seed) lifecycles, then evaluate/analyze.
Each skipped unit requires matching identity and actual artifact checksums.
"""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

DOMAIN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DOMAIN_ROOT))

from pipeline.context import load_context
from pipeline.io import artifacts_match, read_json
from pipeline.units import enumerate_units
from pipeline.worker import marker_path


def unit_complete(context, unit):
    path = marker_path(context, unit)
    try:
        record = read_json(path)
        return (record['status'] == 'complete' and record['identity'] == context.identity
                and artifacts_match(record['artifacts']))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_unit(context, config, unit, python, retries):
    if unit_complete(context, unit):
        print(f'{unit}: verified; skip', flush=True)
        return
    # Refitting a prerequisite invalidates selections made from it. Refuse to
    # mix old downstream evidence; do not erase it automatically.
    if unit.startswith(('upstream:', 'cache:')):
        parts = unit.split(':')
        downstream = [f'refine:{parts[1]}:{parts[2]}:{r}:{s}'
                      for r in context.module('data').applicable_recipes(context, parts[1])
                      for s in context.config['refiner']['seeds']]
        if unit.startswith('upstream:'):
            downstream.append(f'cache:{parts[1]}:{parts[2]}')
        if any(marker_path(context, u).exists() for u in downstream):
            raise RuntimeError(f'{unit} must be rebuilt but dependent markers exist; use a new experiment to preserve evidence')
    log_dir = marker_path(context, unit).parent.parent / 'unit_output'
    log_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries + 1):
        log = log_dir / f'{unit.replace(":", "__")}.attempt{attempt}.log'
        with log.open('a', encoding='utf-8') as handle:
            result = subprocess.run([python, str(DOMAIN_ROOT / 'scripts' / 'run_unit.py'),
                                     '--config', str(config), '--unit', unit],
                                    cwd=DOMAIN_ROOT.parent, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode == 0 and unit_complete(context, unit):
            print(f'{unit}: complete; {log}', flush=True)
            return
        print(f'{unit}: attempt {attempt+1} failed; {log}', flush=True)
    raise RuntimeError(f'{unit} failed after {retries+1} attempts')


def seed_units(context, variant, seed):
    return [f'upstream:{variant}:{seed}', f'cache:{variant}:{seed}',
            *[f'refine:{variant}:{seed}:{recipe}:{ds}'
              for recipe in context.module('data').applicable_recipes(context, variant)
              for ds in context.config['refiner']['seeds']]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpus', nargs='+', default=['0'])
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--variant', choices=['single_task', 'multi_task'])
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    config = Path(args.config).resolve()
    context = load_context(config, DOMAIN_ROOT)
    if context.config['dataset'] != 'cv_nyu_mtan' or args.retries < 0:
        parser.error('requires cv_nyu_mtan config and nonnegative retries')
    if len(set(args.gpus)) != len(args.gpus) or any(not g.isdigit() for g in args.gpus):
        parser.error('--gpus must be distinct nonnegative physical GPU IDs')
    if (args.variant is None) != (args.seed is None):
        parser.error('--variant and --seed must be provided together')
    units = enumerate_units(context)
    if args.dry_run:
        print(f'identity={context.identity}\ndata={context.data_dir}\noutput={context.output_dir}')
        print(f'split={context.config["data"]["split_mode"]}; units={len(units)}; GPUs={args.gpus}')
        for unit in units:
            print(unit)
        return
    # run_unit checks the requested CUDA device even for preparation; actual
    # Parquet conversion remains on CPU. Dry-run never configures CUDA.
    if args.variant is not None:
        if args.seed not in context.config['upstream']['seeds'] or args.variant not in context.config['upstream']['variants']:
            parser.error('seed/variant not configured')
        for unit in seed_units(context, args.variant, args.seed):
            run_unit(context, config, unit, sys.executable, args.retries)
        return
    log_root = context.resolve(context.config.get('log_root', 'cross-domain/logs'))
    log_dir = log_root / context.config['dataset'] / context.config['experiment']
    log_dir.mkdir(parents=True, exist_ok=True)
    def stop_pool(signum, frame):
        raise KeyboardInterrupt(f'pool interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, stop_pool)
    with (log_dir / 'cluster.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('another pool is running this experiment')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus[0]
        # Initialize once before spawning workers. The generic runner refuses
        # changed code/config identity; completed results are never overwritten.
        context.initialize()
        if not unit_complete(context, 'prepare'):
            other = [u for u in units if u != 'prepare']
            if any(marker_path(context, u).exists() for u in other):
                raise RuntimeError('prepare invalid but training markers exist; use a new experiment')
        run_unit(context, config, 'prepare', sys.executable, args.retries)
        # A successful evaluation is immutable: only verify and skip this run.
        evaluation = marker_path(context, 'evaluate')
        if evaluation.exists() and read_json(evaluation).get('status') == 'complete':
            if not unit_complete(context, 'evaluate') or any(not unit_complete(context, u) for u in units if u not in ('evaluate', 'analyze')):
                raise RuntimeError('evaluation exists but prerequisites changed; use a new experiment')
            run_unit(context, config, 'analyze', sys.executable, args.retries)
            print('completed evaluation verified; no Y re-evaluation', flush=True)
            return
        queue = [(v, s) for v in context.config['upstream']['variants'] for s in context.config['upstream']['seeds']]
        running, failed = {}, []
        try:
            while queue or running:
                for gpu in args.gpus:
                    if gpu in running or not queue:
                        continue
                    variant, seed = queue.pop(0)
                    log = log_dir / f'{variant}__seed{seed}.pool.log'
                    handle = log.open('a', encoding='utf-8')
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = gpu
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--config', str(config),
                                                '--variant', variant, '--seed', str(seed), '--retries', str(args.retries)],
                                               env=env, cwd=DOMAIN_ROOT.parent, stdout=handle,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    running[gpu] = (variant, seed, process, handle)
                    print(f'{variant}:{seed} -> physical GPU {gpu}; {log}', flush=True)
                time.sleep(2)
                for gpu, (variant, seed, process, handle) in list(running.items()):
                    rc = process.poll()
                    if rc is None:
                        continue
                    handle.close()
                    del running[gpu]
                    if rc:
                        failed.append(f'{variant}:{seed}')
                    print(f'{variant}:{seed}: rc={rc}', flush=True)
        finally:
            for _, _, process, handle in running.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                handle.close()
        if failed:
            raise SystemExit(f'failed seed lifecycles: {failed}; Y evaluation not run')
        run_unit(context, config, 'evaluate', sys.executable, args.retries)
        run_unit(context, config, 'analyze', sys.executable, args.retries)
        print('cluster pool complete', flush=True)


if __name__ == '__main__':
    main()
