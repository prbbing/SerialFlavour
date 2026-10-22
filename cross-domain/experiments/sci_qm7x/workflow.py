"""Small QM7-X backend for the shared sequential seed lifecycle (no tensor imports)."""
from pathlib import Path
from pipeline.io import read_json, sha256_file, artifact_record, artifacts_match
from pipeline.units import enumerate_units
from experiments.sci_qm7x.common import (
    validate_config, expected_metadata, check_metadata, assert_prepared,
    upstream_dir, cache_dir, refiner_dir, applicable_recipes,
)


def validate(context):
    validate_config(context.config)
    if context.config.get('workflow', {}).get('mode') != 'seed_lifecycle':
        raise ValueError('shared lifecycle requires workflow.mode=seed_lifecycle; use the legacy pool for all-cache configs')
    if context.config.get('workflow', {}).get('max_active_seeds', 1) != 1:
        raise ValueError('QM7-X resource lifecycle permits one active upstream seed')
    run = context.output_dir / 'run_manifest.json'
    if run.exists() and read_json(run).get('identity') != context.identity:
        raise ValueError('run identity changed; choose a new experiment')


def verify_inputs(context):
    """Expensive input verification, once after prepare / at driver startup."""
    assert_prepared(context)
    source = read_json(context.output_dir / 'source_manifest.json')
    downloaded = read_json(context.output_dir / 'download_manifest.json')
    if downloaded['identity'] != context.identity or downloaded['sources'] != source['source_files'] or downloaded['duplicates'] != source['duplicates']:
        raise ValueError('source/download manifest disagreement')
    for entry in [*source['source_files'], source['duplicates']]:
        if sha256_file(entry['path']) != entry['sha256']:
            raise ValueError('local raw source or duplicate list changed')


def plan(context):
    validate(context)
    config = context.config
    if len(config['upstream']['seeds']) != 5 or len(config['refiner']['seeds']) != 5:
        raise ValueError('formal cluster plan requires complete 5 x 5 seeds')
    units = enumerate_units(context)
    w, h = config['model']['width'], config['model']['head_hidden']
    radial, blocks = config['model']['num_gaussians'], config['model']['num_interactions']
    upstream = 18*w + blocks*(4*w*w+(radial+4)*w) + (w+1)*h+h+1+(2*w+1)*h+2*(h+1)
    hidden = config['refiner']['hidden']
    first = hidden[0]
    readout = (w+h+config['refiner']['geometry_bins']+1)*first+(first+1)*first+first+1
    previous = 3*first+w+2
    for output in hidden:
        readout += (previous+1)*output
        previous = output
    readout += previous+1
    atoms = config['data']['max_atoms']
    per_row = atoms*((w+h+5)*4+1)+(w+2)*4
    b_count = sum(config['data']['sizes'][s] for s in ('b_train','b_val'))
    y_count = config['data']['sizes']['y_test']
    count = len(config['upstream']['variants'])*len(config['upstream']['seeds'])
    per_seed = b_count*per_row
    peak_cache = per_seed*(1 if config.get('workflow',{}).get('release_cache',True) else count)
    legacy_total = (b_count+y_count)*per_row*count
    prepared = sum(config['data']['sizes'].values())*(atoms*(8+3*4+1+2*4)+4)
    permanent_models = count*upstream*4 + sum(len(applicable_recipes(context,v)) for v in config['upstream']['variants'])*len(config['upstream']['seeds'])*len(config['refiner']['seeds'])*readout*4
    return {'identity': context.identity, 'config': str(context.config_path), 'data': str(context.data_dir),
            'output': str(context.output_dir), 'structure_budget': sum(config['data']['sizes'].values()),
            'structure_splits': config['data']['sizes'], 'group_counts': 'reported after local HDF5 prepare',
            'variants': config['upstream']['variants'], 'upstream_seeds': config['upstream']['seeds'],
            'downstream_seeds': config['refiner']['seeds'], 'upstream_jobs': count,
            'refine_jobs': sum(u.startswith('refine:') for u in units), 'total_units': len(units),
            'evaluation_rows': sum(1+len(applicable_recipes(context,v))*len(config['refiner']['seeds']) for v in config['upstream']['variants'])*len(config['upstream']['seeds']),
            'parameters_static': {'single_task': upstream, 'multi_task': upstream, 'every_readout': readout},
            'max_active_seeds': 1, 'cache_splits': ['b_train','b_val'], 'stream_Y': True,
            'save_predictions': config.get('evaluation',{}).get('save_predictions',False),
            'cache_tensor_bytes_per_upstream_estimate': per_seed,
            'cache_tensor_bytes_total_estimate': peak_cache,
            'peak_regenerable_cache_bytes_estimate': peak_cache,
            'legacy_all_seeds_B_Y_cache_bytes_estimate': legacy_total,
            'prepared_tensor_bytes_estimate': prepared, 'permanent_model_tensor_bytes_estimate': permanent_models,
            'peak_output_tensor_bytes_estimate': prepared+permanent_models+peak_cache,
            'estimate_excludes': 'raw input archives/HDF5, IDs, serialization overhead, histories, JSON, optional predictions; tensor estimates are not measured peak RSS/VRAM/disk',
            'seed_order': [[v,s] for v in config['upstream']['variants'] for s in config['upstream']['seeds']],
            'units': units, 'execution': 'read-only plan; no model, CUDA, data reads, mkdir, subprocess or download'}


def evaluate_seed(context, variant, seed):
    from experiments.sci_qm7x.evaluate import evaluate_seed as run
    return run(context, variant, seed)


def seed_artifacts(context, variant, seed):
    paths = [upstream_dir(context,variant,seed)/name for name in ('best.pt','training_manifest.json','history.json','history.csv')]
    for recipe in applicable_recipes(context,variant):
        for ds in context.config['refiner']['seeds']:
            directory = refiner_dir(context,variant,seed,recipe,ds)
            paths.extend(directory/name for name in ('best.pt','input_normalization.pt','training_manifest.json','history.json','history.csv'))
    paths.append(cache_dir(context,variant,seed)/'cache_manifest.json')
    directory = context.output_dir/'evaluation'/'seeds'/variant/f'seed{seed}'
    manifest_path = directory/'evaluation_manifest.json'
    manifest = read_json(manifest_path)
    check_metadata(context,manifest)
    if (manifest.get('variant'),manifest.get('upstream_seed'),manifest.get('status')) != (variant,seed,'complete') or not artifacts_match(manifest['artifacts']):
        raise ValueError('seed evaluation evidence is incomplete or corrupt')
    paths.extend(Path(record['path']) for record in manifest['artifacts'])
    paths.append(manifest_path)
    paths.extend(context.output_dir/name for name in ('source_manifest.json','split_manifest.json','target_normalization.json','download_manifest.json','run_manifest.json','resolved_config.json') if (context.output_dir/name).exists())
    return paths


def cleanup_roots(context, variant, seed):
    return [cache_dir(context,variant,seed),context.output_dir/'scratch'/variant/f'seed{seed}']


def cleanup_records(context, variant, seed):
    directory = cache_dir(context,variant,seed)
    manifest = read_json(directory/'cache_manifest.json')
    check_metadata(context,manifest)
    if (manifest['variant'],manifest['seed']) != (variant,seed) or set(manifest['cache_sha256']) != {'b_train.pt','b_val.pt'}:
        raise ValueError('only current seed B tensors are eligible for cleanup')
    records = [artifact_record(directory/name) for name in ('b_train.pt','b_val.pt')]
    if any(record['sha256'] != manifest['cache_sha256'][Path(record['path']).name] for record in records):
        raise ValueError('cleanup candidate cache hash mismatch')
    return records


def aggregate(context):
    from pipeline.lifecycle import verify_receipts
    from experiments.sci_qm7x.evaluate import evaluate
    from experiments.sci_qm7x.analysis import analyze
    verify_receipts(context)
    return evaluate(context)+analyze(context)
