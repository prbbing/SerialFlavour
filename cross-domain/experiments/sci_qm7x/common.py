"""Protocol declarations and artifact identities; safe to import for dry-run."""
from pathlib import Path
from pipeline.io import read_json, sha256_file, json_hash, artifacts_match

SPLITS = ('a_train', 'a_val', 'b_train', 'b_val', 'y_test')
VARIANTS = ('single_task', 'multi_task')
RECIPES = ('embedding', 'embedding_capacity', 'embedding_aux', 'embedding_hidden', 'embedding_aux_shuffle')
PREPARED_FILES = ('prepared.pt', 'split_manifest.json', 'target_normalization.json', 'source_manifest.json')
FIELDS = {'z': 'atNUM', 'coords': 'atXYZ', 'main': 'eH', 'charge': 'hCHG', 'polarizability': 'atPOL'}
UNITS = {'coords': 'angstrom', 'main': 'eV', 'charge': 'elementary_charge', 'polarizability': 'bohr^3'}
SOURCE_URL = 'https://zenodo.org/records/4288677'
PAPER_URL = 'https://www.nature.com/articles/s41597-021-00812-2'
DUPLICATES_MD5 = '5d886ccac38877c8cb26c07704dd1034'


def validate_config(config):
    if config['dataset'] != 'sci_qm7x':
        raise ValueError('wrong dataset')
    data, model = config['data'], config['model']
    if set(data['sizes']) != set(SPLITS) or any(type(data['sizes'][s]) is not int or data['sizes'][s] <= 0 for s in SPLITS):
        raise ValueError('all five positive structure budgets required')
    if not data['sources'] or len({s['path'] for s in data['sources']}) != len(data['sources']):
        raise ValueError('distinct local HDF5 sources required')
    if type(data['sampling_seed']) is not int or type(data['split_seed']) is not int:
        raise ValueError('fixed integer sampling and split seeds required')
    if type(data['max_conformations_per_parent']) is not int or data['max_conformations_per_parent'] < 1:
        raise ValueError('positive conformation cap required')
    if not 1 <= data['max_atoms'] <= 23:
        raise ValueError('QM7-X has at most 23 atoms')
    if data.get('max_file_bytes', 0) > 5_000_000_000 or data.get('max_file_bytes', 0) <= 0:
        raise ValueError('single-file automatic write limit must be in (0, 5 GB]')
    if data.get('duplicates_md5') != DUPLICATES_MD5:
        raise ValueError('official DupMols.dat version 2 checksum required')
    variants = config['upstream']['variants']
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
        raise ValueError('nonempty distinct subset of declared upstream variants required')
    if config.get('workflow', {}).get('mode', 'all_cache') not in ('seed_lifecycle', 'all_cache'):
        raise ValueError('workflow.mode must be seed_lifecycle or all_cache')
    if config['refiner']['recipes'] != list(RECIPES):
        raise ValueError('complete declared readout controls required')
    for section in ('upstream', 'refiner'):
        settings = config[section]
        if len(settings['seeds']) < 2 or len(set(settings['seeds'])) != len(settings['seeds']) or any(type(s) is not int or s < 0 for s in settings['seeds']):
            raise ValueError('distinct nonnegative integer seeds required')
        if settings['epochs'] <= 0 or settings['batch_size'] <= 0 or settings['learning_rate'] <= 0:
            raise ValueError('invalid training budget')
        if settings.get('scheduler') and settings['scheduler']['patience'] >= settings['early_stopping_patience']:
            raise ValueError('scheduler patience must be shorter than early stopping')
    if not config['refiner'].get('include_initial_checkpoint'):
        raise ValueError('retain native epoch 0 candidate')
    if model['width'] < model['head_hidden'] or model['head_hidden'] < 2 or model['num_gaussians'] < 2 or model['cutoff'] <= 0:
        raise ValueError('invalid model dimensions')
    if len(config['upstream']['auxiliary_weights']) != 2 or any(w <= 0 for w in config['upstream']['auxiliary_weights']):
        raise ValueError('two positive local task weights required')


def applicable_recipes(context, variant):
    validate_config(context.config)
    if variant not in VARIANTS:
        raise ValueError('unknown upstream variant')
    return list(RECIPES if variant == 'multi_task' else RECIPES[:2])


def data_identity(context):
    return json_hash({name: sha256_file(context.output_dir / name) for name in PREPARED_FILES})


def assert_prepared(context):
    manifest = read_json(context.output_dir / 'split_manifest.json')
    if manifest['identity'] != context.identity:
        raise ValueError('prepared data/code/config identity mismatch')
    source = read_json(context.output_dir / 'source_manifest.json')
    if source['identity'] != context.identity:
        raise ValueError('source identity mismatch')
    if source['prepared_sha256'] != sha256_file(context.output_dir / 'prepared.pt'):
        raise ValueError('prepared tensor SHA256 mismatch')
    if source['normalization_sha256'] != sha256_file(context.output_dir / 'target_normalization.json'):
        raise ValueError('normalization SHA256 mismatch')
    if source['split_manifest_sha256'] != sha256_file(context.output_dir / 'split_manifest.json'):
        raise ValueError('split manifest SHA256 mismatch')
    all_indices, all_ids = [], []
    for split in SPLITS:
        indices, ids, groups = manifest['indices'][split], manifest['ids'][split], manifest['groups'][split]
        if len(indices) != context.config['data']['sizes'][split] or len(ids) != len(indices) or len(groups) != len(set(groups)):
            raise ValueError('split budget, IDs or group counts mismatch')
        parents = []
        for identity in ids:
            match = __import__('re').fullmatch(r'Geom-m([1-9]\d*)-i[1-9]\d*-c[1-9]\d*-(?:opt|d?[1-9]\d*)', identity)
            if match is None:
                raise ValueError('invalid split geometry ID')
            parents.append('m' + match.group(1))
        if not set(parents) <= set(groups) or sorted(set(parents)) != manifest['represented_groups'][split]:
            raise ValueError('actual split sample parents disagree with declared groups')
        if manifest['group_counts'][split] != len(groups) or manifest['represented_group_counts'][split] != len(set(parents)):
            raise ValueError('group count report mismatch')
        all_indices.extend(indices)
        all_ids.extend(ids)
    if any(type(index) is not int for index in all_indices) or sorted(all_indices) != list(range(len(all_indices))):
        raise ValueError('split indices must partition every prepared row exactly once')
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('duplicate geometry across split samples')
    for left in SPLITS:
        for right in SPLITS:
            if left != right and set(manifest['groups'][left]) & set(manifest['groups'][right]):
                raise ValueError('parent group leakage')
    return manifest


def expected_metadata(context):
    return {'identity': context.identity, 'data_identity': data_identity(context)}


def check_metadata(context, metadata):
    if any(metadata.get(key) != value for key, value in expected_metadata(context).items()):
        raise ValueError('artifact data/code/config identity mismatch')


def upstream_dir(context, variant, seed):
    return context.output_dir / 'upstream' / variant / f'seed{seed}'


def cache_dir(context, variant, seed):
    return context.output_dir / 'cache' / variant / f'seed{seed}'


def refiner_dir(context, variant, up_seed, recipe, seed):
    return context.output_dir / 'refine' / variant / f'up{up_seed}' / recipe / f'seed{seed}'


def verify_unit_markers(context):
    """Never trust generic pool's status-only skip markers without auditing first."""
    from pipeline.worker import marker_path
    from pipeline.units import enumerate_units
    units = enumerate_units(context)
    for unit in units:
        path = marker_path(context, unit)
        if not path.exists():
            continue
        record = read_json(path)
        if record.get('identity') != context.identity:
            raise ValueError(f'stale unit marker: {unit}; choose a new experiment')
        if record.get('status') == 'complete' and not artifacts_match(record.get('artifacts', [])):
            raise ValueError(f'corrupt completed unit: {unit}; choose a new experiment')
    log = context.resolve(context.config.get('log_root', 'cross-domain/logs')) / 'sci_qm7x' / context.config['experiment']
    for variant in context.config['upstream']['variants']:
        for seed in context.config['upstream']['seeds']:
            marker = log / 'seeds' / f'{variant}__seed{seed}.json'
            if not marker.exists() or read_json(marker).get('status') != 'complete':
                continue
            needed = [f'upstream:{variant}:{seed}', f'cache:{variant}:{seed}'] + [
                f'refine:{variant}:{seed}:{recipe}:{ds}' for recipe in applicable_recipes(context, variant)
                for ds in context.config['refiner']['seeds']]
            for unit in needed:
                path = marker_path(context, unit)
                if not path.exists() or read_json(path).get('status') != 'complete':
                    raise ValueError(f'status-only seed marker has incomplete unit: {unit}')


def require_lifecycle_filters(context, keys):
    if context.config.get('workflow',{}).get('mode') == 'seed_lifecycle' and any(context.filters.get(key) is None for key in keys):
        raise ValueError('resource mode requires exact unit filters; use scripts/run_cluster.py or shared run_lifecycle.py')
