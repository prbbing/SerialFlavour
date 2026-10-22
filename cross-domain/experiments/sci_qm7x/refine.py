"""Frozen tensor publication and B-only native-initialized full-H readouts."""
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader
from pipeline.fit import fit, save_checkpoint
from pipeline.io import read_json, write_json, sha256_file
from pipeline.runtime import configure, seed_all, to_device
from experiments.sci_qm7x.common import (
    applicable_recipes, validate_config, assert_prepared, expected_metadata, check_metadata, require_lifecycle_filters,
    upstream_dir, cache_dir, refiner_dir,
)
from experiments.sci_qm7x.data import make_loader
from experiments.sci_qm7x.training import load_upstream, physical_output
from experiments.sci_qm7x.model import Readout

CACHE_SPLITS = ('b_train', 'b_val', 'y_test')


def cache_splits(context):
    return CACHE_SPLITS[:2] if context.config.get('workflow', {}).get('mode') == 'seed_lifecycle' else CACHE_SPLITS
CACHE_FIELDS = {'H', 'g', 'native', 'aux_prediction', 'aux_hidden', 'coords', 'mask', 'main', 'ids', 'metadata'}


def cache(context):
    require_lifecycle_filters(context,('variant','seed'))
    validate_config(context.config)
    split_manifest = assert_prepared(context)
    device, artifacts = configure(context.config['runtime']), []
    norm = read_json(context.output_dir / 'target_normalization.json')
    for variant in context.config['upstream']['variants']:
        if context.filters.get('variant') not in (None, variant):
            continue
        for seed in context.config['upstream']['seeds']:
            if context.filters.get('seed') not in (None, seed):
                continue
            checkpoint = upstream_dir(context, variant, seed) / 'best.pt'
            before = sha256_file(checkpoint)
            model, _ = load_upstream(context, variant, seed, device)
            directory, hashes = cache_dir(context, variant, seed), {}
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise ValueError('upstream was not completely frozen in eval mode')
            with torch.inference_mode():
                for split in cache_splits(context):
                    collected = {key: [] for key in CACHE_FIELDS - {'metadata', 'ids'}}
                    ids = []
                    for batch in make_loader(context, split, seed, context.config['upstream']['batch_size']):
                        batch = to_device(batch, device)
                        output = physical_output(model(batch), norm)
                        values = {'H': output['H'], 'g': output['g'], 'native': output['main'],
                                  'aux_prediction': output['auxiliary'], 'aux_hidden': output['aux_hidden'],
                                  'coords': batch['coords'], 'mask': batch['mask'], 'main': batch['main']}
                        for key, value in values.items():
                            collected[key].append(value.detach().cpu())
                        ids.extend(batch['id'])
                    if ids != split_manifest['ids'][split]:
                        raise ValueError('cache sample order mismatch')
                    payload = {key: torch.cat(values) for key, values in collected.items()}
                    payload['ids'] = ids
                    payload['metadata'] = {**expected_metadata(context), 'variant': variant, 'seed': seed, 'split': split,
                                           'checkpoint_sha256': before, 'auxiliary_truth_stored': False,
                                           'units': {'native': 'eV', 'aux_prediction': ['e', 'bohr^3']}}
                    path = directory / f'{split}.pt'
                    size = sum(v.numel() * v.element_size() for v in payload.values() if torch.is_tensor(v))
                    if size + len(str(ids).encode()) + 1_000_000 > context.config['data']['max_file_bytes']:
                        raise ValueError('cache split exceeds single-file write limit')
                    save_checkpoint(path, payload)
                    if path.stat().st_size > context.config['data']['max_file_bytes']:
                        path.unlink()
                        raise ValueError('serialized cache exceeded write limit')
                    hashes[path.name] = sha256_file(path)
                    artifacts.append(path)
            if sha256_file(checkpoint) != before:
                raise ValueError('frozen upstream checkpoint mutated during cache publication')
            manifest = directory / 'cache_manifest.json'
            write_json(manifest, {**expected_metadata(context), 'variant': variant, 'seed': seed,
                                  'checkpoint_sha256': before, 'cache_sha256': hashes, 'fields': sorted(CACHE_FIELDS),
                                  'unchanged_after_cache': True, 'eval_mode': True, 'all_parameters_frozen': True,
                                  'auxiliary_truth_stored': False, 'geometry_access': 'all recipes receive same coords/mask',
                                  'native_unit': 'eV', 'auxiliary_units': ['e', 'bohr^3']})
            artifacts.append(manifest)
    if not artifacts:
        raise ValueError('no upstream matched cache filters')
    return artifacts


def load_cache(context, variant, seed, split):
    if split not in cache_splits(context):
        raise ValueError('resource lifecycle caches B only; Y must be streamed')
    directory = cache_dir(context, variant, seed)
    manifest = read_json(directory / 'cache_manifest.json')
    check_metadata(context, manifest)
    if manifest['variant'] != variant or manifest['seed'] != seed or manifest.get('auxiliary_truth_stored') is not False:
        raise ValueError('cache variant/seed/frozen-boundary mismatch')
    if set(manifest['cache_sha256']) != {f'{s}.pt' for s in cache_splits(context)}:
        raise ValueError('incomplete cache matrix')
    path = directory / f'{split}.pt'
    if sha256_file(path) != manifest['cache_sha256'][path.name]:
        raise ValueError('cache SHA256 mismatch')
    if manifest['checkpoint_sha256'] != sha256_file(upstream_dir(context, variant, seed) / 'best.pt'):
        raise ValueError('cache upstream checkpoint mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if set(saved) != CACHE_FIELDS:
        raise ValueError('unexpected cache field or missing required field; auxiliary truth forbidden')
    metadata = saved['metadata']
    check_metadata(context, metadata)
    if (metadata['variant'], metadata['seed'], metadata['split'], metadata['checkpoint_sha256']) != (
            variant, seed, split, manifest['checkpoint_sha256']):
        raise ValueError('cache metadata mismatch')
    expected_ids = read_json(context.output_dir / 'split_manifest.json')['ids'][split]
    if saved['ids'] != expected_ids:
        raise ValueError('cache sample identities/order mismatch')
    if len(saved['main']) != len(expected_ids) or not saved['mask'].any(1).all():
        raise ValueError('cache sample/mask mismatch')
    for name, value in saved.items():
        if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'nonfinite cache tensor: {name}')
    return saved


def shuffle_atoms(values, mask, ids, seed):
    """Deterministic nontrivial within-molecule shuffle; channels stay paired."""
    result = values.clone()
    for index, identity in enumerate(ids):
        valid = mask[index].nonzero(as_tuple=True)[0]
        if len(valid) <= 1:
            continue
        digest = int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest(), 16)
        generator = torch.Generator().manual_seed(digest % (2**63 - 1))
        order = torch.randperm(len(valid), generator=generator)
        # Avoid an unchanged control for an individual molecule.
        if torch.equal(order, torch.arange(len(valid))):
            order = order.roll(1)
        result[index, valid] = values[index, valid[order]]
    return result


def features(saved, recipe, seed, extra_width):
    if recipe not in ('embedding', 'embedding_capacity', 'embedding_aux', 'embedding_hidden', 'embedding_aux_shuffle'):
        raise ValueError('unknown recipe')
    extra = saved['H'].new_zeros((*saved['H'].shape[:2], extra_width))
    if recipe == 'embedding_capacity':
        extra = saved['H'][..., :extra_width].clone()
    elif recipe == 'embedding_hidden':
        extra = saved['aux_hidden'].clone()
    elif recipe in ('embedding_aux', 'embedding_aux_shuffle'):
        prediction = saved['aux_prediction']
        if recipe == 'embedding_aux_shuffle':
            prediction = shuffle_atoms(prediction, saved['mask'], saved['ids'], seed)
        extra[..., :2] = prediction
    return {key: saved[key] for key in ('H', 'g', 'native', 'coords', 'mask', 'main')} | {'extra': extra}


def normalization(values):
    stats = {}
    for key in ('H', 'extra', 'g', 'native'):
        value = values[key][values['mask']] if key in ('H', 'extra') else values[key]
        value = value.double()
        stats[key + '_mean'] = value.mean(0).float()
        stats[key + '_std'] = value.std(0, unbiased=False).clamp_min(1e-8).float()
    return stats


class FrozenDataset(Dataset):
    def __init__(self, values, stats):
        self.values = dict(values)
        for key in ('H', 'extra', 'g'):
            self.values[key] = (values[key] - stats[key + '_mean']) / stats[key + '_std']
            if key in ('H', 'extra'):
                self.values[key] = self.values[key] * values['mask'][..., None]
        self.values['native_scaled'] = (values['native'] - stats['native_mean']) / stats['native_std']

    def __len__(self):
        return len(self.values['main'])

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.values.items()}


def frozen_loader(values, stats, batch_size, seed=1, shuffle=False):
    return DataLoader(FrozenDataset(values, stats), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def train(context):
    require_lifecycle_filters(context,('variant','upstream_seed','recipe','downstream_seed'))
    validate_config(context.config)
    assert_prepared(context)
    device, artifacts = configure(context.config['runtime']), []
    settings = context.config['refiner']
    norm = read_json(context.output_dir / 'target_normalization.json')
    for variant in context.config['upstream']['variants']:
        if context.filters.get('variant') not in (None, variant):
            continue
        for up_seed in context.config['upstream']['seeds']:
            if context.filters.get('upstream_seed') not in (None, up_seed):
                continue
            saved = {split: load_cache(context, variant, up_seed, split) for split in ('b_train', 'b_val')}
            for recipe in applicable_recipes(context, variant):
                if context.filters.get('recipe') not in (None, recipe):
                    continue
                for seed in settings['seeds']:
                    if context.filters.get('downstream_seed') not in (None, seed):
                        continue
                    seed_all(seed)
                    width = context.config['model']['head_hidden']
                    train_values = features(saved['b_train'], recipe, seed + 1000, width)
                    val_values = features(saved['b_val'], recipe, seed + 2000, width)
                    stats = normalization(train_values)
                    directory = refiner_dir(context, variant, up_seed, recipe, seed)
                    stats_path = directory / 'input_normalization.pt'
                    save_checkpoint(stats_path, stats)
                    model = Readout(context.config['model'], settings, norm['main_std'])

                    def loss(current, batch):
                        return ((current(batch) - batch['main']) / norm['main_std']).square().mean(), len(batch['main'])

                    def score(current, batches, dev):
                        total, n = 0., 0
                        for batch in batches:
                            batch = to_device(batch, dev)
                            total += (current(batch) - batch['main']).abs().sum().item()
                            n += len(batch['main'])
                        return total / n

                    metadata = {**expected_metadata(context), 'variant': variant, 'upstream_seed': up_seed,
                                'seed': seed, 'recipe': recipe, 'training_split': 'b_train', 'selection_split': 'b_val',
                                'input_normalization_fit': 'b_train_valid_atoms_and_molecules', 'auxiliary_truth_used': False,
                                'upstream_checkpoint_sha256': sha256_file(upstream_dir(context, variant, up_seed) / 'best.pt'),
                                'cache_manifest_sha256': sha256_file(cache_dir(context, variant, up_seed) / 'cache_manifest.json'),
                                'input_normalization_sha256': sha256_file(stats_path), 'native_epoch0_candidate': True,
                                'geometry_access': 'same pairwise radial distances for every recipe',
                                'full_H_access': 'learned attention, mean, max; g, native HOMO and atom count',
                                'extra_width': width, 'capacity_matching': 'identical architecture and all trainable parameters for all recipes',
                                'shuffle_seeds': {'b_train': seed + 1000, 'b_val': seed + 2000, 'y_test': seed + 3000}}
                    artifacts.append(stats_path)
                    artifacts.extend(fit(model, frozen_loader(train_values, stats, settings['batch_size'], seed, True),
                                         frozen_loader(val_values, stats, settings['batch_size'], seed), loss, score,
                                         directory, settings, metadata, device))
    if not artifacts:
        raise ValueError('no configured refiner matched runtime filters')
    return artifacts
