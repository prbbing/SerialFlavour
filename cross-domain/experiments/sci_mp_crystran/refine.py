"""Freeze complete upstream; B readouts use only Eg truth and frozen predictions."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pipeline.fit import fit, save_checkpoint
from pipeline.io import read_json, write_json, sha256_file
from pipeline.runtime import configure, seed_all, to_device
from experiments.sci_mp_crystran.data import make_loader, data_identity, applicable_recipes
from experiments.sci_mp_crystran.model import Readout
from experiments.sci_mp_crystran.training import load_upstream, upstream_dir, physical_output


def cache_dir(context, variant, seed):
    return context.output_dir / 'cache' / variant / f'seed{seed}'


def cache(context):
    device = configure(context.config['runtime'])
    artifacts = []
    norm = read_json(context.output_dir / 'target_normalization.json')
    for variant in context.config['upstream']['variants']:
        if context.filters.get('variant') not in (None, variant):
            continue
        for seed in context.config['upstream']['seeds']:
            if context.filters.get('seed') not in (None, seed):
                continue
            model, _ = load_upstream(context, variant, seed, device)
            checkpoint = upstream_dir(context, variant, seed) / 'best.pt'
            before = sha256_file(checkpoint)
            directory = cache_dir(context, variant, seed)
            directory.mkdir(parents=True, exist_ok=True)
            max_atoms, width = context.config['data']['max_atoms'], context.config['model']['feature_size']
            for split in ('b_train', 'b_val', 'y_test'):
                chunks = []
                ids = []
                with torch.inference_mode():
                    for batch in make_loader(context, split, batch_size=context.config['upstream']['batch_size']):
                        assert 'auxiliary' not in batch
                        batch = to_device(batch, device)
                        output = physical_output(model(batch['z'], batch['coords'], batch['mask']), norm)
                        count, length = batch['mask'].shape
                        h = torch.zeros(count, max_atoms, width)
                        mask = torch.ones(count, max_atoms, dtype=torch.bool)
                        h[:, :length], mask[:, :length] = output['H'].cpu(), batch['mask'].cpu()
                        chunk = {'H': h, 'mask': mask, 'g': output['g'].cpu(), 'native': output['main'].cpu(),
                                 'main': batch['main'].cpu()}
                        if variant == 'multi_task':
                            chunk['aux_prediction'] = output['auxiliary'].cpu()
                            chunk['aux_hidden'] = output['aux_hidden'].cpu()
                        chunks.append(chunk)
                        ids.extend(batch['id'])
                payload = {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
                payload['ids'] = ids
                payload['metadata'] = {'identity': context.identity, 'data_identity': data_identity(context),
                                       'checkpoint_sha256': before, 'split': split, 'variant': variant, 'seed': seed,
                                       'auxiliary_truth_stored': False, 'frozen_parameters': all(not p.requires_grad for p in model.parameters()),
                                       'eval_mode': not model.training, 'units': {'main': 'eV', 'aux_prediction': 'eV/atom'}}
                path = directory / f'{split}.pt'
                save_checkpoint(path, payload)
                artifacts.append(path)
            if sha256_file(checkpoint) != before:
                raise RuntimeError('upstream checkpoint changed during caching')
            manifest = directory / 'cache_manifest.json'
            write_json(manifest, {'checkpoint_sha256': before, 'unchanged_after_cache': True, 'eval_mode': True,
                                  'all_parameters_frozen': True, 'auxiliary_truth_stored': False,
                                  'cache_sha256': {p.name: sha256_file(p) for p in artifacts if p.parent == directory},
                                  'fields': list(payload), 'identity': context.identity})
            artifacts.append(manifest)
    return artifacts


def load_cache(context, variant, seed, split):
    directory = cache_dir(context, variant, seed)
    path = directory / f'{split}.pt'
    manifest = read_json(directory / 'cache_manifest.json')
    if manifest['cache_sha256'][path.name] != sha256_file(path):
        raise ValueError('cache hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    metadata = saved['metadata']
    if metadata['identity'] != context.identity or metadata['data_identity'] != data_identity(context):
        raise ValueError('cache data/code/config identity mismatch')
    if metadata['checkpoint_sha256'] != sha256_file(upstream_dir(context, variant, seed) / 'best.pt'):
        raise ValueError('cache upstream checkpoint mismatch')
    if 'auxiliary' in saved or 'aux_truth' in saved:
        raise ValueError('auxiliary truth in cache is forbidden')
    return saved


def features(saved, recipe, seed):
    if recipe not in ('embedding', 'embedding_capacity', 'embedding_aux', 'embedding_aux_shuffle', 'embedding_hidden'):
        raise ValueError('unknown recipe')
    if recipe == 'embedding':
        extra = torch.zeros_like(saved['main'])
    elif recipe == 'embedding_capacity':
        extra = saved['g'][:, 0].clone()  # deterministic duplicate; same input width as one auxiliary scalar
    elif recipe == 'embedding_hidden':
        # One task-specific hidden scalar, exactly capacity matched to scalar Ef output.
        extra = saved['aux_hidden'][:, 0].clone()
    else:
        extra = saved['aux_prediction'].clone()
        if recipe == 'embedding_aux_shuffle':
            # Independent, predeclared split-wise shuffles, retraining each shuffled readout.
            generator = torch.Generator().manual_seed(seed)
            extra = extra[torch.randperm(len(extra), generator=generator)]
    return {key: saved[key] for key in ('H', 'mask', 'g', 'native', 'main')} | {'extra': extra}


def normalization(batch):
    return {'g_mean': batch['g'].mean(0), 'g_std': batch['g'].std(0, unbiased=False).clamp_min(1e-8),
            'extra_mean': batch['extra'].mean(), 'extra_std': batch['extra'].std(unbiased=False).clamp_min(1e-8)}


class FrozenDataset(Dataset):
    def __init__(self, values, stats):
        self.values = {**values, 'g': (values['g']-stats['g_mean'])/stats['g_std'],
                       'extra': (values['extra']-stats['extra_mean'])/stats['extra_std']}

    def __len__(self):
        return len(self.values['main'])

    def __getitem__(self, i):
        return {key: value[i] for key, value in self.values.items()}


def loader(values, stats, seed, batch_size, shuffle=False):
    return DataLoader(FrozenDataset(values, stats), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def refiner_dir(context, variant, up_seed, recipe, seed):
    return context.output_dir / 'refine' / variant / f'up{up_seed}' / recipe / f'seed{seed}'


def recipes(context, variant):
    return applicable_recipes(context, variant)


def train(context):
    device = configure(context.config['runtime'])
    settings = context.config['refiner']
    norm = read_json(context.output_dir / 'target_normalization.json')
    artifacts = []
    for variant in context.config['upstream']['variants']:
        if context.filters.get('variant') not in (None, variant):
            continue
        for up_seed in context.config['upstream']['seeds']:
            if context.filters.get('upstream_seed') not in (None, up_seed):
                continue
            saved = {s: load_cache(context, variant, up_seed, s) for s in ('b_train', 'b_val')}
            for recipe in recipes(context, variant):
                if context.filters.get('recipe') not in (None, recipe):
                    continue
                for seed in settings['seeds']:
                    if context.filters.get('downstream_seed') not in (None, seed):
                        continue
                    seed_all(seed)
                    train_values = features(saved['b_train'], recipe, seed + 1000)
                    val_values = features(saved['b_val'], recipe, seed + 2000)
                    stats = normalization(train_values)
                    model = Readout(context.config['model']['feature_size'], settings['hidden'], norm['mean'][0], norm['std'][0])

                    def loss(current, batch):
                        return ((current(batch)-batch['main'])/norm['std'][0]).square().mean(), len(batch['main'])

                    def score(current, batches, dev):
                        total, n = 0., 0
                        for batch in batches:
                            batch = to_device(batch, dev)
                            total += float((current(batch)-batch['main']).abs().sum())
                            n += len(batch['main'])
                        return total / n

                    metadata = {'identity': context.identity, 'variant': variant, 'upstream_seed': up_seed, 'seed': seed,
                                'recipe': recipe, 'data_identity': data_identity(context),
                                'cache_identity': {s: sha256_file(cache_dir(context, variant, up_seed) / f'{s}.pt') for s in saved},
                                'training_split': 'b_train', 'selection_split': 'b_val', 'auxiliary_truth_used': False,
                                'input_normalization_fit': 'b_train', 'native_epoch0_candidate': True,
                                'access': 'full H learned masked pooling + g + native Eg, all recipes; one extra scalar',
                                'shuffle_seeds': {'b_train': seed+1000, 'b_val': seed+2000, 'y_test': seed+3000}}
                    directory = refiner_dir(context, variant, up_seed, recipe, seed)
                    artifacts.extend(fit(model, loader(train_values, stats, seed, settings['batch_size'], True),
                                         loader(val_values, stats, seed, settings['batch_size']), loss, score,
                                         directory, settings, metadata, device))
                    path = directory / 'input_normalization.pt'
                    save_checkpoint(path, stats)
                    artifacts.append(path)
    return artifacts