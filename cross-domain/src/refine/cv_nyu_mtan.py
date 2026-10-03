"""Frozen spatial cache and native-initialized, capacity-matched CNN readouts."""

import math
import torch
from torch.utils.data import Dataset, DataLoader
from data.cv_nyu_mtan import make_loader, load_data, processed_dir, applicable_recipes, data_identity
from model.cv_nyu_mtan import head
from training.cv_nyu_mtan import load_upstream, upstream_dir
from evaluate.cv_nyu_mtan import segmentation_loss, score
from pipeline.fit import fit
from pipeline.io import read_json, write_json, sha256_file
from pipeline.runtime import configure, seed_all, to_device


def cache_dir(context, variant, seed):
    return context.output_dir / 'cache' / variant / f'seed{seed}'


def cache(context):
    device = configure(context.config['runtime'])
    artifacts = []
    for variant in context.config['upstream']['variants']:
        if context.filters.get('variant') not in (None, variant):
            continue
        for seed in context.config['upstream']['seeds']:
            if context.filters.get('seed') not in (None, seed):
                continue
            model, _ = load_upstream(context, variant, seed, device, frozen=True)
            before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            checkpoint_hash = sha256_file(upstream_dir(context, variant, seed) / 'best.pt')
            directory = cache_dir(context, variant, seed)
            directory.mkdir(parents=True, exist_ok=True)
            records = {}
            for split in ('b_train', 'b_val', 'y_test'):
                pieces = {}
                with torch.inference_mode():
                    for batch in make_loader(context, split, seed, 2, auxiliary=False):
                        batch = to_device(batch, device)
                        output = model(batch['image'])
                        values = {'embedding': torch.cat((output['semantic_hidden'], output['shared']), 1),
                                  'logits': output['logits'], 'segmentation': batch['segmentation']}
                        if variant == 'multi_task':
                            values.update(aux_prediction=torch.cat((output['depth'], output['normal']), 1), aux_hidden=output['aux_hidden'])
                        for key, value in values.items():
                            if not torch.isfinite(value).all():
                                raise FloatingPointError(f'nonfinite cached {key}')
                            pieces.setdefault(key, []).append(value.cpu())
                indices = read_json(processed_dir(context) / 'splits.json')[split]
                payload = {k: torch.cat(v) for k, v in pieces.items()}
                payload['ids'] = [load_data(context)['ids'][i] for i in indices]
                path = directory / f'{split}.pt'
                torch.save(payload, path)
                artifacts.append(path)
                records[split] = {'sha256': sha256_file(path), 'ids': payload['ids'],
                                  'shapes': {k: list(v.shape) for k, v in payload.items() if isinstance(v, torch.Tensor)}}
            if any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()):
                raise AssertionError('upstream parameters/normalization buffers mutated during cache')
            artifacts.append(write_json(directory / 'manifest.json', {
                'identity': context.identity, 'data_identity': data_identity(context), 'checkpoint_sha256': checkpoint_hash,
                'variant': variant, 'seed': seed, 'frozen': True, 'eval': True, 'state_unchanged': True,
                'auxiliary_truth_cached': False, 'input_normalization': 'none', 'records': records,
            }))
    return artifacts


def load_cache(context, variant, seed, split):
    directory = cache_dir(context, variant, seed)
    manifest = read_json(directory / 'manifest.json')
    if manifest['identity'] != context.identity or manifest['data_identity'] != data_identity(context):
        raise ValueError('cache code/data identity mismatch')
    if manifest['checkpoint_sha256'] != sha256_file(upstream_dir(context, variant, seed) / 'best.pt'):
        raise ValueError('cache upstream checkpoint changed')
    path = directory / f'{split}.pt'
    if manifest['records'][split]['sha256'] != sha256_file(path):
        raise ValueError('cache content changed')
    return torch.load(path, map_location='cpu', weights_only=True)


def feature_channels(context, recipe):
    width = context.config['model']['widths'][0]
    return 2*width if recipe == 'embedding' else 4*width


def features(payload, recipe):
    base = payload['embedding']
    total = 2*base.shape[1]
    if recipe == 'embedding':
        return base
    if recipe == 'embedding_matched':
        # Repeat real spatial features rather than zero-only dead input channels.
        return torch.cat((base, base), 1)
    extra = payload[recipe]
    value = torch.cat((base, extra), 1)
    if value.shape[1] > total:
        raise ValueError('extra feature channels exceed fixed readout budget')
    if value.shape[1] < total:
        padding = base.repeat(1, math.ceil((total-value.shape[1])/base.shape[1]), 1, 1)
        value = torch.cat((value, padding[:, :total-value.shape[1]]), 1)
    return value


class Cached(Dataset):
    def __init__(self, payload, recipe):
        self.x, self.y = features(payload, recipe), payload['segmentation']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {'features': self.x[i], 'segmentation': self.y[i]}


def cache_loader(context, variant, us, split, recipe, ds, batch_size, shuffle=False):
    return DataLoader(Cached(load_cache(context, variant, us, split), recipe), batch_size=batch_size,
                      shuffle=shuffle, generator=torch.Generator().manual_seed(ds), num_workers=0)


def readout_dir(context, variant, us, recipe, ds):
    return context.output_dir / 'refine' / variant / f'seed{us}' / recipe / f'seed{ds}'


def initialize_native(readout, upstream):
    native = upstream.pred_task1
    width = native[0].in_channels
    with torch.no_grad():
        readout[0].weight[:width].zero_()
        readout[0].weight[:width, :width].copy_(native[0].weight)
        readout[0].bias[:width].copy_(native[0].bias)
        readout[1].weight.zero_()
        readout[1].weight[:, :width].copy_(native[1].weight)
        readout[1].bias.copy_(native[1].bias)


def train(context):
    settings = context.config['refiner']
    device = configure(context.config['runtime'])
    artifacts = []
    filters = context.filters
    for variant in context.config['upstream']['variants']:
        if filters.get('variant') not in (None, variant):
            continue
        for us in context.config['upstream']['seeds']:
            if filters.get('upstream_seed') not in (None, us):
                continue
            upstream, _ = load_upstream(context, variant, us, device, frozen=True)
            checkpoint_hash = sha256_file(upstream_dir(context, variant, us) / 'best.pt')
            for recipe in applicable_recipes(context, variant):
                if filters.get('recipe') not in (None, recipe):
                    continue
                for ds in settings['seeds']:
                    if filters.get('downstream_seed') not in (None, ds):
                        continue
                    seed_all(ds)
                    model = head(feature_channels(context, recipe), 13).to(device)
                    initialize_native(model, upstream)
                    validation = load_cache(context, variant, us, 'b_val')
                    with torch.inference_mode():
                        initial_logits = model(features(validation, recipe).to(device))
                        error = (initial_logits.cpu()-validation['logits']).abs().max().item()
                    if error > 1e-5:
                        raise AssertionError(f'native initialization mismatch {error}')

                    def loss(current, batch):
                        return segmentation_loss(current(batch['features']), batch['segmentation']), len(batch['segmentation'])

                    metadata = {'identity': context.identity, 'variant': variant, 'upstream_seed': us,
                                'downstream_seed': ds, 'recipe': recipe, 'checkpoint_sha256': checkpoint_hash,
                                'cache_manifest_sha256': sha256_file(cache_dir(context, variant, us) / 'manifest.json'),
                                'training_split': 'b_train', 'selection_split': 'b_val', 'selection_metric': 'dataset_miou',
                                'supervision': 'semantic only', 'auxiliary_source': 'frozen predictions/hidden only',
                                'native_initialization_max_error': error, 'normalization': 'none',
                                'input_channels': feature_channels(context, recipe), 'upstream_frozen': True}
                    artifacts.extend(fit(model, cache_loader(context, variant, us, 'b_train', recipe, ds, settings['batch_size'], True),
                                         cache_loader(context, variant, us, 'b_val', recipe, ds, settings['batch_size']),
                                         loss, score, readout_dir(context, variant, us, recipe, ds), settings, metadata, device))
                    if sha256_file(upstream_dir(context, variant, us) / 'best.pt') != checkpoint_hash:
                        raise AssertionError('upstream checkpoint changed during B training')
    return artifacts


def load_readout(context, variant, us, recipe, ds, device):
    path = readout_dir(context, variant, us, recipe, ds) / 'best.pt'
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    metadata = checkpoint['metadata']
    if metadata['identity'] != context.identity or metadata['cache_manifest_sha256'] != sha256_file(cache_dir(context, variant, us) / 'manifest.json'):
        raise ValueError('readout source identity mismatch')
    model = head(feature_channels(context, recipe), 13).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()
