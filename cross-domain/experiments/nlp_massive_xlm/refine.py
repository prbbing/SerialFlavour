"""Frozen token caches and capacity-matched intent-only readouts on B."""
import math
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from experiments.nlp_massive_xlm.data import make_loader, data_identity, applicable_recipes, Records, processed_dir
from experiments.nlp_massive_xlm.model import Readout
from experiments.nlp_massive_xlm.training import load_upstream, upstream_dir
from experiments.nlp_massive_xlm.evaluate import score
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
            directory = cache_dir(context, variant, seed)
            directory.mkdir(parents=True, exist_ok=True)
            records = {}
            for split in ('b_train', 'b_val', 'y_test'):
                pieces, ids = {}, []
                with torch.inference_mode():
                    for batch in make_loader(context, split, seed, context.config['upstream']['batch_size']):
                        if 'slots_num' in batch:
                            raise AssertionError('auxiliary truth exposed to cache inference')
                        batch = to_device(batch, device)
                        output = model(batch)
                        values = {'embedding': output['embedding'], 'native_logits': output['logits'],
                                  'intent_num': batch['intent_num'], 'attention_mask': batch['attention_mask'], 'word_mask': batch['word_mask']}
                        if variant == 'multi_task':
                            values['slot_logits'] = output['slot_logits']
                        for key, value in values.items():
                            if not torch.isfinite(value).all():
                                raise FloatingPointError(f'nonfinite cache {key}')
                            pieces.setdefault(key, []).append(value.cpu())
                        ids.extend(batch['ids'])
                payload = {k: torch.cat(v) for k, v in pieces.items()}
                payload['ids'] = ids
                path = directory / f'{split}.pt'
                torch.save(payload, path)
                artifacts.append(path)
                records[split] = {'sha256': sha256_file(path), 'ids': ids,
                                  'shapes': {k: list(v.shape) for k, v in payload.items() if isinstance(v, torch.Tensor)}}
            if any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.state_dict().items()):
                raise AssertionError('frozen upstream mutated during cache')
            artifacts.append(write_json(directory / 'manifest.json', {
                'identity': context.identity, 'data_identity': data_identity(context),
                'checkpoint_sha256': sha256_file(upstream_dir(context, variant, seed) / 'best.pt'),
                'variant': variant, 'seed': seed, 'frozen': True, 'eval': True, 'state_unchanged': True,
                'auxiliary_truth_cached': False, 'records': records, 'same_forward_H_and_predictions': True}))
    return artifacts

def load_cache(context, variant, seed, split):
    directory = cache_dir(context, variant, seed)
    manifest = read_json(directory / 'manifest.json')
    if manifest['identity'] != context.identity or manifest['data_identity'] != data_identity(context):
        raise ValueError('cache identity mismatch')
    if manifest['checkpoint_sha256'] != sha256_file(upstream_dir(context, variant, seed) / 'best.pt'):
        raise ValueError('cache checkpoint changed')
    path = directory / f'{split}.pt'
    if sha256_file(path) != manifest['records'][split]['sha256']:
        raise ValueError('cache contents changed')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if 'slots_num' in payload:
        raise ValueError('auxiliary truth in frozen cache')
    return payload

def features(payload, recipe):
    h = payload['embedding']
    if recipe == 'embedding':
        return h
    slot_channels = 56
    if recipe == 'embedding_matched':
        extra = h.repeat(1, 1, math.ceil(slot_channels / h.shape[-1]))[..., :slot_channels]
    elif recipe == 'aux_prediction':
        extra = payload['slot_logits'].softmax(-1)
        if extra.shape[-1] != slot_channels:
            raise ValueError('unexpected slot channels')
    else:
        raise ValueError(f'unknown readout recipe {recipe}')
    # The extra input of BOTH matched recipes uses the same text-only word mask.
    extra = extra * payload['word_mask'].unsqueeze(-1)
    return torch.cat((h, extra), -1)

def cache_loader(context, variant, us, split, recipe, ds, batch_size, shuffle=False):
    cache = load_cache(context, variant, us, split)
    payload = {'features': features(cache, recipe), 'intent_num': cache['intent_num'],
               'native_logits': cache['native_logits'], 'attention_mask': cache['attention_mask']}
    return DataLoader(Records(payload), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(ds), num_workers=0)

def channels(context, recipe):
    return context.config['model']['hidden_size'] + (0 if recipe == 'embedding' else 56)

def readout_dir(context, variant, us, recipe, ds):
    return context.output_dir / 'refine' / variant / f'seed{us}' / recipe / f'seed{ds}'

def train(context):
    settings = context.config['refiner']
    device = configure(context.config['runtime'])
    artifacts = []
    for variant in context.config['upstream']['variants']:
        if context.filters.get('variant') not in (None, variant):
            continue
        for us in context.config['upstream']['seeds']:
            if context.filters.get('upstream_seed') not in (None, us):
                continue
            checkpoint_hash = sha256_file(upstream_dir(context, variant, us) / 'best.pt')
            for recipe in applicable_recipes(context, variant):
                if context.filters.get('recipe') not in (None, recipe):
                    continue
                for ds in settings['seeds']:
                    if context.filters.get('downstream_seed') not in (None, ds):
                        continue
                    seed_all(ds)
                    model = Readout(channels(context, recipe), 60, settings['width']).to(device)
                    validation = cache_loader(context, variant, us, 'b_val', recipe, ds, settings['batch_size'])
                    model.eval()
                    with torch.inference_mode():
                        batch = to_device(next(iter(validation)), device)
                        error = (model(batch) - batch['native_logits']).abs().max().item()
                    if error != 0:
                        raise AssertionError('native residual initialization mismatch')
                    metadata = {'identity': context.identity, 'data_identity': data_identity(context),
                                'variant': variant, 'upstream_seed': us, 'downstream_seed': ds, 'recipe': recipe,
                                'checkpoint_sha256': checkpoint_hash,
                                'cache_manifest_sha256': sha256_file(cache_dir(context, variant, us) / 'manifest.json'),
                                'training_split': 'b_train', 'selection_split': 'b_val', 'selection_metric': 'intent_accuracy',
                                'supervision': 'intent only', 'upstream_frozen': True, 'auxiliary_truth_visible': False,
                                'native_initialization_max_error': error, 'input_channels': channels(context, recipe),
                                'architecture': 'token linear-GELU, masked mean/max, residual classifier',
                                'normalization': 'none; slot softmax; same text-only word mask in matched extras'}
                    artifacts.extend(fit(model,
                        cache_loader(context, variant, us, 'b_train', recipe, ds, settings['batch_size'], True), validation,
                        lambda current, batch: (F.cross_entropy(current(batch), batch['intent_num']), len(batch['intent_num'])),
                        score, readout_dir(context, variant, us, recipe, ds), settings, metadata, device))
                    if sha256_file(upstream_dir(context, variant, us) / 'best.pt') != checkpoint_hash:
                        raise AssertionError('upstream changed during B training')
    return artifacts

def load_readout(context, variant, us, recipe, ds, device):
    checkpoint = torch.load(readout_dir(context, variant, us, recipe, ds) / 'best.pt', map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    if metadata['identity'] != context.identity or metadata['cache_manifest_sha256'] != sha256_file(cache_dir(context, variant, us) / 'manifest.json'):
        raise ValueError('readout identity mismatch')
    model = Readout(channels(context, recipe), 60, context.config['refiner']['width']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()
