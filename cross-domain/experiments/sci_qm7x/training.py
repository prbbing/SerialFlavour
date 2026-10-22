"""A-only architecture-matched ST/MT supervised training and frozen loading."""
import torch
from pipeline.fit import fit
from pipeline.io import read_json, sha256_file
from pipeline.runtime import configure, seed_all, to_device
from experiments.sci_qm7x.common import (
    validate_config, assert_prepared, expected_metadata, check_metadata, require_lifecycle_filters, upstream_dir,
)
from experiments.sci_qm7x.data import make_loader
from experiments.sci_qm7x.model import MolecularModel


def physical_output(output, normalization):
    return {**output, 'main': output['main'] * normalization['main_std'] + normalization['main_mean'],
            'auxiliary': output['auxiliary'] * output['auxiliary'].new_tensor(normalization['aux_std'])
                         + output['auxiliary'].new_tensor(normalization['aux_mean'])}


def load_upstream(context, variant, seed, device, frozen=True):
    path = upstream_dir(context, variant, seed) / 'best.pt'
    saved = torch.load(path, map_location=device, weights_only=True)
    check_metadata(context, saved['metadata'])
    if saved['metadata']['variant'] != variant or saved['metadata']['seed'] != seed:
        raise ValueError('upstream variant/seed mismatch')
    model = MolecularModel(context.config['model'], variant).to(device)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.eval()
    if frozen:
        model.requires_grad_(False)
    return model, saved['metadata']


def score_main(model, batches, device, normalization):
    total, n = 0., 0
    for batch in batches:
        batch = to_device(batch, device)
        predictions = physical_output(model(batch), normalization)['main']
        total += (predictions - batch['main']).abs().sum().item()
        n += len(predictions)
    if not n:
        raise ValueError('empty validation loader')
    return total / n


def train(context):
    require_lifecycle_filters(context,('variant','seed'))
    validate_config(context.config)
    assert_prepared(context)
    settings = context.config['upstream']
    norm = read_json(context.output_dir / 'target_normalization.json')
    device, artifacts = configure(context.config['runtime']), []
    for variant in settings['variants']:
        if context.filters.get('variant') not in (None, variant):
            continue
        for seed in settings['seeds']:
            if context.filters.get('seed') not in (None, seed):
                continue
            seed_all(seed)
            model = MolecularModel(context.config['model'], variant)
            multi = variant == 'multi_task'

            def loss(current, batch):
                output = current(batch)
                objective = (output['main'] - (batch['main'] - norm['main_mean']) / norm['main_std']).square().mean()
                if multi:
                    targets = (batch['auxiliary'] - output['auxiliary'].new_tensor(norm['aux_mean'])) / output['auxiliary'].new_tensor(norm['aux_std'])
                    per_task = (output['auxiliary'] - targets).square()[batch['mask']].mean(0)
                    objective = objective + (per_task * per_task.new_tensor(settings['auxiliary_weights'])).sum()
                return objective, len(batch['main'])

            metadata = {**expected_metadata(context), 'variant': variant, 'seed': seed, 'normalization': norm,
                        'training_split': 'a_train', 'selection_split': 'a_val', 'selection_metric': 'homo_mae_eV',
                        'auxiliary_truth_used': multi, 'model': context.config['model'], 'pretrained': False,
                        'architecture': 'package-local project SchNet-style, identical ST/MT including dormant ST local heads',
                        'frozen_boundary': 'all encoder, main and local heads; eval mode; no BatchNorm/dropout',
                        'source_manifest_sha256': sha256_file(context.output_dir / 'source_manifest.json')}
            artifacts.extend(fit(model, make_loader(context, 'a_train', seed, settings['batch_size'], True, multi),
                                 make_loader(context, 'a_val', seed, settings['batch_size']), loss,
                                 lambda m, l, d: score_main(m, l, d, norm), upstream_dir(context, variant, seed),
                                 settings, metadata, device))
    if not artifacts:
        raise ValueError('no configured upstream matched runtime filters')
    return artifacts
