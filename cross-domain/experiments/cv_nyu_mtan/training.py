"""A-only STAN/MTAN optimization using the shared fit loop."""

import torch
from experiments.cv_nyu_mtan.data import make_loader, data_identity
from experiments.cv_nyu_mtan.model import MTAN
from experiments.cv_nyu_mtan.evaluate import segmentation_loss, score
from pipeline.fit import fit
from pipeline.runtime import configure, seed_all


def upstream_dir(context, variant, seed):
    return context.output_dir / 'upstream' / variant / f'seed{seed}'


def load_upstream(context, variant, seed, device, frozen=False):
    checkpoint = torch.load(upstream_dir(context, variant, seed) / 'best.pt', map_location=device, weights_only=True)
    metadata = checkpoint['metadata']
    if metadata['identity'] != context.identity or metadata['data_identity'] != data_identity(context):
        raise ValueError('upstream source identity mismatch')
    model = MTAN(context.config['model']['widths'], variant == 'multi_task').to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if frozen:
        model.requires_grad_(False)
    return model, metadata


def masked_mean(values, mask):
    # A task with no valid targets in this batch contributes zero, not fake labels.
    return values[mask].mean() if mask.any() else values.sum()*0


def train(context):
    settings = context.config['upstream']
    device = configure(context.config['runtime'])
    artifacts = []
    filters = context.filters
    for variant in settings['variants']:
        if filters.get('variant') not in (None, variant):
            continue
        for seed in settings['seeds']:
            if filters.get('seed') not in (None, seed):
                continue
            seed_all(seed)
            multi = variant == 'multi_task'
            model = MTAN(context.config['model']['widths'], multi)

            def loss(current, batch):
                output = current(batch['image'])
                objective = segmentation_loss(output['logits'], batch['segmentation'])
                if multi:
                    objective = objective + masked_mean((output['depth'] - batch['depth']).abs(), batch['depth'] > 0)
                    valid = batch['normal'].square().sum(1) > 0
                    objective = objective + masked_mean(1-(output['normal']*batch['normal']).sum(1), valid)
                return objective, len(batch['image'])

            metadata = {'identity': context.identity, 'variant': variant, 'seed': seed,
                        'data_identity': data_identity(context), 'model': context.config['model'],
                        'pretrained': False, 'selection_split': 'a_val', 'selection_metric': 'dataset_miou',
                        'training_split': 'a_train', 'loss': 'CE + masked depth L1 + masked normal cosine, equal weights' if multi else 'CE',
                        'architecture': 'five-scale width-reduced author MTAN' if multi else 'five-scale one-task attention STAN',
                        'reference_commit': 'c36c30baa18968dec74fe9039abcfd4f132edfa1'}
            artifacts.extend(fit(model, make_loader(context, 'a_train', seed, settings['batch_size'], True, multi),
                                 make_loader(context, 'a_val', seed, settings['batch_size'], auxiliary=False),
                                 loss, score, upstream_dir(context, variant, seed), settings, metadata, device))
    return artifacts
