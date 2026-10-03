"""A-only upstream training via the common validation/checkpoint fitter."""
import torch
from pipeline.fit import fit
from pipeline.io import read_json
from pipeline.runtime import configure, seed_all, to_device
from experiments.sci_mp_crystran.data import make_loader, data_identity
from experiments.sci_mp_crystran.model import CrystalTransformer, REFERENCE_COMMIT


def upstream_dir(context, variant, seed):
    return context.output_dir / 'upstream' / variant / f'seed{seed}'


def load_upstream(context, variant, seed, device, frozen=True):
    path = upstream_dir(context, variant, seed) / 'best.pt'
    saved = torch.load(path, map_location=device, weights_only=True)
    metadata = saved['metadata']
    if metadata['identity'] != context.identity or metadata['data_identity'] != data_identity(context):
        raise ValueError('upstream identity mismatch')
    model = CrystalTransformer(context.config['model'], variant).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    if frozen:
        model.requires_grad_(False)
    return model, metadata


def physical_output(output, normalization):
    result = dict(output)
    result['main'] = output['main'] * normalization['std'][0] + normalization['mean'][0]
    if 'auxiliary' in output:
        result['auxiliary'] = output['auxiliary'] * normalization['std'][1] + normalization['mean'][1]
    return result


def score_main(model, loader, device, normalization):
    total, n = 0., 0
    for batch in loader:
        batch = to_device(batch, device)
        output = physical_output(model(batch['z'], batch['coords'], batch['mask']), normalization)
        total += float((output['main'] - batch['main']).abs().sum())
        n += len(batch['main'])
    return total / n


def train(context):
    settings = context.config['upstream']
    norm = read_json(context.output_dir / 'target_normalization.json')
    device = configure(context.config['runtime'])
    artifacts = []
    for variant in settings['variants']:
        if context.filters.get('variant') not in (None, variant):
            continue
        for seed in settings['seeds']:
            if context.filters.get('seed') not in (None, seed):
                continue
            seed_all(seed)
            multi = variant == 'multi_task'
            model = CrystalTransformer(context.config['model'], variant)

            def loss(current, batch):
                output = current(batch['z'], batch['coords'], batch['mask'])
                objective = (output['main'] - (batch['main'] - norm['mean'][0])/norm['std'][0]).square().mean()
                if multi:
                    auxiliary = (batch['auxiliary'] - norm['mean'][1])/norm['std'][1]
                    objective = objective + settings['auxiliary_weight'] * (output['auxiliary']-auxiliary).square().mean()
                return objective, len(batch['main'])

            metadata = {'identity': context.identity, 'data_identity': data_identity(context),
                        'variant': variant, 'seed': seed, 'training_split': 'a_train', 'selection_split': 'a_val',
                        'normalization': norm, 'reference_commit': REFERENCE_COMMIT, 'pretrained': False,
                        'model': context.config['model'], 'selection_metric': 'bandgap_mae_eV',
                        'optimizer_deviation': 'common AdamW fitter instead of author SGD/StepLR',
                        'augmentation': 'none; deterministic released site order/cartesian coordinates'}
            artifacts.extend(fit(model, make_loader(context, 'a_train', seed, settings['batch_size'], True, multi),
                                 make_loader(context, 'a_val', seed, settings['batch_size']), loss,
                                 lambda m, l, d: score_main(m, l, d, norm), upstream_dir(context, variant, seed),
                                 settings, metadata, device))
    return artifacts