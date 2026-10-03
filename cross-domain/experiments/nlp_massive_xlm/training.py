"""A-only intent ST or intent+slot MT, using the shared validation fit loop."""
import torch
from torch.nn import functional as F

from experiments.nlp_massive_xlm.data import make_loader, data_identity, OFFICIAL_COMMIT, pretrained_enabled
from experiments.nlp_massive_xlm.model import build
from experiments.nlp_massive_xlm.evaluate import score
from pipeline.fit import fit
from pipeline.runtime import configure, seed_all

def upstream_dir(context, variant, seed):
    return context.output_dir / 'upstream' / variant / f'seed{seed}'

def load_upstream(context, variant, seed, device, frozen=False):
    checkpoint = torch.load(upstream_dir(context, variant, seed) / 'best.pt', map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    if metadata['identity'] != context.identity or metadata['data_identity'] != data_identity(context):
        raise ValueError('upstream identity mismatch')
    model = build(context, variant).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if frozen:
        model.requires_grad_(False)
    return model, metadata

def upstream_loss(model, batch, multi):
    if not multi and 'slots_num' in batch:
        raise ValueError('single-task loader must not expose auxiliary labels')
    output = model(batch)
    objective = F.cross_entropy(output['logits'], batch['intent_num'])
    if multi:
        valid = batch['slots_num'] != -100
        if not valid.any():
            raise ValueError('empty slot supervision')
        objective = objective + F.cross_entropy(output['slot_logits'][valid], batch['slots_num'][valid])
    return objective, len(batch['intent_num'])

def train(context):
    settings = context.config['upstream']
    device = configure(context.config['runtime'])
    artifacts = []
    for variant in settings['variants']:
        if context.filters.get('variant') not in (None, variant):
            continue
        for seed in settings['seeds']:
            if context.filters.get('seed') not in (None, seed):
                continue
            seed_all(seed)
            model = build(context, variant, initialize=True)
            multi = variant == 'multi_task'
            metadata = {'identity': context.identity, 'data_identity': data_identity(context),
                        'variant': variant, 'seed': seed, 'official_commit': OFFICIAL_COMMIT,
                        'architecture': 'unmodified author parallel heads + XLMRobertaConfig',
                        'model': context.config['model'], 'pretrained_encoder': pretrained_enabled(context),
                        'pretrained_source': getattr(model, 'pretrained_metadata', None),
                        'training_split': 'a_train', 'selection_split': 'a_val', 'selection_metric': 'intent_accuracy',
                        'supervision': 'intent CE + first-subword slot CE (weight 1)' if multi else 'intent CE only; slot head frozen',
                        'optimizer': 'NLP AdamW with accumulation/linear warmup-decay' if settings.get('loop') == 'nlp_finetune' else 'shared fit AdamW, no warmup; smoke adaptation',
                        'auxiliary_truth_visible': multi}
            if settings.get('loop') == 'nlp_finetune':
                from experiments.nlp_massive_xlm.finetune import fit as fit_nlp
                artifacts.extend(fit_nlp(model,
                    make_loader(context, 'a_train', seed, settings['batch_size'], True, auxiliary=multi),
                    make_loader(context, 'a_val', seed, settings['batch_size']),
                    lambda current, batch: upstream_loss(current, batch, multi),
                    upstream_dir(context, variant, seed), settings, metadata, device))
                continue
            artifacts.extend(fit(model,
                make_loader(context, 'a_train', seed, settings['batch_size'], True, auxiliary=multi),
                make_loader(context, 'a_val', seed, settings['batch_size']),
                lambda current, batch: upstream_loss(current, batch, multi), score,
                upstream_dir(context, variant, seed), settings, metadata, device))
    return artifacts
