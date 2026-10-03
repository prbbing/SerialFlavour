"""Only locked Y evaluation after both selections have finished."""
import numpy as np
import torch
from pipeline.io import read_json, write_json, sha256_file
from pipeline.metrics import regression_metrics
from pipeline.runtime import configure, to_device
from experiments.sci_mp_crystran.model import Readout
from experiments.sci_mp_crystran.refine import load_cache, features, loader, recipes, refiner_dir, cache_dir
from experiments.sci_mp_crystran.training import upstream_dir


def evaluate(context):
    device = configure(context.config['runtime'])
    settings = context.config['refiner']
    norm = read_json(context.output_dir / 'target_normalization.json')
    records, arrays = [], {}
    expected_ids = read_json(context.output_dir / 'split_manifest.json')['ids']['y_test']
    for variant in context.config['upstream']['variants']:
        for up_seed in context.config['upstream']['seeds']:
            saved = load_cache(context, variant, up_seed, 'y_test')
            if saved['ids'] != expected_ids:
                raise ValueError('Y material identities/order mismatch')
            truth = saved['main'].numpy()
            arrays[f'{variant}_up{up_seed}_truth'] = truth
            arrays[f'{variant}_up{up_seed}_ids'] = np.asarray(saved['ids'])
            native_metrics = regression_metrics(truth, saved['native'].numpy())
            records.append({'variant': variant, 'upstream_seed': up_seed, 'recipe': 'native', 'seed': None,
                            'metrics': native_metrics, 'parameters': read_json(upstream_dir(context, variant, up_seed)/'training_manifest.json')['parameters']})
            arrays[f'{variant}_up{up_seed}_native'] = saved['native'].numpy()
            b_cache_hashes = {split: sha256_file(cache_dir(context, variant, up_seed) / f'{split}.pt')
                              for split in ('b_train', 'b_val')}
            for recipe in recipes(context, variant):
                for seed in settings['seeds']:
                    directory = refiner_dir(context, variant, up_seed, recipe, seed)
                    checkpoint = torch.load(directory / 'best.pt', map_location=device, weights_only=True)
                    metadata = checkpoint['metadata']
                    if (metadata['identity'] != context.identity or metadata['variant'] != variant
                            or metadata['upstream_seed'] != up_seed or metadata['seed'] != seed
                            or metadata['recipe'] != recipe):
                        raise ValueError('readout identity mismatch')
                    if metadata['cache_identity'] != b_cache_hashes:
                        raise ValueError('readout B cache identity mismatch')
                    stats = torch.load(directory / 'input_normalization.pt', weights_only=True)
                    model = Readout(context.config['model']['feature_size'], settings['hidden'], norm['mean'][0], norm['std'][0]).to(device)
                    model.load_state_dict(checkpoint['state_dict'])
                    model.eval()
                    predictions = []
                    with torch.inference_mode():
                        for batch in loader(features(saved, recipe, seed+3000), stats, seed, settings['batch_size']):
                            predictions.append(model(to_device(batch, device)).cpu().numpy())
                    prediction = np.concatenate(predictions)
                    metrics = regression_metrics(truth, prediction)
                    records.append({'variant': variant, 'upstream_seed': up_seed, 'recipe': recipe, 'seed': seed,
                                    'metrics': metrics, 'mae_delta_vs_native': metrics['mae']-native_metrics['mae'],
                                    'selected_epoch': checkpoint['epoch'], 'b_val_mae': checkpoint['validation_metric'],
                                    'checkpoint_sha256': sha256_file(directory/'best.pt'),
                                    'parameters': sum(p.numel() for p in model.parameters())})
                    arrays[f'{variant}_up{up_seed}_{recipe}_seed{seed}'] = prediction
    path = context.output_dir / 'evaluation.json'
    write_json(path, {'identity': context.identity, 'split': 'y_test', 'selection_uses_y': False,
                      'units': 'eV', 'records': records})
    predictions_path = context.output_dir / 'predictions.npz'
    np.savez_compressed(predictions_path, **arrays)
    return [path, predictions_path]