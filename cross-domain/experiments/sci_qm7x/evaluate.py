"""Locked Y evaluation: reject an incomplete/mismatched matrix before scoring."""
import torch
from pipeline.fit import save_checkpoint
from pipeline.io import read_json, write_json, sha256_file, artifact_record, artifacts_match
from pipeline.runtime import configure, to_device
from experiments.sci_qm7x.common import (
    validate_config, assert_prepared, expected_metadata, check_metadata, applicable_recipes,
    upstream_dir, cache_dir, refiner_dir,
)
from experiments.sci_qm7x.refine import load_cache, features, frozen_loader, cache_splits, FrozenDataset
from experiments.sci_qm7x.data import make_loader
from experiments.sci_qm7x.model import Readout


def matrix_keys(context):
    keys = []
    for variant in context.config['upstream']['variants']:
        for us in context.config['upstream']['seeds']:
            keys.append((variant, us, 'native', None))
            for recipe in applicable_recipes(context, variant):
                for ds in context.config['refiner']['seeds']:
                    keys.append((variant, us, recipe, ds))
    return keys


def verify_matrix(context, variant_filter=None, seed_filter=None):
    """Audit all required inputs before any Y prediction or aggregation."""
    validate_config(context.config)
    if context.config.get('workflow', {}).get('mode') == 'seed_lifecycle' and variant_filter is None:
        from pipeline.lifecycle import verify_receipts, receipt_path
        receipts = verify_receipts(context)
        return {str(receipt_path(context,r['variant'],r['upstream_seed'])): sha256_file(receipt_path(context,r['variant'],r['upstream_seed'])) for r in receipts}
    assert_prepared(context)
    hashes = {}
    for variant in context.config['upstream']['variants']:
        if variant_filter not in (None, variant):
            continue
        for us in context.config['upstream']['seeds']:
            if seed_filter not in (None, us):
                continue
            up = upstream_dir(context, variant, us) / 'best.pt'
            saved = torch.load(up, map_location='cpu', weights_only=True)
            check_metadata(context, saved['metadata'])
            if (saved['metadata']['variant'], saved['metadata']['seed']) != (variant, us):
                raise ValueError('wrong upstream checkpoint variant/seed')
            for split in cache_splits(context):
                load_cache(context, variant, us, split)
            cache_manifest = cache_dir(context, variant, us) / 'cache_manifest.json'
            hashes[str(up)] = sha256_file(up)
            hashes[str(cache_manifest)] = sha256_file(cache_manifest)
            for recipe in applicable_recipes(context, variant):
                for ds in context.config['refiner']['seeds']:
                    directory = refiner_dir(context, variant, us, recipe, ds)
                    checkpoint = directory / 'best.pt'
                    norm_path = directory / 'input_normalization.pt'
                    meta = torch.load(checkpoint, map_location='cpu', weights_only=True)['metadata']
                    check_metadata(context, meta)
                    if (meta['variant'], meta['upstream_seed'], meta['recipe'], meta['seed']) != (variant, us, recipe, ds):
                        raise ValueError('refiner checkpoint matrix identity mismatch')
                    if (meta['training_split'], meta['selection_split'], meta['auxiliary_truth_used'], meta['native_epoch0_candidate']) != ('b_train', 'b_val', False, True):
                        raise ValueError('refiner supervision/native-initialization protocol mismatch')
                    if (meta['upstream_checkpoint_sha256'] != hashes[str(up)]
                            or meta['cache_manifest_sha256'] != hashes[str(cache_manifest)]
                            or meta['input_normalization_sha256'] != sha256_file(norm_path)):
                        raise ValueError('refiner input provenance mismatch')
                    training = read_json(directory / 'training_manifest.json')
                    check_metadata(context, training)
                    if any(training.get(key) != value for key, value in meta.items()):
                        raise ValueError('training manifest and checkpoint provenance mismatch')
                    if meta.get('input_normalization_fit') != 'b_train_valid_atoms_and_molecules':
                        raise ValueError('readout normalization was not fitted on B train')
                    hashes[str(directory / 'training_manifest.json')] = sha256_file(directory / 'training_manifest.json')
                    hashes[str(checkpoint)] = sha256_file(checkpoint)
                    hashes[str(norm_path)] = sha256_file(norm_path)
    return hashes


def regression(target, prediction):
    difference = prediction.double() - target.double()
    if not len(difference) or not torch.isfinite(difference).all():
        raise ValueError('invalid regression predictions')
    return {'mae_eV': difference.abs().mean().item(), 'rmse_eV': difference.square().mean().sqrt().item(),
            'bias_eV': difference.mean().item(), 'n_structures': len(difference)}


def evaluate(context):
    if context.config.get('workflow', {}).get('mode') == 'seed_lifecycle':
        return aggregate_seed_metrics(context)
    provenance = verify_matrix(context)
    directory = context.output_dir / 'evaluation'
    complete = directory / 'evaluation_manifest.json'
    if complete.exists():
        record = read_json(complete)
        check_metadata(context, record)
        if record['provenance'] != provenance or not artifacts_match(record['artifacts']):
            raise ValueError('locked Y evidence or prerequisites changed; choose a new experiment')
        return [complete] + [__import__('pathlib').Path(row['path']) for row in record['artifacts']]
    device, rows, artifacts = configure(context.config['runtime']), [], []
    norm = read_json(context.output_dir / 'target_normalization.json')
    settings = context.config['refiner']
    diagnostics = []
    for variant in context.config['upstream']['variants']:
        for us in context.config['upstream']['seeds']:
            saved = load_cache(context, variant, us, 'y_test')
            prediction_path = directory / 'predictions' / f'{variant}__up{us}__native.pt'
            save_checkpoint(prediction_path, {'ids': saved['ids'], 'target_eV': saved['main'], 'prediction_eV': saved['native'],
                                              'metadata': {**expected_metadata(context), 'variant': variant, 'upstream_seed': us, 'recipe': 'native'}})
            artifacts.append(prediction_path)
            rows.append({'variant': variant, 'upstream_seed': us, 'recipe': 'native', 'downstream_seed': None,
                         **regression(saved['main'], saved['native']), 'prediction_file': str(prediction_path),
                         'prediction_sha256': sha256_file(prediction_path)})
            for recipe in applicable_recipes(context, variant):
                for ds in settings['seeds']:
                    run = refiner_dir(context, variant, us, recipe, ds)
                    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
                    stats = torch.load(run / 'input_normalization.pt', map_location='cpu', weights_only=True)
                    values = features(saved, recipe, ds + 3000, context.config['model']['head_hidden'])
                    model = Readout(context.config['model'], settings, norm['main_std']).to(device)
                    model.load_state_dict(checkpoint['state_dict'], strict=True)
                    model.eval().requires_grad_(False)
                    predictions = []
                    with torch.inference_mode():
                        for batch in frozen_loader(values, stats, settings['batch_size']):
                            predictions.append(model(to_device(batch, device)).cpu())
                    prediction = torch.cat(predictions)
                    prediction_path = directory / 'predictions' / f'{variant}__up{us}__{recipe}__ds{ds}.pt'
                    save_checkpoint(prediction_path, {'ids': saved['ids'], 'target_eV': saved['main'], 'prediction_eV': prediction,
                                                      'metadata': {**expected_metadata(context), 'variant': variant, 'upstream_seed': us,
                                                                   'recipe': recipe, 'downstream_seed': ds, 'selected_epoch': checkpoint['epoch']}})
                    artifacts.append(prediction_path)
                    rows.append({'variant': variant, 'upstream_seed': us, 'recipe': recipe, 'downstream_seed': ds,
                                 **regression(saved['main'], prediction), 'selected_epoch': checkpoint['epoch'],
                                 'prediction_file': str(prediction_path), 'prediction_sha256': sha256_file(prediction_path)})
            # Y local labels are fetched ONLY here under an explicit diagnostics flag.
            if context.config.get('evaluation', {}).get('auxiliary_diagnostics', False) and variant == 'multi_task':
                errors, start, atoms = torch.zeros(2, dtype=torch.float64), 0, 0
                for batch in make_loader(context, 'y_test', batch_size=settings['batch_size'], auxiliary=True, diagnostics=True):
                    n = len(batch['main'])
                    if batch['id'] != saved['ids'][start:start + n]:
                        raise ValueError('diagnostic label/sample mismatch')
                    delta = (saved['aux_prediction'][start:start + n] - batch['auxiliary']).double()
                    errors += delta.abs()[batch['mask']].sum(0)
                    atoms += batch['mask'].sum().item()
                    start += n
                diagnostics.append({'variant': variant, 'upstream_seed': us, 'valid_atoms': atoms,
                                    'hCHG_mae_e': (errors[0] / atoms).item(), 'atPOL_mae_bohr3': (errors[1] / atoms).item(),
                                    'truth_source': 'prepared y_test; explicit diagnostics flag; never refiner input'})
    observed = [(row['variant'], row['upstream_seed'], row['recipe'], row['downstream_seed']) for row in rows]
    if len(set(observed)) != len(observed) or set(observed) != set(matrix_keys(context)):
        raise ValueError('incomplete or duplicated evaluation matrix')
    metrics = directory / 'metrics.json'
    write_json(metrics, {**expected_metadata(context), 'split': 'y_test', 'main_task': 'eH', 'unit': 'eV',
                         'rows': rows, 'expected_rows': len(matrix_keys(context)), 'provenance': provenance,
                         'auxiliary_truth_in_predictions': False})
    artifacts.append(metrics)
    if diagnostics:
        path = directory / 'auxiliary_diagnostics.json'
        write_json(path, {**expected_metadata(context), 'rows': diagnostics, 'selection_used': False})
        artifacts.append(path)
    write_json(complete, {**expected_metadata(context), 'provenance': provenance,
                          'artifacts': [artifact_record(path) for path in artifacts], 'status': 'complete'})
    return artifacts + [complete]


class RegressionAccumulator:
    """Float64 error sums; only targets/predictions of the current batch are needed."""
    def __init__(self):
        self.n, self.absolute, self.squared, self.signed = 0, 0., 0., 0.

    def update(self, target, prediction):
        delta = (prediction.detach().cpu().double() - target.detach().cpu().double())
        if not len(delta) or not torch.isfinite(delta).all():
            raise ValueError('invalid regression predictions')
        self.n += len(delta)
        self.absolute += delta.abs().sum().item()
        self.squared += delta.square().sum().item()
        self.signed += delta.sum().item()

    def result(self):
        if not self.n:
            raise ValueError('empty Y evaluation')
        return {'mae_eV': self.absolute/self.n, 'rmse_eV': (self.squared/self.n)**0.5,
                'bias_eV': self.signed/self.n, 'n_structures': self.n}


def stream_rows(upstream, candidates, batches, normalization, device, extra_width, expected_ids,
                save_predictions=False, diagnostics=False):
    """One frozen upstream forward per Y batch shared by every preselected readout.

    candidate models and their B-train statistics are supplied by evaluate_seed.
    No Y statistic is fitted and no complete Y hidden tensor is retained.
    """
    from experiments.sci_qm7x.training import physical_output
    keys = [('native',None)] + [(c['recipe'],c['downstream_seed']) for c in candidates]
    accumulators = {key: RegressionAccumulator() for key in keys}
    predictions = {key: [] for key in keys} if save_predictions else None
    targets = [] if save_predictions else None
    ids, local_errors, atoms = [], torch.zeros(2,dtype=torch.float64), 0
    if upstream.training or any(p.requires_grad for p in upstream.parameters()):
        raise ValueError('Y requires a completely frozen eval-mode upstream')
    with torch.inference_mode():
        for batch in batches:
            ids.extend(batch['id'])
            # Auxiliary truth is accessed solely by the scoring branch below.
            inputs = {key: batch[key] for key in ('z','coords','mask','main')}
            output = physical_output(upstream(to_device(inputs,device)),normalization)
            saved = {key: output[source].cpu() for key,source in (
                ('H','H'),('g','g'),('native','main'),('aux_prediction','auxiliary'),('aux_hidden','aux_hidden'))}
            saved.update({key: batch[key] for key in ('coords','mask','main')})
            saved['ids'] = batch['id']
            accumulators[('native',None)].update(batch['main'],saved['native'])
            if save_predictions:
                predictions[('native',None)].append(saved['native'])
                targets.append(batch['main'])
            for candidate in candidates:
                key = candidate['recipe'], candidate['downstream_seed']
                values = features(saved,key[0],key[1]+3000,extra_width)
                normalized = FrozenDataset(values,candidate['stats']).values
                prediction = candidate['model'](to_device(normalized,device)).cpu()
                accumulators[key].update(batch['main'],prediction)
                if save_predictions:
                    predictions[key].append(prediction)
            if diagnostics:
                delta = (saved['aux_prediction']-batch['auxiliary']).double()
                local_errors += delta.abs()[batch['mask']].sum(0)
                atoms += batch['mask'].sum().item()
    if ids != expected_ids:
        raise ValueError('streamed Y sample identities/order mismatch')
    rows = [{'recipe':recipe,'downstream_seed':ds,**accumulators[(recipe,ds)].result()} for recipe,ds in keys]
    diagnostic = {'valid_atoms':atoms,'hCHG_mae_e':(local_errors[0]/atoms).item(),
                  'atPOL_mae_bohr3':(local_errors[1]/atoms).item(),
                  'truth_source':'prepared y_test; explicit diagnostics flag; never refiner input'} if diagnostics else None
    payload = {'ids':ids,'target_eV':torch.cat(targets),
               'predictions':{key:torch.cat(value) for key,value in predictions.items()}} if save_predictions else None
    return rows,diagnostic,payload


def evaluate_seed(context, variant, seed):
    """Lock one completed B-trained matrix, then stream Y in an independent process."""
    from pipeline.lifecycle import verify_seed_ready
    from experiments.sci_qm7x.training import load_upstream
    if variant not in context.config['upstream']['variants'] or seed not in context.config['upstream']['seeds']:
        raise ValueError('unplanned upstream seed')
    verify_seed_ready(context,variant,seed)
    provenance = verify_matrix(context,variant,seed)
    directory = context.output_dir/'evaluation'/'seeds'/variant/f'seed{seed}'
    complete = directory/'evaluation_manifest.json'
    if complete.exists():
        previous = read_json(complete)
        check_metadata(context,previous)
        if (previous.get('variant'),previous.get('upstream_seed'),previous.get('status')) != (variant,seed,'complete') or previous['provenance'] != provenance or not artifacts_match(previous['artifacts']):
            raise ValueError('locked seed evaluation evidence changed; choose a new experiment')
        return [__import__('pathlib').Path(row['path']) for row in previous['artifacts']]+[complete]
    device = configure(context.config['runtime'])
    norm = read_json(context.output_dir/'target_normalization.json')
    settings = context.config['refiner']
    candidates = []
    for recipe in applicable_recipes(context,variant):
        for ds in settings['seeds']:
            directory_run = refiner_dir(context,variant,seed,recipe,ds)
            checkpoint = torch.load(directory_run/'best.pt',map_location='cpu',weights_only=True)
            model = Readout(context.config['model'],settings,norm['main_std']).to(device)
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            model.eval().requires_grad_(False)
            candidates.append({'recipe':recipe,'downstream_seed':ds,'model':model,'selected_epoch':checkpoint['epoch'],
                               'stats':torch.load(directory_run/'input_normalization.pt',map_location='cpu',weights_only=True)})
    upstream,_ = load_upstream(context,variant,seed,device)
    diagnostics = context.config.get('evaluation',{}).get('auxiliary_diagnostics',False) and variant == 'multi_task'
    prediction_enabled = context.config.get('evaluation',{}).get('save_predictions',False)
    expected_ids = read_json(context.output_dir/'split_manifest.json')['ids']['y_test']
    loader = make_loader(context,'y_test',batch_size=settings['batch_size'],auxiliary=diagnostics,diagnostics=diagnostics)
    rows,diagnostic,payload = stream_rows(upstream,candidates,loader,norm,device,context.config['model']['head_hidden'],
                                        expected_ids,prediction_enabled,diagnostics)
    selected = {(c['recipe'],c['downstream_seed']):c['selected_epoch'] for c in candidates}
    artifacts = []
    for row in rows:
        row.update(variant=variant,upstream_seed=seed)
        if row['recipe'] != 'native':
            row['selected_epoch'] = selected[(row['recipe'],row['downstream_seed'])]
        if prediction_enabled:
            path = directory/'predictions'/f"{row['recipe']}__ds{row['downstream_seed']}.pt"
            save_checkpoint(path,{'ids':payload['ids'],'target_eV':payload['target_eV'],
                                  'prediction_eV':payload['predictions'][(row['recipe'],row['downstream_seed'])],
                                  'metadata':{**expected_metadata(context),**{key:row[key] for key in ('variant','upstream_seed','recipe','downstream_seed')}}})
            artifacts.append(path)
            row.update(prediction_file=str(path),prediction_sha256=sha256_file(path))
    expected = {(variant,seed,'native',None)}|{(variant,seed,recipe,ds) for recipe in applicable_recipes(context,variant) for ds in settings['seeds']}
    observed = [(r['variant'],r['upstream_seed'],r['recipe'],r['downstream_seed']) for r in rows]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError('incomplete or duplicate seed evaluation matrix')
    metrics = directory/'metrics.json'
    write_json(metrics,{**expected_metadata(context),'variant':variant,'upstream_seed':seed,'split':'y_test',
                        'rows':rows,'expected_rows':len(expected),'provenance':provenance,
                        'evaluation_mode':'stream_Y_shared_frozen_batch','auxiliary_truth_in_predictions':False,
                        'normalization_fit':'B train only; saved per recipe/downstream seed','selection_used':False})
    artifacts.append(metrics)
    if diagnostic is not None:
        path = directory/'auxiliary_diagnostics.json'
        write_json(path,{**expected_metadata(context),'rows':[dict(diagnostic,variant=variant,upstream_seed=seed)],'selection_used':False})
        artifacts.append(path)
    write_json(complete,{**expected_metadata(context),'variant':variant,'upstream_seed':seed,
                         'provenance':provenance,'artifacts':[artifact_record(p) for p in artifacts],'status':'complete'})
    return artifacts+[complete]


def aggregate_seed_metrics(context):
    """Only durable receipts authorize aggregation; intentionally deleted B caches are unnecessary."""
    from pipeline.lifecycle import verify_receipts
    verify_receipts(context)
    provenance = verify_matrix(context)
    directory = context.output_dir/'evaluation'
    complete = directory/'evaluation_manifest.json'
    if complete.exists():
        manifest = read_json(complete)
        check_metadata(context,manifest)
        if manifest['provenance'] != provenance or not artifacts_match(manifest['artifacts']):
            raise ValueError('locked evaluation evidence changed')
        return [__import__('pathlib').Path(row['path']) for row in manifest['artifacts']]+[complete]
    rows,diagnostics = [],[]
    for variant in context.config['upstream']['variants']:
        for seed in context.config['upstream']['seeds']:
            seed_dir = directory/'seeds'/variant/f'seed{seed}'
            seed_metrics = read_json(seed_dir/'metrics.json')
            check_metadata(context,seed_metrics)
            rows.extend(seed_metrics['rows'])
            if (seed_dir/'auxiliary_diagnostics.json').exists():
                diagnostics.extend(read_json(seed_dir/'auxiliary_diagnostics.json')['rows'])
    observed = [(r['variant'],r['upstream_seed'],r['recipe'],r['downstream_seed']) for r in rows]
    if len(set(observed)) != len(observed) or set(observed) != set(matrix_keys(context)):
        raise ValueError('aggregation refuses incomplete/duplicate seed matrix')
    metrics = directory/'metrics.json'
    write_json(metrics,{**expected_metadata(context),'split':'y_test','main_task':'eH','unit':'eV','rows':rows,
                        'expected_rows':len(matrix_keys(context)),'provenance':provenance,'auxiliary_truth_in_predictions':False})
    artifacts = [metrics]
    if diagnostics:
        path = directory/'auxiliary_diagnostics.json'
        write_json(path,{**expected_metadata(context),'rows':diagnostics,'selection_used':False})
        artifacts.append(path)
    write_json(complete,{**expected_metadata(context),'provenance':provenance,
                         'artifacts':[artifact_record(p) for p in artifacts],'status':'complete'})
    return artifacts+[complete]
