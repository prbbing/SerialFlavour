"""Dataset-level confusion matrix and locked Y evaluation."""

import torch
from torch.nn import functional as F
from pipeline.io import write_json
from pipeline.runtime import configure, to_device


def segmentation_loss(logits, target):
    mask = (target >= 0) & (target < 13)
    if not mask.any():
        raise ValueError('batch contains no valid semantic pixels')
    return F.cross_entropy(logits, target, ignore_index=-1)


def metrics(model, loader, device, auxiliary=False):
    confusion = torch.zeros(13, 13, dtype=torch.int64)
    depth_abs, depth_rel, depth_count = 0., 0., 0
    angles = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            output = model(batch['image']) if 'image' in batch else model(batch['features'])
            logits = output['logits'] if isinstance(output, dict) else output
            target = batch['segmentation']
            pred = logits.argmax(1)
            mask = (target >= 0) & (target < 13)
            confusion += torch.bincount((13*target[mask] + pred[mask]).cpu(), minlength=169).reshape(13, 13)
            if auxiliary:
                depth = batch['depth']
                valid = depth > 0
                errors = (output['depth'] - depth).abs()[valid]
                depth_abs += errors.sum().item()
                depth_rel += (errors / depth[valid]).sum().item()
                depth_count += valid.sum().item()
                normal = batch['normal']
                valid = normal.square().sum(1) > 0
                cosine = (output['normal'] * normal).sum(1)[valid].clamp(-1, 1)
                angles.append(torch.rad2deg(torch.acos(cosine)).cpu())
    intersection = confusion.diag().double()
    union = confusion.sum(0) + confusion.sum(1) - confusion.diag()
    supported = union > 0
    if not supported.any():
        raise ValueError('no valid evaluation pixels')
    result = {'miou': (intersection[supported] / union[supported]).mean().item(),
              'pixel_accuracy': (intersection.sum() / confusion.sum()).item(),
              'classes_with_union': supported.sum().item(), 'valid_pixels': confusion.sum().item(),
              'confusion_matrix': confusion.tolist()}
    if auxiliary:
        angle = torch.cat(angles) if angles else torch.empty(0)
        result.update(depth_abs_error=depth_abs/depth_count if depth_count else None,
                      depth_relative_error=depth_rel/depth_count if depth_count else None,
                      depth_valid_pixels=depth_count,
                      normal_mean_degrees=angle.mean().item() if len(angle) else None,
                      normal_median_degrees=torch.quantile(angle, .5).item() if len(angle) else None,
                      normal_valid_pixels=len(angle))
    return result


def score(model, loader, device):
    return metrics(model, loader, device)['miou']


def evaluate(context):
    from data.cv_nyu_mtan import make_loader, applicable_recipes
    from training.cv_nyu_mtan import load_upstream
    from refine.cv_nyu_mtan import cache_loader, load_readout
    device = configure(context.config['runtime'])
    rows = []
    for variant in context.config['upstream']['variants']:
        for us in context.config['upstream']['seeds']:
            model, _ = load_upstream(context, variant, us, device, frozen=True)
            native = metrics(model, make_loader(context, 'y_test', us, 2, auxiliary=variant == 'multi_task'), device,
                             auxiliary=variant == 'multi_task')
            rows.append({'variant': variant, 'upstream_seed': us, 'recipe': 'native', 'downstream_seed': None, **native})
            for recipe in applicable_recipes(context, variant):
                for ds in context.config['refiner']['seeds']:
                    readout = load_readout(context, variant, us, recipe, ds, device)
                    value = metrics(readout, cache_loader(context, variant, us, 'y_test', recipe, ds, 2), device)
                    rows.append({'variant': variant, 'upstream_seed': us, 'recipe': recipe, 'downstream_seed': ds, **value})
    return [write_json(context.output_dir / 'evaluation.json', {
        'identity': context.identity, 'split': 'y_test', 'selection': 'A_val/B_val only', 'results': rows})]
