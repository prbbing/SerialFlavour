"""Intent accuracy/macro-F1 and official-style contiguous-type slot spans."""
import torch
from torch.nn import functional as F
from pipeline.io import write_json
from pipeline.runtime import configure, to_device

def spans(sequence):
    result, start, current = set(), 0, 0
    for i, value in enumerate(list(sequence) + [0]):
        value = int(value)
        if value != current:
            if current != 0:
                result.add((current, start, i))
            current, start = value, i
    return result

def metrics(model, loader, device, auxiliary=False):
    confusion = torch.zeros(60, 60, dtype=torch.long)
    predictions, nll = [], 0.0
    correct_spans, gold_spans, predicted_spans, exact_frames = 0, 0, 0, 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            output = model(batch)
            logits = output['logits'] if isinstance(output, dict) else output
            pred, target = logits.argmax(-1), batch['intent_num']
            confusion += torch.bincount((60 * target + pred).cpu(), minlength=3600).reshape(60, 60)
            nll += F.cross_entropy(logits, target, reduction='sum').item()
            predictions.extend(pred.cpu().tolist())
            if auxiliary:
                slot_preds = output['slot_logits'].argmax(-1)
                for j in range(len(target)):
                    valid = batch['word_mask'][j]
                    gold = batch['slots_num'][j][valid].cpu().tolist()
                    predicted = slot_preds[j][valid].cpu().tolist()
                    a, b = spans(gold), spans(predicted)
                    correct_spans += len(a & b)
                    gold_spans += len(a)
                    predicted_spans += len(b)
                    exact_frames += int(pred[j] == target[j] and predicted == gold)
    n = int(confusion.sum())
    if n == 0:
        raise ValueError('empty evaluation split')
    tp = confusion.diag().double()
    denom = confusion.sum(0) + confusion.sum(1)
    f1 = 2 * tp / denom.clamp_min(1)
    result = {'accuracy': tp.sum().item() / n, 'macro_f1_60': f1.mean().item(),
              'main_nll': nll / n, 'n': n, 'correct': int(tp.sum()), 'classes_present': int((confusion.sum(1) > 0).sum()),
              'intent_predictions': predictions}
    if auxiliary:
        result.update(slot_span_micro_f1=2 * correct_spans / (gold_spans + predicted_spans) if gold_spans + predicted_spans else 0.0,
                      gold_spans=gold_spans, predicted_spans=predicted_spans, matched_spans=correct_spans,
                      semantic_frame_accuracy=exact_frames / n,
                      slot_metric='first-subword decoded word-level contiguous-type spans, Other excluded, no BIO head')
    return result

def score(model, loader, device):
    return metrics(model, loader, device)['accuracy']

def evaluate(context):
    from experiments.nlp_massive_xlm.data import make_loader, applicable_recipes, load_split
    from experiments.nlp_massive_xlm.training import load_upstream
    from experiments.nlp_massive_xlm.refine import cache_loader, load_readout
    device = configure(context.config['runtime'])
    rows = []
    ids = load_split(context, 'y_test')['ids']
    for variant in context.config['upstream']['variants']:
        for us in context.config['upstream']['seeds']:
            model, _ = load_upstream(context, variant, us, device, frozen=True)
            multi = variant == 'multi_task'
            value = metrics(model, make_loader(context, 'y_test', us, context.config['upstream']['batch_size'], auxiliary=multi), device, auxiliary=multi)
            rows.append({'variant': variant, 'upstream_seed': us, 'recipe': 'native', 'downstream_seed': None, **value})
            del model
            for recipe in applicable_recipes(context, variant):
                for ds in context.config['refiner']['seeds']:
                    readout = load_readout(context, variant, us, recipe, ds, device)
                    value = metrics(readout, cache_loader(context, variant, us, 'y_test', recipe, ds, context.config['refiner']['batch_size']), device)
                    rows.append({'variant': variant, 'upstream_seed': us, 'recipe': recipe, 'downstream_seed': ds, **value})
    return [write_json(context.output_dir / 'evaluation.json', {
        'identity': context.identity, 'split': 'y_test', 'ids': ids, 'selection': 'A_val/B_val only',
        'scope': ('full official en-US test' if context.config['data'].get('split_mode') == 'official_grouped_full' else
                  'locked small subset of official test, not complete official test'), 'results': rows})]
