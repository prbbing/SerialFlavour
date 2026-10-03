"""Paired smoke contrasts, with no multi-seed uncertainty claims."""

import numpy as np
from pipeline.io import read_json, write_json


def analyze(context):
    rows = read_json(context.output_dir / 'evaluation.json')['results']
    summaries = []
    for variant in context.config['upstream']['variants']:
        recipes = sorted({r['recipe'] for r in rows if r['variant'] == variant})
        for recipe in recipes:
            seed_means = []
            for seed in context.config['upstream']['seeds']:
                seed_means.append(float(np.mean([r['miou'] for r in rows if r['variant'] == variant and r['recipe'] == recipe and r['upstream_seed'] == seed])))
            summaries.append({'variant': variant, 'recipe': recipe, 'miou_mean': float(np.mean(seed_means)),
                              'upstream_seed_means': seed_means, 'upstream_sample_sd': float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None})
    contrasts = []
    for r in rows:
        if r['recipe'] not in ('aux_prediction', 'aux_hidden'):
            continue
        baseline = next(b for b in rows if b['variant'] == r['variant'] and b['upstream_seed'] == r['upstream_seed']
                        and b['downstream_seed'] == r['downstream_seed'] and b['recipe'] == 'embedding_matched')
        contrasts.append({'recipe': r['recipe'], 'upstream_seed': r['upstream_seed'], 'downstream_seed': r['downstream_seed'],
                          'miou_difference': r['miou'] - baseline['miou']})
    return [write_json(context.output_dir / 'summary.json', {
        'identity': context.identity, 'summaries': summaries, 'paired_contrasts': contrasts,
        'scope': 'real data CPU smoke; no claim of method efficacy, no scene-grouped generalization',
        'aggregation': 'downstream seeds averaged within upstream seed, SD over upstream means with ddof=1; null if n=1',
    })]
