"""Paired contrasts and nested seed statistics; smoke does not prove efficacy."""
import numpy as np
from pipeline.io import read_json, write_json

def analyze(context):
    rows = read_json(context.output_dir / 'evaluation.json')['results']
    summaries, contrasts = [], []
    for variant in context.config['upstream']['variants']:
        for recipe in sorted({r['recipe'] for r in rows if r['variant'] == variant}):
            means = [float(np.mean([r['accuracy'] for r in rows if r['variant'] == variant and r['recipe'] == recipe and r['upstream_seed'] == us]))
                     for us in context.config['upstream']['seeds']]
            summaries.append({'variant': variant, 'recipe': recipe, 'accuracy_mean': float(np.mean(means)),
                              'upstream_seed_means': means, 'upstream_sample_sd': float(np.std(means, ddof=1)) if len(means) > 1 else None})
    for row in rows:
        if row['recipe'] != 'aux_prediction':
            continue
        baseline = next(r for r in rows if r['variant'] == row['variant'] and r['upstream_seed'] == row['upstream_seed']
                        and r['downstream_seed'] == row['downstream_seed'] and r['recipe'] == 'embedding_matched')
        contrasts.append({'upstream_seed': row['upstream_seed'], 'downstream_seed': row['downstream_seed'],
                          'comparison': 'MT aux_prediction minus MT embedding_matched',
                          'accuracy_difference': row['accuracy'] - baseline['accuracy']})
    native = {(r['variant'], r['upstream_seed']): r for r in rows if r['recipe'] == 'native'}
    native_contrasts = [{'upstream_seed': us, 'comparison': 'MT native minus ST native',
                         'accuracy_difference': native[('multi_task', us)]['accuracy'] - native[('single_task', us)]['accuracy']}
                        for us in context.config['upstream']['seeds']]
    return [write_json(context.output_dir / 'summary.json', {
        'identity': context.identity, 'summaries': summaries, 'paired_contrasts': contrasts,
        'native_contrasts': native_contrasts,
        'aggregation': 'average downstream seeds within upstream seed, sample SD ddof=1 over upstream means; null for n=1',
        'scope': 'real-data CPU engineering smoke with random reduced XLM-R; no claim about pretrained XLM-R or statistical significance',
        'auxiliary_information': 'deterministic frozen outputs, not new Shannon information relative to complete H',
    })]
