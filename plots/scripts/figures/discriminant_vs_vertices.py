"""Independent figure: discriminant_vs_vertices. Default: prepare its data, then plot it."""
from pathlib import Path
import sys

# Allow `python /absolute/path/to/this_script.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_common import REPO, DATA, SEEDS, STRUCTURE, read_json
from pipeline_common import MODEL_COLORS
from figure_runtime import run_cli

import numpy as np
import pandas as pd

RESULTS = REPO/'local/results'
E1V2 = RESULTS/'parallel_refine_e1_ex_v2'
SUPPLEMENTARY = ('FG1_48d', 'FG2_32d')
EXPERIMENT = 'experiment1_ex_v2_p122k_n3m_a075_b025'
STYLES = {
    'F1': (MODEL_COLORS['F1'], 'o'), 'F3': (MODEL_COLORS['F3'], 's'), 'FG2': (MODEL_COLORS['FG2'], '^'),
    'FG1_48d': (MODEL_COLORS['FG1_48d'], 'X'), 'FG2_32d': (MODEL_COLORS['FG2_32d'], 'h'),
}
DISPLAY_MODELS = ('F1', 'F3', 'FG1_48d')

FIGURE_ID = STRUCTURE
KIND = 'structure'

# ======================================================================
# 1. Data preparation: source selection, statistics, validation; no plotting.
# ======================================================================

def finite_records(frame):
    """Represent missing bins as JSON null, preserving their count and seed flags."""
    return frame.astype(object).where(pd.notnull(frame), None).to_dict('records')

def discriminants(probabilities):
    p = probabilities.astype('float64')
    assert np.isfinite(p).all() and np.min(p) >= 0 and np.max(p) <= 1
    assert np.max(np.abs(p.sum(axis=1)-1)) < 1e-5
    p = np.clip(p, 1e-12, None)
    p /= p.sum(axis=1, keepdims=True)
    return np.stack((np.log(p[:, 0]/(.2*p[:, 1]+.8*p[:, 2])),
                     np.log(p[:, 1]/(.3*p[:, 0]+.7*p[:, 2]))), axis=1)


def prepare_supplementary():
    """Pair each new refiner with the upstream probabilities used in its evaluation."""
    feature_path = E1V2/'probe_substructure/tables/jet_level_features.npz'
    with np.load(feature_path, allow_pickle=False) as f:
        identity = {key: f[key] for key in ('y', 'source_index', 'event_number')}
        counts = f['n_sv_ge2']
    y = identity['y']
    sources, rows, overall, audit = [feature_path], [], [], []

    def load_scores(path):
        with np.load(path, allow_pickle=False) as f:
            for key, expected in identity.items():
                np.testing.assert_array_equal(f[key], expected, err_msg=f'{path}: {key}')
            result = discriminants(f['probabilities'])
        sources.append(path)
        return result

    for seed in SEEDS:
        original_dir = E1V2/EXPERIMENT/f'evaluation/parallel_seed{seed}'
        folder = E1V2/(EXPERIMENT+'_refine_inputs_v1')/f'evaluation/parallel_seed{seed}'
        original_manifest = original_dir/'F1_embed/dnn_seed1/evaluation_manifest.json'
        original = read_json(original_manifest)['feature_cache']
        sources.append(original_manifest)
        upstream = load_scores(folder/'parallel/test_predictions.npz')
        upstream_manifest_path = folder/'parallel/evaluation_manifest.json'
        paired = read_json(upstream_manifest_path)['feature_cache']
        sources.append(upstream_manifest_path)
        for field in ('checkpoint_sha256', 'source_index_sha256', 'split_index_sha256'):
            assert paired[field] == original[field], (seed, field)
        for field in ('labels', 'event_number'):
            assert paired['arrays'][field]['sha256'] == original['arrays'][field]['sha256']
        original_scores = load_scores(original_dir/'parallel/test_predictions.npz')
        audit.append(dict(upstream_seed=seed, checkpoint_sha256=paired['checkpoint_sha256'],
            source_index_sha256=paired['source_index_sha256'],
            original_feature_sha256=original['arrays']['features']['sha256'],
            paired_feature_sha256=paired['arrays']['features']['sha256'],
            max_abs_upstream_discriminant_difference=float(np.max(np.abs(upstream-original_scores))),
            mean_upstream_discriminant_difference_by_flavour={
                str(flavour): (upstream-original_scores)[y == flavour].mean(axis=0).tolist()
                for flavour in range(3)}))
        for model in SUPPLEMENTARY:
            delta = np.zeros_like(upstream)
            for downstream in SEEDS:
                leaf = folder/model/f'dnn_seed{downstream}'
                manifest_path = leaf/'evaluation_manifest.json'
                manifest = read_json(manifest_path)
                assert manifest['recipe'] == model and manifest['parallel_seed'] == seed
                assert manifest['downstream_seed'] == downstream and manifest['split'] == 'y_test'
                assert manifest['feature_cache'] == paired, (model, seed, downstream)
                sources.append(manifest_path)
                delta += (load_scores(leaf/'test_predictions.npz')-upstream)/len(SEEDS)
            # Mirror the original per-jet cache precision before bin aggregation.
            delta = delta.astype('float32').astype('float64')
            assert np.isfinite(delta).all()
            for flavour in range(3):
                mask = y == flavour
                flavour_mean = delta[mask].mean(axis=0)
                for ti, tag in enumerate(('b', 'c')):
                    overall.append(dict(flavour=flavour, model=model, tag=tag,
                        upstream_seed=seed, n_jets=int(mask.sum()), mean_delta=float(flavour_mean[ti])))
                for count in range(7):
                    selected = mask & (counts == count)
                    n = int(selected.sum())
                    values = delta[selected].mean(axis=0) if n else [np.nan, np.nan]
                    for ti, tag in enumerate(('b', 'c')):
                        rows.append(dict(feature='n_sv_ge2', count=count, flavour=flavour,
                            model=model, tag=tag, upstream_seed=seed, n_jets=n,
                            mean_delta=float(values[ti])))
            print(f'Prepared {model}, upstream seed {seed}', flush=True)
    return pd.DataFrame(rows), pd.DataFrame(overall), audit, sources


def prepare_data():
    folder = E1V2 / 'probe_substructure/tables'
    summaries, sources = [], []
    extra, extra_overall, audit, extra_sources = prepare_supplementary()
    for stem, keys in [('discriminant_vs_structure_a075', ['feature', 'count', 'flavour', 'model', 'tag']),
                       ('discriminant_overall_a075', ['flavour', 'model', 'tag'])]:
        source = folder / f'{stem}_per_seed.csv'
        frame = pd.read_csv(source)
        frame = frame[frame.upstream_seed.isin(SEEDS)]
        if 'feature' in frame:
            frame = frame[frame.feature == 'n_sv_ge2']
        assert set(frame.upstream_seed) == set(SEEDS)
        assert not frame.duplicated(keys+['upstream_seed']).any()
        summary = frame.groupby(keys, as_index=False).agg(mean=('mean_delta', 'mean'), sd=('mean_delta', 'std'),
                                                        n_seeds=('mean_delta', 'count'), n_jets=('n_jets', 'first'))
        assert set(summary.n_seeds) <= {0, 5}
        reference = pd.read_csv(DATA / f'{stem}_seed1to5_summary.csv')
        pd.testing.assert_frame_equal(summary, reference, check_exact=False, rtol=1e-12, atol=1e-12)
        combined = pd.concat([frame, extra if 'feature' in frame else extra_overall], ignore_index=True)
        assert not combined.duplicated(keys+['upstream_seed']).any()
        assert set(combined.model) == set(STYLES)
        assert combined.groupby(keys).upstream_seed.apply(set).apply(lambda s: s == set(SEEDS)).all()
        assert (combined.groupby(keys).n_jets.nunique() == 1).all()
        summary = combined.groupby(keys, as_index=False).agg(mean=('mean_delta', 'mean'), sd=('mean_delta', 'std'),
                                                        n_seeds=('mean_delta', 'count'), n_jets=('n_jets', 'first'))
        assert set(summary.n_seeds) <= {0, 5}
        summaries.append(finite_records(summary))
        sources.extend([source, DATA/f'{stem}_seed1to5_summary.csv'])
    quantile_source = folder / 'structure_quantiles_n_sv_ge2_a075.csv'
    quantiles = pd.read_csv(quantile_source)
    guides = []
    for (flavour, position), group in quantiles.groupby(['flavour', 'position'], sort=True):
        guides.append(dict(flavour=int(flavour), position=float(position),
                           label='/'.join(f'{100*q:.0f}%' for q in group['quantile'])))
    return dict(summary=summaries[0], overall=summaries[1], quantiles=guides,
                feature='n_sv_ge2', limit=6, upstream_seeds=SEEDS, models=list(STYLES),
                supplementary_per_seed=finite_records(extra),
                supplementary_overall_per_seed=finite_records(extra_overall),
                supplementary_upstream_audit=audit,
                aggregation='Mean of five downstream discriminant shifts per jet; mean and sample SD across upstream seeds 1-5.',
                supplementary_baseline='Each supplementary refiner uses its own paired regenerated upstream predictions; original F1/F3/FG2 values are preserved.',
                discriminant_coefficients={'f_c': .2, 'f_b': .3}), sources+extra_sources+[quantile_source]

# ======================================================================
# 2. Drawing: consume only prepared data; do not aggregate or refit here.
# ======================================================================

def draw(data):
    import matplotlib.pyplot as plt
    summary = pd.DataFrame(data['summary'])
    overall = pd.DataFrame(data['overall'])
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout='constrained')
    for flavour in range(3):
        for ti, tag in enumerate(('b', 'c')):
            ax = axes[flavour, ti]
            sign = -1 if flavour != ti else 1
            direction = '+' if sign == 1 else '-'
            for model in DISPLAY_MODELS:
                colour, marker = STYLES[model]
                frame = summary[(summary.flavour == flavour) & (summary.model == model) &
                                (summary.tag == tag)].sort_values('count')
                y = sign*frame['mean'].where(frame.n_seeds == 5)
                sd = frame.sd.where(frame.n_seeds == 5)
                ax.plot(frame['count'], y, marker=marker, ms=4, color=colour, label=model)
                ax.fill_between(frame['count'], y-sd, y+sd, color=colour, alpha=.12)
                baseline = overall[(overall.flavour == flavour) & (overall.model == model) & (overall.tag == tag)].iloc[0]
                ax.axhline(sign*baseline['mean'], color=colour, ls='--', lw=1.3, label=f'{model} overall mean')
            ax.axhline(0, color='.5', lw=.8)
            ax.set(xlim=(0, data['limit']), xlabel='Truth secondary vertices (at least 2 selected tracks)',
                   ylabel=rf'Mean favourable change (${direction}\Delta D_{tag}$)',
                   title=rf'{("b jets", "c jets", "light jets")[flavour]} / $D_{tag}$')
            ax.set_xticks(np.arange(0, int(data['limit'])+1))
            for guide in data['quantiles']:
                if guide['flavour'] != flavour:
                    continue
                position = guide['position']; at_left = position == 0
                ax.axvline(position, color='#666666', ls='--', lw=1, alpha=.8, zorder=3, clip_on=False)
                ax.annotate(guide['label'], (position, .98), xycoords=ax.get_xaxis_transform(),
                            xytext=(3 if at_left else 0, 0), textcoords='offset points',
                            ha='left' if at_left else 'center', va='top', fontsize=8, color='#444444',
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1))
            ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Matplotlib fills columns: each curve/mean pair makes one column,
    # placing the three marker entries above their dashed overall means.
    fig.legend(handles, labels, loc='outside upper center', ncol=len(DISPLAY_MODELS),
               fontsize=9, frameon=False)
    return fig, {'bbox_inches': 'tight'}


if __name__ == '__main__':
    run_cli(FIGURE_ID, KIND, prepare_data, draw)
