"""Independent figure: overlay_3m_width_scan_six_working_points. Default: prepare its data, then plot it."""
from pathlib import Path
import sys

# Allow `python /absolute/path/to/this_script.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_common import REPO, DATA, SEEDS, OVERLAY_3M_WIDTHS, read_json
from pipeline_common import MODEL_COLORS, UPSTREAM_FIT_COLOR, WIDTH_REFERENCE_STYLE
from figure_runtime import run_cli

from statistics import mean, stdev
import numpy as np
import pandas as pd

RESULTS = REPO/'local/results'
E1V2 = RESULTS/'parallel_refine_e1_ex_v2'

FIGURE_ID = 'wp_rejection_summary_by_recipe_3m_with_upstream_fit_overlay_width_scan_six_working_points_2026-11-07'
KIND = 'overlay'

# ======================================================================
# 1. Data preparation: source selection, statistics, validation; no plotting.
# ======================================================================

WORKING_POINTS = [
    ('b_light_70', 'b/light 70%'), ('b_light_85', 'b/light 85%'),
    ('c_light_30', 'c/light 30%'), ('b_c_70', 'b/c 70%'),
    ('b_c_85', 'b/c 85%'), ('c_b_30', 'c/b 30%'),
]
WP_SPECS = {
    'b_light_70': ('b-jet', 'light-jet', .70),
    'b_light_85': ('b-jet', 'light-jet', .85),
    'c_light_30': ('c-jet', 'light-jet', .30),
    'b_c_70': ('b-jet', 'c-jet', .70),
    'b_c_85': ('b-jet', 'c-jet', .85),
    'c_b_30': ('c-jet', 'b-jet', .30),
}
OVERLAY_3M_REFINERS = ('F1', 'F2_origin', 'F2_vertex', 'F3')


def _select_working_point(summary, key):
    signal, background, efficiency = WP_SPECS[key]
    rows = [r for r in summary['curves'] if r['signal'] == signal and
            r['background'] == background and
            np.isclose(r['target_signal_efficiency'], efficiency, rtol=0, atol=1e-12)]
    assert len(rows) == 1, f'No unique working point: {key}'
    return rows[0]


def _prepare_input_refiners(parent):
    """Use completed five-upstream groups; leave unavailable allocations blank."""
    rows, sources, coverage, pending, baseline_audit = [], [], {}, [], []
    paired_upstream = {}
    for model in ('F2_origin', 'F2_vertex'):
        coverage[model] = []
        for a in range(95, 49, -5):
            folder = E1V2/f'experiment1_ex_v2_p122k_n3m_a{a:03d}_b{100-a:03d}_refine_inputs_v1/evaluation'
            paths = {s: folder/f'parallel_seed{s}/{model}/rejection_comparison_dnn_seed_mean.json' for s in SEEDS}
            missing = [s for s, path in paths.items() if not path.exists()]
            if missing:
                pending.append(dict(model=model, a_percent=a, b_percent=100-a, missing_upstream_seeds=missing))
                continue
            groups = {s: read_json(path) for s, path in paths.items()}
            for group in groups.values():
                assert 'mean over 5 downstream seeds' in group['definition']['seed_aggregation']
            aggregate_path = folder/model/'rejection_comparison_parallel_seed_mean.json'
            aggregate = read_json(aggregate_path)
            sources.extend([*paths.values(), aggregate_path])
            coverage[model].append(a)
            for key, label in WORKING_POINTS:
                cells = {str(s): _select_working_point(group, key) for s, group in groups.items()}
                selected = {s: float(cell['dnn_rejection']) for s, cell in cells.items()}
                values = list(selected.values())
                assert np.isfinite(values).all() and min(values) >= 1
                upstream = next(r for r in parent['rows'] if r['working_point'] == key and
                                r['source_model'] == 'Parallel' and r['a_percent'] == a)
                paired = [c['parallel_rejection'] for c in cells.values()]
                if (a, key) in paired_upstream:
                    np.testing.assert_allclose(paired, paired_upstream[a, key], rtol=1e-12)
                else:
                    paired_upstream[a, key] = paired
                    original = [upstream['per_upstream_seed_rejection'][str(s)] for s in SEEDS]
                    baseline_audit.append(dict(a_percent=a, working_point=key,
                        existing_per_upstream_seed_rejection=dict(zip(map(str, SEEDS), original)),
                        input_refiner_per_upstream_seed_rejection=dict(zip(map(str, SEEDS), paired)),
                        max_abs_relative_difference=float(np.max(np.abs(np.array(paired)/original-1)))))
                saved = _select_working_point(aggregate, key)
                np.testing.assert_allclose([mean(values), stdev(values)],
                    [saved['dnn_rejection'], saved['dnn_rejection_sample_sd']], rtol=1e-12)
                np.testing.assert_allclose([mean(paired), stdev(paired)],
                    [saved['parallel_rejection'], saved['parallel_rejection_sample_sd']], rtol=1e-12)
                rows.append(dict(total_m=3, working_point=key, label=label, a_percent=a, b_percent=100-a,
                    allocation=f'{a}/{100-a}', source_model=model, n_upstream_seeds=5,
                    mean=mean(values), sample_sd=stdev(values), per_upstream_seed_rejection=selected))
    # The available 75/25 manifests verify checkpoint and Y identities. Cache values
    # were regenerated and differ; retain the observed baseline differences above.
    identity_audit = []
    for seed in SEEDS:
        original_path = E1V2/f'experiment1_ex_v2_p122k_n3m_a075_b025/evaluation/parallel_seed{seed}/F1_embed/dnn_seed1/evaluation_manifest.json'
        if not original_path.exists():
            continue
        original = read_json(original_path)['feature_cache']
        for model in ('F2_origin', 'F2_vertex'):
            path = E1V2/f'experiment1_ex_v2_p122k_n3m_a075_b025_refine_inputs_v1/evaluation/parallel_seed{seed}/{model}/dnn_seed1/evaluation_manifest.json'
            if not path.exists():
                continue
            current = read_json(path)['feature_cache']
            for field in ('checkpoint_sha256', 'source_index_sha256', 'split_index_sha256'):
                assert current[field] == original[field], f'75/25 upstream/Y identity mismatch: {model}, seed {seed}'
            for field in ('labels', 'event_number'):
                assert current['arrays'][field]['sha256'] == original['arrays'][field]['sha256']
            identity_audit.append(dict(a_percent=75, model=model, upstream_seed=seed,
                checkpoint_sha256=current['checkpoint_sha256'],
                source_index_sha256=current['source_index_sha256'],
                split_index_sha256=current['split_index_sha256'],
                existing_feature_sha256=original['arrays']['features']['sha256'],
                input_refiner_feature_sha256=current['arrays']['features']['sha256']))
            sources.extend([original_path, path])
    return rows, sources, coverage, pending, baseline_audit, identity_audit

def prepare_data():
    """Read existing v2 summaries only; no prediction or Y evaluation is run."""
    source = E1V2/'analysis/data/rejection_log_values.csv'
    table = pd.read_csv(source)
    table = table[table.baseline_version == 'own_upstream']
    rows, fits = [], []
    sources = [source]
    for key, label in WORKING_POINTS:
        signal, background, efficiency = WP_SPECS[key]
        selected = table[(table.signal == signal) & (table.background == background) &
                         np.isclose(table.signal_efficiency, efficiency, rtol=0, atol=1e-12)]
        for a in range(100, 49, -5):
            for model in ('Parallel', 'F1', 'F3'):
                if a == 100 and model != 'Parallel':
                    continue
                saved = selected[(selected.a_percent == a) &
                    (selected.model == ('Transformer' if model == 'Parallel' else model))]
                assert len(saved) == 1, (key, a, model)
                values = saved.iloc[0][[f'rejection_seed{s}' for s in SEEDS]].to_numpy(float)
                assert np.isfinite(values).all() and np.min(values) >= 1
                np.testing.assert_allclose([mean(values), stdev(values)],
                    saved.iloc[0][['rejection_mean', 'rejection_sample_sd']].to_numpy(float), rtol=1e-12)
                rows.append(dict(total_m=3, working_point=key, label=label,
                    a_percent=a, b_percent=100-a, allocation=f'{a}/{100-a}', source_model=model,
                    n_upstream_seeds=5, mean=mean(values), sample_sd=stdev(values),
                    per_upstream_seed_rejection=dict(zip(map(str, SEEDS), values.tolist()))))
        upstream = sorted((r for r in rows if r['working_point'] == key and
                           r['source_model'] == 'Parallel'), key=lambda r: -r['a_percent'])
        a = np.array([r['a_percent'] for r in upstream], float)
        assert a.tolist() == list(range(100, 49, -5))
        y = np.array([r['mean'] for r in upstream])
        t = (a-75)/25
        coeff = np.polynomial.polynomial.polyfit(t, y, 2)
        independent = np.linalg.lstsq(np.column_stack([np.ones_like(t), t, t*t]), y, rcond=None)[0]
        np.testing.assert_allclose(coeff, independent, rtol=1e-11, atol=1e-10)
        dense = np.linspace(100, 50, 501)
        fitted = np.polynomial.polynomial.polyval((dense-75)/25, coeff)
        fits.append(dict(total_m=3, working_point=key, coefficients_c0_c1_c2=coeff.tolist(),
            curve=[dict(a_percent=float(x), fitted_rejection=float(v)) for x, v in zip(dense, fitted)]))
    payload = dict(rows=rows, fits=fits, sizes=[3], models=['Parallel', *OVERLAY_3M_REFINERS],
        upstream_seeds=SEEDS, working_points=[dict(key=k, label=v) for k, v in WORKING_POINTS],
        curve_linestyle='-', aggregation='Average five downstream seeds within each upstream seed; equal-weight mean and sample SD (ddof=1) across upstream seeds 1-5.',
        fit_definition='Equal-weight quadratic OLS in t=(A-75)/25; descriptive only.',
        refiner_selection_note='F1, F2_origin, F2_vertex and F3. No missing seed or allocation is imputed.')
    extra, extra_sources, coverage, pending, baseline_audit, identity_audit = _prepare_input_refiners(payload)
    sources.extend(extra_sources)
    payload['rows'].extend(extra)
    payload.update(refiner_coverage=coverage, refiner_pending=pending,
        input_refiner_upstream_audit=baseline_audit, input_refiner_identity_audit=identity_audit,
        input_refiner_baseline_note='F2 absolute rejections use their actual paired-regenerated-cache summaries without rescaling. Retain the original upstream curve; paired baseline differences are audited.')
    folder = RESULTS/'parallel_refine_e2_v2/analysis'
    summary_source, b85_source = folder/'width_scan_wp_summary.json', folder/'width_scan_b85.json'
    summary, b85 = read_json(summary_source), read_json(b85_source)
    identities = summary['shared_test_identities']
    assert len(identities) == 1 and b85['test_identity_hashes'] == [identities[0][0]]
    payload['width_reference_test_identity'] = identities[0]
    payload['reference_lines'] = []
    metric_fields = {'b_light_70': 'b_vs_light_at_b70_rejection',
                     'b_c_70': 'b_vs_c_at_b70_rejection',
                     'c_light_30': 'c_vs_light_at_c30_rejection',
                     'c_b_30': 'c_vs_b_at_c30_rejection'}
    for model in ('122K', '142K', '164K', '188K'):
        width = int(model[:-1])
        wp_rows = summary['per_seed'][model]
        b_rows = [r for r in b85['rows'] if str(r['model']) == str(width)]
        wp_index, b_index = {r['seed']: r for r in wp_rows}, {r['seed']: r for r in b_rows}
        assert len(wp_rows) == len(wp_index) == len(b_rows) == len(b_index) == 5
        assert set(wp_index) == set(b_index) == set(SEEDS)
        for seed in SEEDS:
            for metric, cell in [('b_vs_light_at_b70_rejection', 'light_b70'), ('b_vs_c_at_b70_rejection', 'c_b70')]:
                np.testing.assert_allclose(wp_index[seed][metric], b_index[seed][cell]['rejection'], rtol=1e-12)
        config_source = REPO/f'configs/parallel_refine/experiments/experiment2_v2/experiment2_v2_p{142 if width == 122 else width}k_n3m_a100_b000.json'
        config = read_json(config_source)['experiment']['markers']['variables']
        assert config['total_dataset_jets'] == 3000000 and config['a_fraction_percent'] == 100 and config['b_fraction_percent'] == 0
        parameters = config['baseline_parallel_parameters'] if width == 122 else config['parallel_parameters']
        sources.append(config_source)
        for key, _ in WORKING_POINTS:
            if key in ('b_light_85', 'b_c_85'):
                cell_name = 'light_b85' if key == 'b_light_85' else 'c_b85'
                for seed in SEEDS:
                    cell = b_index[seed][cell_name]
                    np.testing.assert_allclose(cell['rejection'], cell['total']/cell['passed'], rtol=1e-12)
                values = [float(b_index[s][cell_name]['rejection']) for s in SEEDS]
            else:
                values = [float(wp_index[s][metric_fields[key]]) for s in SEEDS]
            assert np.isfinite(values).all() and min(values) >= 1
            reference = dict(working_point=key, parameter_label=f'{width}k', parameters=parameters,
                allocation='100/0', total_m=3, model='upstream Transformer', seeds=SEEDS,
                missing_seeds=[], partial=False, n_seeds=5, mean=mean(values), sample_sd=stdev(values),
                per_upstream_seed_rejection=dict(zip(map(str, SEEDS), values)))
            if width == 122:
                original = next(r for r in rows if r['working_point']==key and r['source_model']=='Parallel' and r['a_percent']==100)
                np.testing.assert_allclose(values, [original['per_upstream_seed_rejection'][str(s)] for s in SEEDS], rtol=1e-12)
            payload['reference_lines'].append(reference)
    sources.extend([summary_source, b85_source])
    payload['reference_pending'] = b85['pending']
    payload['reference_definition'] = '100/0 Transformer mean over upstream seeds 1-5 at 122k/142k/164k/188k parameters. Shorter, denser dashes encode larger parameter counts.'
    # Verify every existing top-row value and fit against the saved three-panel figure.
    top_source = DATA/'prepared'/(OVERLAY_3M_WIDTHS+'.json')
    top = read_json(top_source)['payload']
    row_key = lambda r: (r['working_point'], r['a_percent'], r['source_model'])
    indexed = {row_key(r): r for r in payload['rows']}
    for original in top['rows']:
        current = indexed[row_key(original)]
        np.testing.assert_allclose([current['mean'], current['sample_sd']], [original['mean'], original['sample_sd']], rtol=1e-12)
        np.testing.assert_allclose(list(current['per_upstream_seed_rejection'].values()), list(original['per_upstream_seed_rejection'].values()), rtol=1e-12)
    for original in top['fits']:
        current = next(f for f in fits if f['working_point']==original['working_point'])
        np.testing.assert_allclose([r['fitted_rejection'] for r in current['curve']], [r['fitted_rejection'] for r in original['curve']], rtol=1e-12, atol=1e-10)
    for original in top['reference_lines']:
        current = next(r for r in payload['reference_lines'] if (r['working_point'],r['parameter_label'])==(original['working_point'],original['parameter_label']))
        assert current == original
    sources.append(top_source)
    payload['caption'] = 'Six established working points at total 3M. Upstream seeds 1-5; each refiner group averages five downstream seeds. Error bars show upstream-seed sample SD; upstream fits are descriptive. Top row matches the existing light-background figure. F2 paired-cache differences are audited; no rescaling or imputation.'
    return payload, sources

# ======================================================================
# 2. Drawing: consume only prepared data; do not aggregate or refit here.
# ======================================================================

MODEL_STYLE = {
    'Parallel': {'color': MODEL_COLORS['Parallel'], 'marker': 'o', 'offset': -.16},
    'F1': {'color': MODEL_COLORS['F1'], 'marker': 's', 'offset': -.08},
    'F2_origin': {'color': MODEL_COLORS['F2_origin'], 'marker': 'D', 'offset': 0.},
    'F2_vertex': {'color': MODEL_COLORS['F2_vertex'], 'marker': 'v', 'offset': .08},
    'F3': {'color': MODEL_COLORS['F3'], 'marker': '^', 'offset': .16},
}


def draw(data):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    sizes = data['sizes']
    rows, fits = data['rows'], data['fits']
    models_to_draw = data.get('models', list(MODEL_STYLE))
    model_styles = {model: {**style, 'color': MODEL_COLORS[model]}
                    for model, style in data.get('model_styles', MODEL_STYLE).items()}
    references = data.get('reference_lines', [])
    references_by_width = {r['parameter_label']: r for r in references}
    styles = {1: '-', 3: '--', 5: ':'}
    allocations = list(range(100, 49, -5))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.unicode_minus': False,
                         'text.color': '#293744', 'axes.labelcolor': '#293744', 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 3, figsize=(18, 13.5))
    for ax, (key, label) in zip(axes.flat, WORKING_POINTS):
        bounds = []
        x = np.arange(11, dtype=float)
        for size in sizes:
            for model in models_to_draw:
                style = model_styles[model]
                chosen = {r['a_percent']: r for r in rows if r['total_m'] == size and
                          r['working_point'] == key and r['source_model'] == model}
                y = np.array([chosen[a]['mean'] if a in chosen else np.nan for a in allocations])
                sd = np.array([chosen[a]['sample_sd'] if a in chosen else np.nan for a in allocations])
                positions = x+style['offset']; positions[0] = 0
                container = ax.errorbar(positions, y, yerr=sd, color=style['color'], ecolor=style['color'],
                    marker=style['marker'], linestyle=data.get('curve_linestyle', styles[size]), markerfacecolor='white' if model == 'Parallel' else style['color'],
                    markersize=4.4, capsize=2, linewidth=1.6, elinewidth=.75, zorder=3)
                for artist in [*container[1], *container[2]]:
                    artist.set_alpha(.30)
                finite = np.isfinite(y)
                bounds.extend((y-sd)[finite]); bounds.extend((y+sd)[finite])
            fit = next(f for f in fits if f['total_m'] == size and f['working_point'] == key)
            a = np.array([v['a_percent'] for v in fit['curve']])
            y = np.array([v['fitted_rejection'] for v in fit['curve']])
            nominal = (100-a)/5
            display = nominal+np.interp(nominal, [0, 1, 10], [0, model_styles['Parallel']['offset'], model_styles['Parallel']['offset']])
            ax.plot(display, y, color=UPSTREAM_FIT_COLOR, linestyle=data.get('curve_linestyle', styles[size]), linewidth=2, alpha=.85, zorder=5)
            bounds.extend(y)
        for reference in references:
            if reference['working_point'] == key:
                colour, dash = WIDTH_REFERENCE_STYLE[reference['parameter_label']]
                line = ax.axhline(reference['mean'], color=colour, linestyle=dash, dash_capstyle='round', linewidth=1.4, alpha=.90, zorder=2)
                line.set_gid(f'100_0_transformer_{reference["parameter_label"]}')
                bounds.append(reference['mean'])
        ax.set_box_aspect(1)
        ax.set(title=label, xlabel='A/B data allocation (%)', ylabel='Rejection')
        ax.set_xticks(x, [f'{a}/{100-a}' for a in allocations], rotation=45, ha='right', fontsize=8.5)
        ax.set_xlim(-.5, 10.5)
        lo, hi = min(bounds), max(bounds); pad = (hi-lo)*.12
        ax.set_ylim(max(0, lo-pad), hi+pad)
        ax.grid(axis='y', color='#E5E9ED', linewidth=.8)
        ax.set_axisbelow(True); ax.spines[['top', 'right']].set_visible(False)
    coverage = data.get('refiner_coverage', {})
    labels = {}
    for model in models_to_draw:
        available = coverage.get(model)
        label = 'Upstream Transformer' if model == 'Parallel' else model
        if available is not None and len(available) < 10:
            if available and available == list(range(max(available), min(available)-1, -5)):
                label += f' ({max(available)}/{100-max(available)}–{min(available)}/{100-min(available)})'
            else:
                missing = [a for a in range(95, 49, -5) if a not in available]
                label += ' (missing '+', '.join(f'{a}/{100-a}' for a in missing)+')'
        labels[model] = label
    models = [Line2D([0], [0], color=s['color'], marker=s['marker'], linestyle='-',
               markerfacecolor='white' if m == 'Parallel' else s['color'],
               label=labels[m]) for m in models_to_draw for s in [model_styles[m]]]
    models.append(Line2D([0], [0], color=UPSTREAM_FIT_COLOR, linewidth=2,
                  label='Upstream quadratic fit' if len(sizes) != 2 else 'Upstream quadratic fit (1M / 3M)'))
    fig.legend(handles=models, loc='upper center', bbox_to_anchor=(.5, .885), ncol=len(models),
               frameon=False, fontsize=10, columnspacing=1.7)
    if references:
        handles = [Line2D([0], [0], color=WIDTH_REFERENCE_STYLE[label][0], linestyle=WIDTH_REFERENCE_STYLE[label][1],
                         linewidth=1.4, dash_capstyle='round', label=f'100/0 {label}'+('*' if any(r['partial'] for r in references if r['parameter_label'] == label) else ''))
                   for label in references_by_width]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .838), ncol=4,
                   frameon=False, fontsize=10, columnspacing=2.2)
    else:
        fig.legend(handles=[Line2D([0], [0], color='#53616E', linestyle=styles[size], linewidth=2,
                                  label=f'{size}M') for size in sizes],
                   loc='upper center', bbox_to_anchor=(.5, .838), ncol=3, frameon=False, fontsize=10, columnspacing=2.2)
    fig.subplots_adjust(left=.052, right=.985, top=.90, bottom=.08, wspace=.25, hspace=.35)
    return fig, {'dpi': 300}


if __name__ == '__main__':
    run_cli(FIGURE_ID, KIND, prepare_data, draw)
