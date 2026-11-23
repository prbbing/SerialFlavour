"""Independent figure: overlay_1m_5m. Default: prepare its data, then plot it."""
from math import isclose, isfinite
from pathlib import Path
import sys

# Allow `python /absolute/path/to/this_script.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_common import (DATA, DEFAULT_GRAPH_REFINER, MODEL_COLORS, OVERLAY_123, REPO,
                             SEEDS, UPSTREAM_FIT_COLOR, WIDTH_REFERENCE_STYLE, WORKING_POINTS,
                             read_json)
from figure_runtime import run_cli

import copy
from statistics import mean, stdev
import numpy as np

RESULTS = REPO/'local/results'
E1V2 = RESULTS/'parallel_refine_e1_ex_v2'

FIGURE_ID = 'wp_rejection_summary_by_recipe_1m_5m_with_upstream_fit_overlay_2026-11-07'
KIND = 'overlay'
WP_SPECS = {
    'b_light_70': ('b-jet', 'light-jet', .70),
    'b_light_85': ('b-jet', 'light-jet', .85),
    'c_light_30': ('c-jet', 'light-jet', .30),
    'b_c_70': ('b-jet', 'c-jet', .70),
    'b_c_85': ('b-jet', 'c-jet', .85),
    'c_b_30': ('c-jet', 'b-jet', .30),
}

# ======================================================================
# 1. Data preparation: source selection, statistics, validation; no plotting.
# ======================================================================

def prepare_overlay(source, reference_path, sizes, working_points):
    original = read_json(source)
    rows = [copy.deepcopy(row) for row in original['rows'] if row['total_m'] in sizes]
    for row in rows:
        selected = {str(s): float(row['per_upstream_seed_rejection'][str(s)]) for s in SEEDS}
        values = list(selected.values())
        row.update(per_upstream_seed_rejection=selected, n_upstream_seeds=5,
                   mean=mean(values), sample_sd=stdev(values))
    fits = []
    for size in sizes:
        for key, _ in working_points:
            upstream = sorted((r for r in rows if r['total_m'] == size and r['working_point'] == key
                               and r['source_model'] == 'Parallel'), key=lambda r: -r['a_percent'])
            assert [r['a_percent'] for r in upstream] == list(range(100, 49, -5))
            a = np.array([r['a_percent'] for r in upstream], float)
            y = np.array([r['mean'] for r in upstream])
            t = (a-75)/25
            coeff = np.polynomial.polynomial.polyfit(t, y, 2)
            independent = np.linalg.lstsq(np.column_stack([np.ones_like(t), t, t*t]), y, rcond=None)[0]
            np.testing.assert_allclose(coeff, independent, rtol=1e-11, atol=1e-10)
            dense = np.linspace(100, 50, 501)
            curve = np.polynomial.polynomial.polyval((dense-75)/25, coeff)
            fits.append(dict(total_m=size, working_point=key, coefficients_c0_c1_c2=coeff.tolist(),
                             curve=[dict(a_percent=float(x), fitted_rejection=float(y)) for x, y in zip(dense, curve)]))
    # Validate against the existing paper summaries before changing any rendering.
    reference = original if reference_path == source else read_json(reference_path)
    reference_rows = [r for r in reference['rows'] if r['total_m'] in sizes]
    assert len(rows) == len(reference_rows)
    for row, saved in zip(rows, reference_rows):
        assert (row['total_m'], row['working_point'], row['a_percent'], row['source_model']) == (
            saved['total_m'], saved['working_point'], saved['a_percent'], saved['source_model'])
        np.testing.assert_allclose([row['mean'], row['sample_sd']], [saved['mean'], saved['sample_sd']], rtol=1e-12)
    for fit in fits:
        saved = next(f for f in reference['fits'] if (f['total_m'], f['working_point']) == (fit['total_m'], fit['working_point']))
        np.testing.assert_allclose([p['fitted_rejection'] for p in fit['curve']], [p['fitted_rejection'] for p in saved['curve']], rtol=1e-11, atol=1e-9)
    payload = dict(rows=rows, fits=fits, sizes=sizes, upstream_seeds=SEEDS,
                   aggregation='Downstream seed averages within upstream groups; equal-weight upstream mean and sample SD (ddof=1).',
                   fit_definition='Equal-weight quadratic OLS in t=(A-75)/25; descriptive only.',
                   cross_allocation_note='1M Y identity hashes differ across allocations; no common test sample is assumed.')
    return payload, list(dict.fromkeys((source, reference_path)))

def prepare_light_overlay():
    source = E1V2 / 'analysis/data/wp_rejection_summary_by_recipe_1m_3m_5m_with_upstream_fit_2026-11-02.json'
    return prepare_overlay(source, DATA / f'{OVERLAY_123}.json', [1, 5], WORKING_POINTS)


def _select_wp(summary, key):
    signal, background, efficiency = WP_SPECS[key]
    row, = (r for r in summary['curves'] if r['signal'] == signal and r['background'] == background
            and isclose(r['target_signal_efficiency'], efficiency, rel_tol=0, abs_tol=1e-12))
    return row


def refiner_rows(evaluation, size, allocation, model, working_points):
    seed_paths = [evaluation / f'parallel_seed{s}/{model}/rejection_comparison_dnn_seed_mean.json'
                  for s in SEEDS]
    aggregate_path = evaluation / f'{model}/rejection_comparison_parallel_seed_mean.json'
    paths = [*seed_paths, aggregate_path]
    if not all(path.is_file() for path in paths):
        return [], []

    groups = [read_json(path) for path in seed_paths]
    aggregate = read_json(aggregate_path)
    assert all('mean over 5 downstream seeds' in g['definition']['seed_aggregation'] for g in groups)
    rows = []
    for key, label in working_points:
        values = [float(_select_wp(group, key)['dnn_rejection']) for group in groups]
        assert all(isfinite(value) and value >= 1 for value in values)
        avg, sd = mean(values), stdev(values)
        saved = _select_wp(aggregate, key)
        assert isclose(avg, saved['dnn_rejection'], rel_tol=1e-12)
        assert isclose(sd, saved['dnn_rejection_sample_sd'], rel_tol=1e-12)
        rows.append(dict(total_m=size, working_point=key, label=label,
                         a_percent=allocation, b_percent=100-allocation,
                         allocation=f'{allocation}/{100-allocation}', source_model=model,
                         n_upstream_seeds=len(SEEDS), mean=avg, sample_sd=sd,
                         per_upstream_seed_rejection=dict(zip(map(str, SEEDS), values))))
    return rows, paths


def refiner_evaluation(size, a, suffix):
    name = f'experiment1_ex_v2_p122k_n{size}m_a{a:03d}_b{100-a:03d}'
    return E1V2 / f'{name}_{suffix}/evaluation'


def add_graph(payload, sources, working_points):
    payload['rows'] = [row for row in payload['rows'] if row['source_model'] != 'FG2']
    for size in payload['sizes']:
        for a in range(95, 49, -5):
            evaluation = refiner_evaluation(size, a, 'refiner_ablation_v1')
            rows, paths = refiner_rows(evaluation, size, a, DEFAULT_GRAPH_REFINER, working_points)
            if not rows:
                raise FileNotFoundError(f'Missing {DEFAULT_GRAPH_REFINER} summaries in {evaluation}')
            payload['rows'].extend(rows)
            sources.extend(paths)


def prepare_data():
    payload, sources = prepare_light_overlay()
    add_graph(payload, sources, WORKING_POINTS)
    payload['models'] = ['Parallel', 'F1', 'F3', DEFAULT_GRAPH_REFINER]
    return payload, sources

# ======================================================================
# 2. Drawing: consume only prepared data; do not aggregate or refit here.
# ======================================================================

MODEL_STYLE = {
    'Parallel': {'color': MODEL_COLORS['Parallel'], 'marker': 'o', 'offset': -.12},
    'F1': {'color': MODEL_COLORS['F1'], 'marker': 's', 'offset': -.04},
    'F3': {'color': MODEL_COLORS['F3'], 'marker': '^', 'offset': .04},
    DEFAULT_GRAPH_REFINER: {'color': MODEL_COLORS[DEFAULT_GRAPH_REFINER], 'marker': 'D', 'offset': .12},
}


def draw(data, working_points=WORKING_POINTS):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
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
    nrows = len(working_points) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 7.8 if nrows == 1 else 13.5))
    for ax, (key, label) in zip(axes.flat, working_points, strict=True):
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
    models = [Line2D([0], [0], color=s['color'], marker=s['marker'], linestyle='-',
               markerfacecolor='white' if m == 'Parallel' else s['color'],
               label='Upstream Transformer' if m == 'Parallel' else m) for m in models_to_draw for s in [model_styles[m]]]
    models.append(Line2D([0], [0], color=UPSTREAM_FIT_COLOR, linewidth=2,
                  label='Upstream quadratic fit (1M / 5M)'))
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
    if nrows == 1:
        fig.subplots_adjust(left=.052, right=.985, top=.74, bottom=.235, wspace=.25)
    else:
        fig.subplots_adjust(left=.052, right=.985, top=.90, bottom=.08, wspace=.25, hspace=.35)
    return fig, {'dpi': 300}


if __name__ == '__main__':
    run_cli(FIGURE_ID, KIND, prepare_data, draw)
