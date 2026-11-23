"""Independent 75/25, 3M rejection curves: F1, F2_origin, F2_vertex, F3.

Default: prepare this figure's data, then plot. Supports --prepare-only/--plot-only.
The earlier F1/F3/FG2 figure and its input data are preserved.
"""
from pathlib import Path
import sys

# Allow `python /absolute/path/to/this_script.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_common import (REPO, DATA, SEEDS, MAP, WORKING_POINTS, SPECS,
                             OVERLAY_123, OVERLAY_13, OVERLAY_3M_WIDTHS,
                             STRUCTURE, HEATMAP, E3, read_json)
from pipeline_common import MODEL_COLORS
from figure_runtime import run_cli

import copy
from functools import lru_cache
from statistics import mean, stdev
import numpy as np
import pandas as pd

RESULTS = REPO/'local/results'
E1V2 = RESULTS/'parallel_refine_e1_ex_v2'
MODELS = ('Transformer', 'F1', 'F2_origin', 'F2_vertex', 'F3')
SUPPLEMENTARY = ('F2_origin', 'F2_vertex')

FIGURE_ID = 'p122_a075_b025_n3m_rejc_F1_F2origin_F2vertex_F3_2026-11-04'
KIND = 'curves'

# ======================================================================
# 1. Data preparation: source selection, statistics, validation; no plotting.
# ======================================================================

@lru_cache(maxsize=None)
def curve_table():
    return pd.read_csv(E1V2 / 'analysis/data/rejection_log_values.csv')

def prepare_data():
    version = 'own_upstream'
    source = E1V2 / 'analysis/data/rejection_log_values.csv'
    table = curve_table()
    selected = table[(table.baseline_version == version) & (table.a_percent == 75)]
    folder = E1V2/'experiment1_ex_v2_p122k_n3m_a075_b025_refine_inputs_v1/evaluation'
    groups, aggregates = {}, {}
    sources = [source]
    for model in SUPPLEMENTARY:
        paths = [folder/f'parallel_seed{s}/{model}/rejection_comparison_dnn_seed_mean.json' for s in SEEDS]
        groups[model] = [read_json(p) for p in paths]
        for group in groups[model]:
            assert 'mean over 5 downstream seeds' in group['definition']['seed_aggregation']
        aggregate_path = folder/model/'rejection_comparison_parallel_seed_mean.json'
        aggregates[model] = read_json(aggregate_path)
        sources.extend([*paths, aggregate_path])

    def cells(summary, signal, background, x):
        rows = sorted((r for r in summary['curves'] if r['signal'] == signal and
                       r['background'] == background), key=lambda r: r['target_signal_efficiency'])
        np.testing.assert_allclose([r['target_signal_efficiency'] for r in rows], x, rtol=0, atol=1e-12)
        return rows

    panels, baseline_audit = [], []
    for title, _, signal, background, lo, hi in SPECS:
        curves = []
        panel_frame = selected[(selected.signal == signal) & (selected.background == background)]
        reference = panel_frame[panel_frame.model == 'Transformer'].sort_values('signal_efficiency')
        assert len(reference) and not reference.signal_efficiency.duplicated().any()
        x = reference.signal_efficiency.to_numpy(float)
        denominator = reference[[f'baseline_rejection_seed{s}' for s in SEEDS]].to_numpy(float).T
        old_upstream = table[(table.baseline_version == 'own_upstream') & (table.a_percent == 75) &
                            (table.signal == signal) & (table.background == background) &
                            (table.model == 'Transformer')].sort_values('signal_efficiency')
        old_arr = old_upstream[[f'rejection_seed{s}' for s in SEEDS]].to_numpy(float).T
        native_upstream = None
        for model in MODELS:
            if model in SUPPLEMENTARY:
                group_cells = [cells(g, signal, background, x) for g in groups[model]]
                arr = np.array([[r['dnn_rejection'] for r in rows] for rows in group_cells], dtype=float)
                native = np.array([[r['parallel_rejection'] for r in rows] for rows in group_cells], dtype=float)
                aggregate = cells(aggregates[model], signal, background, x)
                for field, values in [('dnn_rejection', arr), ('parallel_rejection', native)]:
                    np.testing.assert_allclose(values.mean(axis=0), [r[field] for r in aggregate], rtol=1e-12)
                    np.testing.assert_allclose(values.std(axis=0, ddof=1),
                        [r[field+'_sample_sd'] for r in aggregate], rtol=1e-12, atol=1e-12)
                if native_upstream is None:
                    native_upstream = native
                    baseline_audit.append(dict(signal=signal, background=background, x=x.tolist(),
                        per_upstream_seed_rejection={str(s): values.tolist() for s, values in zip(SEEDS, native)},
                        max_abs_relative_difference_vs_original_75_25=float(np.max(np.abs(native/old_arr-1)))))
                else:
                    np.testing.assert_allclose(native, native_upstream, rtol=1e-12)
            else:
                frame = panel_frame[panel_frame.model == model].sort_values('signal_efficiency')
                np.testing.assert_allclose(frame.signal_efficiency, x, rtol=0, atol=1e-12)
                assert not frame.signal_efficiency.duplicated().any()
                arr = frame[[f'rejection_seed{s}' for s in SEEDS]].to_numpy(float).T
                np.testing.assert_allclose(denominator,
                    frame[[f'baseline_rejection_seed{s}' for s in SEEDS]].to_numpy(float).T, rtol=1e-12)
            assert arr.shape == denominator.shape == (5, len(x))
            assert np.isfinite(arr).all() and np.isfinite(denominator).all()
            assert np.min(arr) >= 1 and np.min(denominator) >= 1
            ratio = arr/denominator
            arrays = [arr.mean(axis=0), arr.std(axis=0, ddof=1), ratio.mean(axis=0), ratio.std(axis=0, ddof=1)]
            if model not in SUPPLEMENTARY:
                for values, field in zip(arrays, ['rejection_mean', 'rejection_sample_sd', 'paired_ratio_mean', 'paired_ratio_sample_sd']):
                    np.testing.assert_allclose(values, frame[field], rtol=1e-12, atol=1e-12)
            curves.append(dict(model=model, x=x.tolist(),
                               per_upstream_seed_rejection={str(s): values.tolist() for s, values in zip(SEEDS, arr)},
                               per_upstream_seed_baseline={str(s): values.tolist() for s, values in zip(SEEDS, denominator)},
                               **{k: v.tolist() for k, v in zip(['mean', 'sd', 'ratio', 'ratio_sd'], arrays)}))
        panels.append(dict(title=title, signal=signal, lo=lo, hi=hi, curves=curves))
    return dict(panels=panels, version=version, protocol='v2', upstream_seeds=SEEDS,
                models=list(MODELS), aggregation='Average five downstream seeds within each upstream seed, then mean and sample SD (ddof=1) across upstream seeds 1-5.',
                ratio_definition='Mean and sample SD of same-number upstream-seed ratios to the selected original Transformer baseline: 75/25 for own_upstream, 100/0 for 100_0. This is not a ratio of grand means.',
                supplementary_upstream_audit=baseline_audit,
                supplementary_upstream_note='F2 paired upstream curves differ from the original 75/25 curve. They are recorded for auditing; all ratios use the selected original Transformer baseline, consistently with F1/F3. Absolute rejection is not rescaled.'), sources

# ======================================================================
# 2. Drawing: consume only prepared data; do not aggregate or refit here.
# ======================================================================

def draw(data):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    v2 = data['protocol'] == 'v2'
    version = data['version']
    if v2:
        plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none'})
    styles = {'Transformer': (MODEL_COLORS['Transformer'], '-'), 'Parallel': (MODEL_COLORS['Parallel'], '-'),
              'Parallel (100/0)': (MODEL_COLORS['Parallel (100/0)'], '-'), 'F1': (MODEL_COLORS['F1'], '--'),
              'F2_origin': (MODEL_COLORS['F2_origin'], '-.'), 'F2_vertex': (MODEL_COLORS['F2_vertex'], ':'),
              'F3': (MODEL_COLORS['F3'], '-')}
    fig = plt.figure(figsize=(13.5, 13.0))
    outer = fig.add_gridspec(2, 2, left=.08, right=.98, bottom=.105, top=.82, hspace=.22, wspace=.23)
    for index, panel in enumerate(data['panels']):
        inner = outer[index//2, index%2].subgridspec(2, 1, height_ratios=(3, 1), hspace=.06)
        top = fig.add_subplot(inner[0]); bottom = fig.add_subplot(inner[1], sharex=top)
        top.tick_params(axis='x', labelbottom=False)
        bands = []
        for curve in panel['curves']:
            model = curve['model']; colour, style = styles[model]
            x, y, sd, ratio, rsd = [np.asarray(curve[k], float) for k in ['x', 'mean', 'sd', 'ratio', 'ratio_sd']]
            label = 'Transformer (75/25)' if model == 'Transformer' else model
            top.plot(x, y, color=colour, ls=style, lw=1.8, label=label)
            top.fill_between(x, np.maximum(1, y-sd), y+sd, color=colour, alpha=.10)
            bottom.plot(x, ratio, color=colour, ls=style, lw=1.7)
            if model != 'Parallel (100/0)':
                bottom.fill_between(x, ratio-rsd, ratio+rsd, color=colour, alpha=.10)
            bands.extend([float(np.min(ratio-rsd)), float(np.max(ratio+rsd))])
        lo, hi = panel['lo'], panel['hi']
        top.set(title=panel['title'], yscale='log', xlim=(lo, hi), ylabel='Background rejection')
        top.set_ylim(bottom=1)
        bottom.axhline(1, color='#666666' if v2 or version == '100_0' else '#666', ls='--', lw=1)
        if v2:
            lower, upper = min(bands), max(bands); pad = max(.025, .08*(upper-lower))
            bottom.set_ylim(max(0, min(.7 if lo > .5 else .6, lower-pad)), max(1.05, upper+pad))
        else:
            bottom.set_ylim(bottom=.7 if lo > .5 else .6)
        bottom.set_xlabel(r'$\epsilon$ signal efficiency')
        bottom.set_ylabel(('R / upstream' if version == 'own_upstream' else 'R / 100/0') if v2 else
                          ('R / Parallel' if version == 'own_upstream' else 'R / 100/0 Parallel'))
        for axis in (top, bottom):
            axis.grid(axis='y', color='#DDDDDD', lw=.5)
            for efficiency in ((.70, .85) if panel['signal'] == 'b-jet' else (.30,)) if v2 else ((.70 if lo > .5 else .30),):
                axis.axvline(efficiency, color='#888888' if v2 or version == '100_0' else '#888',
                             ls=':', lw=1 if axis is top or v2 else .9)
    handles, labels = top.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .895), ncol=len(labels), frameon=False)
    return fig, {'dpi': 180, 'bbox_inches': 'tight'}


if __name__ == '__main__':
    run_cli(FIGURE_ID, KIND, prepare_data, draw)
