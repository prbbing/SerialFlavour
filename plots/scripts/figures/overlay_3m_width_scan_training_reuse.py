"""Overlay disjoint-B and upstream-training-reuse refiners at six working points."""
from math import isfinite
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_runtime import checked_input, run_cli
from figures import overlay_3m_width_scan_six_working_points as original
from figures.overlay_1m_5m import refiner_evaluation, refiner_rows
from pipeline_common import (
    DEFAULT_GRAPH_REFINER, MAP, MODEL_COLORS, SEEDS, UPSTREAM_FIT_COLOR,
    WIDTH_REFERENCE_STYLE, read_json,
)

FIGURE_ID = 'wp_rejection_3m_width_scan_6wp_training_reuse'
KIND = 'overlay'
ALLOCATIONS = tuple(range(100, 49, -5))
RECIPES = {'F1': 'F1_embed', 'F3': 'F3_embed_aux', DEFAULT_GRAPH_REFINER: DEFAULT_GRAPH_REFINER}
AGGREGATE = 'rejection_comparison_parallel_seed_mean.json'
WORKING_POINTS = original.WORKING_POINTS


def prepare_data(recipes=RECIPES, working_points=WORKING_POINTS, *, include_100_0_reuse=True):
    _, base_path = checked_input(original.FIGURE_ID, KIND)
    base = read_json(base_path)['payload']
    sources = [base_path]
    wp_keys = {key for key, _ in working_points}
    rows_at_wp = [r for r in base['rows'] if r['working_point'] in wp_keys]
    disjoint = [r for r in rows_at_wp if r['source_model'] in recipes]
    if DEFAULT_GRAPH_REFINER in recipes:
        for a in ALLOCATIONS[1:]:
            evaluation = refiner_evaluation(3, a, 'refine_inputs_v1')
            rows, paths = refiner_rows(evaluation, 3, a, DEFAULT_GRAPH_REFINER, working_points)
            if not rows:
                raise FileNotFoundError(f'Missing complete {DEFAULT_GRAPH_REFINER} results: {evaluation}')
            disjoint.extend(rows)
            sources.extend(paths)
    assert {(r['a_percent'], r['working_point'], r['source_model']) for r in disjoint} == {
        (a, wp, model) for a in ALLOCATIONS[1:] for wp, _ in working_points for model in recipes
    }
    assert len(disjoint) == len(ALLOCATIONS[1:]) * len(working_points) * len(recipes)

    upstream = [r for r in rows_at_wp if r['source_model'] == 'Parallel']
    upstream_means = {(r['a_percent'], r['working_point']): r['mean'] for r in upstream}
    reused = []
    reuse_allocations = ALLOCATIONS if include_100_0_reuse else ALLOCATIONS[1:]
    for a in reuse_allocations:
        reuse_percent = 25 if a == 100 else 100 - a
        evaluation = refiner_evaluation(3, a, f'refiner{reuse_percent:02d}')
        for model, recipe in recipes.items():
            path = evaluation / recipe / AGGREGATE
            summary = read_json(path)
            if 'mean over 5 Parallel seeds' not in summary['definition']['seed_aggregation']:
                raise ValueError(f'Expected a five-upstream-seed summary: {path}')
            sources.append(path)
            for wp, label in working_points:
                cell = original._select_working_point(summary, wp)
                avg, sd = cell['dnn_rejection'], cell['dnn_rejection_sample_sd']
                paired = cell['parallel_rejection']
                if not all(isfinite(v) for v in (avg, sd, paired)) or not (avg >= 1 and sd >= 0 and paired >= 1):
                    raise ValueError(f'Invalid rejection summary at {wp}: {path}')
                reused.append(dict(total_m=3, working_point=wp, label=label,
                    a_percent=a, b_percent=100-a, source_model=model, recipe=recipe,
                    n_upstream_seeds=len(SEEDS), mean=avg, sample_sd=sd,
                    reuse_percent=reuse_percent, paired_upstream_mean=paired,
                    paired_upstream_relative_difference=paired / upstream_means[a, wp] - 1))

    fits = [r for r in base['fits'] if r['working_point'] in wp_keys]
    references = [r for r in base['reference_lines'] if r['working_point'] in wp_keys]
    reuse_note = 'At 100/0, the isolated refiner markers use 25% of A. ' if include_100_0_reuse else ''
    return dict(upstream=upstream, fits=fits, reference_lines=references,
                disjoint=disjoint, reused=reused, upstream_seeds=SEEDS,
                caption=(
                    f'{len(working_points)} working points at 3M. Refiner means average downstream seeds within each '
                    'upstream seed first. Error bars show sample SD across five upstream seeds. '
                    'For 95/5 through 50/50, refiners use either disjoint B or the same number of '
                    f'training jets sampled from A by event. {reuse_note}'
                    'The original upstream scan, quadratic fits and 100/0 width '
                    'references are shown. Refiner rejection values are taken directly from their '
                    'evaluations. The reused results retain their paired upstream means in the data.'
                )), sources


def draw(data, recipes=RECIPES, working_points=WORKING_POINTS):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'text.color': '#293744', 'axes.labelcolor': '#293744'})
    single_row = len(working_points) == 3
    fig, axes = plt.subplots(1 if single_row else 2, 3, figsize=(18, 7.2 if single_row else 13.5))
    models = ('Parallel', *recipes)
    markers = dict(zip(('Parallel', *RECIPES), ('o', 's', '^', 'D')))
    offsets = dict(zip(('Parallel', *RECIPES), (-.32, -.18, .01, .20)))
    x = np.arange(len(ALLOCATIONS))

    def series(ax, rows, wp, model, *, reused=False):
        selected = {r['a_percent']: r for r in rows if r['working_point'] == wp
                    and r['source_model'] == model}
        allocations = [a for a in ALLOCATIONS if a in selected]
        positions = np.array([(100 - a) / 5 for a in allocations]) + offsets[model]
        if model != 'Parallel':
            positions += .045 if reused else -.045
        color = MODEL_COLORS[model]
        y = [selected[a]['mean'] for a in allocations]
        sd = [selected[a]['sample_sd'] for a in allocations]
        bars = ax.errorbar(positions, y, yerr=sd, color=color, marker=markers[model],
                           markerfacecolor='white' if reused or model == 'Parallel' else color,
                           linestyle='none', markersize=4.6, capsize=1.7, elinewidth=.7,
                           zorder=5 if reused else 4)
        for artist in (*bars[1], *bars[2]):
            artist.set_alpha(.26)
        start = int(reused and allocations[:1] == [100])
        ax.plot(positions[start:], y[start:], color=color, linestyle='--' if reused else '-',
                linewidth=1.45, zorder=3)

    fit_by_wp = {fit['working_point']: fit for fit in data['fits']}
    for ax, (wp, label) in zip(axes.flat, working_points):
        for reference in data['reference_lines']:
            if reference['working_point'] == wp:
                color, dash = WIDTH_REFERENCE_STYLE[reference['parameter_label']]
                ax.axhline(reference['mean'], color=color, linestyle=dash, linewidth=1.2, zorder=1)
        fit = fit_by_wp[wp]['curve']
        ax.plot([(100 - cell['a_percent']) / 5 + offsets['Parallel'] for cell in fit],
                [cell['fitted_rejection'] for cell in fit], color=UPSTREAM_FIT_COLOR,
                linewidth=1.8, alpha=.85, zorder=2)
        series(ax, data['upstream'], wp, 'Parallel')
        for model in recipes:
            series(ax, data['disjoint'], wp, model)
            series(ax, data['reused'], wp, model, reused=True)
        ax.set(title=label, xlabel='A/B data allocation (%)', ylabel='Rejection', xlim=(-.60, 10.56))
        ax.set_xticks(x, [f'{a}/{100-a}' for a in ALLOCATIONS], rotation=45, ha='right', fontsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E5E9ED', linewidth=.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_box_aspect(1)

    model_handles = [
        Line2D([], [], color=MODEL_COLORS[model], marker=markers[model],
               markerfacecolor='white' if model == 'Parallel' else MODEL_COLORS[model],
               label='Upstream Transformer' if model == 'Parallel' else MAP[model])
        for model in models
    ]
    model_handles.append(Line2D([], [], color=UPSTREAM_FIT_COLOR, label='Upstream quadratic fit'))
    reuse_label = 'Reused A training data'
    if any(row['a_percent'] == 100 for row in data['reused']):
        reuse_label += ' (25% of A at 100/0)'
    protocol_handles = [
        Line2D([], [], color='#454545', marker='o', label='Disjoint B training data'),
        Line2D([], [], color='#454545', marker='o', markerfacecolor='white',
               linestyle='--', label=reuse_label),
    ]
    width_handles = [
        Line2D([], [], color=color, linestyle=dash, label=f'100/0 {width}')
        for width, (color, dash) in WIDTH_REFERENCE_STYLE.items()
    ]
    legend_heights = (.996, .944, .892) if single_row else (.996, .963, .930)
    for handles, y in zip((model_handles, protocol_handles, width_handles), legend_heights):
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, y), ncol=len(handles),
                   frameon=False, columnspacing=1.8, handlelength=2.6)
    fig.subplots_adjust(left=.055, right=.985, top=.820 if single_row else .874,
                        bottom=.15 if single_row else .08, wspace=.24, hspace=.35)
    return fig, {'dpi': 300}


if __name__ == '__main__':
    run_cli(FIGURE_ID, KIND, prepare_data, draw)
