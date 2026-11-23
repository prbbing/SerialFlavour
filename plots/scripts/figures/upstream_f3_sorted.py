"""Independent F3/upstream comparison for the E3 auxiliary-loss weight scan."""
from pathlib import Path
import math
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_common import REPO, SEEDS, WORKING_POINTS, read_json
from pipeline_common import MODEL_COLORS, UPSTREAM_FIT_COLOR
from figure_runtime import run_cli

import numpy as np

FIGURE_ID = 'parallel_f3_absolute_rejection_sorted_by_upstream'
KIND = 'e3'

# ======================================================================
# 1. Data preparation: validate saved summaries; no plotting or evaluation.
# ======================================================================

def prepare_data():
    root = REPO/'local/results/parallel_refine_e3_v2'
    source = root/'analysis/data/f3_absolute_rejection_sorted_line.json'
    reference = root/'analysis/data/absolute_rejection_sorted_line.json'
    payload, original = read_json(source), read_json(reference)
    if payload['recipe'] != 'F3' or payload['sort_definition'] != original['sort_definition']:
        raise ValueError('F3 recipe or sort definition differs from the reference figure')
    sources = [source, reference]
    summaries = {}
    panels = []
    expected_weights = {(a, b) for a in (.2, .5, 1., 1.5) for b in (.2, .5, 1., 1.5)}
    fields = {'upstream_rejection': 'parallel_rejection',
              'upstream_sample_sd': 'parallel_rejection_sample_sd',
              'f3_rejection': 'dnn_rejection', 'f3_sample_sd': 'dnn_rejection_sample_sd'}
    for key, label in WORKING_POINTS:
        rows = sorted((r for r in payload['rows'] if r['working_point'] == key), key=lambda r: r['rank'])
        baseline = sorted((r for r in original['rows'] if r['working_point'] == key), key=lambda r: r['rank'])
        if len(rows) != 16 or [r['rank'] for r in rows] != list(range(1, 17)):
            raise ValueError(f'{key}: expected all 16 ranked configurations')
        if {(r['origin_weight'], r['pair_weight']) for r in rows} != expected_weights:
            raise ValueError(f'{key}: incomplete or duplicate loss-weight grid')
        ordered = sorted(rows, key=lambda r: (r['upstream_rejection'], r['origin_weight'], r['pair_weight']))
        if rows != ordered or len(baseline) != len(rows):
            raise ValueError(f'{key}: upstream sort order or coverage differs')
        signal, efficiency = ('c-jet', .3) if key == 'c_light_30' else ('b-jet', .7 if key == 'b_light_70' else .85)
        for row, paired in zip(rows, baseline):
            for field in ('rank', 'experiment', 'origin_weight', 'pair_weight', 'upstream_rejection', 'upstream_sample_sd'):
                if row[field] != paired[field]:
                    raise ValueError(f'{key}: {field} differs from the original upstream curve')
            experiment = row['experiment']
            if experiment not in summaries:
                path = root/experiment/'evaluation/F3_embed_aux/rejection_comparison_parallel_seed_mean.json'
                summaries[experiment] = read_json(path)
                sources.append(path)
            matches = [r for r in summaries[experiment]['curves']
                       if r['signal'] == signal and r['background'] == 'light-jet'
                       and math.isclose(r['target_signal_efficiency'], efficiency, abs_tol=1e-9)]
            if len(matches) != 1:
                raise ValueError(f'{experiment}: missing or duplicate working point {key}')
            for field, saved_field in fields.items():
                value = row[field]
                if not math.isfinite(value) or value < 0 or (not field.endswith('_sd') and value == 0):
                    raise ValueError(f'{experiment}: invalid {field}')
                if not math.isclose(value, matches[0][saved_field], rel_tol=1e-12):
                    raise ValueError(f'{experiment}: {field} disagrees with evaluation summary')
        maximum = max(r['upstream_rejection'] for r in rows)
        if maximum != original['upstream_max_reference'][key]:
            raise ValueError(f'{key}: upstream maximum reference differs')
        panels.append(dict(key=key, label=label, rows=rows, upstream_max_reference=maximum))
    return dict(panels=panels, recipe='F3', sort_definition=payload['sort_definition'],
                upstream_seeds=SEEDS, downstream_seed_count_per_upstream=5,
                uncertainty='Sample SD (ddof=1) across five upstream-seed group means; '
                            'each downstream group averages five downstream seeds.'), sources

# ======================================================================
# 2. Drawing: consume only prepared data; match the existing FG2 figure.
# ======================================================================

def draw(data):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 7.2))
    fig.subplots_adjust(left=.065, right=.98, bottom=.25, top=.77, wspace=.26)
    for index, (panel, ax) in enumerate(zip(data['panels'], axes)):
        rows = panel['rows']; x = np.arange(len(rows))
        up = [r['upstream_rejection'] for r in rows]; down = [r['f3_rejection'] for r in rows]
        ax.axhline(panel['upstream_max_reference'], color=UPSTREAM_FIT_COLOR, linestyle='-.', linewidth=1.3, label='Maximum upstream mean')
        ax.set_box_aspect(1)
        ax.errorbar(x, up, yerr=[r['upstream_sample_sd'] for r in rows], capsize=3, elinewidth=1, capthick=1,
                    color=MODEL_COLORS['Parallel'], marker='o', markersize=4.5, linewidth=1.8, label='Upstream Parallel')
        ax.errorbar(x, down, yerr=[r['f3_sample_sd'] for r in rows], capsize=3, elinewidth=1, capthick=1,
                    color=MODEL_COLORS['F3'], marker='s', markersize=4.5, linewidth=1.8, label='Downstream F3')
        ax.set_title(f'({chr(97+index)}) {panel["label"]}', fontsize=11)
        ax.set_xticks(x, [f'{r["origin_weight"]:.1f}/{r["pair_weight"]:.1f}' for r in rows], rotation=50, ha='right', fontsize=8)
        ax.set_xlabel('Origin / pair weights (upstream R ascending)', fontsize=9); ax.set_ylabel('Rejection', fontsize=10)
        ax.grid(axis='y', color='#dce2e8', linewidth=.7); ax.spines[['top', 'right']].set_visible(False)
        bounds = [r[field]+sign*r[sd] for r in rows for sign in [-1, 1]
                  for field, sd in [('upstream_rejection', 'upstream_sample_sd'), ('f3_rejection', 'f3_sample_sd')]]
        low, high = min(bounds), max(bounds); pad = (high-low)*.15
        ax.set_ylim(low-pad, high+pad)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .9), ncol=3, frameon=False, fontsize=10)
    return fig, {'dpi': 300}


if __name__ == '__main__':
    run_cli(FIGURE_ID, KIND, prepare_data, draw)
