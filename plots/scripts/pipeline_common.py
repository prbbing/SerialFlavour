"""Paths, serialization and contracts shared by the two independent stages."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA = ROOT / 'data'
PREPARED = DATA / 'prepared'
FIGURE = ROOT / 'figure'
SEEDS = [1, 2, 3, 4, 5]
DEFAULT_GRAPH_REFINER = 'FG1_48d'
# One series palette for every current figure. Baselines remain neutral.
MODEL_COLORS = {
    'Transformer': '#777777',
    'Parallel': '#777777',
    'Parallel (100/0)': '#777777',
    'F1': '#3B78A5',
    'F2_origin': '#B18B3C',
    'F2_vertex': '#8172AA',
    'F3': '#4A9885',
    'FG1_48d': '#C5694B',
    'FG2': '#B66D91',
    'FG2_32d': '#718552',
}
UPSTREAM_FIT_COLOR = '#454545'
WIDTH_REFERENCE_STYLE = {
    # Smaller models use wider, sparser dashes; larger models shorter, denser dots.
    '122k': ('#A6A6A6', (0, (8, 5))),
    '142k': ('#858585', (0, (5, 4))),
    '164k': ('#636363', (0, (2.5, 3))),
    '188k': ('#404040', (0, (1, 2))),
}
MAP = {
    'F1': 'DNN-embed',
    'F2_origin': 'DNN-origin',
    'F2_vertex': 'DNN-pair',
    'F3': 'DNN-ori-pair',
    'FG2': 'DNN-graph-2layer',
    'FG1_48d': 'DNN-graph',
    'FG2_32d': 'DNN-graph-2layer_small',
}
# Convert internal model IDs directly to the seven current display names.
MODEL_NAME_PATTERN = re.compile(
    r'(?<![A-Za-z0-9_-])('
    + '|'.join(re.escape(name) for name in sorted(MAP, key=len, reverse=True))
    + r')(?![A-Za-z0-9_-])'
)


def display_name(text):
    return MODEL_NAME_PATTERN.sub(lambda match: MAP[match[0]], text)


OVERLAY_123 = 'wp_rejection_summary_by_recipe_1m_3m_5m_with_upstream_fit_overlay_2026-11-02'
OVERLAY_13 = 'wp_rejection_summary_by_recipe_1m_3m_with_upstream_fit_overlay_2026-10-28'
OVERLAY_15_SIX_WP = 'wp_rejection_summary_by_recipe_1m_5m_with_upstream_fit_overlay_six_working_points_2026-11-11'
OVERLAY_3M_WIDTHS = 'wp_rejection_summary_by_recipe_3m_with_upstream_fit_overlay_width_scan_2026-11-04'
STRUCTURE = 'discriminant_change_vs_n_sv_ge2_a075'
HEATMAP = 'four_experiments_absolute_rejection_heatmap_light_only_clean'
E3 = 'parallel_fg2_absolute_rejection_sorted_by_upstream'
# Stable scientific IDs map to concise, content-based image filenames.
FIGURE_FILENAMES = {
    'discriminant_change_vs_n_sv_ge2_a075': 'discriminant_vs_vertices_3m_75_25.png',
    'four_experiments_absolute_rejection_heatmap_light_only_clean': 'rejection_heatmap_3m_122k_variants.png',
    'p122_a075_b025_n3m_rejc': 'rejection_3m_75_25_embed_ori_ver_graph.png',
    'p122_a075_b025_n3m_rejc_100_baseline': 'rejection_3m_75_25_embed_ori_ver_graph_vs_100_0.png',
    'p122_a075_b025_n3m_rejc_F1_F2origin_F2vertex_F3_2026-11-04': 'rejection_3m_75_25_four_refiners.png',
    'p122_a075_b025_n3m_rejc_100_baseline_F1_F2origin_F2vertex_F3_2026-11-04': 'rejection_3m_75_25_four_refiners_vs_100_0.png',
    'parallel_f3_absolute_rejection_sorted_by_upstream': 'rejection_aux_weights_ori_ver.png',
    'parallel_fg2_absolute_rejection_sorted_by_upstream': 'rejection_aux_weights_graph_2layer.png',
    'pipeline': 'pipeline.png',
    'wp_rejection_summary_by_recipe_1m_3m_5m_with_upstream_fit_overlay_2026-11-02': 'wp_rejection_1m_3m_5m.png',
    'wp_rejection_summary_by_recipe_1m_3m_with_upstream_fit_overlay_2026-10-28': 'wp_rejection_1m_3m.png',
    'wp_rejection_summary_by_recipe_1m_5m_with_upstream_fit_overlay_2026-11-07': 'wp_rejection_1m_5m.png',
    OVERLAY_15_SIX_WP: 'wp_rejection_1m_5m_6wp.png',
    'wp_rejection_summary_by_recipe_3m_with_upstream_fit_overlay_width_scan_2026-11-04': 'wp_rejection_3m_width_scan.png',
    'wp_rejection_summary_by_recipe_3m_with_upstream_fit_overlay_width_scan_all_refiners_2026-11-04': 'wp_rejection_3m_width_scan_graph.png',
    'wp_rejection_summary_by_recipe_3m_with_upstream_fit_overlay_width_scan_six_working_points_2026-11-07': 'wp_rejection_3m_width_scan_6wp.png',
}
FIGURE_IDS_BY_FILENAME = {filename: identity for identity, filename in FIGURE_FILENAMES.items()}


def figure_filename(figure_id):
    return FIGURE_FILENAMES.get(figure_id, figure_id+'.png')


def figure_identity(filename):
    return FIGURE_IDS_BY_FILENAME.get(filename, Path(filename).stem)


WORKING_POINTS = [('b_light_70', 'b/light 70%'), ('b_light_85', 'b/light 85%'),
                  ('c_light_30', 'c/light 30%')]
SPECS = [('b tagging: c rejection', 'b/c', 'b-jet', 'c-jet', .60, 1.),
         ('b tagging: light rejection', 'b/light', 'b-jet', 'light-jet', .60, 1.),
         ('c tagging: b rejection', 'c/b', 'c-jet', 'b-jet', .10, .40),
         ('c tagging: light rejection', 'c/light', 'c-jet', 'light-jet', .10, .40)]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def source_record(path):
    path = Path(path).resolve()
    relative = path.relative_to(REPO).as_posix()
    validate_result_source(relative)
    return {'path': relative, 'sha256': sha(path)}


def validate_result_source(relative):
    """Reject non-v2 result provenance without opening original data in plot mode."""
    path = Path(relative.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Expected repository-relative source: {relative}')
    parts = path.parts
    if parts[:2] == ('local', 'results'):
        if len(parts) < 3 or not parts[2].endswith('_v2'):
            raise ValueError(f'Only v2 result sources are allowed: {relative}')


def checked_child(directory, relative):
    path = (Path(directory) / relative).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError(f'Path escapes prepared data directory: {relative}')
    return path
