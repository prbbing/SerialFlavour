"""1M and 5M allocation overlays for the six working points."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figure_runtime import run_cli
from figures import overlay_1m_5m
from pipeline_common import OVERLAY_15_SIX_WP, WORKING_POINTS

FIGURE_ID = OVERLAY_15_SIX_WP
KIND = 'overlay'
HEAVY_WORKING_POINTS = [('b_c_70', 'b/c 70%'), ('b_c_85', 'b/c 85%'), ('c_b_30', 'c/b 30%')]
ALL_WORKING_POINTS = WORKING_POINTS + HEAVY_WORKING_POINTS
MODEL_STYLE = overlay_1m_5m.MODEL_STYLE


def prepare_data():
    payload, sources = overlay_1m_5m.prepare_light_overlay()
    heavy_source = overlay_1m_5m.E1V2 / (
        'analysis/data/wp_rejection_summary_by_recipe_1m_3m_5m_heavy_background_with_upstream_fit_2026-11-02.json'
    )
    heavy, heavy_sources = overlay_1m_5m.prepare_overlay(
        heavy_source, heavy_source, payload['sizes'], HEAVY_WORKING_POINTS,
    )
    sources.extend(heavy_sources)
    payload['rows'].extend(heavy['rows'])
    payload['fits'].extend(heavy['fits'])
    overlay_1m_5m.add_graph(payload, sources, ALL_WORKING_POINTS)
    payload['models'] = list(MODEL_STYLE)
    return payload, sources


def draw(data):
    return overlay_1m_5m.draw(data, ALL_WORKING_POINTS)


if __name__ == '__main__':
    run_cli(FIGURE_ID, KIND, prepare_data, draw)
