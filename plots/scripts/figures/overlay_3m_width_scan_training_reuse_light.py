"""Light-jet rejection with DNN-embed and DNN-ori-pair."""
from functools import partial

from overlay_3m_width_scan_training_reuse_embed_ori_ver import KIND, SELECTED_RECIPES, draw, prepare_data, run_cli
from pipeline_common import WORKING_POINTS

FIGURE_ID = 'wp_rejection_3m_width_scan_3wp_light_training_reuse_embed_ori_ver'

if __name__ == '__main__':
    run_cli(FIGURE_ID, KIND,
            partial(prepare_data, recipes=SELECTED_RECIPES, working_points=WORKING_POINTS,
                    include_100_0_reuse=False),
            partial(draw, recipes=SELECTED_RECIPES, working_points=WORKING_POINTS))
