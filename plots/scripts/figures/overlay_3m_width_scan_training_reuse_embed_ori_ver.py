"""Six working points for training-data reuse with DNN-embed and DNN-ori-pair."""
from functools import partial

from overlay_3m_width_scan_training_reuse import KIND, RECIPES, draw, prepare_data, run_cli

FIGURE_ID = 'wp_rejection_3m_width_scan_6wp_training_reuse_embed_ori_ver'
SELECTED_RECIPES = {model: RECIPES[model] for model in ('F1', 'F3')}

if __name__ == '__main__':
    run_cli(FIGURE_ID, KIND, partial(prepare_data, recipes=SELECTED_RECIPES),
            partial(draw, recipes=SELECTED_RECIPES))
