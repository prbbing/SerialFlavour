"""
Staged transformer jet-flavour classifier: track-origin prediction ->
differentiable secondary-vertex fit -> jet-flavour classification, each
stage driven by its own dedicated transformer encoder.

Architecture
------------
Stage 1: track-origin prediction (encoder 1)
Stage 2: differentiable secondary-vertex fits (encoder 2)
Stage 3: jet-flavour classification (encoder 3)

See transformer_jet_classifier_origin_vertex_flavour.py for the full
architecture description and documentation.

Usage: python train_origin_vertex.py [--config config.json]
"""
import argparse
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

from src.config import load_config, _DEFAULTS
from src.data import create_dataloaders
from src.models import build_model
from src.losses import compute_origin_class_weights
from src.training import run_training, validate_epoch
from src.plotting import (
    plot_input_variables,
    plot_training_summary,
    plot_origin_confusion_matrix,
    plot_vertex_fit,
    plot_pair_vertexing,
    plot_output_probabilities,
    plot_discriminant_roc,
    plot_c_discriminant_roc,
    plot_track_vertex_assignment,
    plot_refine_vtx_weight_history,
    plot_gradient_diagnostics,
    plot_vertex_metrics_history,
    plot_vertex_loss_components,
)

from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser(
    description="Train staged transformer jet classifier: track-origin "
                "prediction -> differentiable vertex fit -> jet classification, "
                "each stage with its own encoder."
)
parser.add_argument("--config", default=None,
                    help="Path to JSON config file. Keys override built-in defaults.")
parser.add_argument("--eval-only", action="store_true",
                    help="Skip training; load saved weights and run evaluation only.")
parser.add_argument("--weights", "--weights-path", dest="weights_path", default=None,
                    help="Checkpoint .pt file for --eval-only. Defaults to "
                         "<train_plot_dir>/last.pt.")
parser.add_argument("--output-dir", default=None,
                    help="Override train_plot_dir in training mode, or the parent "
                         "directory for eval_<pt_name> in eval-only mode.")
args = parser.parse_args()

config, cfg_dict = load_config(args.config)

# ----------------------------------------------------------------
# Device selection (shared by train & eval)
#   gpu_ids: [-1]        → auto  (cpu fallback)
#   gpu_ids: [0]         → single GPU 0
#   gpu_ids: [0, 1, 2]   → DataParallel on GPUs 0,1,2
# ----------------------------------------------------------------
gpu_ids = config.gpu_ids
if torch.cuda.is_available() and gpu_ids != [-1]:
    DEVICE   = f"cuda:{gpu_ids[0]}"
    use_dp   = len(gpu_ids) > 1
else:
    DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
    use_dp   = False
    gpu_ids  = [0]
# ----------------------------------------------------------------

# ── _Tee utility (shared) ──────────────────────────────────────────
class _Tee:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def write(self, msg):
        self.a.write(msg)
        self.b.write(msg)
    def flush(self):
        self.a.flush()
        self.b.flush()

# ── _run_evaluation (shared by train and eval-only modes) ────────────
def _run_evaluation(pred_arrays, cfg, plot_dir, history=None):
    all_preds    = pred_arrays["all_preds"]
    all_true     = pred_arrays["all_true"]
    all_probs    = pred_arrays["all_probs"]
    origin_preds = pred_arrays["origin_preds"]
    origin_true  = pred_arrays["origin_true"]

    print("\nJet classification report:")
    print(classification_report(all_true, all_preds,
                                target_names=cfg.jet_class_names))
    print("Jet confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_true, all_preds))

    print("\nTrack-origin classification report:")
    print(classification_report(origin_true, origin_preds,
                                target_names=cfg.origin_class_names,
                                labels=list(range(cfg.n_origin_classes)),
                                zero_division=0))

    if history is not None:
        plot_training_summary(history, all_true, all_preds,
                              cfg.jet_class_names, plot_dir, cfg.epochs,
                              model_type=cfg.model_type)
        plot_refine_vtx_weight_history(history, plot_dir)
        plot_gradient_diagnostics(history, plot_dir)
        plot_vertex_metrics_history(history, plot_dir)
        plot_vertex_loss_components(history, plot_dir)

    plot_output_probabilities(all_probs, all_true, cfg.jet_class_names,
                              cfg.colours, plot_dir)

    plot_origin_confusion_matrix(origin_true, origin_preds,
                                 cfg.origin_class_names, plot_dir)

    plot_discriminant_roc(all_probs, all_true, cfg.jet_class_names,
                          cfg.disc_bkg_weights, cfg.colours, plot_dir)

    plot_c_discriminant_roc(all_probs, all_true, cfg.jet_class_names,
                            cfg.c_disc_bkg_weights, cfg.colours, plot_dir)

    if "lxy_pred" in pred_arrays:
        plot_vertex_fit(pred_arrays["lxy_pred"], pred_arrays["lxy_true"],
                        pred_arrays["dz_pred"], pred_arrays["dz_true"],
                        pred_arrays["vtx_valid"], all_true, cfg, plot_dir)
    if "vtx_weight" in pred_arrays:
        plot_track_vertex_assignment(
            pred_arrays["vtx_weight"], pred_arrays["origin_full"],
            pred_arrays["mask_full"], all_true,
            cfg.origin_class_names, cfg.vertex_leg_names,
            cfg.vertex_legs, cfg.n_vertex_legs,
            cfg.leg_owner_cls, cfg.jet_class_names,
            cfg.colours, plot_dir,
            leg_origin_probs=pred_arrays.get("leg_origin_probs"),
            gate=pred_arrays.get("gate"),
            refine=pred_arrays.get("refine"))
    if "pair_logits" in pred_arrays:
        plot_pair_vertexing(pred_arrays["pair_logits"],
                            pred_arrays["pair_target"],
                            pred_arrays["pair_mask"],
                            cfg.jet_class_names, all_true,
                            cfg.colours, plot_dir)

    print(f"\nAll outputs saved to {plot_dir}")

# ══════════════════════════════════════════════════════════════════════
# Eval-only mode — load weights, validate, run all diagnostic plots.
# ══════════════════════════════════════════════════════════════════════
if args.eval_only:
    os.makedirs(config.cache_dir, exist_ok=True)

    if args.weights_path is not None:
        weights_path = os.path.abspath(os.path.expanduser(args.weights_path))
    else:
        weights_path = os.path.abspath(
            os.path.join(config.plot_dir, "last.pt"))

    pt_name = os.path.splitext(os.path.basename(weights_path))[0]
    eval_parent_dir = (os.path.abspath(os.path.expanduser(args.output_dir))
                       if args.output_dir is not None
                       else os.path.dirname(weights_path))
    eval_plot_dir = os.path.join(eval_parent_dir, f"eval_{pt_name}/")
    os.makedirs(eval_plot_dir, exist_ok=True)

    _orig_stdout = sys.stdout
    log_file = open(os.path.join(eval_plot_dir, "eval_log.md"), "w")
    sys.stdout = _Tee(_orig_stdout, log_file)

    print(f"Device: {gpu_ids}  |  DataParallel: {use_dp}")
    print(f"Eval-only — weights: {weights_path}")

    print("Loading data...")
    _, val_loader, _, test_data, _, y_test = create_dataloaders(config, DEVICE)

    model = build_model(config).to(DEVICE)
    if use_dp:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    model.load_state_dict(
        torch.load(weights_path, map_location=DEVICE, weights_only=True))
    print(f"Loaded weights from {weights_path}")

    if config.model_type == "staged_origin_vertex_jet":
        print(f"  Vertex fit coordinates: {config.vertex_fit_coords}")
        print(f"  Stage-3 extra inputs: {config.stage3_extra_inputs}")
        print(f"  Stage-3 tagging fields ({len(config.tagging_fields)}): "
              f"{config.tagging_fields}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"Device: {DEVICE}  |  Test: {len(y_test):,}\n")

    criterion_jet = nn.CrossEntropyLoss()
    criterion_origin = nn.CrossEntropyLoss(ignore_index=-1)
    *_, pred_arrays = validate_epoch(
        model, val_loader, criterion_jet, criterion_origin,
        config.n_origin_classes, config.lambda_jet,
        config.lambda_origin, config.lambda_vertex,
        config.fit_lxy, config.fit_dz, DEVICE)

    _run_evaluation(pred_arrays, config, eval_plot_dir)

    sys.stdout = _orig_stdout
    log_file.close()
    sys.exit(0)

# ══════════════════════════════════════════════════════════════════════
# Training mode
# ══════════════════════════════════════════════════════════════════════

if args.output_dir is not None:
    config.plot_dir = args.output_dir
    
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
config.plot_dir = f"{config.plot_dir.rstrip('/')}_{ts}/"

os.makedirs(config.plot_dir, exist_ok=True)
os.makedirs(config.cache_dir, exist_ok=True)

_orig_stdout = sys.stdout
log_file = open(os.path.join(config.plot_dir, "training_log.md"), "w")
sys.stdout = _Tee(_orig_stdout, log_file)

print(f"Device: {gpu_ids}  |  DataParallel: {use_dp}")

cfg_dict["train_plot_dir"] = config.plot_dir
_cfg_save_path = os.path.join(config.plot_dir, "config.json")
with open(_cfg_save_path, "w") as f:
    json.dump(cfg_dict, f, indent=4)
print(f"Config saved to {_cfg_save_path}")

print("Loading data...")
train_loader, val_loader, train_data, test_data, y_train, y_test = create_dataloaders(
    config, DEVICE)

plot_input_variables(train_data["X"], train_data["mask"], train_data["y"],
                     config.track_fields, config.top_k,
                     config.jet_class_names, config.colours, config.plot_dir)

model = build_model(config).to(DEVICE)

# DataParallel wrapping for multi-GPU
if use_dp:
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    print(f"DataParallel enabled on GPUs: {gpu_ids}")

print(f"Model type: {config.model_type}")
if config.model_type == "staged_origin_vertex_jet":
    print(f"  Vertex fit coordinates: {config.vertex_fit_coords}")
    print(f"  Stage-3 extra inputs: {config.stage3_extra_inputs}")
    print(f"  Stage-3 tagging fields ({len(config.tagging_fields)}): {config.tagging_fields}")
    print(f"  Calibrate vertex fit (learnable per-leg scale): {config.calibrate_vertex_fit}")

# ----------------------------------------------------------------
# Training setup
# ----------------------------------------------------------------

optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
criterion_jet = nn.CrossEntropyLoss()

_origin_weights = compute_origin_class_weights(
    train_data, config.n_origin_classes, DEVICE)
print("Origin class weights:")
for _i, (_n, _w) in enumerate(zip(config.origin_class_names, _origin_weights.cpu())):
    print(f"  {_i:2d}  {_n:<18s}  {_w:.3f}")

criterion_origin = nn.CrossEntropyLoss(ignore_index=-1, weight=_origin_weights)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params:,}")
print(f"Device: {DEVICE}  |  Train: {len(y_train):,}  |  Test: {len(y_test):,}\n")

history, pred_arrays = run_training(
    model, train_loader, val_loader, optimiser,
    criterion_jet, criterion_origin,
    config, DEVICE, config.epochs, config.n_origin_classes, len(y_train),
    config.vertex_leg_names, config.calibrate_vertex_fit)

# ── run evaluation (training mode) ────────────────────────────────────
_run_evaluation(pred_arrays, config, config.plot_dir, history=history)

sys.stdout = _orig_stdout
log_file.close()
