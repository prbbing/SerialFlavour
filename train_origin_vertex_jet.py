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
from src.training import run_training
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
)

from sklearn.metrics import classification_report, confusion_matrix

parser = argparse.ArgumentParser(
    description="Train staged transformer jet classifier: track-origin "
                "prediction -> differentiable vertex fit -> jet classification, "
                "each stage with its own encoder."
)
parser.add_argument("--config", default=None,
                    help="Path to JSON config file. Keys override built-in defaults.")
args = parser.parse_args()

config, cfg_dict = load_config(args.config)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
config.plot_dir = f"{config.plot_dir.rstrip('/')}_{ts}/"

os.makedirs(config.plot_dir, exist_ok=True)
os.makedirs(config.cache_dir, exist_ok=True)

_orig_stdout = sys.stdout
log_file = open(os.path.join(config.plot_dir, "training_log.md"), "w")

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

sys.stdout = _Tee(_orig_stdout, log_file)

# ----------------------------------------------------------------
# Device selection
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
print(f"Device: {gpu_ids}  |  DataParallel: {use_dp}")
# ----------------------------------------------------------------

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

torch.save(model.state_dict(), os.path.join(config.plot_dir, config.model_name))
print(f"Saved {config.model_name}")

# -----------------------------------------------------------------
# Evaluate model on test set and produce plots
# -----------------------------------------------------------------

all_preds   = pred_arrays["all_preds"]
all_true    = pred_arrays["all_true"]
all_probs   = pred_arrays["all_probs"]
origin_preds = pred_arrays["origin_preds"]
origin_true  = pred_arrays["origin_true"]

print("\nJet classification report:")
print(classification_report(all_true, all_preds, target_names=config.jet_class_names))
print("Jet confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(all_true, all_preds))

print("\nTrack-origin classification report:")
print(classification_report(origin_true, origin_preds, target_names=config.origin_class_names,
                            labels=list(range(config.n_origin_classes)), zero_division=0))

plot_training_summary(history, all_true, all_preds, config.jet_class_names,
                      config.plot_dir, config.epochs, model_type=config.model_type)

plot_origin_confusion_matrix(origin_true, origin_preds,
                             config.origin_class_names, config.plot_dir)

# ----------------------------------------------------------------
# model: staged_origin_vertex_jet — Lxy/dz vertex-fit evaluation
# ----------------------------------------------------------------
if "lxy_pred" in pred_arrays:
    plot_vertex_fit(pred_arrays["lxy_pred"], pred_arrays["lxy_true"],
                    pred_arrays["dz_pred"], pred_arrays["dz_true"],
                    pred_arrays["vtx_valid"], all_true, config, config.plot_dir)
# ----------------------------------------------------------------
# model: parallel_origin_vertex_jet — pair-vertexing evaluation
# ----------------------------------------------------------------
if "pair_logits" in pred_arrays:
    plot_pair_vertexing(pred_arrays["pair_logits"], pred_arrays["pair_target"],
                        pred_arrays["pair_mask"], config.jet_class_names,
                        all_true, config.colours, config.plot_dir)
# ----------------------------------------------------------------

plot_output_probabilities(all_probs, all_true, config.jet_class_names,
                          config.colours, config.plot_dir)

plot_discriminant_roc(all_probs, all_true, config.jet_class_names,
                      config.disc_bkg_weights, config.colours, config.plot_dir)

plot_c_discriminant_roc(all_probs, all_true, config.jet_class_names,
                        config.c_disc_bkg_weights, config.colours, config.plot_dir)

if "vtx_weight" in pred_arrays:
    plot_track_vertex_assignment(
        pred_arrays["vtx_weight"], pred_arrays["origin_full"],
        pred_arrays["mask_full"], all_true,
        config.origin_class_names, config.vertex_leg_names,
        config.vertex_legs, config.n_vertex_legs,
        config.leg_owner_cls, config.jet_class_names,
        config.colours, config.plot_dir)

print(f"\nAll outputs saved to {config.plot_dir}")

sys.stdout = _orig_stdout
log_file.close()
