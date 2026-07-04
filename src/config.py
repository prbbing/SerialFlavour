"""
Configuration module: defaults, validation, and loading.

The _DEFAULTS dict centralises every tunable parameter.
A JSON config file can override any subset of keys via --config.
The Config class unpacks the dict into typed attributes and computes
derived quantities (feature indices, projection matrices, etc.).
"""
import argparse
import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# All tunable parameters live here.  Keys present in a JSON config file
# override the corresponding entries; keys not present inherit the defaults.
# ---------------------------------------------------------------------------
_DEFAULTS = {
    # -- model dispatch -----------------------------------------------------
    "model_type": "staged_origin_vertex_jet",  # which architecture to build
    # -- device -------------------------------------------------------------
    "gpu_ids":    [-1],  # [-1]=auto; [0]=GPU0; [0,1,2]=DataParallel
    # -- data pre-processing ------------------------------------------------
    "top_k":      40,    # tracks per jet (ranked by |d0 significance|)
    "batch_size": 600,
    # -- transformer hyper-parameters ---------------------------------------
    "d_model":    32,    # embedding dimension (all sub-layers)
    "n_heads":    2,     # attention heads (must divide d_model)
    "n_layers":   2,     # encoder layers *per stage*
    "d_ffn":      64,    # feed-forward hidden dim
    "dropout":    0.1,   # dropout probability
    "gate_temp":  0.1,   # sigmoid temperature for vertex-leg origin gating
                         # smaller → sharper step at p=0.5
    # -- task sizes ---------------------------------------------------------
    "n_origin_classes": 8,
    "n_jet_classes":    3,   # b / c / light (tau excluded)

    # -- Stage-2 vertex legs ------------------------------------------------
    # Each vertex leg is built from tracks predicted to belong to a subset
    # of origin classes.  Default merges "From b" and "From b->c" into one
    # leg because the model struggles to separate them at Stage 1.
    "vertex_legs": {
        "b_vertex": ["From b", "From b->c"],
        "c_vertex": ["From c"],
    },
    # Truth mapping for each leg: which jet flavour owns this leg (5=b, 4=c)
    # and which slot in truth_hadrons provides the Lxy/dz target.
    "vertex_targets": {
        "b_vertex":  {"flavour": 5, "slot": 0},   # leading B-hadron
        "c_vertex":  {"flavour": 4, "slot": 0},   # leading C-hadron
    },
    # Coordinates to fit & supervise.  Non-empty subset of ["Lxy","dz"].
    "vertex_fit_coords": ["Lxy", "dz"],

    # Learnable multiplicative calibration per leg & active coordinate.
    #   pred_calibrated = pred * exp(log_scale)
    # Absorbs systematic biases from the closed-form fit via cheap params.
    "calibrate_vertex_fit": True,

    # Extra per-track signals from Stages 1/2 to feed into encoder 3.
    # Subset of ["origin_probs", "vtx_weight"]; [] = raw track features only.
    "stage3_extra_inputs": ["origin_probs", "vtx_weight"],

    # -- feature field sets ------------------------------------------------
    # Stage-2 vertexing input features (impact parameters + uncertainties).
    "vertex_fields": [
        "qOverP", "deta", "dphi", "d0", "z0SinTheta",
        "d0Uncertainty", "z0SinThetaUncertainty",
        "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
    ],
    # Full set of 21 track features read from HDF5.
    "track_fields": [
        "qOverP", "deta", "dphi", "d0", "z0SinTheta",
        "d0Uncertainty", "z0SinThetaUncertainty",
        "qOverPUncertainty", "thetaUncertainty", "phiUncertainty",
        "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
        "numberOfPixelHits", "numberOfSCTHits",
        "numberOfInnermostPixelLayerHits", "numberOfNextToInnermostPixelLayerHits",
        "numberOfInnermostPixelLayerSharedHits", "numberOfInnermostPixelLayerSplitHits",
        "numberOfPixelSharedHits", "numberOfPixelSplitHits", "numberOfSCTSharedHits",
    ],
    # Subset of track_fields fed into encoder 3 (jet classification).
    # Default = all track_fields; narrow to test whether hit-count features
    # matter after Stages 1/2 already consume them.
    "tagging_fields": [
        "qOverP", "deta", "dphi", "d0", "z0SinTheta",
        "d0Uncertainty", "z0SinThetaUncertainty",
        "qOverPUncertainty", "thetaUncertainty", "phiUncertainty",
        "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
        "numberOfPixelHits", "numberOfSCTHits",
        "numberOfInnermostPixelLayerHits", "numberOfNextToInnermostPixelLayerHits",
        "numberOfInnermostPixelLayerSharedHits", "numberOfInnermostPixelLayerSplitHits",
        "numberOfPixelSharedHits", "numberOfPixelSplitHits", "numberOfSCTSharedHits",
    ],

    # -- loss weights -------------------------------------------------------
    # total = lambda_jet * CE(jet) + lambda_origin * CE(origin) + lambda_vertex * vtx_loss
    "lambda_jet":    1.0,
    "lambda_origin": 1.0,
    "lambda_vertex": 1.0,

    # -- label mapping ------------------------------------------------------
    # HadronConeExclTruthLabelID values: 5=b, 4=c, 0=light, 15=tau (excluded)
    "flavour_to_label": {"5": 0, "4": 1, "0": 2},
    "jet_class_names":  ["b-jet", "c-jet", "light-jet"],
    "origin_class_names": [
        "Pileup", "Fake", "Primary", "From b",
        "From b->c", "From c", "From tau", "Other secondary",
    ],
    # Plotting colours for each jet class.
    "colours": {
        "b-jet": "#1f77b4", "c-jet": "#ff7f0e", "light-jet": "#2ca02c",
    },
    # Background weights in the b-tagging discriminant: log(p_b / Σ w_i p_i)
    "disc_bkg_weights": {"c-jet": 0.3, "light-jet": 0.7},

    # -- training -----------------------------------------------------------
    "train_file":      "/data/yuyang/opendata/gn2_tt/mc-flavtag-ttbar-small.h5",
    "n_train":         120_000,       # balanced across classes
    "n_test":          40_000,        # natural distribution (last N)
    "epochs":          40,
    "lr":              1e-3,          # Adam learning rate
    "num_workers":     0,             # DataLoader workers
    "model_name":      "model.pt",
    "train_plot_dir":  "./results/",
    "train_cache_dir": ".track_cache/",
}

# ===========================================================================
# Config — validated, typed container for every parameter derived from
# _DEFAULTS + JSON.  Constructed by load_config().
# ===========================================================================
class Config:
    def __init__(self, cfg_dict):
        self._raw = cfg_dict  # keep the raw dict for serialisation

        # -- dispatch & device ----------------------------------------------
        self.model_type         = cfg_dict["model_type"]
        self.gpu_ids            = cfg_dict["gpu_ids"]

        # -- data & training ------------------------------------------------
        self.train_file         = cfg_dict["train_file"]
        self.n_train            = cfg_dict["n_train"]
        self.n_test             = cfg_dict["n_test"]
        self.batch_size         = cfg_dict["batch_size"]
        self.epochs             = cfg_dict["epochs"]
        self.lr                 = cfg_dict["lr"]
        self.num_workers        = cfg_dict["num_workers"]
        self.top_k              = cfg_dict["top_k"]
        self.model_name         = cfg_dict["model_name"]
        self.plot_dir           = cfg_dict["train_plot_dir"]
        self.cache_dir          = cfg_dict["train_cache_dir"]

        # -- transformer ----------------------------------------------------
        self.d_model            = cfg_dict["d_model"]
        self.n_heads            = cfg_dict["n_heads"]
        self.n_layers           = cfg_dict["n_layers"]
        self.d_ffn              = cfg_dict["d_ffn"]
        self.dropout            = cfg_dict["dropout"]
        self.gate_temp          = cfg_dict["gate_temp"]

        # -- task sizes -----------------------------------------------------
        self.n_origin_classes   = cfg_dict["n_origin_classes"]
        self.n_jet_classes      = cfg_dict["n_jet_classes"]

        # -- vertex legs & coordinates -------------------------------------
        self.vertex_legs        = cfg_dict["vertex_legs"]
        self.vertex_leg_names   = list(cfg_dict["vertex_legs"].keys())
        self.n_vertex_legs      = len(self.vertex_leg_names)
        self.vertex_targets     = cfg_dict["vertex_targets"]
        self.vertex_fit_coords  = cfg_dict["vertex_fit_coords"]
        self.fit_lxy            = "Lxy" in cfg_dict["vertex_fit_coords"]
        self.fit_dz             = "dz"  in cfg_dict["vertex_fit_coords"]
        self.n_vtx_coords       = len(cfg_dict["vertex_fit_coords"])
        self.calibrate_vertex_fit = cfg_dict["calibrate_vertex_fit"]

        # -- stage-3 extra inputs -------------------------------------------
        self.stage3_extra_inputs  = cfg_dict["stage3_extra_inputs"]
        self.stage3_use_origin_probs = "origin_probs" in cfg_dict["stage3_extra_inputs"]
        self.stage3_use_vtx_weight   = "vtx_weight"   in cfg_dict["stage3_extra_inputs"]

        # -- feature fields & indices --------------------------------------
        self.vertex_fields      = cfg_dict["vertex_fields"]
        self.track_fields       = cfg_dict["track_fields"]
        self.tagging_fields     = cfg_dict["tagging_fields"]

        # -- loss weights ---------------------------------------------------
        self.lambda_jet         = cfg_dict["lambda_jet"]
        self.lambda_origin      = cfg_dict["lambda_origin"]
        self.lambda_vertex      = cfg_dict["lambda_vertex"]

        # -- label / display ------------------------------------------------
        self.flavour_to_label   = {int(k): v for k, v in cfg_dict["flavour_to_label"].items()}
        self.jet_class_names    = cfg_dict["jet_class_names"]
        self.origin_class_names = cfg_dict["origin_class_names"]
        self.colours            = cfg_dict["colours"]
        self.disc_bkg_weights   = cfg_dict["disc_bkg_weights"]

        # -- validation ----------------------------------------------------
        assert set(self.vertex_fit_coords) <= {"Lxy", "dz"} and len(self.vertex_fit_coords) >= 1, \
            "vertex_fit_coords must be a non-empty subset of ['Lxy', 'dz']"
        assert set(self.stage3_extra_inputs) <= {"origin_probs", "vtx_weight"}, \
            "stage3_extra_inputs must be a subset of ['origin_probs', 'vtx_weight']"
        assert set(self.tagging_fields) <= set(self.track_fields) and len(self.tagging_fields) >= 1, \
            "tagging_fields must be a non-empty subset of track_fields"

        # -- derived quantities ---------------------------------------------
        self.n_feats = len(self.track_fields)   # total input dimensionality

        # Projection matrix: origin class c → vertex leg l.
        # entry [c, l] = 1 if origin class c contributes to leg l.
        # Used in Stage 2 to convert 8-class soft probs into per-leg gate signals.
        _leg_matrix = np.zeros((self.n_origin_classes, self.n_vertex_legs), dtype=np.float32)
        for _li, _lname in enumerate(self.vertex_leg_names):
            _cls_list = self.vertex_legs[_lname]
            if isinstance(_cls_list, str):
                _cls_list = [_cls_list]
            for _cname in _cls_list:
                _leg_matrix[self.origin_class_names.index(_cname), _li] = 1.0
        self.vertex_leg_origin_matrix = _leg_matrix

        # Column indices of specific feature subsets within the full track feature array.
        self.vertex_feat_idx  = [self.track_fields.index(f) for f in self.vertex_fields]
        self.tagging_feat_idx = [self.track_fields.index(f) for f in self.tagging_fields]
        self.d0_idx          = self.track_fields.index("d0")
        self.d0_unc_idx      = self.track_fields.index("d0Uncertainty")
        self.dphi_idx        = self.track_fields.index("dphi")
        self.z0st_idx        = self.track_fields.index("z0SinTheta")
        self.z0st_unc_idx    = self.track_fields.index("z0SinThetaUncertainty")

        # Mapping from vertex leg name → owner jet class index, for plotting.
        _flavour_to_class_name = {v: k for k, v in
            {name: int(fl) for fl, name in
             [(lbl, self.jet_class_names[idx]) for lbl, idx in self.flavour_to_label.items()]}.items()}
        self.leg_owner_cls = {
            lname: self.jet_class_names.index(
                _flavour_to_class_name[self.vertex_targets[lname]["flavour"]]
            )
            for lname in self.vertex_leg_names
        }

# ===========================================================================
# load_config — entry point: merges _DEFAULTS + optional JSON file → Config.
# ===========================================================================
def load_config(config_path=None):
    """Return (Config, raw_dict) from built-in defaults merged with JSON file.

    Args:
        config_path: optional path to a JSON file whose keys override defaults.

    Raises:
        ValueError: if the JSON contains keys unknown to _DEFAULTS.
    """
    cfg_dict = dict(_DEFAULTS)
    if config_path is not None:
        with open(config_path) as f:
            file_cfg = json.load(f)
        unknown = set(file_cfg) - set(_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")
        cfg_dict.update(file_cfg)
    config = Config(cfg_dict)
    return config, cfg_dict
