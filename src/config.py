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
    "use_pair_target": False,  # load dense (K, K) pair supervision
    # -- transformer hyper-parameters ---------------------------------------
    "d_model":    32,    # embedding dimension (all sub-layers)
    "n_heads":    2,     # attention heads (must divide d_model)
    "n_layers":   2,     # encoder layers *per stage*
    "d_ffn":      64,    # feed-forward hidden dim
    "dropout":    0.1,   # dropout probability
    "gate_temp":  0.1,   # sigmoid temperature for vertex-leg origin gating
                         # smaller → sharper step at p=0.5
    "delta_w_amp": 0.5,  # bounded residual amplitude for fix-refine Stage-2 weights
    # -- residual-refine Stage 2 -------------------------------------------
    # Per-track MLP inputs. "geometry" contributes 4 values per vertex leg:
    # [r_d0, r_z, r_d0^2+r_z^2, initial_gate]. Optional additions are the
    # full origin-probability vector and raw q/p.
    "residual_refine_inputs": ["geometry", "origin_probs"],
    "residual_refine_hidden_dims": [32, 16],
    "residual_refine_alpha": 1.0,
    "residual_vertex_detach": True,
    # -- track-origin ablation ---------------------------------------------
    # geometry/kinematics -> assignment Transformer -> b/c/other
    "track_assignment_fields": [
        "d0", "z0SinTheta", "d0Uncertainty", "z0SinThetaUncertainty",
        "theta", "dphi", "qOverP",
    ],
    "track_assignment_n_layers": 2,
    # -- unified jet-only architecture ------------------------------------
    # "mlp" matches Parallel.init_net exactly; "linear" matches the
    # projection used by a Staged Stage-3 track path.
    "jet_only_track_init": "mlp",
    # "attention" matches Parallel pooling; "cls" matches Staged Stage 3.
    "jet_only_pooling": "attention",
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

    # Vertex fitting algorithm for fix_dz models:
    #   "two_step" — keep existing Lxy/flight_phi fit, replace dz with
    #                two-step WLS using fitted Lxy geometry
    #   "wls_3d"   — joint 3D WLS solving (X,Y,Z) from all tracks
    "vertex_fit_method": "wls_3d",
    "vertex_fit_reg": 1e-6,    # Tikhonov regularisation for 3D WLS

    # Learnable multiplicative calibration per leg & active coordinate.
    #   pred_calibrated = pred * exp(log_scale)
    # Absorbs systematic biases from the closed-form fit via cheap params.
    "calibrate_vertex_fit": True,

    # Extra per-track signals from Stages 1/2 to feed into encoder 3.
    # Subset of ["origin_probs", "vtx_weight"]; [] = raw track features only.
    "stage3_extra_inputs": ["origin_probs", "vtx_weight"],
    # Whether fitted per-leg vertex coordinates are embedded as Stage-3 tokens.
    # True preserves the original [CLS] + vertex tokens + track tokens sequence.
    "stage3_use_vertex_tokens": True,
    # Stop jet-loss gradients at both Stage-3 vertex inputs (per-track weights
    # and fitted vertex tokens), while keeping the vertex-loss graph intact.
    "detach_vertex_from_jet": False,

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
        "qOverPUncertainty", "theta", "thetaUncertainty", "phiUncertainty",
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
        "qOverPUncertainty", "theta", "thetaUncertainty", "phiUncertainty",
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
    "lambda_pair":   1.0,

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
    # Background weights in the c-tagging discriminant: log(p_c / Σ w_i p_i)
    "c_disc_bkg_weights": {"b-jet": 0.5, "light-jet": 0.5},

    # -- training -----------------------------------------------------------
    "train_file":      "/data/yuyang/opendata/gn2_tt/mc-flavtag-ttbar-small.h5",
    "n_train":         120_000,       # balanced across classes
    "n_test":          40_000,        # natural distribution (last N)
    "epochs":          100,
    "checkpoint_interval": 20,  # save epoch_N.pt every N epochs
    "lr":              1e-3,          # Adam learning rate
    "num_workers":     4,             # DataLoader workers
    "model_name":      "model.pt",
    "train_plot_dir":  "./results/",
    "train_cache_dir": ".track_cache/",
    "tensorboard_log_dir": "results/tb_runs",  # null=disabled
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
        self.use_pair_target    = cfg_dict["use_pair_target"]
        self.epochs             = cfg_dict["epochs"]
        self.checkpoint_interval = cfg_dict["checkpoint_interval"]
        self.lr                 = cfg_dict["lr"]
        self.num_workers        = cfg_dict["num_workers"]
        self.top_k              = cfg_dict["top_k"]
        self.model_name         = cfg_dict["model_name"]
        self.plot_dir           = cfg_dict["train_plot_dir"]
        self.cache_dir          = cfg_dict["train_cache_dir"]
        self.tensorboard_log_dir = cfg_dict["tensorboard_log_dir"]

        # -- transformer ----------------------------------------------------
        self.d_model            = cfg_dict["d_model"]
        self.n_heads            = cfg_dict["n_heads"]
        self.n_layers           = cfg_dict["n_layers"]
        self.d_ffn              = cfg_dict["d_ffn"]
        self.dropout            = cfg_dict["dropout"]
        self.gate_temp          = cfg_dict["gate_temp"]
        self.delta_w_amp        = cfg_dict["delta_w_amp"]
        self.residual_refine_inputs = cfg_dict["residual_refine_inputs"]
        self.residual_refine_hidden_dims = cfg_dict["residual_refine_hidden_dims"]
        self.residual_refine_alpha = cfg_dict["residual_refine_alpha"]
        self.residual_vertex_detach = cfg_dict["residual_vertex_detach"]
        self.track_assignment_fields = cfg_dict["track_assignment_fields"]
        self.track_assignment_n_layers = cfg_dict["track_assignment_n_layers"]
        self.jet_only_track_init = cfg_dict["jet_only_track_init"]
        self.jet_only_pooling = cfg_dict["jet_only_pooling"]

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
        self.vertex_fit_method  = cfg_dict["vertex_fit_method"]   # "two_step" or "wls_3d"
        self.vertex_fit_reg     = cfg_dict["vertex_fit_reg"]       # λ for 3D WLS

        # -- stage-3 extra inputs -------------------------------------------
        self.stage3_extra_inputs  = cfg_dict["stage3_extra_inputs"]
        self.stage3_use_origin_probs = "origin_probs" in cfg_dict["stage3_extra_inputs"]
        self.stage3_use_vtx_weight   = "vtx_weight"   in cfg_dict["stage3_extra_inputs"]
        self.stage3_use_vertex_tokens = cfg_dict["stage3_use_vertex_tokens"]
        self.detach_vertex_from_jet = cfg_dict["detach_vertex_from_jet"]

        # -- feature fields & indices --------------------------------------
        self.vertex_fields      = cfg_dict["vertex_fields"]
        self.track_fields       = cfg_dict["track_fields"]
        self.tagging_fields     = cfg_dict["tagging_fields"]

        # -- loss weights ---------------------------------------------------
        self.lambda_jet         = cfg_dict["lambda_jet"]
        self.lambda_origin      = cfg_dict["lambda_origin"]
        self.lambda_vertex      = cfg_dict["lambda_vertex"]
        self.lambda_pair        = cfg_dict["lambda_pair"]

        # -- label / display ------------------------------------------------
        self.flavour_to_label   = {int(k): v for k, v in cfg_dict["flavour_to_label"].items()}
        self.jet_class_names    = cfg_dict["jet_class_names"]
        self.origin_class_names = cfg_dict["origin_class_names"]
        self.colours            = cfg_dict["colours"]
        self.disc_bkg_weights   = cfg_dict["disc_bkg_weights"]
        self.c_disc_bkg_weights = cfg_dict["c_disc_bkg_weights"]

        # -- validation ----------------------------------------------------
        assert set(self.vertex_fit_coords) <= {"Lxy", "dz"} and len(self.vertex_fit_coords) >= 1, \
            "vertex_fit_coords must be a non-empty subset of ['Lxy', 'dz']"
        assert set(self.stage3_extra_inputs) <= {"origin_probs", "vtx_weight"}, \
            "stage3_extra_inputs must be a subset of ['origin_probs', 'vtx_weight']"
        assert type(self.stage3_use_vertex_tokens) is bool, \
            "stage3_use_vertex_tokens must be a boolean"
        assert type(self.detach_vertex_from_jet) is bool, \
            "detach_vertex_from_jet must be a boolean"
        assert set(self.residual_refine_inputs) <= {"geometry", "origin_probs", "qoverp"}, \
            "residual_refine_inputs must be a subset of ['geometry', 'origin_probs', 'qoverp']"
        assert "geometry" in self.residual_refine_inputs, \
            "residual_refine_inputs must include 'geometry'"
        assert (len(self.residual_refine_hidden_dims) == 2
                and all(type(v) is int and v > 0 for v in self.residual_refine_hidden_dims)), \
            "residual_refine_hidden_dims must contain two positive integers"
        assert self.residual_refine_alpha > 0, \
            "residual_refine_alpha must be positive"
        assert type(self.residual_vertex_detach) is bool, \
            "residual_vertex_detach must be a boolean"
        if self.model_type == "staged_origin_vertex_jet_track_ablation":
            assert (len(self.track_assignment_fields) >= 1
                    and set(self.track_assignment_fields) <= set(self.track_fields)), \
                "track_assignment_fields must be a non-empty subset of track_fields"
            assert type(self.track_assignment_n_layers) is int and self.track_assignment_n_layers > 0, \
                "track_assignment_n_layers must be a positive integer"
        if self.model_type == "jet_only_transformer":
            assert self.jet_only_track_init in {"mlp", "linear"}, \
                "jet_only_track_init must be 'mlp' or 'linear'"
            assert self.jet_only_pooling in {"attention", "cls"}, \
                "jet_only_pooling must be 'attention' or 'cls'"
        assert set(self.tagging_fields) <= set(self.track_fields) and len(self.tagging_fields) >= 1, \
            "tagging_fields must be a non-empty subset of track_fields"
        assert type(self.checkpoint_interval) is int and self.checkpoint_interval > 0, \
            "checkpoint_interval must be a positive integer"
        assert type(self.use_pair_target) is bool, \
            "use_pair_target must be a boolean"

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
        self.theta_idx       = self.track_fields.index("theta") if "theta" in self.track_fields else -1
        self.qoverp_idx      = self.track_fields.index("qOverP") if "qOverP" in self.track_fields else -1
        self.track_assignment_feat_idx = (
            [self.track_fields.index(field) for field in self.track_assignment_fields]
            if set(self.track_assignment_fields) <= set(self.track_fields) else [])

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
