"""
Staged transformer jet-flavour classifier: track-origin prediction →
differentiable secondary-vertex fit → jet-flavour classification, each
stage driven by its own dedicated transformer encoder.

Architecture
------------
Stage 1 — track-origin prediction (encoder 1)
    All track features are projected and passed through a dedicated
    transformer encoder (no CLS token — this is a per-track task). A shared
    origin head produces a soft distribution over 8 track-origin classes
    for every track. Truth labels come from `ftagTruthOriginLabel`.

Stage 2 — three separate differentiable secondary-vertex fits (encoder 2)
    Rather than a single inclusive secondary vertex, three *origin-gated*
    vertex fits are performed in parallel, each using only the tracks that
    Stage 1 predicts (softly) to originate from the corresponding decay leg:

        • b-vertex      — built from tracks predicted "From b"     (origin 3)
        • cascade vertex — built from tracks predicted "From b->c" (origin 4)
        • c-vertex      — built from tracks predicted "From c"     (origin 5)

    The soft origin probability for the relevant class, p_origin(track),
    acts as the (differentiable) track-selection weight for each fit. The
    predicted soft origin distribution is also turned into a differentiable
    per-track embedding (weighted sum over learnable class vectors) and
    combined with the vertexing-relevant track features (impact parameters,
    their uncertainties, lifetime-signed significances); a second dedicated
    encoder refines these representations and produces, per track, three
    learned refinement weights (one per vertex leg) in [0, 1]. For each leg k:

        wᵢ = p_origin_k(trackᵢ) · refine_k(trackᵢ) / σ_d0,ᵢ²
        L̂xy_k = (Σᵢ wᵢ |d0ᵢ|) / (Σᵢ wᵢ)        (closed-form, fully differentiable)

    giving the transverse flight distance Lxy for each of the three vertex
    legs. (Only Lxy — not the longitudinal z position — is fitted: Lxy is
    the more physically discriminating quantity for b/c-tagging, and trying
    to additionally reconstruct z from a single-track-parameter weighted
    average was found to be an overly ambitious target for this simple
    closed-form fitter.) Truth targets are taken from the leading
    `truth_hadrons` slots: for b-jets, slot 0 is the B-hadron decay vertex
    (→ b-vertex target) and slot 1, when present, is the cascade c-hadron
    from the B decay (→ cascade-vertex target); for c-jets, slot 0 is the
    C-hadron decay vertex (→ c-vertex target). Legs with no matching truth
    hadron are masked out of the vertex loss.

Stage 3 — jet-flavour classification (encoder 3)
    The three fitted vertex legs' Lxy values are each embedded (with a
    shared projection) into their own "vertex token". These three tokens,
    together with a CLS token, are prepended to the full set of projected
    track features. A third dedicated encoder produces the final jet
    representation; the CLS token feeds a 3-class jet-flavour head
    (b / c / light, from HadronConeExclTruthLabelID — tau jets are
    excluded for now).

The three stages are connected only through differentiable intermediate
quantities (soft origin probabilities, fitted Lxy values), so the whole
network is trained end-to-end with a combined loss:

    λ_jet · CE(jet) + λ_origin · CE(origin) + λ_vertex · Σ_legs Lxy_loss

Data: mc-flavtag-ttbar-small.h5

Returns (from the model):
    jet_logits, origin_logits, vtx_weight, lxy_pred
    (lxy_pred / vtx_weight have a leading dimension of 3 — one entry per
    vertex leg, in the order [b, cascade(b->c), c])
"""
import argparse
import hashlib
import json
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# ── defaults ───────────────────────────────────────────────────────────
_DEFAULTS = {
    "top_k":      40,
    "batch_size": 600,
    "d_model":    32,
    "n_heads":    2,
    "n_layers":   2,
    "d_ffn":      64,
    "dropout":    0.1,
    "gate_temp":  0.1,   # sigmoid gate temperature for vertex-leg origin gating
                         # (smaller → sharper step at p=0.5; 0.1 ≈ hard threshold)

    "n_origin_classes": 8,
    "n_jet_classes":    3,   # b / c / light — tau jets are excluded for now

    # Stage-2 vertex legs: name -> list of origin-class names whose soft
    # probabilities are summed to form the sigmoid gate weight for that leg.
    # "From b" and "From b->c" are merged here because the model struggles to
    # separate them; split them into separate legs if origin classification
    # improves.  To restore the three-leg fit, change to:
    #   "b_vertex":  ["From b"],
    #   "bc_vertex": ["From b->c"],
    #   "c_vertex":  ["From c"],
    # and add a matching entry in vertex_targets below.
    "vertex_legs": {
        "b_vertex": ["From b", "From b->c"],
        "c_vertex": ["From c"],
    },

    # Truth Lxy target for each vertex leg:
    #   flavour — HadronConeExclTruthLabelID value of the jet type that owns
    #             this vertex (5=b, 4=c)
    #   slot    — index into truth_hadrons (0=leading hadron, 1=first cascade)
    # For the three-leg case add:
    #   "bc_vertex": {"flavour": 5, "slot": 1},
    "vertex_targets": {
        "b_vertex":  {"flavour": 5, "slot": 0},
        "c_vertex":  {"flavour": 4, "slot": 0},
    },

    # Which vertex coordinates to fit and supervise.
    # Any non-empty subset of ["Lxy", "dz"].
    # "Lxy"  — transverse flight distance (from d0 + dphi)
    # "dz"   — longitudinal displacement  (from z0SinTheta)
    # Both together give the full 3-D picture and help separate secondaries
    # from primaries even when one coordinate is small.
    "vertex_fit_coords": ["Lxy", "dz"],

    # Whether to give each vertex leg a learnable per-coordinate multiplicative
    # calibration scale (one nn.Parameter per leg per active fit coordinate),
    # applied as  pred_calibrated = pred * exp(log_scale)  to lxy_pred / dz_pred
    # before both the vertex-fit loss and the Stage-3 vtx_summary token. This
    # lets the network absorb known systematic biases in the closed-form
    # Stage-2 fit (e.g. from how vertex_legs groups origin classes / truth
    # vertices) via a single cheap parameter per leg, rather than distorting
    # Stage 1/2 origin classification to compensate.
    "calibrate_vertex_fit": True,

    # Extra per-track signals from Stages 1/2 to concatenate onto the raw
    # track features feeding encoder 3 (jet classification), so the final
    # stage can directly see what Stage 1/2 inferred about each track.
    # Any subset of ["origin_probs", "vtx_weight"]; [] = original behaviour
    # (encoder3 sees only raw track features + vertex-fit summary tokens).
    #   "origin_probs" — Stage-1 soft origin-class probabilities (per track)
    #   "vtx_weight"   — Stage-2 per-leg vertex-fit gating weight (per track)
    "stage3_extra_inputs": ["origin_probs", "vtx_weight"],

    # vertexing-relevant track features fed (alongside the predicted soft
    # origin embedding) into encoder 2 for the differentiable vertex fit
    "vertex_fields": [
        "qOverP", "deta", "dphi", "d0", "z0SinTheta",
        "d0Uncertainty", "z0SinThetaUncertainty",
        "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
    ],

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

    # Raw per-track features fed into encoder 3 (jet-flavour classification).
    # Must be a subset of track_fields. Defaults to all track_fields (original
    # behaviour). Narrow this down to e.g. just kinematic/IP variables to see
    # whether the hit-count features matter for final tagging once Stage 1/2
    # already use them.
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

    # loss weights
    "lambda_jet":    1.0,
    "lambda_origin": 1.0,
    "lambda_vertex": 1.0,

    # tau jets (HadronConeExclTruthLabelID == 15) are excluded for now
    "flavour_to_label": {"5": 0, "4": 1, "0": 2},
    "jet_class_names":  ["b-jet", "c-jet", "light-jet"],
    "origin_class_names": [
        "Pileup", "Fake", "Primary", "From b",
        "From b->c", "From c", "From tau", "Other secondary",
    ],
    "colours": {
        "b-jet": "#1f77b4", "c-jet": "#ff7f0e", "light-jet": "#2ca02c",
    },
    # background weights for the b-tagging discriminant log(p_b / Σ w_i p_i)
    "disc_bkg_weights": {"c-jet": 0.3, "light-jet": 0.7},

    # training-only
    "train_file":      "../opendata_tt/mc-flavtag-ttbar-small.h5",
    "n_train":         120_000,
    "n_test":          40_000,
    "epochs":          40,
    "lr":              1e-3,
    "num_workers":     0,
    "model_name":      "transformer_jet_classifier_staged_origin_vertex_jet.pt",
    "train_plot_dir":  "./transformer_results_staged_origin_vertex_jet/",
    "train_cache_dir": ".track_cache_staged_origin_vertex_jet/",
}

# ── args & config file ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Train staged transformer jet classifier: track-origin "
                "prediction -> differentiable vertex fit -> jet classification, "
                "each stage with its own encoder."
)
parser.add_argument("--config", default=None,
                    help="Path to JSON config file. Keys override built-in defaults.")
args = parser.parse_args()

cfg = dict(_DEFAULTS)
if args.config is not None:
    with open(args.config) as _f:
        _file_cfg = json.load(_f)
    _unknown = set(_file_cfg) - set(_DEFAULTS)
    if _unknown:
        raise ValueError(f"Unknown config keys: {_unknown}")
    cfg.update(_file_cfg)

# ── unpack config ──────────────────────────────────────────────────────
TRAIN_FILE         = cfg["train_file"]
N_TRAIN            = cfg["n_train"]
N_TEST             = cfg["n_test"]
BATCH_SIZE         = cfg["batch_size"]
EPOCHS             = cfg["epochs"]
LR                 = cfg["lr"]
NUM_WORKERS        = cfg["num_workers"]
TOP_K              = cfg["top_k"]
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME         = cfg["model_name"]
PLOT_DIR           = cfg["train_plot_dir"]
CACHE_DIR          = cfg["train_cache_dir"]
D_MODEL            = cfg["d_model"]
N_HEADS            = cfg["n_heads"]
N_LAYERS           = cfg["n_layers"]
D_FFN              = cfg["d_ffn"]
DROPOUT            = cfg["dropout"]
GATE_TEMP          = cfg["gate_temp"]
N_ORIGIN_CLASSES   = cfg["n_origin_classes"]
N_JET_CLASSES      = cfg["n_jet_classes"]
VERTEX_LEGS        = cfg["vertex_legs"]                         # {name: [origin_class_names]}
VERTEX_LEG_NAMES   = list(VERTEX_LEGS.keys())
N_VERTEX_LEGS      = len(VERTEX_LEG_NAMES)
VERTEX_TARGETS     = cfg["vertex_targets"]                      # {name: {flavour, slot}}
VERTEX_FIT_COORDS  = cfg["vertex_fit_coords"]                   # subset of ["Lxy", "dz"]
assert set(VERTEX_FIT_COORDS) <= {"Lxy", "dz"} and len(VERTEX_FIT_COORDS) >= 1, \
    "vertex_fit_coords must be a non-empty subset of ['Lxy', 'dz']"
N_VTX_COORDS       = len(VERTEX_FIT_COORDS)
FIT_LXY            = "Lxy" in VERTEX_FIT_COORDS
FIT_DZ             = "dz"  in VERTEX_FIT_COORDS
CALIBRATE_VERTEX_FIT = cfg["calibrate_vertex_fit"]              # learnable per-leg fit-bias scale
STAGE3_EXTRA_INPUTS = cfg["stage3_extra_inputs"]                # subset of ["origin_probs", "vtx_weight"]
assert set(STAGE3_EXTRA_INPUTS) <= {"origin_probs", "vtx_weight"}, \
    "stage3_extra_inputs must be a subset of ['origin_probs', 'vtx_weight']"
STAGE3_USE_ORIGIN_PROBS = "origin_probs" in STAGE3_EXTRA_INPUTS
STAGE3_USE_VTX_WEIGHT   = "vtx_weight"   in STAGE3_EXTRA_INPUTS
VERTEX_FIELDS      = cfg["vertex_fields"]
TRACK_FIELDS       = cfg["track_fields"]
TAGGING_FIELDS     = cfg["tagging_fields"]
assert set(TAGGING_FIELDS) <= set(TRACK_FIELDS) and len(TAGGING_FIELDS) >= 1, \
    "tagging_fields must be a non-empty subset of track_fields"
LAMBDA_JET         = cfg["lambda_jet"]
LAMBDA_ORIGIN      = cfg["lambda_origin"]
LAMBDA_VERTEX      = cfg["lambda_vertex"]
FLAVOUR_TO_LABEL   = {int(k): v for k, v in cfg["flavour_to_label"].items()}
JET_CLASS_NAMES    = cfg["jet_class_names"]
ORIGIN_CLASS_NAMES = cfg["origin_class_names"]
COLOURS            = cfg["colours"]
DISC_BKG_WEIGHTS   = cfg["disc_bkg_weights"]

# Build a (n_origin_classes, n_legs) projection matrix: entry [c, l] = 1 if
# origin class c contributes to vertex leg l (summed before sigmoid gate).
import numpy as _np
_leg_matrix = _np.zeros((N_ORIGIN_CLASSES, N_VERTEX_LEGS), dtype=_np.float32)
for _li, _lname in enumerate(VERTEX_LEG_NAMES):
    _cls_list = VERTEX_LEGS[_lname]
    if isinstance(_cls_list, str):
        _cls_list = [_cls_list]
    for _cname in _cls_list:
        _leg_matrix[ORIGIN_CLASS_NAMES.index(_cname), _li] = 1.0
VERTEX_LEG_ORIGIN_MATRIX = _leg_matrix   # shape (N_ORIGIN_CLASSES, N_VERTEX_LEGS)

N_FEATS = len(TRACK_FIELDS)

os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

_cfg_save_path = os.path.join(PLOT_DIR, "config.json")
with open(_cfg_save_path, "w") as _f:
    json.dump(cfg, _f, indent=4)
print(f"Config saved to {_cfg_save_path}")

# ── data loading ──────────────────────────────────────────────────────
def _cache_key(idx):
    h = hashlib.md5(idx.tobytes()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"tracks_{h}_nom.npz")


def load_tracks(path, idx):
    """Returns a dict of numpy arrays:
        X         (N, K, F)  track features
        mask      (N, K)     validity mask
        y         (N,)       jet flavour label (0..3)
        origin    (N, K)     track origin label (0..7, -1 = padding/no truth)
        vtx_lxy   (N, 2)     per-leg truth decay-vertex Lxy   [b (b+bc merged), c]
        vtx_valid (N, 2)     per-leg validity mask
    """
    cp = _cache_key(idx)
    if os.path.exists(cp):
        d = np.load(cp)
        return {k: d[k] for k in d.files}

    with h5py.File(path, "r") as f:
        flavour_id = f["jets"]["HadronConeExclTruthLabelID"][idx]
        keep_jet   = np.isin(flavour_id, list(FLAVOUR_TO_LABEL.keys()))
        fidx       = idx[keep_jet]

        valid  = f["tracks"]["valid"][fidx]
        d0     = f["tracks"]["lifetimeSignedD0"][fidx].astype(np.float32)
        ip2d   = f["tracks"]["lifetimeSignedD0Significance"][fidx].astype(np.float32)
        origin = f["tracks"]["ftagTruthOriginLabel"][fidx].astype(np.int8)
        arrs   = {fld: f["tracks"][fld][fidx].astype(np.float32) for fld in TRACK_FIELDS}

        th_valid_all = f["truth_hadrons"]["valid"][fidx]
        th_lxy_all   = f["truth_hadrons"]["Lxy"][fidx].astype(np.float32)
        th_dz_all    = f["truth_hadrons"]["decayVertexZ"][fidx].astype(np.float32)
        flavour_id_keep = flavour_id[keep_jet]

    keep = valid & (np.abs(d0) < 3.5)

    sort_key = ip2d.copy()
    sort_key[~keep] = -np.inf
    order = np.argsort(-sort_key, axis=1)

    feats       = np.stack([arrs[fld] for fld in TRACK_FIELDS], axis=-1)
    topk_idx    = order[:, :TOP_K]
    rows        = np.arange(len(fidx))[:, None]
    topk_feat   = feats[rows, topk_idx]
    topk_valid  = keep[rows, topk_idx]
    topk_feat   = np.where(topk_valid[:, :, None], topk_feat, 0.0).astype(np.float32)
    topk_origin = origin[rows, topk_idx]
    topk_origin = np.where(topk_valid, topk_origin, -1).astype(np.int64)

    # Drop jets that end up with zero surviving tracks after the
    # valid & |d0| < 3.5 selection. A fully-padded row forces every
    # attention query in the encoders to attend over an all-masked key
    # set, and softmax over an all "-inf" row yields NaN. Multiplying by
    # the validity mask afterwards does NOT clean this up (NaN * 0 == NaN),
    # so the NaN silently propagates through soft_probs / vtx_weight /
    # lxy_pred / vtx_tokens into that jet's CLS output, and a single NaN
    # row poisons the whole batch's mean CE loss — which is exactly the
    # `val_loss = nan` symptom observed during training.
    has_track = topk_valid.any(axis=1)
    if not has_track.all():
        print(f"  load_tracks: dropping {(~has_track).sum()} jet(s) with no surviving tracks")
    topk_feat       = topk_feat[has_track]
    topk_valid      = topk_valid[has_track]
    topk_origin     = topk_origin[has_track]
    flavour_id_keep = flavour_id_keep[has_track]
    th_valid_all    = th_valid_all[has_track]
    th_lxy_all      = th_lxy_all[has_track]
    th_dz_all       = th_dz_all[has_track]

    labels = np.array([FLAVOUR_TO_LABEL[v] for v in flavour_id_keep], dtype=np.int64)

    # `truth_hadrons` is ordered by decay chain: for b-jets, slot 0 is the
    # primary B-hadron (-> b-vertex target) and slot 1, when present, is the
    # cascade c-hadron produced in the B decay (-> cascade-vertex target);
    # for c-jets, slot 0 is the primary C-hadron (-> c-vertex target). This
    # is a pragmatic mapping (the same simplification used for the single
    # leading-hadron target in the original design, now split per leg).
    is_b = (flavour_id_keep == 5)
    is_c = (flavour_id_keep == 4)

    def _leg(valid_mask, slot):
        v    = valid_mask & th_valid_all[:, slot]
        lxy  = np.where(v, th_lxy_all[:, slot], 0.0).astype(np.float32)
        dz   = np.where(v, th_dz_all[:, slot],  0.0).astype(np.float32)
        return (np.nan_to_num(lxy, nan=0.0),
                np.nan_to_num(dz,  nan=0.0),
                v.astype(np.bool_))

    # Build one truth column per vertex leg, driven entirely by VERTEX_TARGETS
    _lxy_cols, _dz_cols, _valid_cols = [], [], []
    for _lname in VERTEX_LEG_NAMES:
        _tgt      = VERTEX_TARGETS[_lname]
        _jet_mask = (flavour_id_keep == _tgt["flavour"])
        _lxy_col, _dz_col, _valid_col = _leg(_jet_mask, _tgt["slot"])
        _lxy_cols.append(_lxy_col)
        _dz_cols.append(_dz_col)
        _valid_cols.append(_valid_col)

    vtx_lxy   = np.stack(_lxy_cols,   axis=-1)   # (N, L)
    vtx_dz    = np.stack(_dz_cols,    axis=-1)   # (N, L)  signed z-displacement
    vtx_valid = np.stack(_valid_cols, axis=-1)

    np.savez(cp, X=topk_feat, mask=topk_valid, y=labels, origin=topk_origin,
             vtx_lxy=vtx_lxy, vtx_dz=vtx_dz, vtx_valid=vtx_valid)
    return {"X": topk_feat, "mask": topk_valid, "y": labels, "origin": topk_origin,
            "vtx_lxy": vtx_lxy, "vtx_dz": vtx_dz, "vtx_valid": vtx_valid}


class JetDataset(Dataset):
    def __init__(self, d):
        self.X         = torch.from_numpy(d["X"])
        self.mask      = torch.from_numpy(d["mask"])
        self.y         = torch.from_numpy(d["y"])
        self.origin    = torch.from_numpy(d["origin"])
        self.vtx_lxy   = torch.from_numpy(d["vtx_lxy"])
        self.vtx_dz    = torch.from_numpy(d["vtx_dz"])
        self.vtx_valid = torch.from_numpy(d["vtx_valid"])
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (self.X[i], self.mask[i], self.y[i], self.origin[i],
                self.vtx_lxy[i], self.vtx_dz[i], self.vtx_valid[i])


# ── model ──────────────────────────────────────────────────────────────
class StagedOriginVertexJetTransformer(nn.Module):
    """
    Three-stage transformer pipeline, each stage with its own encoder:

      Stage 1 (encoder1): all track features -> per-track origin prediction
      Stage 2 (encoder2): predicted origin + vertexing-relevant track features
                          -> differentiable secondary-vertex fit (Lxy, z)
      Stage 3 (encoder3): fitted vertex quantities + all track features
                          -> CLS token -> jet-flavour classification

    Stages communicate only through differentiable intermediate tensors
    (soft origin probabilities; fitted vertex position), so gradients from
    any of the three losses flow through the whole network.
    """
    def __init__(self, in_dim, d_model, n_heads, n_layers, d_ffn, dropout,
                 n_origin_classes, n_jet_classes,
                 vertex_feat_indices, d0_idx, d0_unc_idx, dphi_idx,
                 z0st_idx, z0st_unc_idx,
                 vertex_leg_origin_matrix, gate_temp=0.1,
                 fit_lxy=True, fit_dz=True,
                 stage3_use_origin_probs=False, stage3_use_vtx_weight=False,
                 tagging_feat_indices=None,
                 calibrate_vertex_fit=True):
        super().__init__()
        self.n_origin_classes = n_origin_classes
        self.n_vertex_legs    = vertex_leg_origin_matrix.shape[1]
        self.gate_temp        = gate_temp
        self.fit_lxy          = fit_lxy
        self.fit_dz           = fit_dz
        self.stage3_use_origin_probs = stage3_use_origin_probs
        self.stage3_use_vtx_weight   = stage3_use_vtx_weight
        self.calibrate_vertex_fit    = calibrate_vertex_fit
        n_vtx_coords          = int(fit_lxy) + int(fit_dz)  # 1 or 2

        # Learnable per-leg multiplicative calibration scale for the
        # closed-form Stage-2 vertex fit, applied as
        #   pred_calibrated = pred * exp(log_scale)
        # before both the vertex-fit loss and the Stage-3 vtx_summary token.
        # Initialised to log_scale=0 (scale=1, i.e. no-op at init).
        if calibrate_vertex_fit and fit_lxy:
            self.lxy_log_scale = nn.Parameter(torch.zeros(self.n_vertex_legs))
        if calibrate_vertex_fit and fit_dz:
            self.dz_log_scale = nn.Parameter(torch.zeros(self.n_vertex_legs))

        # ── Stage 1: track-origin prediction ──
        self.input_proj1 = nn.Linear(in_dim, d_model)
        _enc1_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder1 = nn.TransformerEncoder(_enc1_layer, num_layers=n_layers)
        self.origin_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_origin_classes))

        # learnable, differentiable per-class embedding vectors — turn the
        # predicted soft origin distribution into a track-level embedding
        self.class_embed = nn.Parameter(torch.zeros(n_origin_classes, d_model))
        nn.init.trunc_normal_(self.class_embed, std=0.02)

        # ── Stage 2: differentiable secondary-vertex fit ──
        n_vtx_feats = len(vertex_feat_indices)
        self.input_proj2 = nn.Linear(n_vtx_feats, d_model)
        _enc2_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder2 = nn.TransformerEncoder(_enc2_layer, num_layers=n_layers)
        # one learned refinement weight per vertex leg (b, cascade b->c, c)
        self.vertex_weight_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, self.n_vertex_legs))
        # dz prediction head: one output per vertex leg
        self.dz_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, self.n_vertex_legs))

        # ── Stage 3: jet-flavour classification ──
        if tagging_feat_indices is None:
            tagging_feat_indices = list(range(in_dim))
        self.register_buffer("tagging_feat_idx",
            torch.tensor(tagging_feat_indices, dtype=torch.long))
        _stage3_in_dim = len(tagging_feat_indices)
        if stage3_use_origin_probs:
            _stage3_in_dim += n_origin_classes
        if stage3_use_vtx_weight:
            _stage3_in_dim += self.n_vertex_legs
        self.input_proj3 = nn.Linear(_stage3_in_dim, d_model)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # vertex summary token: active coords → d_model  (1 or 2 inputs)
        self.vertex_embed = nn.Sequential(
            nn.Linear(n_vtx_coords, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        _enc3_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder3 = nn.TransformerEncoder(_enc3_layer, num_layers=n_layers)
        self.jet_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_jet_classes))

        self.register_buffer("vertex_feat_idx",
            torch.tensor(vertex_feat_indices, dtype=torch.long))
        self.register_buffer("vertex_leg_origin_matrix",
            torch.tensor(vertex_leg_origin_matrix, dtype=torch.float32))
        self.d0_idx, self.d0_unc_idx, self.dphi_idx = d0_idx, d0_unc_idx, dphi_idx
        self.z0st_idx, self.z0st_unc_idx = z0st_idx, z0st_unc_idx

    def forward(self, x, mask):
        B, K, _ = x.shape
        track_padding_mask = ~mask
        mask_f = mask.unsqueeze(-1).float()

        # ── Stage 1: track-origin prediction (encoder 1) ──
        h1 = self.input_proj1(x)
        h1 = self.encoder1(h1, src_key_padding_mask=track_padding_mask)
        origin_logits = self.origin_head(h1)                       # (B, K, C)
        soft_probs = torch.softmax(origin_logits, dim=-1) * mask_f

        # ── Stage 2: three origin-gated differentiable vertex fits (encoder 2) ──
        soft_embed = soft_probs @ self.class_embed                 # (B, K, d_model)
        vtx_feats  = x.index_select(-1, self.vertex_feat_idx)      # (B, K, n_vtx_feats)
        h2 = self.input_proj2(vtx_feats) + soft_embed
        h2 = self.encoder2(h2, src_key_padding_mask=track_padding_mask)

        # learned per-leg refinement weight in [0, 1], gated by the
        # predicted soft probability of the leg's origin class — i.e. the
        # b-vertex fit only "sees" tracks Stage 1 thinks come from b, etc.
        refine = torch.sigmoid(self.vertex_weight_head(h2))             # (B, K, L)
        # Sum soft probs across the origin classes that belong to each vertex leg
        # (e.g. b_vertex = P("From b") + P("From b->c")), then apply a sigmoid
        # gate centred at 0.5 so only tracks with combined probability > 0.5
        # contribute meaningfully.  gate_temp controls sharpness — smaller →
        # closer to a hard step function while remaining fully differentiable.
        leg_origin_probs = soft_probs @ self.vertex_leg_origin_matrix    # (B, K, L)
        gate       = torch.sigmoid((leg_origin_probs - 0.5) / self.gate_temp)
        vtx_weight = refine * gate * mask_f                              # (B, K, L)

        # ── Lxy fit (transverse) ─────────────────────────────────────
        # |d0ᵢ| ≈ Lxy · |sin Δφᵢ|, flight φ from vtx_weight-gated circular mean.
        if self.fit_lxy:
            d0      = x[..., self.d0_idx].unsqueeze(-1)
            d0_unc  = x[..., self.d0_unc_idx].abs().clamp(min=1e-3).unsqueeze(-1)
            dphi    = x[..., self.dphi_idx].unsqueeze(-1)
            sin_dphi   = torch.sin(dphi)
            cos_dphi   = torch.cos(dphi)
            sum_sin    = (vtx_weight * sin_dphi).sum(1)
            sum_cos    = (vtx_weight * cos_dphi).sum(1)
            flight_phi = torch.atan2(sum_sin, sum_cos).unsqueeze(1)
            delta_phi  = dphi - flight_phi
            sin_d      = torch.sin(delta_phi)
            inv_var_d0 = vtx_weight / d0_unc.pow(2)
            num_lxy    = (inv_var_d0 * sin_d.abs() * d0.abs()).sum(1)
            den_lxy    = (inv_var_d0 * sin_d.pow(2)).sum(1).clamp(min=1e-6)
            lxy_pred   = num_lxy / den_lxy                              # (B, L)
            if self.calibrate_vertex_fit:
                lxy_pred = lxy_pred * torch.exp(self.lxy_log_scale).unsqueeze(0)
        else:
            lxy_pred = vtx_weight.new_zeros(vtx_weight.shape[0], self.n_vertex_legs)

        # ── dz fit (longitudinal) ─────────────────────────────────────
        # z0·sinθ ≈ dz; signed inverse-variance-weighted mean over gated tracks.
        if self.fit_dz:
            z0st     = x[..., self.z0st_idx].unsqueeze(-1)
            z0st_unc = x[..., self.z0st_unc_idx].abs().clamp(min=1e-3).unsqueeze(-1)
            inv_var_z0 = vtx_weight / z0st_unc.pow(2)
            num_dz     = (inv_var_z0 * z0st).sum(1)
            den_dz     = inv_var_z0.sum(1).clamp(min=1e-6)
            dz_pred    = num_dz / den_dz                                # (B, L) signed
            if self.calibrate_vertex_fit:
                dz_pred = dz_pred * torch.exp(self.dz_log_scale).unsqueeze(0)
        else:
            dz_pred = vtx_weight.new_zeros(vtx_weight.shape[0], self.n_vertex_legs)

        # ── Stage 3: jet-flavour classification (encoder 3) ──
        # Build vertex summary from only the active coordinates
        _vtx_parts = []
        if self.fit_lxy:
            _vtx_parts.append(torch.log1p(lxy_pred.clamp(min=0)))
        if self.fit_dz:
            _vtx_parts.append(torch.log1p(dz_pred.abs()))
        vtx_summary = torch.stack(_vtx_parts, dim=-1)                   # (B, L, N_VTX_COORDS)
        vtx_tokens  = self.vertex_embed(vtx_summary)                    # (B, L, d_model)

        # Optionally let encoder3 see what Stages 1/2 inferred per track:
        # the Stage-1 soft origin-class probabilities and/or the Stage-2
        # per-leg vertex-fit gating weight. Raw track features are first
        # restricted to the configured tagging_fields subset.
        x_tag = x.index_select(-1, self.tagging_feat_idx)
        _h3_input_parts = [x_tag]
        if self.stage3_use_origin_probs:
            _h3_input_parts.append(soft_probs)
        if self.stage3_use_vtx_weight:
            _h3_input_parts.append(vtx_weight)
        h3_input  = torch.cat(_h3_input_parts, dim=-1) if len(_h3_input_parts) > 1 else x_tag
        h3_tracks = self.input_proj3(h3_input)
        cls_tok   = self.cls_token.expand(B, -1, -1)
        h3_in     = torch.cat([cls_tok, vtx_tokens, h3_tracks], dim=1)

        extra_valid = torch.ones(B, 1 + self.n_vertex_legs, dtype=torch.bool, device=x.device)
        src_key_padding_mask3 = ~torch.cat([extra_valid, mask], dim=1)

        h3 = self.encoder3(h3_in, src_key_padding_mask=src_key_padding_mask3)
        jet_logits = self.jet_head(h3[:, 0])

        return {
            "jet_logits":    jet_logits,
            "origin_logits": origin_logits,
            "vtx_weight":    vtx_weight,    # (B, K, L)
            "lxy_pred":      lxy_pred,      # (B, L)
            "dz_pred":       dz_pred,       # (B, L)  signed
            "refine":        refine,        # (B, K, L)
        }

    def calibration_scales(self):
        """Return the current learned per-leg calibration scales (exp(log_scale))
        as plain Python lists, for logging/inspection. Empty if disabled."""
        out = {}
        if self.calibrate_vertex_fit and self.fit_lxy:
            out["lxy"] = torch.exp(self.lxy_log_scale).detach().cpu().tolist()
        if self.calibrate_vertex_fit and self.fit_dz:
            out["dz"] = torch.exp(self.dz_log_scale).detach().cpu().tolist()
        return out


# ── balanced index sampling ────────────────────────────────────────────
print("Loading data...")
rng = np.random.default_rng(42)

_flavour_cache = os.path.join(CACHE_DIR, "all_flavours.npy")
if os.path.exists(_flavour_cache):
    all_flavours = np.load(_flavour_cache)
else:
    with h5py.File(TRAIN_FILE, "r") as f:
        all_flavours = f["jets"]["HadronConeExclTruthLabelID"][:]
    np.save(_flavour_cache, all_flavours)

valid_mask = np.isin(all_flavours, list(FLAVOUR_TO_LABEL.keys()))
valid_idx  = rng.permutation(np.where(valid_mask)[0])

test_idx    = np.sort(valid_idx[-N_TEST:])
pool_idx    = valid_idx[:-N_TEST]
pool_labels = np.array([FLAVOUR_TO_LABEL[v] for v in all_flavours[pool_idx]])

n_classes_jet = len(JET_CLASS_NAMES)
n_per_class   = N_TRAIN // n_classes_jet
train_idx = np.sort(np.concatenate([
    rng.choice(pool_idx[pool_labels == cls],
               size=min(n_per_class, (pool_labels == cls).sum()),
               replace=False)
    for cls in range(n_classes_jet)
]))

train_data = load_tracks(TRAIN_FILE, train_idx)
test_data  = load_tracks(TRAIN_FILE, test_idx)
y_train, y_test = train_data["y"], test_data["y"]

print("Train — " + "  ".join(f"{name}:{(y_train==i).sum():,}"
                             for i, name in enumerate(JET_CLASS_NAMES)))
print("Test  — " + "  ".join(f"{name}:{(y_test==i).sum():,}"
                             for i, name in enumerate(JET_CLASS_NAMES)))

_pin = DEVICE == "cuda"
_pw  = NUM_WORKERS > 0
train_loader = DataLoader(
    JetDataset(train_data), batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=_pin, num_workers=NUM_WORKERS, persistent_workers=_pw)
val_loader = DataLoader(
    JetDataset(test_data), batch_size=BATCH_SIZE,
    pin_memory=_pin, num_workers=NUM_WORKERS, persistent_workers=_pw)

# ── input variable plot ────────────────────────────────────────────────
X_train, mask_train = train_data["X"], train_data["mask"]
tracks_flat = X_train.reshape(-1, N_FEATS)
labels_rep  = np.repeat(y_train, TOP_K)
nonzero     = mask_train.ravel()

n_cols = min(N_FEATS, 4)
n_rows = (N_FEATS + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = np.array(axes).ravel()
fig.suptitle("Input variables by jet flavour (training sample)", fontweight="bold")
for fi, fld in enumerate(TRACK_FIELDS):
    col = tracks_flat[:, fi]
    valid_col = col[nonzero]
    clip = np.percentile(np.abs(valid_col), 99) if valid_col.size else 1.0
    for cls_idx, name in enumerate(JET_CLASS_NAMES):
        m = (labels_rep == cls_idx) & nonzero
        axes[fi].hist(col[m], bins=80, range=(-clip, clip),
                      histtype="step", label=name, color=COLOURS[name],
                      linewidth=1.5, density=True)
    axes[fi].set_title(fld, fontsize=8)
    axes[fi].set_xlabel(fld, fontsize=7)
    axes[fi].set_ylabel("Density", fontsize=7)
    if fi == 0:
        axes[fi].legend(fontsize=7)
for ax in axes[N_FEATS:]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR + "input_variables.png", dpi=150, bbox_inches="tight")
print("Saved input_variables.png")

# ── model / optimiser / losses ─────────────────────────────────────────
_vertex_feat_idx  = [TRACK_FIELDS.index(f) for f in VERTEX_FIELDS]
_tagging_feat_idx = [TRACK_FIELDS.index(f) for f in TAGGING_FIELDS]
_d0_idx          = TRACK_FIELDS.index("d0")
_d0_unc_idx      = TRACK_FIELDS.index("d0Uncertainty")
_dphi_idx        = TRACK_FIELDS.index("dphi")
_z0st_idx        = TRACK_FIELDS.index("z0SinTheta")
_z0st_unc_idx    = TRACK_FIELDS.index("z0SinThetaUncertainty")

model = StagedOriginVertexJetTransformer(
    in_dim=N_FEATS, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, d_ffn=D_FFN, dropout=DROPOUT,
    n_origin_classes=N_ORIGIN_CLASSES, n_jet_classes=N_JET_CLASSES,
    vertex_feat_indices=_vertex_feat_idx,
    d0_idx=_d0_idx, d0_unc_idx=_d0_unc_idx, dphi_idx=_dphi_idx,
    z0st_idx=_z0st_idx, z0st_unc_idx=_z0st_unc_idx,
    vertex_leg_origin_matrix=VERTEX_LEG_ORIGIN_MATRIX,
    gate_temp=GATE_TEMP, fit_lxy=FIT_LXY, fit_dz=FIT_DZ,
    stage3_use_origin_probs=STAGE3_USE_ORIGIN_PROBS,
    stage3_use_vtx_weight=STAGE3_USE_VTX_WEIGHT,
    tagging_feat_indices=_tagging_feat_idx,
    calibrate_vertex_fit=CALIBRATE_VERTEX_FIT,
).to(DEVICE)
print(f"Vertex fit coordinates: {VERTEX_FIT_COORDS}")
print(f"Stage-3 extra inputs: {STAGE3_EXTRA_INPUTS}")
print(f"Stage-3 tagging fields ({len(TAGGING_FIELDS)}): {TAGGING_FIELDS}")
print(f"Calibrate vertex fit (learnable per-leg scale): {CALIBRATE_VERTEX_FIT}")
optimiser = torch.optim.Adam(model.parameters(), lr=LR)

criterion_jet    = nn.CrossEntropyLoss()

# Inverse-frequency class weights for origin loss: rare classes (Fake, From tau)
# have very few tracks and would otherwise be drowned out by Primary/Pileup.
# Weights are computed from the training split and clipped at 20x the median
# to avoid extreme values for near-absent classes.
_origin_flat = train_data["origin"].ravel()                      # (N*K,)
_origin_flat = _origin_flat[_origin_flat >= 0]                   # drop padding (-1)
_counts = np.bincount(_origin_flat, minlength=N_ORIGIN_CLASSES).astype(np.float64)
_counts = np.maximum(_counts, 1)                                 # avoid /0
_inv_freq = 1.0 / _counts
_inv_freq = _inv_freq / _inv_freq.mean()                         # normalise to mean=1
_inv_freq = np.clip(_inv_freq, None, 20.0 * np.median(_inv_freq))  # cap at 20× median
_origin_weights = torch.tensor(_inv_freq, dtype=torch.float32).to(DEVICE)
print("Origin class weights:")
for _i, (_n, _w) in enumerate(zip(ORIGIN_CLASS_NAMES, _origin_weights.cpu())):
    print(f"  {_i:2d}  {_n:<18s}  {_w:.3f}")

criterion_origin = nn.CrossEntropyLoss(ignore_index=-1, weight=_origin_weights)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params:,}")
print(f"Device: {DEVICE}  |  Train: {len(y_train):,}  |  Test: {len(y_test):,}\n")


def vertex_loss_fn(lxy_pred, dz_pred, vtx_lxy, vtx_dz, vtx_valid):
    """Smooth-L1 vertex loss over active coordinates (controlled by FIT_LXY / FIT_DZ).
    All tensor inputs shape (B, L); skips coordinates disabled in config."""
    total = lxy_pred.new_tensor(0.0)
    for leg in range(lxy_pred.shape[-1]):
        v = vtx_valid[:, leg]
        if not v.any():
            continue
        if FIT_LXY:
            lp = torch.log1p(lxy_pred[v, leg].clamp(min=0))
            lt = torch.log1p(vtx_lxy[v, leg].clamp(min=0))
            total = total + F.smooth_l1_loss(lp, lt)
        if FIT_DZ:
            zp = torch.log1p(dz_pred[v, leg].abs()) * dz_pred[v, leg].sign()
            zt = torch.log1p(vtx_dz[v, leg].abs())  * vtx_dz[v, leg].sign()
            total = total + F.smooth_l1_loss(zp, zt)
    return total


def _compute_epoch_refine_vtx_stats(vtx_weight, refine, origin_full, mask_full, all_true):
    norig  = ORIGIN_CLASS_NAMES
    _leg_owner = {}
    for lname in VERTEX_LEG_NAMES:
        _flv = VERTEX_TARGETS[lname]["flavour"]
        _leg_owner[lname] = JET_CLASS_NAMES.index(
            {int(fl): name for name, fl in
             {JET_CLASS_NAMES[i]: int(lbl)
              for lbl, i in FLAVOUR_TO_LABEL.items()}.items()}[_flv])
    stats = {}
    for leg in range(N_VERTEX_LEGS):
        leg_name  = VERTEX_LEG_NAMES[leg]
        owner_cls = _leg_owner[leg_name]
        leg_origin_names = VERTEX_LEGS[leg_name]
        if isinstance(leg_origin_names, str):
            leg_origin_names = [leg_origin_names]
        leg_origin_ids = [norig.index(c) for c in leg_origin_names]

        owner_mask = (all_true == owner_cls)
        if not owner_mask.any():
            continue
        jet_vtx_w = vtx_weight[owner_mask, :, leg]
        jet_ref   = refine[owner_mask, :, leg]
        jet_orig  = origin_full[owner_mask]
        jet_mask  = mask_full[owner_mask]
        match_mask = np.isin(jet_orig, leg_origin_ids) & jet_mask
        other_mask = ~np.isin(jet_orig, leg_origin_ids) & jet_mask & (jet_orig >= 0)
        s = leg_name.replace("_vertex", "")
        if match_mask.any():
            stats[f"val_{s}_refine_match_mean"]      = jet_ref[match_mask].mean()
            stats[f"val_{s}_vtx_weight_match_mean"]  = jet_vtx_w[match_mask].mean()
        if other_mask.any():
            stats[f"val_{s}_refine_other_mean"]      = jet_ref[other_mask].mean()
            stats[f"val_{s}_vtx_weight_other_mean"]  = jet_vtx_w[other_mask].mean()
    return stats


# ── training loop ─────────────────────────────────────────────────────
history = {
    "train_loss": [], "train_jet_loss": [], "train_origin_loss": [], "train_vertex_loss": [],
    "val_loss": [], "val_acc": [], "val_origin_acc": [],
    "train_refine_mean": [], "train_vtx_weight_mean": [],
    "val_b_refine_match_mean": [], "val_b_refine_other_mean": [],
    "val_b_vtx_weight_match_mean": [], "val_b_vtx_weight_other_mean": [],
    "val_c_refine_match_mean": [], "val_c_refine_other_mean": [],
    "val_c_vtx_weight_match_mean": [], "val_c_vtx_weight_other_mean": [],
}

for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_loss = tot_jet = tot_origin = tot_vertex = 0.0
    tot_refine = tot_vtxw = 0.0

    for X_b, mask_b, y_b, origin_b, lxy_b, dz_b, vvalid_b in train_loader:
        X_b, mask_b, y_b   = X_b.to(DEVICE), mask_b.to(DEVICE), y_b.to(DEVICE)
        origin_b           = origin_b.to(DEVICE)
        lxy_b              = lxy_b.to(DEVICE)
        dz_b               = dz_b.to(DEVICE)
        vvalid_b           = vvalid_b.to(DEVICE)

        optimiser.zero_grad()
        out = model(X_b, mask_b)

        jet_loss    = criterion_jet(out["jet_logits"], y_b)
        origin_loss = criterion_origin(
            out["origin_logits"].reshape(-1, N_ORIGIN_CLASSES), origin_b.reshape(-1))
        vtx_loss    = vertex_loss_fn(out["lxy_pred"], out["dz_pred"], lxy_b, dz_b, vvalid_b)

        loss = LAMBDA_JET * jet_loss + LAMBDA_ORIGIN * origin_loss + LAMBDA_VERTEX * vtx_loss
        loss.backward()
        optimiser.step()

        tot_loss   += loss.item()        * len(y_b)
        tot_jet    += jet_loss.item()    * len(y_b)
        tot_origin += origin_loss.item() * len(y_b)
        tot_vertex += vtx_loss.item()    * len(y_b)

        valid_3d = mask_b.unsqueeze(-1).expand_as(out["refine"])
        tot_refine += out["refine"][valid_3d].float().mean().item()      * len(y_b)
        tot_vtxw   += out["vtx_weight"][valid_3d].float().mean().item()  * len(y_b)

    train_loss        = tot_loss   / len(y_train)
    train_jet_loss    = tot_jet    / len(y_train)
    train_origin_loss = tot_origin / len(y_train)
    train_vertex_loss = tot_vertex / len(y_train)
    train_refine      = tot_refine / len(y_train)
    train_vtxw        = tot_vtxw   / len(y_train)

    # ── validation ──
    model.eval()
    val_loss, correct, origin_correct, origin_total = 0.0, 0, 0, 0
    all_preds, all_true, all_probs = [], [], []
    all_origin_preds, all_origin_true = [], []
    all_lxy_pred, all_lxy_true, all_dz_pred, all_dz_true, all_vtx_valid = [], [], [], [], []
    all_vtx_weight, all_origin_full, all_mask_full, all_refine = [], [], [], []
    with torch.no_grad():
        for X_b, mask_b, y_b, origin_b, lxy_b, dz_b, vvalid_b in val_loader:
            X_b, mask_b, y_b = X_b.to(DEVICE), mask_b.to(DEVICE), y_b.to(DEVICE)
            origin_b   = origin_b.to(DEVICE)
            lxy_b      = lxy_b.to(DEVICE)
            dz_b       = dz_b.to(DEVICE)
            vvalid_b   = vvalid_b.to(DEVICE)

            out = model(X_b, mask_b)
            jet_loss    = criterion_jet(out["jet_logits"], y_b)
            origin_loss = criterion_origin(
                out["origin_logits"].reshape(-1, N_ORIGIN_CLASSES), origin_b.reshape(-1))
            vtx_loss    = vertex_loss_fn(out["lxy_pred"], out["dz_pred"], lxy_b, dz_b, vvalid_b)
            loss = LAMBDA_JET * jet_loss + LAMBDA_ORIGIN * origin_loss + LAMBDA_VERTEX * vtx_loss
            val_loss += loss.item() * len(y_b)

            preds = out["jet_logits"].argmax(dim=1)
            correct += (preds == y_b).sum().item()
            all_preds.append(preds.cpu())
            all_true.append(y_b.cpu())
            all_probs.append(torch.softmax(out["jet_logits"], dim=1).cpu())

            origin_preds = out["origin_logits"].argmax(dim=-1)
            origin_mask  = origin_b >= 0
            origin_correct += ((origin_preds == origin_b) & origin_mask).sum().item()
            origin_total   += origin_mask.sum().item()
            all_origin_preds.append(origin_preds[origin_mask].cpu())
            all_origin_true.append(origin_b[origin_mask].cpu())

            all_lxy_pred.append(out["lxy_pred"].cpu())
            all_lxy_true.append(lxy_b.cpu())
            all_dz_pred.append(out["dz_pred"].cpu())
            all_dz_true.append(dz_b.cpu())
            all_vtx_valid.append(vvalid_b.cpu())

            all_vtx_weight.append(out["vtx_weight"].cpu())
            all_origin_full.append(origin_b.cpu())
            all_mask_full.append(mask_b.cpu())
            all_refine.append(out["refine"].cpu())

    val_loss       /= len(y_test)
    val_acc         = correct / len(y_test)
    val_origin_acc  = origin_correct / max(origin_total, 1)

    history["train_loss"].append(train_loss)
    history["train_jet_loss"].append(train_jet_loss)
    history["train_origin_loss"].append(train_origin_loss)
    history["train_vertex_loss"].append(train_vertex_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_origin_acc"].append(val_origin_acc)

    history["train_refine_mean"].append(train_refine)
    history["train_vtx_weight_mean"].append(train_vtxw)

    _v0 = torch.cat(all_vtx_weight).numpy()
    _r0 = torch.cat(all_refine).numpy()
    _of = torch.cat(all_origin_full).numpy()
    _mf = torch.cat(all_mask_full).numpy().astype(bool)
    _at = torch.cat(all_true).numpy()
    _vtx_stats = _compute_epoch_refine_vtx_stats(_v0, _r0, _of, _mf, _at)
    _expected_val_keys = [
        "val_b_refine_match_mean", "val_b_refine_other_mean",
        "val_b_vtx_weight_match_mean", "val_b_vtx_weight_other_mean",
        "val_c_refine_match_mean", "val_c_refine_other_mean",
        "val_c_vtx_weight_match_mean", "val_c_vtx_weight_other_mean",
    ]
    for _k in _expected_val_keys:
        history[_k].append(_vtx_stats.get(_k, 0.0))

    _calib_str = ""
    if CALIBRATE_VERTEX_FIT:
        _scales = model.calibration_scales()
        _parts = []
        for _coord, _vals in _scales.items():
            _per_leg = ", ".join(f"{_n}={_v:.3f}" for _n, _v in zip(VERTEX_LEG_NAMES, _vals))
            _parts.append(f"{_coord}_scale=[{_per_leg}]")
        if _parts:
            _calib_str = "  " + "  ".join(_parts)

    print(f"Epoch {epoch:02d}/{EPOCHS}  "
          f"loss={train_loss:.4f} (jet={train_jet_loss:.4f} "
          f"origin={train_origin_loss:.4f} vtx={train_vertex_loss:.4f})  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
          f"origin_acc={val_origin_acc:.4f}{_calib_str}")
    if _vtx_stats:
        print(f"         refine train={train_refine:.4f}  "
              f"val b(m={_vtx_stats.get('val_b_refine_match_mean', 0):.4f} "
              f"o={_vtx_stats.get('val_b_refine_other_mean', 0):.4f})  "
              f"c(m={_vtx_stats.get('val_c_refine_match_mean', 0):.4f} "
              f"o={_vtx_stats.get('val_c_refine_other_mean', 0):.4f})  "
              f"vtx_w b(m={_vtx_stats.get('val_b_vtx_weight_match_mean', 0):.4f} "
              f"o={_vtx_stats.get('val_b_vtx_weight_other_mean', 0):.4f})  "
              f"c(m={_vtx_stats.get('val_c_vtx_weight_match_mean', 0):.4f} "
              f"o={_vtx_stats.get('val_c_vtx_weight_other_mean', 0):.4f})")

# ── save model ────────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(PLOT_DIR, MODEL_NAME))
print(f"Saved {MODEL_NAME}")

# ── final evaluation arrays ────────────────────────────────────────────
all_preds  = torch.cat(all_preds).numpy()
all_true   = torch.cat(all_true).numpy()
all_probs  = torch.cat(all_probs).numpy()
origin_preds = torch.cat(all_origin_preds).numpy()
origin_true  = torch.cat(all_origin_true).numpy()
lxy_pred   = torch.cat(all_lxy_pred).numpy()
lxy_true   = torch.cat(all_lxy_true).numpy()
dz_pred    = torch.cat(all_dz_pred).numpy()
dz_true    = torch.cat(all_dz_true).numpy()
vtx_valid  = torch.cat(all_vtx_valid).numpy().astype(bool)
vtx_weight = torch.cat(all_vtx_weight).numpy()
origin_full = torch.cat(all_origin_full).numpy()
mask_full = torch.cat(all_mask_full).numpy().astype(bool)
refine = torch.cat(all_refine).numpy()

print("\nJet classification report:")
print(classification_report(all_true, all_preds, target_names=JET_CLASS_NAMES))
print("Jet confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(all_true, all_preds))

print("\nTrack-origin classification report (Stage 1):")
print(classification_report(origin_true, origin_preds, target_names=ORIGIN_CLASS_NAMES,
                            labels=list(range(N_ORIGIN_CLASSES)), zero_division=0))

# ── plot: training summary ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Staged origin → vertex → jet transformer — training summary", fontweight="bold")
ep = range(1, EPOCHS + 1)
axes[0].plot(ep, history["train_loss"],        label="train (total)")
axes[0].plot(ep, history["train_jet_loss"],    label="train jet CE",    linestyle="--")
axes[0].plot(ep, history["train_origin_loss"], label="train origin CE", linestyle=":")
axes[0].plot(ep, history["train_vertex_loss"], label="train vertex",    linestyle="-.")
axes[0].plot(ep, history["val_loss"],          label="val (total)")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=7)

axes[1].plot(ep, history["val_acc"],        label="jet accuracy")
axes[1].plot(ep, history["val_origin_acc"], label="origin accuracy (Stage 1)")
axes[1].set_title("Validation accuracy"); axes[1].set_xlabel("Epoch")
axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8)

cm = confusion_matrix(all_true, all_preds, normalize="true")
im = axes[2].imshow(cm, cmap="Blues", vmin=0, vmax=1)
axes[2].set_xticks(range(N_JET_CLASSES)); axes[2].set_yticks(range(N_JET_CLASSES))
axes[2].set_xticklabels(JET_CLASS_NAMES, rotation=45, ha="right", fontsize=8)
axes[2].set_yticklabels(JET_CLASS_NAMES, fontsize=8)
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
axes[2].set_title("Jet confusion matrix (normalised)")
for i in range(N_JET_CLASSES):
    for j in range(N_JET_CLASSES):
        axes[2].text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                     color="white" if cm[i,j] > 0.5 else "black", fontsize=8)
plt.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig(PLOT_DIR + "training_summary.png", dpi=150, bbox_inches="tight")
print("Saved training_summary.png")

# ── plot: refine & vtx_weight per epoch ──────────────────────────────
if any(k.startswith("val_") and "refine" in k for k in history):
    fig, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("refine & vtx_weight per epoch", fontweight="bold")
    ep = range(1, EPOCHS + 1)
    for ax, prefix in [(ax_r, "refine"), (ax_v, "vtx_weight")]:
        key_pairs = [
            (f"val_b_{prefix}_match_mean", f"val_b_{prefix}_other_mean", "b", "#1f77b4"),
            (f"val_c_{prefix}_match_mean", f"val_c_{prefix}_other_mean", "c", "#2ca02c"),
        ]
        for mkey, okey, label, color in key_pairs:
            if (mkey in history and okey in history and
                    len(history[mkey]) == len(ep)):
                ax.plot(ep, history[mkey], color=color, linestyle="-",
                        linewidth=1.5, label=f"{label} match")
                ax.plot(ep, history[okey], color=color, linestyle="--",
                        linewidth=1.5, label=f"{label} other")
        tk = f"train_{prefix}_mean"
        if tk in history and len(history[tk]) == len(ep):
            ax.plot(ep, history[tk], color="grey", linestyle=":",
                    linewidth=1.0, label="train overall")
        ax.set_xlabel("Epoch"); ax.set_ylabel(prefix)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7); ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR + "refine_vtx_weight_history.png",
                dpi=150, bbox_inches="tight")
    print("Saved refine_vtx_weight_history.png")

# ── plot: track-origin confusion matrix (Stage 1) ─────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
fig.suptitle("Track-origin confusion matrix — Stage 1 (normalised)", fontweight="bold")
cm_o = confusion_matrix(origin_true, origin_preds, normalize="true",
                        labels=list(range(N_ORIGIN_CLASSES)))
im = ax.imshow(cm_o, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(N_ORIGIN_CLASSES)); ax.set_yticks(range(N_ORIGIN_CLASSES))
ax.set_xticklabels(ORIGIN_CLASS_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ORIGIN_CLASS_NAMES, fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(N_ORIGIN_CLASSES):
    for j in range(N_ORIGIN_CLASSES):
        ax.text(j, i, f"{cm_o[i,j]:.2f}", ha="center", va="center",
                color="white" if cm_o[i,j] > 0.5 else "black", fontsize=7)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(PLOT_DIR + "origin_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved origin_confusion_matrix.png")

# ── plot: vertex fitter — predicted vs truth, per leg & per flavour (Stage 2) ──
# For jets where a matching truth hadron exists, show overlaid truth/predicted
# histograms (solid=truth, dashed=predicted) split by true jet flavour. For jets where no truth
# hadron exists for that leg (e.g. light jets, or the "wrong" flavour
# for that leg), there is nothing to compare against — instead show the
# distribution of the *fitted* quantity itself, again split by true flavour,
# which is diagnostic of whether the network has learned to suppress
# spurious vertex fits for jets that shouldn't have one.
# Auto-generate plot titles from config (leg name + contributing origin classes)
_leg_titles = {
    lname: f"{lname.replace('_', '-')} (tracks: {', '.join(VERTEX_LEGS[lname]) if isinstance(VERTEX_LEGS[lname], list) else VERTEX_LEGS[lname]})"
    for lname in VERTEX_LEG_NAMES
}
# Left panel: restrict to the jet flavour that owns each vertex leg
# Derive owner jet class from VERTEX_TARGETS (config-driven)
_flavour_to_class_name = {v: k for k, v in
    {name: int(fl) for fl, name in
     [(lbl, JET_CLASS_NAMES[idx]) for lbl, idx in FLAVOUR_TO_LABEL.items()]}.items()}
_leg_owner_cls = {
    lname: JET_CLASS_NAMES.index(
        _flavour_to_class_name[VERTEX_TARGETS[lname]["flavour"]]
    )
    for lname in VERTEX_LEG_NAMES
}
for leg in range(N_VERTEX_LEGS):
    leg_name   = VERTEX_LEG_NAMES[leg]
    owner_cls  = _leg_owner_cls[leg_name]
    owner_name = JET_CLASS_NAMES[owner_cls]
    has_truth  = vtx_valid[:, leg] & (lxy_true[:, leg] > 0) & (all_true == owner_cls)
    no_truth   = ~has_truth

    if FIT_LXY:
        fig, (ax_cmp, ax_dist) = plt.subplots(1, 2, figsize=(13, 5.5))
        fig.suptitle(f"Differentiable origin-gated vertex fit — {_leg_titles[leg_name]}  "
                     r"— $L_{xy}$ (test sample, by true jet flavour)", fontweight="bold")

        # -- predicted vs. truth, where a truth target exists (owner jet class only) --
        if has_truth.any():
            _clip = max(np.percentile(lxy_true[has_truth, leg], 99),
                        np.percentile(lxy_pred[has_truth, leg], 99), 1e-3)
            _bins = np.linspace(0, _clip, 61)
            ax_cmp.hist(lxy_true[has_truth, leg], bins=_bins, histtype="step",
                        color=COLOURS[owner_name], linestyle="-", linewidth=1.5,
                        density=True, label=f"{owner_name} (truth)")
            ax_cmp.hist(lxy_pred[has_truth, leg], bins=_bins, histtype="step",
                        color=COLOURS[owner_name], linestyle="--", linewidth=1.5,
                        density=True, label=f"{owner_name} (pred)")
            ax_cmp.set_xlim(0, _clip)
        else:
            ax_cmp.text(0.5, 0.5, "no jets with a matching truth hadron", ha="center",
                        va="center", transform=ax_cmp.transAxes, fontsize=9, color="grey")
        ax_cmp.set_xlabel(r"$L_{xy}$ [mm]"); ax_cmp.set_ylabel("Density")
        ax_cmp.set_title(r"Truth (solid) vs. predicted (dashed) — truth hadron exists")
        ax_cmp.legend(fontsize=6); ax_cmp.grid(True, linestyle="--", alpha=0.3)

        # -- fitted-value distributions, where no truth target exists --
        if no_truth.any():
            _lxy_clip = np.percentile(lxy_pred[no_truth, leg], 99)
            for cls_idx, cls_name in enumerate(JET_CLASS_NAMES):
                m = no_truth & (all_true == cls_idx)
                if not m.any():
                    continue
                ax_dist.hist(lxy_pred[m, leg], bins=60, range=(0, _lxy_clip),
                             histtype="step", color=COLOURS[cls_name], label=cls_name,
                             linewidth=1.5, density=True)
        else:
            ax_dist.text(0.5, 0.5, "every jet has a matching truth hadron", ha="center",
                         va="center", transform=ax_dist.transAxes, fontsize=9, color="grey")
        ax_dist.set_xlabel(r"Predicted $L_{xy}$ [mm]"); ax_dist.set_ylabel("Density")
        ax_dist.set_title(r"Fitted-value distribution (no truth hadron)")
        ax_dist.legend(fontsize=7); ax_dist.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOT_DIR + f"vertex_fit_{leg_name}.png", dpi=150, bbox_inches="tight")
        print(f"Saved vertex_fit_{leg_name}.png")

    # ── dz plot: truth vs predicted ──────────────────────────────────
    if FIT_DZ:
        fig_dz, axes_dz = plt.subplots(1, 2, figsize=(13, 5.5))
        fig_dz.suptitle(f"Vertex dz — {_leg_titles[leg_name]}  "
                        r"— $d_z$ (test sample, by true jet flavour)", fontweight="bold")

        ax_dz_cmp, ax_dz_dist = axes_dz

        if has_truth.any():
            _dz_all   = np.concatenate([dz_true[has_truth, leg], dz_pred[has_truth, leg]])
            _dz_edge  = np.percentile(np.abs(_dz_all[np.isfinite(_dz_all)]), 99)
            _dz_bins  = np.linspace(-_dz_edge, _dz_edge, 61)
            ax_dz_cmp.hist(dz_true[has_truth, leg], bins=_dz_bins, histtype="step",
                           color=COLOURS[owner_name], linestyle="-", linewidth=1.5,
                           density=True, label=f"{owner_name} (truth)")
            ax_dz_cmp.hist(dz_pred[has_truth, leg], bins=_dz_bins, histtype="step",
                           color=COLOURS[owner_name], linestyle="--", linewidth=1.5,
                           density=True, label=f"{owner_name} (pred)")
        else:
            ax_dz_cmp.text(0.5, 0.5, "no jets with a matching truth hadron", ha="center",
                           va="center", transform=ax_dz_cmp.transAxes, fontsize=9, color="grey")
        ax_dz_cmp.set_xlabel(r"$d_z$ [mm]"); ax_dz_cmp.set_ylabel("Density")
        ax_dz_cmp.set_title(r"Truth (solid) vs. predicted (dashed) — truth hadron exists")
        ax_dz_cmp.legend(fontsize=6); ax_dz_cmp.grid(True, linestyle="--", alpha=0.3)

        if no_truth.any():
            _dz_clip  = np.percentile(np.abs(dz_pred[no_truth, leg]), 99)
            for cls_idx, cls_name in enumerate(JET_CLASS_NAMES):
                m = no_truth & (all_true == cls_idx)
                if not m.any():
                    continue
                ax_dz_dist.hist(dz_pred[m, leg], bins=60, range=(-_dz_clip, _dz_clip),
                                histtype="step", color=COLOURS[cls_name], label=cls_name,
                                linewidth=1.5, density=True)
        else:
            ax_dz_dist.text(0.5, 0.5, "every jet has a matching truth hadron", ha="center",
                            va="center", transform=ax_dz_dist.transAxes, fontsize=9, color="grey")
        ax_dz_dist.set_xlabel(r"Predicted $d_z$ [mm]"); ax_dz_dist.set_ylabel("Density")
        ax_dz_dist.set_title(r"Fitted-value distribution (no truth hadron)")
        ax_dz_dist.legend(fontsize=7); ax_dz_dist.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOT_DIR + f"vertex_fit_{leg_name}_dz.png", dpi=150, bbox_inches="tight")
        print(f"Saved vertex_fit_{leg_name}_dz.png")

# ── plot: track-to-vertex assignment (Stage 2) ──────────────────────────
print("\n=== Track-to-vertex assignment efficiency ===")
_leg_owner_cls = {
    lname: JET_CLASS_NAMES.index(
        {int(fl): name for name, fl in
         {JET_CLASS_NAMES[i]: int(lbl)
          for lbl, i in FLAVOUR_TO_LABEL.items()}.items()}[VERTEX_TARGETS[lname]["flavour"]]
    )
    for lname in VERTEX_LEG_NAMES
}
for leg in range(N_VERTEX_LEGS):
    leg_name  = VERTEX_LEG_NAMES[leg]
    owner_cls = _leg_owner_cls[leg_name]
    owner_name = JET_CLASS_NAMES[owner_cls]
    leg_origin_names = VERTEX_LEGS[leg_name]
    if isinstance(leg_origin_names, str):
        leg_origin_names = [leg_origin_names]
    leg_origin_ids = [ORIGIN_CLASS_NAMES.index(c) for c in leg_origin_names]

    owner_mask = (all_true == owner_cls)
    if not owner_mask.any():
        print(f"  {leg_name}: no {owner_name} jets in test set")
        continue

    jet_vtx_w = vtx_weight[owner_mask, :, leg]
    jet_ref   = refine[owner_mask, :, leg]
    jet_orig  = origin_full[owner_mask]
    jet_mask  = mask_full[owner_mask]

    match_mask = np.isin(jet_orig, leg_origin_ids) & jet_mask
    other_mask = (~np.isin(jet_orig, leg_origin_ids)) & jet_mask & (jet_orig >= 0)

    match_weights = jet_vtx_w[match_mask]
    other_weights = jet_vtx_w[other_mask]

    if len(match_weights):
        print(f"  {leg_name}  vtx_w match:  "
              f"min={match_weights.min():.4f}  mean={match_weights.mean():.4f}  "
              f"P25={np.percentile(match_weights,25):.4f}  "
              f"P50={np.percentile(match_weights,50):.4f}  "
              f"P75={np.percentile(match_weights,75):.4f}  "
              f"max={match_weights.max():.4f}")
    if len(other_weights):
        print(f"  {leg_name}  vtx_w other:  "
              f"min={other_weights.min():.4f}  mean={other_weights.mean():.4f}  "
              f"P50={np.percentile(other_weights,50):.4f}  "
              f"P90={np.percentile(other_weights,90):.4f}  "
              f"max={other_weights.max():.4f}")

    _match_ref = jet_ref[match_mask]
    _other_ref = jet_ref[other_mask]
    if len(_match_ref):
        print(f"  {leg_name}  refine match:  "
              f"min={_match_ref.min():.4f}  mean={_match_ref.mean():.4f}  "
              f"P50={np.percentile(_match_ref,50):.4f}  "
              f"max={_match_ref.max():.4f}")
    if len(_other_ref):
        print(f"  {leg_name}  refine other:  "
              f"min={_other_ref.min():.4f}  mean={_other_ref.mean():.4f}  "
              f"P50={np.percentile(_other_ref,50):.4f}  "
              f"max={_other_ref.max():.4f}")

    _parts = [w for w in [match_weights, other_weights] if len(w)]
    all_w = np.concatenate(_parts) if _parts else np.array([])
    if len(all_w):
        print(f"  {leg_name}  vtx_w range=[{all_w.min():.5f}, {all_w.max():.5f}]  "
              f"mean={all_w.mean():.5f}  median={np.median(all_w):.5f}  "
              f"P99={np.percentile(all_w, 99):.5f}")

    for thr in [0.5, 0.8]:
        eff = (match_weights > thr).mean() if len(match_weights) > 0 else 0.0
        fp  = (other_weights > thr).mean() if len(other_weights) > 0 else 0.0
        print(f"  {leg_name} (thr>{thr:.1f}): assignment={eff:.3f}  "
              f"false-positive={fp:.4f}  n_match={len(match_weights)}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.suptitle(f"Track-to-vertex weight — {leg_name}  ({owner_name} jets only)",
                 fontweight="bold")
    if len(match_weights) > 0:
        ax.hist(match_weights, bins=40, range=(0, 1), histtype="step",
                color=COLOURS[owner_name], linewidth=1.5, density=True,
                label=f"True {leg_name.replace('_','-')} origin  (n={len(match_weights)})")
    if len(other_weights) > 0:
        ax.hist(other_weights, bins=40, range=(0, 1), histtype="step",
                color="grey", linestyle="--", linewidth=1.5, density=True,
                label=f"Other origin  (n={len(other_weights)})")
    ax.set_xlabel("vtx_weight"); ax.set_ylabel("Density")
    ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR + f"track_vtx_assignment_{leg_name}.png",
                dpi=150, bbox_inches="tight")
    print(f"Saved track_vtx_assignment_{leg_name}.png")

# ── plot: output probabilities (Stage 3) ──────────────────────────────
fig, axes = plt.subplots(1, N_JET_CLASSES, figsize=(5 * N_JET_CLASSES, 4))
fig.suptitle("Jet output probabilities by true flavour (test sample)", fontweight="bold")
for cls_idx, cls_name in enumerate(JET_CLASS_NAMES):
    ax = axes[cls_idx]
    for true_idx, true_name in enumerate(JET_CLASS_NAMES):
        ax.hist(all_probs[all_true == true_idx, cls_idx], bins=50, range=(0, 1),
                histtype="step", label=true_name, color=COLOURS[true_name],
                linewidth=1.5, density=True)
    ax.set_title(f"P({cls_name})"); ax.set_xlabel("Probability")
    ax.set_ylabel("Density"); ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(PLOT_DIR + "output_probs.png", dpi=150, bbox_inches="tight")
print("Saved output_probs.png")

# ── plot: b-tagging discriminant & ROC ─────────────────────────────────
b_idx = JET_CLASS_NAMES.index("b-jet")
bkg_idxs = [i for i in range(N_JET_CLASSES) if i != b_idx]
w = np.array([DISC_BKG_WEIGHTS[JET_CLASS_NAMES[i]] for i in bkg_idxs])
w = w / w.sum()

pb   = all_probs[:, b_idx]
pbkg = all_probs[:, bkg_idxs] @ w
disc = np.log(pb / (pbkg + 1e-10))
finite = np.isfinite(disc)
clip = np.percentile(np.abs(disc[finite]), 99)

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(r"$\log(p_b\,/\,\sum_i w_i\,p_i)$ — test sample", fontweight="bold")
for true_idx, true_name in enumerate(JET_CLASS_NAMES):
    mfin = (all_true == true_idx) & finite
    ax.hist(disc[mfin], bins=80, range=(-clip, clip), histtype="step",
            label=true_name, color=COLOURS[true_name], linewidth=1.5, density=True)
ax.set_xlabel(r"$\log(p_b\,/\,\sum_i w_i\,p_i)$"); ax.set_ylabel("Density"); ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR + "discriminant.png", dpi=150, bbox_inches="tight")
print("Saved discriminant.png")

fig, axes = plt.subplots(1, len(bkg_idxs), figsize=(6 * len(bkg_idxs), 5))
fig.suptitle(r"ROC curves — $\log(p_b\,/\,\sum_i w_i\,p_i)$", fontweight="bold")
for ax, bkg_idx in zip(np.atleast_1d(axes), bkg_idxs):
    bkg_name = JET_CLASS_NAMES[bkg_idx]
    mroc   = (all_true == b_idx) | (all_true == bkg_idx)
    labels = (all_true[mroc] == b_idx).astype(int)
    score  = disc[mroc]
    fin    = np.isfinite(score)
    fpr, tpr, _ = roc_curve(labels[fin], score[fin])
    ax.plot(tpr, fpr, color="#1f77b4", linewidth=1.5, label=f"AUC={auc(fpr,tpr):.3f}")
    ax.set_xlabel("b-jet efficiency (TPR)")
    ax.set_ylabel(f"{bkg_name} rate (FPR)")
    ax.set_title(f"b vs {bkg_name}"); ax.set_yscale("log")
    ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(PLOT_DIR + "roc.png", dpi=150, bbox_inches="tight")
print("Saved roc.png")

print(f"\nAll outputs saved to {PLOT_DIR}")
