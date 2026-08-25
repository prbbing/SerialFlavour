"""
Data loading: HDF5 I/O, track pre-processing, caching, PyTorch Dataset.

All heavy data work happens in load_tracks(), which reads the raw HDF5
records, applies quality cuts, selects top-K tracks by impact-parameter
significance, drops jets with zero surviving tracks (to prevent NaN
propagation in attention), builds per-leg vertex truth targets, and
optionally constructs pair-compatibility targets (for parallel models).
Results are cached as .npz files keyed by index-hash for instant reload.
"""
import hashlib
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import dataloader_generator, seed_dataloader_worker
from .data_split import (
    DataSplits,
    JET_EVENT_FIELD,
    SPLIT_VERSION,
    split_indices_by_event,
)


def _cache_key(idx, track_fields, cache_dir, data_seed=42):
    """Hash indices, fields and data seed into a deterministic cache name."""
    fields_bytes = ",".join(track_fields).encode()
    seed_bytes = (
        f"\0data_seed={data_seed}\0split_version={SPLIT_VERSION}").encode()
    h = hashlib.md5(
        np.asarray(idx).tobytes() + fields_bytes + seed_bytes
    ).hexdigest()[:12]
    return os.path.join(cache_dir, f"tracks_{h}_nom.npz")


def load_tracks(path, idx, flavour_to_label, track_fields,
                vertex_leg_names, vertex_targets, top_k, cache_dir,
                data_seed=42):
    """Load and pre-process tracks for the jets specified by *idx*.

    Processing pipeline:
      1. HDF5 read: jet flavour, track features, truth-hadron vertices.
      2. Quality cut:  valid & |d0| < 3.5 mm.
      3. Top-K selection: rank by |lifetimeSignedD0Significance| descending.
      4. Zero-pad invalid tracks, set origin = -1 for padded positions.
      5. Drop jets with zero surviving tracks (avoid NaN in attention).
      6. Build per-leg truth: Lxy / dz / validity from truth_hadrons.
      7. (optional) Build pair target: (K,K) binary matrix —
         1 if tracks i,j share the same ftagTruthVertexIndex.

    Returns:
        dict with keys:
            X          (N, K, F) float32  — track features
            mask       (N, K)    bool_    — track validity mask
            y          (N,)      int64    — jet flavour label (0=b,1=c,2=light)
            origin     (N, K)    int64    — track origin label (-1 = padding)
            vtx_lxy    (N, L)    float32  — truth Lxy per vertex leg
            vtx_dz     (N, L)    float32  — truth dz  per vertex leg
            vtx_valid  (N, L)    bool_    — truth validity per leg
            pair_target (N, K, K) float32 — pair-compatibility labels
    """
    cp = _cache_key(idx, track_fields, cache_dir, data_seed)
    if os.path.exists(cp):
        d = np.load(cp)
        return {k: d[k] for k in d.files}

    # ── raw HDF5 read ─────────────────────────────────────────────────────
    with h5py.File(path, "r") as f:
        # jet-level
        flavour_id = f["jets"]["HadronConeExclTruthLabelID"][idx]
        keep_jet   = np.isin(flavour_id, list(flavour_to_label.keys()))
        fidx       = idx[keep_jet]          # indices of kept jets

        # track-level
        valid  = f["tracks"]["valid"][fidx]
        d0     = f["tracks"]["lifetimeSignedD0"][fidx].astype(np.float32)
        ip2d   = f["tracks"]["lifetimeSignedD0Significance"][fidx].astype(np.float32)
        origin = f["tracks"]["ftagTruthOriginLabel"][fidx].astype(np.int8)
        arrs   = {fld: f["tracks"][fld][fidx].astype(np.float32) for fld in track_fields}

        # truth hadrons (decay-vertex targets)
        th_valid_all = f["truth_hadrons"]["valid"][fidx]
        th_lxy_all   = f["truth_hadrons"]["Lxy"][fidx].astype(np.float32)
        th_dz_all    = f["truth_hadrons"]["decayVertexZ"][fidx].astype(np.float32)
        flavour_id_keep = flavour_id[keep_jet]

        # ----------------------------------------------------------------
        # model: parallel_origin_vertex_jet — per-track vertex index for
        #          pair-compatibility targets
        # ----------------------------------------------------------------
        vtx_idx = f["tracks"]["ftagTruthVertexIndex"][fidx].astype(np.int64)
        # ----------------------------------------------------------------

    # ── quality cut --------------------------------------------------------
    keep = valid & (np.abs(d0) < 3.5)

    # ── top-K selection by |d0 significance| -------------------------------
    sort_key = ip2d.copy()
    sort_key[~keep] = -np.inf          # unqualified tracks pushed to the end
    order = np.argsort(-sort_key, axis=1)

    feats       = np.stack([arrs[fld] for fld in track_fields], axis=-1)
    topk_idx    = order[:, :top_k]
    rows        = np.arange(len(fidx))[:, None]
    topk_feat   = feats[rows, topk_idx]
    topk_valid  = keep[rows, topk_idx]
    topk_feat   = np.where(topk_valid[:, :, None], topk_feat, 0.0).astype(np.float32)
    topk_origin = origin[rows, topk_idx]
    topk_origin = np.where(topk_valid, topk_origin, -1).astype(np.int64)

    # ----------------------------------------------------------------
    # model: parallel_origin_vertex_jet
    # ----------------------------------------------------------------
    topk_vidx    = vtx_idx[rows, topk_idx]
    topk_vidx    = np.where(topk_valid, topk_vidx, -1).astype(np.int64)
    # ----------------------------------------------------------------

    # ── drop jets with zero surviving tracks (NaN guard) ─────────────────
    # A fully padded row would cause softmax over all -inf → NaN.
    # NaN * 0 == NaN, so validity masking does not clean it — must drop.
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

    # ----------------------------------------------------------------
    # model: parallel_origin_vertex_jet
    # ----------------------------------------------------------------
    topk_vidx       = topk_vidx[has_track]
    # ----------------------------------------------------------------

    labels = np.array([flavour_to_label[v] for v in flavour_id_keep], dtype=np.int64)

    # ── per-leg vertex truth -----------------------------------------------
    # truth_hadrons is decay-chain ordered:
    #   b-jet: slot 0 = primary B-hadron  → b-vertex target
    #          slot 1 = cascade c-hadron  → cascade-vertex target
    #   c-jet: slot 0 = primary C-hadron  → c-vertex target

    def _leg(valid_mask, slot):
        """Extract Lxy, dz, validity for a hadron slot."""
        v    = valid_mask & th_valid_all[:, slot]
        lxy  = np.where(v, th_lxy_all[:, slot], 0.0).astype(np.float32)
        dz   = np.where(v, th_dz_all[:, slot],  0.0).astype(np.float32)
        return (np.nan_to_num(lxy, nan=0.0),
                np.nan_to_num(dz,  nan=0.0),
                v.astype(np.bool_))

    _lxy_cols, _dz_cols, _valid_cols = [], [], []
    for _lname in vertex_leg_names:
        _tgt      = vertex_targets[_lname]
        _jet_mask = (flavour_id_keep == _tgt["flavour"])
        _lxy_col, _dz_col, _valid_col = _leg(_jet_mask, _tgt["slot"])
        _lxy_cols.append(_lxy_col)
        _dz_cols.append(_dz_col)
        _valid_cols.append(_valid_col)

    vtx_lxy   = np.stack(_lxy_cols,   axis=-1)   # (N, L)
    vtx_dz    = np.stack(_dz_cols,    axis=-1)   # (N, L)
    vtx_valid = np.stack(_valid_cols, axis=-1)   # (N, L)

    # ----------------------------------------------------------------
    # model: parallel_origin_vertex_jet — pair target (N, K, K)
    #   pair_target[i, p, q] = 1  if tracks p,q share same vertex index
    #                        = 0  if both valid but different indices
    #                        = -1 if either track is padded/invalid
    # ----------------------------------------------------------------
    valid_pair  = topk_valid[:, :, None] & topk_valid[:, None, :]
    same_vertex = topk_vidx[:, :, None] == topk_vidx[:, None, :]
    pair_target = np.where(valid_pair, same_vertex.astype(np.float32), -1.0)
    # ----------------------------------------------------------------

    # cache and return
    np.savez(cp, X=topk_feat, mask=topk_valid, y=labels, origin=topk_origin,
             vtx_lxy=vtx_lxy, vtx_dz=vtx_dz, vtx_valid=vtx_valid,
             pair_target=pair_target)
    return {"X": topk_feat, "mask": topk_valid, "y": labels, "origin": topk_origin,
            "vtx_lxy": vtx_lxy, "vtx_dz": vtx_dz, "vtx_valid": vtx_valid,
            "pair_target": pair_target}


# ===========================================================================
# JetDataset — wraps a dict of numpy arrays as a torch Dataset.
# __getitem__ returns an 8-tuple; the extra pair_target (position 7) is
# ignored by models that do not emit pair_logits.
# ===========================================================================
class JetDataset(Dataset):
    def __init__(self, d):
        self.X         = torch.from_numpy(d["X"])
        self.mask      = torch.from_numpy(d["mask"])
        self.y         = torch.from_numpy(d["y"])
        self.origin    = torch.from_numpy(d["origin"])
        self.vtx_lxy   = torch.from_numpy(d["vtx_lxy"])
        self.vtx_dz    = torch.from_numpy(d["vtx_dz"])
        self.vtx_valid = torch.from_numpy(d["vtx_valid"])

        # ----------------------------------------------------------------
        # model: parallel_origin_vertex_jet — fallback if absent from cache
        # ----------------------------------------------------------------
        self.pair_target = torch.from_numpy(d.get("pair_target",
            np.full((d["mask"].shape[0], d["mask"].shape[1], d["mask"].shape[1]),
                    -1.0, dtype=np.float32)))
        # ----------------------------------------------------------------

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (self.X[i], self.mask[i], self.y[i], self.origin[i],
                self.vtx_lxy[i], self.vtx_dz[i], self.vtx_valid[i],
                self.pair_target[i])


# ===========================================================================
# create_dataloaders — event-disjoint train/validation/test construction.
# ===========================================================================
def create_dataloaders(config, device):
    """Build event-disjoint train, validation and independent test loaders."""
    os.makedirs(config.cache_dir, exist_ok=True)
    print("  [1/4] Reading flavour and event metadata from HDF5 ...")
    with h5py.File(config.train_file, "r") as f:
        jets = f["jets"]
        if isinstance(jets, h5py.Group):
            has_event_number = JET_EVENT_FIELD in jets
        else:
            has_event_number = JET_EVENT_FIELD in (jets.dtype.names or ())
        if not has_event_number:
            raise KeyError(
                f"jets is missing required field: {JET_EVENT_FIELD}")
        all_flavours = jets["HadronConeExclTruthLabelID"][:]
        event_numbers = jets[JET_EVENT_FIELD][:]
    split = split_indices_by_event(config, all_flavours, event_numbers)

    # ── load tracks into memory ──────────────────────────────────────────
    print("  [2/4] Loading training tracks ...")
    train_data = load_tracks(config.train_file, split.train,
                             config.flavour_to_label, config.track_fields,
                             config.vertex_leg_names, config.vertex_targets,
                             config.top_k, config.cache_dir, config.data_seed)
    print("  [3/4] Loading validation tracks ...")
    validation_data = load_tracks(config.train_file, split.validation,
                             config.flavour_to_label, config.track_fields,
                             config.vertex_leg_names, config.vertex_targets,
                             config.top_k, config.cache_dir, config.data_seed)
    print("  [4/4] Loading test tracks ...")
    test_data = load_tracks(config.train_file, split.test,
                             config.flavour_to_label, config.track_fields,
                             config.vertex_leg_names, config.vertex_targets,
                             config.top_k, config.cache_dir, config.data_seed)
    y_train = train_data["y"]
    y_validation = validation_data["y"]
    y_test = test_data["y"]

    print("Train — " + "  ".join(f"{name}:{(y_train==i).sum():,}"
                                  for i, name in enumerate(config.jet_class_names)))
    print("Validation — " + "  ".join(
        f"{name}:{(y_validation==i).sum():,}"
        for i, name in enumerate(config.jet_class_names)))
    print("Test  — " + "  ".join(f"{name}:{(y_test==i).sum():,}"
                                  for i, name in enumerate(config.jet_class_names)))

    # ── DataLoaders ──────────────────────────────────────────────────────
    _pin = str(device).startswith("cuda")
    _pw  = config.num_workers > 0  # persistent_workers avoids respawning
    train_loader = DataLoader(
        JetDataset(train_data), batch_size=config.batch_size, shuffle=True,
        pin_memory=_pin, num_workers=config.num_workers, persistent_workers=_pw,
        generator=dataloader_generator(config.seed),
        worker_init_fn=seed_dataloader_worker)
    validation_loader = DataLoader(
        JetDataset(validation_data), batch_size=config.batch_size,
        pin_memory=_pin, num_workers=config.num_workers, persistent_workers=_pw,
        generator=dataloader_generator(config.seed + 1),
        worker_init_fn=seed_dataloader_worker)
    test_loader = DataLoader(
        JetDataset(test_data), batch_size=config.batch_size,
        pin_memory=_pin, num_workers=config.num_workers, persistent_workers=_pw,
        generator=dataloader_generator(config.seed + 2),
        worker_init_fn=seed_dataloader_worker)

    summary = dict(split.summary)
    summary["retained_jets_after_track_filter"] = {
        "train": int(len(y_train)),
        "validation": int(len(y_validation)),
        "test": int(len(y_test)),
    }
    summary["dropped_jets_no_valid_tracks"] = {
        name: summary["selected_jets"][name] - retained
        for name, retained in summary[
            "retained_jets_after_track_filter"].items()
    }
    summary["retained_class_counts"] = {
        split_name: {
            name: int((labels == index).sum())
            for index, name in enumerate(config.jet_class_names)
        }
        for split_name, labels in (
            ("train", y_train), ("validation", y_validation),
            ("test", y_test))
    }
    return DataSplits(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        train_data=train_data,
        validation_data=validation_data,
        test_data=test_data,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        summary=summary)
