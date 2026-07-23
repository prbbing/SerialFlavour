"""Accelerated, output-compatible data preparation for SerialFlavour.

This module keeps the cache contract of :mod:`src.data` while avoiding the
expensive ``compound_dataset[field][indices]`` access pattern.  Real OpenData
files store ``jets``, ``tracks`` and ``truth_hadrons`` as gzip-compressed
compound datasets.  Reading one field at a time scans and decompresses the
same chunks repeatedly.  Here, required fields are read together in bounded,
chunk-aligned blocks and each block is fully pre-processed before the next one
is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections.abc import Iterable, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import dataloader_generator, seed_dataloader_worker


_JET_FLAVOUR_FIELD = "HadronConeExclTruthLabelID"
_TRACK_AUX_FIELDS = (
    "valid",
    "lifetimeSignedD0",
    "lifetimeSignedD0Significance",
    "ftagTruthOriginLabel",
    "ftagTruthVertexIndex",
)
_TRUTH_FIELDS = ("valid", "Lxy", "decayVertexZ")


def _cache_key(idx, track_fields, cache_dir, data_seed=42):
    """Return exactly the same seeded cache path as ``src.data._cache_key``."""
    fields_bytes = ",".join(track_fields).encode()
    seed_bytes = f"\0data_seed={data_seed}".encode()
    h = hashlib.md5(
        np.asarray(idx).tobytes() + fields_bytes + seed_bytes
    ).hexdigest()[:12]
    return os.path.join(cache_dir, f"tracks_{h}_nom.npz")


def _unique(items: Iterable[str]) -> list[str]:
    """Deduplicate field names without changing their first-seen order."""
    return list(dict.fromkeys(items))


def _field_names(obj: h5py.Group | h5py.Dataset) -> set[str]:
    if isinstance(obj, h5py.Group):
        return set(obj.keys())
    return set(obj.dtype.names or ())


def _require_fields(obj: h5py.Group | h5py.Dataset,
                    names: Sequence[str], object_name: str) -> None:
    missing = set(names) - _field_names(obj)
    if missing:
        raise KeyError(f"{object_name} is missing field(s): {sorted(missing)}")


def _row_count(obj: h5py.Group | h5py.Dataset, field: str) -> int:
    return len(obj[field]) if isinstance(obj, h5py.Group) else len(obj)


def _read_fields(obj: h5py.Group | h5py.Dataset,
                 names: Sequence[str], start: int, stop: int) -> dict[str, np.ndarray]:
    """Read several fields over one contiguous row interval.

    A compound dataset is accessed through one multi-field HDF5 selection, so
    each compressed chunk is decompressed once per block.  Group-based test
    files retain their normal one-dataset-per-field behaviour.
    """
    names = _unique(names)
    if isinstance(obj, h5py.Group):
        return {name: obj[name][start:stop] for name in names}

    block = obj.fields(names)[start:stop]
    return {name: block[name] for name in names}


def _normalise_indices(idx, n_rows: int) -> np.ndarray:
    idx = np.asarray(idx)
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
        raise TypeError("idx must be a one-dimensional integer array")
    if idx.size and (idx[0] < 0 or idx[-1] >= n_rows):
        raise IndexError(f"idx values must lie in [0, {n_rows})")
    if idx.size > 1 and np.any(idx[1:] <= idx[:-1]):
        raise ValueError("idx must be strictly increasing, as required by the HDF5 loader")
    return idx


def _block_slices(idx: np.ndarray, n_rows: int, chunk_rows: int,
                  block_rows: int | None):
    """Yield chunk-aligned source ranges and positions within ``idx``."""
    if not idx.size:
        return

    if block_rows is None:
        block_rows = chunk_rows * 16
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    block_rows = max(chunk_rows, (block_rows // chunk_rows) * chunk_rows)

    pos = 0
    while pos < len(idx):
        logical_end = (int(idx[pos]) // block_rows + 1) * block_rows
        end_pos = int(np.searchsorted(idx, logical_end, side="left"))
        first = int(idx[pos])
        last = int(idx[end_pos - 1])
        start = first // chunk_rows * chunk_rows
        stop = min(((last + 1 + chunk_rows - 1) // chunk_rows) * chunk_rows, n_rows)
        yield pos, end_pos, start, stop
        pos = end_pos


def _allocate_outputs(n: int, top_k: int, n_fields: int,
                      n_legs: int) -> dict[str, np.ndarray]:
    return {
        "X": np.empty((n, top_k, n_fields), dtype=np.float32),
        "mask": np.empty((n, top_k), dtype=np.bool_),
        "y": np.empty(n, dtype=np.int64),
        "origin": np.empty((n, top_k), dtype=np.int64),
        "vtx_lxy": np.empty((n, n_legs), dtype=np.float32),
        "vtx_dz": np.empty((n, n_legs), dtype=np.float32),
        "vtx_valid": np.empty((n, n_legs), dtype=np.bool_),
        # Kept unconditionally to preserve src.data's cache/output contract.
        "pair_target": np.empty((n, top_k, top_k), dtype=np.float32),
    }


def _save_cache_atomic(path: str, arrays: dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=".tracks_", suffix=".npz", dir=os.path.dirname(os.path.abspath(path)),
        delete=False)
    tmp_path = handle.name
    handle.close()
    try:
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def load_tracks(path, idx, flavour_to_label, track_fields,
                vertex_leg_names, vertex_targets, top_k, cache_dir,
                *, force: bool = False, block_rows: int | None = None,
                progress: bool = False, include_pair_target: bool = True,
                data_seed: int = 42):
    """Accelerated drop-in equivalent of :func:`src.data.load_tracks`.

    Output keys, shapes, dtypes, row order, top-K ordering, padding semantics,
    pair targets and cache filenames match the original implementation.
    ``idx`` must be strictly increasing, which is how ``create_dataloaders``
    constructs both train and test indices. When ``include_pair_target`` is
    false, the dense member is omitted from the returned mapping but remains
    present in newly written caches for compatibility with pair models.
    """
    idx_for_key = np.asarray(idx)
    cp = _cache_key(idx_for_key, track_fields, cache_dir, data_seed)
    if os.path.exists(cp) and not force:
        with np.load(cp) as cached:
            keys = cached.files
            if not include_pair_target:
                keys = [key for key in keys if key != "pair_target"]
            return {key: cached[key] for key in keys}

    track_fields = list(track_fields)
    vertex_leg_names = list(vertex_leg_names)
    track_read_fields = _unique((*_TRACK_AUX_FIELDS, *track_fields))

    with h5py.File(path, "r") as f:
        jets = f["jets"]
        tracks = f["tracks"]
        truth_hadrons = f["truth_hadrons"]
        n_rows = _row_count(jets, _JET_FLAVOUR_FIELD)
        if (_row_count(tracks, _TRACK_AUX_FIELDS[0]) != n_rows
                or _row_count(truth_hadrons, _TRUTH_FIELDS[0]) != n_rows):
            raise ValueError("jets, tracks and truth_hadrons must have the same first dimension")

        idx = _normalise_indices(idx_for_key, n_rows)
        _require_fields(jets, [_JET_FLAVOUR_FIELD], "jets")
        _require_fields(tracks, track_read_fields, "tracks")
        _require_fields(truth_hadrons, _TRUTH_FIELDS, "truth_hadrons")

        source_track_count = tracks[_TRACK_AUX_FIELDS[0]].shape[1] \
            if isinstance(tracks, h5py.Group) else tracks.shape[1]
        # Preserve NumPy's ``order[:, :top_k]`` behaviour when top_k is larger
        # than the stored track dimension.
        output_k = min(top_k, source_track_count)

        track_chunks = (tracks[_TRACK_AUX_FIELDS[0]].chunks
                        if isinstance(tracks, h5py.Group) else tracks.chunks)
        chunk_rows = track_chunks[0] if track_chunks else min(2048, max(n_rows, 1))
        outputs = _allocate_outputs(
            len(idx), output_k, len(track_fields), len(vertex_leg_names))
        out_pos = 0
        dropped = 0
        last_percent = -1

        for input_start, input_stop, start, stop in _block_slices(
                idx, n_rows, chunk_rows, block_rows):
            selected = idx[input_start:input_stop]
            local_rows = selected.astype(np.int64, copy=False) - start

            jet_block = _read_fields(jets, [_JET_FLAVOUR_FIELD], start, stop)
            flavour_id = jet_block[_JET_FLAVOUR_FIELD][local_rows]
            keep_flavour = np.isin(flavour_id, list(flavour_to_label.keys()))
            if not keep_flavour.any():
                continue
            local_rows = local_rows[keep_flavour]
            flavour_id = flavour_id[keep_flavour]

            track_block = _read_fields(tracks, track_read_fields, start, stop)
            truth_block = _read_fields(truth_hadrons, _TRUTH_FIELDS, start, stop)

            valid = track_block["valid"][local_rows]
            d0 = track_block["lifetimeSignedD0"][local_rows].astype(np.float32)
            ip2d = track_block["lifetimeSignedD0Significance"][local_rows].astype(np.float32)
            origin = track_block["ftagTruthOriginLabel"][local_rows].astype(np.int8)
            vtx_idx = track_block["ftagTruthVertexIndex"][local_rows].astype(np.int64)

            keep = valid & (np.abs(d0) < 3.5)
            sort_key = ip2d.copy()
            sort_key[~keep] = -np.inf
            order = np.argsort(-sort_key, axis=1)
            topk_idx = order[:, :top_k]
            rows = np.arange(len(local_rows))[:, None]

            feats = np.stack([
                track_block[field][local_rows].astype(np.float32)
                for field in track_fields
            ], axis=-1)
            topk_feat = feats[rows, topk_idx]
            topk_valid = keep[rows, topk_idx]
            topk_feat = np.where(
                topk_valid[:, :, None], topk_feat, 0.0).astype(np.float32)
            topk_origin = origin[rows, topk_idx]
            topk_origin = np.where(topk_valid, topk_origin, -1).astype(np.int64)
            topk_vidx = vtx_idx[rows, topk_idx]
            topk_vidx = np.where(topk_valid, topk_vidx, -1).astype(np.int64)

            has_track = topk_valid.any(axis=1)
            dropped += int((~has_track).sum())
            if not has_track.any():
                continue

            topk_feat = topk_feat[has_track]
            topk_valid = topk_valid[has_track]
            topk_origin = topk_origin[has_track]
            topk_vidx = topk_vidx[has_track]
            flavour_id = flavour_id[has_track]
            kept_local_rows = local_rows[has_track]
            n_kept = len(flavour_id)
            dest = slice(out_pos, out_pos + n_kept)

            outputs["X"][dest] = topk_feat
            outputs["mask"][dest] = topk_valid
            outputs["origin"][dest] = topk_origin
            outputs["y"][dest] = np.array(
                [flavour_to_label[value] for value in flavour_id], dtype=np.int64)

            th_valid = truth_block["valid"][kept_local_rows]
            th_lxy = truth_block["Lxy"][kept_local_rows].astype(np.float32)
            th_dz = truth_block["decayVertexZ"][kept_local_rows].astype(np.float32)
            for leg_idx, leg_name in enumerate(vertex_leg_names):
                target = vertex_targets[leg_name]
                valid_leg = ((flavour_id == target["flavour"])
                             & th_valid[:, target["slot"]])
                lxy = np.where(
                    valid_leg, th_lxy[:, target["slot"]], 0.0).astype(np.float32)
                dz = np.where(
                    valid_leg, th_dz[:, target["slot"]], 0.0).astype(np.float32)
                outputs["vtx_lxy"][dest, leg_idx] = np.nan_to_num(lxy, nan=0.0)
                outputs["vtx_dz"][dest, leg_idx] = np.nan_to_num(dz, nan=0.0)
                outputs["vtx_valid"][dest, leg_idx] = valid_leg.astype(np.bool_)

            valid_pair = topk_valid[:, :, None] & topk_valid[:, None, :]
            same_vertex = topk_vidx[:, :, None] == topk_vidx[:, None, :]
            outputs["pair_target"][dest] = np.where(
                valid_pair, same_vertex.astype(np.float32), -1.0)
            out_pos += n_kept

            if progress and len(idx):
                percent = int(100 * input_stop / len(idx))
                if percent // 10 != last_percent // 10 or input_stop == len(idx):
                    print(f"    processed {input_stop:,}/{len(idx):,} selected jets ({percent}%)")
                    last_percent = percent

    if dropped:
        print(f"  load_tracks: dropping {dropped} jet(s) with no surviving tracks")

    result = {key: value[:out_pos] for key, value in outputs.items()}
    _save_cache_atomic(cp, result)
    if not include_pair_target:
        # The on-disk cache remains output-compatible with pair models, but
        # staged callers must not retain the dense target after cache creation.
        result.pop("pair_target", None)
    return result


class JetDataset(Dataset):
    """Same eight-item dataset interface as :class:`src.data.JetDataset`."""

    def __init__(self, data, *, include_pair_target: bool | None = None):
        self.X = torch.from_numpy(data["X"])
        self.mask = torch.from_numpy(data["mask"])
        self.y = torch.from_numpy(data["y"])
        self.origin = torch.from_numpy(data["origin"])
        self.vtx_lxy = torch.from_numpy(data["vtx_lxy"])
        self.vtx_dz = torch.from_numpy(data["vtx_dz"])
        self.vtx_valid = torch.from_numpy(data["vtx_valid"])
        if include_pair_target is None:
            include_pair_target = "pair_target" in data
        if include_pair_target:
            if "pair_target" not in data:
                raise KeyError("pair_target is required but missing from dataset data")
            self.pair_target = torch.from_numpy(data["pair_target"])
            self.empty_pair_target = None
        else:
            self.pair_target = None
            # DataLoader collates one empty vector per sample into shape (B, 0),
            # preserving the existing eight-item batch contract at negligible
            # memory cost. Staged loss branches never inspect its contents.
            self.empty_pair_target = torch.empty(0, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        pair_target = (self.pair_target[index]
                       if self.pair_target is not None
                       else self.empty_pair_target)
        return (self.X[index], self.mask[index], self.y[index], self.origin[index],
                self.vtx_lxy[index], self.vtx_dz[index], self.vtx_valid[index],
                pair_target)


def _load_all_flavours(config):
    os.makedirs(config.cache_dir, exist_ok=True)
    flavour_cache = os.path.join(config.cache_dir, "all_flavours.npy")
    if os.path.exists(flavour_cache):
        print("  [1/3] Loading flavour labels (cached)")
        return np.load(flavour_cache)

    print("  [1/3] Reading flavour labels from HDF5 ...")
    with h5py.File(config.train_file, "r") as f:
        jets = f["jets"]
        _require_fields(jets, [_JET_FLAVOUR_FIELD], "jets")
        if isinstance(jets, h5py.Group):
            all_flavours = jets[_JET_FLAVOUR_FIELD][:]
        else:
            all_flavours = jets.fields(_JET_FLAVOUR_FIELD)[:]
    np.save(flavour_cache, all_flavours)
    print("  [1/3] Cached flavour labels")
    return all_flavours


def _split_indices(config, all_flavours):
    """Reproduce ``src.data.create_dataloaders`` index selection exactly."""
    rng = np.random.default_rng(config.data_seed)
    valid_mask = np.isin(all_flavours, list(config.flavour_to_label.keys()))
    valid_idx = rng.permutation(np.where(valid_mask)[0])
    test_idx = np.sort(valid_idx[-config.n_test:])
    pool_idx = valid_idx[:-config.n_test]
    pool_labels = np.array([
        config.flavour_to_label[value] for value in all_flavours[pool_idx]
    ])
    n_per_class = config.n_train // len(config.jet_class_names)
    train_idx = np.sort(np.concatenate([
        rng.choice(pool_idx[pool_labels == cls],
                   size=min(n_per_class, (pool_labels == cls).sum()),
                   replace=False)
        for cls in range(len(config.jet_class_names))
    ]))
    return train_idx, test_idx


def generate_cache_from_config(config_path=None, *, train_file=None,
                               cache_dir=None, force=False, block_rows=None,
                               progress=True):
    """Generate train/test cache files using a normal project JSON config."""
    from src.config import load_config

    config, _ = load_config(config_path)
    if train_file is not None:
        config.train_file = train_file
    if cache_dir is not None:
        config.cache_dir = cache_dir

    all_flavours = _load_all_flavours(config)
    train_idx, test_idx = _split_indices(config, all_flavours)

    def build_one(indices):
        cache_path = _cache_key(
            indices, config.track_fields, config.cache_dir, config.data_seed)
        if os.path.exists(cache_path) and not force:
            # NPZ members are lazy. Read only the small label array when a
            # cache already exists instead of materialising a multi-GiB cache.
            with np.load(cache_path) as cached:
                return cache_path, len(cached["y"])
        data = load_tracks(
            config.train_file, indices, config.flavour_to_label,
            config.track_fields, config.vertex_leg_names, config.vertex_targets,
            config.top_k, config.cache_dir, force=force, block_rows=block_rows,
            progress=progress, data_seed=config.data_seed)
        return cache_path, len(data["y"])

    print("  [2/3] Building training-track cache ...")
    train_path, n_train = build_one(train_idx)
    print("  [3/3] Building test-track cache ...")
    test_path, n_test = build_one(test_idx)

    paths = {
        "train": train_path,
        "test": test_path,
        "flavours": os.path.join(config.cache_dir, "all_flavours.npy"),
    }
    print(f"  train cache: {paths['train']} ({n_train:,} jets)")
    print(f"  test cache:  {paths['test']} ({n_test:,} jets)")
    return paths


def create_dataloaders(config, device, *, force=False, block_rows=None,
                       progress=False):
    """Build loaders with the same return contract as ``src.data``."""
    from .models import model_requires_pair_target

    pair_target_required = model_requires_pair_target(config.model_type)
    include_pair_target = config.use_pair_target
    if not include_pair_target and pair_target_required:
        raise ValueError(
            f"model_type '{config.model_type}' requires pair_target; set "
            "use_pair_target=true in its config")

    all_flavours = _load_all_flavours(config)
    train_idx, test_idx = _split_indices(config, all_flavours)
    print("  [2/3] Loading training tracks ...")
    train_data = load_tracks(
        config.train_file, train_idx, config.flavour_to_label,
        config.track_fields, config.vertex_leg_names, config.vertex_targets,
        config.top_k, config.cache_dir, force=force, block_rows=block_rows,
        progress=progress, include_pair_target=include_pair_target,
        data_seed=config.data_seed)
    print("  [3/3] Loading test tracks ...")
    test_data = load_tracks(
        config.train_file, test_idx, config.flavour_to_label,
        config.track_fields, config.vertex_leg_names, config.vertex_targets,
        config.top_k, config.cache_dir, force=force, block_rows=block_rows,
        progress=progress, include_pair_target=include_pair_target,
        data_seed=config.data_seed)
    y_train, y_test = train_data["y"], test_data["y"]

    pin_memory = str(device).startswith("cuda")
    persistent_workers = config.num_workers > 0
    pair_status = "loaded" if include_pair_target else "skipped"
    print(f"  Pair targets: {pair_status} for model_type={config.model_type}")
    train_loader = DataLoader(
        JetDataset(train_data, include_pair_target=include_pair_target),
        batch_size=config.batch_size, shuffle=True,
        pin_memory=pin_memory, num_workers=config.num_workers,
        persistent_workers=persistent_workers,
        generator=dataloader_generator(config.seed),
        worker_init_fn=seed_dataloader_worker)
    val_loader = DataLoader(
        JetDataset(test_data, include_pair_target=include_pair_target),
        batch_size=config.batch_size,
        pin_memory=pin_memory, num_workers=config.num_workers,
        persistent_workers=persistent_workers,
        generator=dataloader_generator(config.seed + 1),
        worker_init_fn=seed_dataloader_worker)
    return train_loader, val_loader, train_data, test_data, y_train, y_test


def cache_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate SerialFlavour track caches with the accelerated loader.")
    parser.add_argument("--config", default=None,
                        help="JSON config path; omitted means src.config defaults.")
    parser.add_argument("--train-file", default=None,
                        help="Optional override for train_file from the config.")
    parser.add_argument("--cache-dir", default=None,
                        help="Optional override for train_cache_dir from the config.")
    parser.add_argument("--block-rows", type=int, default=None,
                        help="Maximum source rows per read block (default: 16 HDF5 chunks).")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild and atomically replace matching track caches.")
    args = parser.parse_args(argv)
    generate_cache_from_config(
        args.config, train_file=args.train_file, cache_dir=args.cache_dir,
        force=args.force, block_rows=args.block_rows, progress=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(cache_cli())
