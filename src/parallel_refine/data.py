"""Top-k Parallel inputs for the experiment-owned A/B/Y splits."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import dataloader_generator, seed_dataloader_worker
from src.data import (
    _block_slices,
    _normalise_indices,
    _read_fields,
    _require_fields,
    _row_count,
    _save_cache_atomic,
    _unique,
)

from src.parallel_refine.splits import load_split_bundle


CACHE_VERSION = "parallel_refine_topk_normalized_v4"
NORMALIZATION_VERSION = "parallel_refine_a_train_standardization_v1"
JET_AUX_FIELDS = (
    "HadronConeExclTruthLabelID", "eventNumber", "pt_btagJes", "eta_btagJes",
)
TRACK_AUX_FIELDS = (
    "valid", "lifetimeSignedD0", "lifetimeSignedD0Significance",
    "z0SinTheta", "ftagTruthOriginLabel",
)


def _source_identity(path: str) -> str:
    stat = os.stat(path)
    return f"{os.path.abspath(path)}:{stat.st_size}:{stat.st_mtime_ns}"


def _cache_path(config, split_name: str, indices: np.ndarray) -> Path:
    payload = b"\0".join((
        _source_identity(config.train_file).encode(),
        split_name.encode(),
        np.asarray(indices, dtype=np.int64).tobytes(),
        ",".join(config.track_fields).encode(),
        ",".join(config.jet_fields).encode(),
        str(config.top_k).encode(),
        json.dumps(config.normalization, sort_keys=True).encode(),
        json.dumps(config.truth_vertex, sort_keys=True).encode(),
        CACHE_VERSION.encode(),
    ))
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return Path(config.cache_dir) / f"{split_name}_{digest}.npz"


def _allocate(
        n: int, k: int, n_track_features: int,
        n_jet_features: int) -> dict[str, np.ndarray]:
    return {
        "X": np.empty((n, k, n_track_features), dtype=np.float32),
        "jet_X": np.empty((n, n_jet_features), dtype=np.float32),
        "mask": np.empty((n, k), dtype=np.bool_),
        "y": np.empty(n, dtype=np.int64),
        "origin": np.empty((n, k), dtype=np.int64),
        "truth_pair": np.empty((n, k, k), dtype=np.float32),
        "jet_pt": np.empty(n, dtype=np.float32),
        "jet_eta": np.empty(n, dtype=np.float32),
        "event_number": np.empty(n, dtype=np.int64),
        "source_index": np.empty(n, dtype=np.int64),
    }


def _normalization_path(config, bundle) -> Path:
    payload = b"\0".join((
        _source_identity(config.train_file).encode(),
        bundle.summary["index_sha256"]["a_train"].encode(),
        ",".join(config.track_fields).encode(),
        ",".join(config.jet_fields).encode(),
        str(config.top_k).encode(),
        json.dumps(config.normalization, sort_keys=True).encode(),
        CACHE_VERSION.encode(),
        NORMALIZATION_VERSION.encode(),
    ))
    digest = hashlib.sha256(payload).hexdigest()[:20]
    return Path(config.cache_dir) / f"normalization_{digest}.npz"


def _standardization_statistics(arrays, config):
    track_sum = np.zeros(len(config.track_fields), dtype=np.float64)
    track_square_sum = np.zeros(len(config.track_fields), dtype=np.float64)
    track_count = 0
    for start in range(0, len(arrays["X"]), 8192):
        stop = min(start + 8192, len(arrays["X"]))
        values = arrays["X"][start:stop][arrays["mask"][start:stop]]
        track_sum += values.sum(axis=0, dtype=np.float64)
        track_square_sum += np.square(
            values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        track_count += len(values)
    if track_count == 0 or len(arrays["jet_X"]) == 0:
        raise ValueError("cannot derive normalization from an empty A-train sample")

    track_mean = track_sum / track_count
    track_variance = np.maximum(
        track_square_sum / track_count - np.square(track_mean), 0.0)
    jet_mean = arrays["jet_X"].mean(axis=0, dtype=np.float64)
    jet_variance = arrays["jet_X"].var(axis=0, dtype=np.float64)
    epsilon = float(config.normalization["epsilon"])
    track_std = np.sqrt(track_variance)
    jet_std = np.sqrt(jet_variance)
    track_std[track_std < epsilon] = 1.0
    jet_std[jet_std < epsilon] = 1.0
    return {
        "track_mean": track_mean.astype(np.float32),
        "track_std": track_std.astype(np.float32),
        "jet_mean": jet_mean.astype(np.float32),
        "jet_std": jet_std.astype(np.float32),
    }


def _apply_standardization(arrays, statistics):
    for start in range(0, len(arrays["X"]), 8192):
        stop = min(start + 8192, len(arrays["X"]))
        values = arrays["X"][start:stop]
        values -= statistics["track_mean"]
        values /= statistics["track_std"]
        values[~arrays["mask"][start:stop]] = 0.0
    arrays["jet_X"] -= statistics["jet_mean"]
    arrays["jet_X"] /= statistics["jet_std"]


def _load_statistics(path):
    with np.load(path) as values:
        required = {"track_mean", "track_std", "jet_mean", "jet_std"}
        if set(values.files) != required:
            raise ValueError(f"normalization cache schema mismatch: {path}")
        return {name: values[name] for name in required}


def _load_raw_processed_split(
        config, split_name, bundle, *, block_rows=None, progress=False):
    indices = np.asarray(bundle.arrays[split_name], dtype=np.int64)
    track_fields = list(config.track_fields)
    jet_fields = list(config.jet_fields)
    truth_vertex_field = config.truth_vertex["field"]
    track_read_fields = _unique((
        *TRACK_AUX_FIELDS, truth_vertex_field, *track_fields))
    jet_read_fields = _unique((*JET_AUX_FIELDS, *jet_fields))
    with h5py.File(config.train_file, "r") as handle:
        jets = handle["jets"]
        tracks = handle["tracks"]
        n_rows = _row_count(jets, JET_AUX_FIELDS[0])
        if _row_count(tracks, TRACK_AUX_FIELDS[0]) != n_rows:
            raise ValueError("jets and tracks must have the same first dimension")
        indices = _normalise_indices(indices, n_rows)
        _require_fields(jets, jet_read_fields, "jets")
        _require_fields(tracks, track_read_fields, "tracks")
        source_k = (
            tracks["valid"].shape[1]
            if isinstance(tracks, h5py.Group) else tracks.shape[1])
        output_k = min(config.top_k, source_k)
        chunks = tracks["valid"].chunks if isinstance(tracks, h5py.Group) else tracks.chunks
        chunk_rows = chunks[0] if chunks else min(2048, max(n_rows, 1))
        outputs = _allocate(
            len(indices), output_k, len(track_fields), len(jet_fields))
        out_pos = 0

        for begin, end, start, stop in _block_slices(
                indices, n_rows, chunk_rows, block_rows):
            selected = indices[begin:end]
            local = selected - start
            jet_block = _read_fields(jets, jet_read_fields, start, stop)
            track_block = _read_fields(tracks, track_read_fields, start, stop)
            flavour = jet_block["HadronConeExclTruthLabelID"][local]
            keep_flavour = np.isin(flavour, list(config.flavour_to_label))
            if not keep_flavour.any():
                continue
            selected = selected[keep_flavour]
            local = local[keep_flavour]
            flavour = flavour[keep_flavour]

            valid = track_block["valid"][local].astype(bool)
            d0 = track_block["lifetimeSignedD0"][local].astype(np.float32)
            z0_sin_theta = track_block[
                "z0SinTheta"][local].astype(np.float32)
            significance = track_block[
                "lifetimeSignedD0Significance"][local].astype(np.float32)
            keep = (
                valid
                & (np.abs(d0) < 3.5)
                & (np.abs(z0_sin_theta) < 5.0))
            rank = np.abs(significance)
            rank[~keep] = -np.inf
            order = np.argsort(-rank, axis=1)[:, :output_k]
            rows = np.arange(len(local))[:, None]
            top_mask = keep[rows, order]
            has_track = top_mask.any(axis=1)
            if not has_track.any():
                continue

            selected = selected[has_track]
            local = local[has_track]
            flavour = flavour[has_track]
            order = order[has_track]
            top_mask = top_mask[has_track]
            rows = np.arange(len(local))[:, None]
            count = len(local)
            destination = slice(out_pos, out_pos + count)

            features = np.stack([
                track_block[name][local].astype(np.float32)
                for name in track_fields
            ], axis=-1)
            outputs["X"][destination] = np.where(
                top_mask[..., None], features[rows, order], 0.0)
            outputs["jet_X"][destination] = np.stack([
                jet_block[name][local].astype(np.float32)
                for name in jet_fields
            ], axis=-1)
            outputs["mask"][destination] = top_mask
            outputs["y"][destination] = np.asarray([
                config.flavour_to_label[int(value)] for value in flavour
            ], dtype=np.int64)
            origin = track_block[
                "ftagTruthOriginLabel"][local].astype(np.int64)[rows, order]
            vertex_index = track_block[
                truth_vertex_field][local].astype(np.int64)[rows, order]
            outputs["origin"][destination] = np.where(top_mask, origin, -1)
            truth_vertex_valid = top_mask & (vertex_index >= 0)
            valid_pair = (
                truth_vertex_valid[:, :, None]
                & truth_vertex_valid[:, None, :])
            same_vertex = vertex_index[:, :, None] == vertex_index[:, None, :]
            off_diagonal = ~np.eye(output_k, dtype=bool)[None, :, :]
            outputs["truth_pair"][destination] = np.where(
                valid_pair & off_diagonal, same_vertex.astype(np.float32), -1.0)
            outputs["jet_pt"][destination] = jet_block[
                "pt_btagJes"][local].astype(np.float32)
            outputs["jet_eta"][destination] = jet_block[
                "eta_btagJes"][local].astype(np.float32)
            outputs["event_number"][destination] = jet_block[
                "eventNumber"][local].astype(np.int64)
            outputs["source_index"][destination] = selected
            out_pos += count
            if progress:
                print(f"  {split_name}: {end:,}/{len(indices):,}")

    return {name: value[:out_pos] for name, value in outputs.items()}


def _normalization_statistics(
        config, bundle, split_name, raw, *, force=False,
        block_rows=None, progress=False):
    path = _normalization_path(config, bundle)
    source_split = config.normalization["source_split"]
    if path.exists() and not (force and split_name == source_split):
        return _load_statistics(path)
    source = raw
    if split_name != source_split:
        source = _load_raw_processed_split(
            config, source_split, bundle,
            block_rows=block_rows, progress=progress)
    statistics = _standardization_statistics(source, config)
    _save_cache_atomic(str(path), statistics)
    return statistics


def load_processed_split(
        config, split_name: str, *, force: bool = False,
        block_rows: int | None = None, progress: bool = False,
        fields: tuple[str, ...] | list[str] | None = None):
    bundle = load_split_bundle(config.split_dir, config=config)
    if split_name not in bundle.arrays:
        raise KeyError(f"unknown Parallel Refine split {split_name!r}")
    indices = np.asarray(bundle.arrays[split_name], dtype=np.int64)
    path = _cache_path(config, split_name, indices)
    if path.exists() and not force:
        with np.load(path) as cached:
            selected = cached.files if fields is None else fields
            missing = set(selected) - set(cached.files)
            if missing:
                raise KeyError(
                    f"processed cache is missing fields: {sorted(missing)}")
            return {name: cached[name] for name in selected}

    result = _load_raw_processed_split(
        config, split_name, bundle,
        block_rows=block_rows, progress=progress)
    statistics = _normalization_statistics(
        config, bundle, split_name, result, force=force,
        block_rows=block_rows, progress=progress)
    _apply_standardization(result, statistics)
    _save_cache_atomic(str(path), result)
    if fields is None:
        return result
    missing = set(fields) - set(result)
    if missing:
        raise KeyError(f"processed cache is missing fields: {sorted(missing)}")
    return {name: result[name] for name in fields}


class ParallelRefineDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.tensors = {
            name: torch.from_numpy(value) for name, value in arrays.items()
        }

    def __len__(self):
        return len(self.tensors["y"])

    def __getitem__(self, index):
        return {name: value[index] for name, value in self.tensors.items()}


def create_loader(
        config, split_name: str, *, shuffle: bool | None = None,
        force: bool = False, block_rows: int | None = None,
        progress: bool = False, batch_size: int | None = None,
        fields: tuple[str, ...] | list[str] | None = None):
    arrays = load_processed_split(
        config, split_name, force=force, block_rows=block_rows,
        progress=progress, fields=fields)
    if shuffle is None:
        shuffle = split_name.endswith("train")
    split_offset = {
        "a_train": 0, "a_val": 1, "b_train": 2, "b_val": 3, "y_test": 4,
    }[split_name]
    loader = DataLoader(
        ParallelRefineDataset(arrays),
        batch_size=batch_size or config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=dataloader_generator(config.seed + split_offset),
        worker_init_fn=seed_dataloader_worker,
    )
    return loader, arrays
