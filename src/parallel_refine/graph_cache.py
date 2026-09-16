"""Checkpoint-bound per-track graph inputs for FG downstream refiners.

This module deliberately lives beside, rather than inside, ``cache.py``:
the compact structured-pool cache remains the stable input for F0--F4 and
F1O/F1V, while graph recipes need unpooled pair predictions and node states.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.parallel_refine.cache import (
    generate_frozen_and_graph_cache, sha256_array, sha256_file)
from src.parallel_refine.config import SeedRun, StudyConfig, active_parallel_config
from src.parallel_refine.splits import load_split_bundle


GRAPH_CACHE_VERSION = "parallel_refine_pair_graph_v1"


@dataclass(frozen=True)
class GraphFeatureCache:
    directory: Path
    pair_probs: np.ndarray
    track_mask: np.ndarray
    origin_probs: np.ndarray
    track_embedding: np.ndarray
    labels: np.ndarray
    source_index: np.ndarray
    event_number: np.ndarray
    manifest: dict


def graph_cache_directory(study, run, split, checkpoint, split_index_sha256):
    identity = (
        f"{GRAPH_CACHE_VERSION}_{sha256_file(checkpoint)[:12]}_"
        f"{split_index_sha256[:12]}")
    return Path(study.cache["graph_root"]) / run.output_name / split / identity


class _GraphWriter:
    """Atomic field-wise writer; fields remain independently mmap-loadable."""

    def __init__(self, directory: Path, *, length: int, tracks: int,
                 embedding_dim: int, dtype: str):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.length = int(length)
        self.cursor = 0
        self.marker = directory / ".building"
        descriptor = os.open(self.marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        shapes = {
            "pair_probs": (length, tracks, tracks),
            "track_mask": (length, tracks),
            "origin_probs": (length, tracks, 8),
            "track_embedding": (length, tracks, embedding_dim),
            "labels": (length,),
            "source_index": (length,),
            "event_number": (length,),
        }
        dtypes = {
            "pair_probs": dtype,
            "track_mask": "bool",
            "origin_probs": dtype,
            "track_embedding": dtype,
            "labels": "int64",
            "source_index": "int64",
            "event_number": "int64",
        }
        self.temporary = {
            name: directory / f".{name}.{os.getpid()}.npy" for name in shapes}
        self.arrays = {
            name: np.lib.format.open_memmap(
                self.temporary[name], mode="w+", dtype=dtypes[name], shape=shape)
            for name, shape in shapes.items()
        }
        self.hashers = {name: hashlib.sha256() for name in self.arrays}

    def write(self, **arrays):
        size = len(arrays["labels"])
        stop = self.cursor + size
        if stop > self.length:
            raise ValueError("graph cache writer received too many rows")
        for name, target in self.arrays.items():
            values = np.asarray(arrays[name], dtype=target.dtype)
            if values.shape != target[self.cursor:stop].shape:
                raise ValueError(f"graph cache shape mismatch for {name}")
            target[self.cursor:stop] = values
            self.hashers[name].update(np.ascontiguousarray(values).tobytes())
        self.cursor = stop

    def finalize(self, metadata):
        if self.cursor != self.length:
            raise ValueError(
                f"graph cache writer expected {self.length} rows, got {self.cursor}")
        specs = {}
        for name, array in list(self.arrays.items()):
            array.flush()
            specs[name] = {
                "file": f"{name}.npy", "shape": list(array.shape),
                "dtype": str(array.dtype), "sha256": self.hashers[name].hexdigest(),
            }
            del self.arrays[name]
            os.replace(self.temporary[name], self.directory / f"{name}.npy")
        temporary = self.directory / f".manifest.{os.getpid()}.json"
        temporary.write_text(
            json.dumps({**metadata, "arrays": specs}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(temporary, self.directory / "manifest.json")
        self.marker.unlink(missing_ok=True)

    def abort(self):
        self.arrays.clear()
        for path in self.temporary.values():
            path.unlink(missing_ok=True)
        self.marker.unlink(missing_ok=True)


def _load_graph_arrays(directory, manifest):
    arrays = {
        name: np.load(directory / spec["file"], mmap_mode="r")
        for name, spec in manifest["arrays"].items()
    }
    required = {
        "pair_probs", "track_mask", "origin_probs", "track_embedding",
        "labels", "source_index", "event_number",
    }
    if set(arrays) != required:
        raise ValueError("graph cache field set mismatch")
    for name, values in arrays.items():
        spec = manifest["arrays"][name]
        if list(values.shape) != spec["shape"] or str(values.dtype) != spec["dtype"]:
            raise ValueError(f"graph cache field mismatch: {name}")
    return GraphFeatureCache(directory=directory, manifest=manifest, **arrays)


def load_graph_cache(study: StudyConfig, run: SeedRun, split: str) -> GraphFeatureCache:
    checkpoint = study.checkpoint(run)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Parallel checkpoint: {checkpoint}")
    active_config = active_parallel_config(study, run)
    bundle = load_split_bundle(active_config.split_dir, config=active_config)
    split_hash = bundle.summary["index_sha256"][split]
    directory = graph_cache_directory(study, run, split, checkpoint, split_hash)
    if (directory / ".building").exists():
        raise RuntimeError(f"graph cache is still being built: {directory}")
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing graph cache: {directory}; run scripts/generate_cache.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "version": GRAPH_CACHE_VERSION,
        "study_name": study.cache_identity_name,
        "parallel_seed": run.seed,
        "parallel_output_name": run.output_name,
        "split": split,
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_index_sha256": split_hash,
        "top_k": int(active_config.top_k),
        "storage_dtype": study.cache.get("graph_dtype", "float32"),
    }
    mismatches = {key: (manifest.get(key), value) for key, value in expected.items()
                  if manifest.get(key) != value}
    if mismatches:
        raise ValueError(f"graph cache identity mismatch: {mismatches}")
    result = _load_graph_arrays(directory, manifest)
    if sha256_array(np.asarray(result.source_index)) != manifest["source_index_sha256"]:
        raise ValueError("graph cache source_index hash mismatch")
    return result


@torch.no_grad()
def generate_graph_cache(study: StudyConfig, run: SeedRun, split: str,
                         device: torch.device, *, force: bool = False):
    """Run a frozen Parallel checkpoint once and persist graph-only inputs."""
    _, graph = generate_frozen_and_graph_cache(
        study, run, split, device, force=force,
        include_frozen=False, include_graph=True)
    return graph
