"""Checkpoint- and split-bound fixed feature tables for downstream refiners."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.parallel_refine.config import (
    FEATURE_RECIPES, GRAPH_RECIPES, SeedRun, StudyConfig,
    active_parallel_config)
from src.parallel_refine.data import create_loader
from src.parallel_refine.upstream import (
    build_parallel, checkpoint_config, frozen_parallel_outputs)
from src.parallel_refine.splits import load_split_bundle


FEATURE_SCHEMA_VERSION = "parallel_refine_structured_pool_v4"
_FEATURE_EPS = 1e-12


@dataclass(frozen=True)
class FeatureTable:
    values: torch.Tensor
    names: tuple[str, ...]
    groups: dict[str, tuple[int, int]]


def _append_group(pieces, names, groups, group, values, group_names):
    start = sum(piece.shape[-1] for piece in pieces)
    pieces.append(values)
    names.extend(group_names)
    groups[group] = (start, start + values.shape[-1])


def _pair_track_features(probabilities, embedding, mask):
    _, tracks, _ = probabilities.shape
    eye = torch.eye(tracks, dtype=torch.bool, device=probabilities.device)
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1) & ~eye.unsqueeze(0)
    weights = torch.where(pair_mask, probabilities, torch.zeros_like(probabilities))
    probability_sum = weights.sum(dim=-1, keepdim=True)
    neighbour_count = pair_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    probability_mean = probability_sum / neighbour_count.to(probabilities.dtype)
    probability_max = probabilities.masked_fill(~pair_mask, float("-inf")).max(
        dim=-1, keepdim=True).values
    probability_max = torch.where(
        torch.isfinite(probability_max), probability_max,
        torch.zeros_like(probability_max))
    weighted_sum = torch.matmul(weights, embedding)
    weighted_embedding = torch.where(
        probability_sum > 0,
        weighted_sum / probability_sum.clamp(min=_FEATURE_EPS),
        torch.zeros_like(weighted_sum))
    return torch.cat([
        weighted_embedding, probability_mean, probability_max, probability_sum,
    ], dim=-1)


def build_feature_table(output: dict[str, torch.Tensor]) -> FeatureTable:
    """Build the complete F4 table without consulting labels or truth fields."""
    mask = output["track_mask"].bool()
    jet_probability = output["flavour_probs"]
    embedding = output["track_embedding"]
    attention = output["pool_attention"]
    embedding_features = output["pooled_embedding"]
    dimension = embedding.shape[-1]
    origin = output["origin_probs"]
    origin_features = (origin * attention).sum(dim=1)
    pair_features = (_pair_track_features(output["pair_probs"], embedding, mask)
                     * attention).sum(dim=1)
    pieces, names, groups = [], [], {}
    _append_group(pieces, names, groups, "jet_probability", jet_probability,
                  tuple(f"jet_prob_{name}" for name in ("b", "c", "light")))
    _append_group(pieces, names, groups, "embedding", embedding_features,
                  tuple(f"pooled_{index}" for index in range(dimension)))
    aux_names = (
        tuple(f"origin_attention_{index}" for index in range(origin.shape[-1]))
        + tuple(f"pair_weighted_embedding_{index}" for index in range(dimension))
        + ("pair_match_mean", "pair_match_max", "pair_match_sum"))
    _append_group(pieces, names, groups, "aux", torch.cat(
        [origin_features, pair_features], dim=-1), aux_names)
    values = torch.cat(pieces, dim=-1)
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite structured pooled feature encountered")
    return FeatureTable(values, tuple(names), groups)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def cache_directory(
        study: StudyConfig, run: SeedRun, split: str, checkpoint: str | Path,
        split_index_sha256: str) -> Path:
    identity = (
        f"{FEATURE_SCHEMA_VERSION}_{sha256_file(checkpoint)[:12]}_"
        f"{split_index_sha256[:12]}")
    return Path(study.cache["root"]) / run.output_name / split / identity


@dataclass(frozen=True)
class FrozenFeatureCache:
    directory: Path
    features: np.ndarray
    labels: np.ndarray
    source_index: np.ndarray
    event_number: np.ndarray
    manifest: dict[str, Any]

    def recipe_columns(self, recipe: str) -> np.ndarray:
        if recipe == "F1O":
            prefixes = ("pooled_", "origin_attention_")
            return self._columns_with_prefixes(prefixes)
        if recipe == "F1V":
            prefixes = ("pooled_", "pair_weighted_embedding_")
            return self._columns_with_prefixes(prefixes)
        if recipe == "F1OJ":
            # Graph-only context selector: F1O plus the frozen jet posterior.
            prefixes = ("jet_prob_", "pooled_", "origin_attention_")
            return self._columns_with_prefixes(prefixes)
        groups = FEATURE_RECIPES[recipe]
        indices = []
        for group in groups:
            start, stop = self.manifest["groups"][group]
            indices.extend(range(start, stop))
        return np.asarray(indices, dtype=np.int64)

    def _columns_with_prefixes(self, prefixes: tuple[str, ...]) -> np.ndarray:
        """Select dynamic-width fields without changing legacy cache groups."""
        names = self.manifest["feature_names"]
        columns = [
            index for prefix in prefixes for index, name in enumerate(names)
            if name.startswith(prefix)
        ]
        if not columns:
            raise ValueError(f"frozen cache has no fields for {prefixes}")
        return np.asarray(columns, dtype=np.int64)

    def recipe_features(self, recipe: str) -> np.ndarray:
        return self.features[:, self.recipe_columns(recipe)]

    def recipe_names(self, recipe: str) -> list[str]:
        return [self.manifest["feature_names"][index]
                for index in self.recipe_columns(recipe)]


class _Writer:
    def __init__(self, directory: Path, length: int, feature_dim: int, dtype: str):
        self.directory = directory
        self.length = int(length)
        self.cursor = 0
        directory.mkdir(parents=True, exist_ok=True)
        self.marker = directory / ".building"
        descriptor = os.open(
            self.marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(str(os.getpid()))
        self.temporary = {
            "features": directory / f".features.{os.getpid()}.npy",
            "labels": directory / f".labels.{os.getpid()}.npy",
            "source_index": directory / f".source_index.{os.getpid()}.npy",
            "event_number": directory / f".event_number.{os.getpid()}.npy",
        }
        self.arrays = {
            "features": np.lib.format.open_memmap(
                self.temporary["features"], mode="w+", dtype=dtype,
                shape=(length, feature_dim)),
            "labels": np.lib.format.open_memmap(
                self.temporary["labels"], mode="w+", dtype="int64", shape=(length,)),
            "source_index": np.lib.format.open_memmap(
                self.temporary["source_index"], mode="w+", dtype="int64", shape=(length,)),
            "event_number": np.lib.format.open_memmap(
                self.temporary["event_number"], mode="w+", dtype="int64", shape=(length,)),
        }
        self.hashers = {name: hashlib.sha256() for name in self.arrays}

    def write(self, features, labels, source_index, event_number):
        arrays = {
            "features": np.asarray(features, dtype=self.arrays["features"].dtype),
            "labels": np.asarray(labels, dtype=np.int64),
            "source_index": np.asarray(source_index, dtype=np.int64),
            "event_number": np.asarray(event_number, dtype=np.int64),
        }
        batch_size = len(arrays["labels"])
        stop = self.cursor + batch_size
        if stop > self.length:
            raise ValueError("feature writer received too many rows")
        for name, values in arrays.items():
            self.arrays[name][self.cursor:stop] = values
            self.hashers[name].update(np.ascontiguousarray(values).tobytes())
        self.cursor = stop

    def finalize(self, metadata):
        if self.cursor != self.length:
            raise ValueError(
                f"feature writer expected {self.length} rows, got {self.cursor}")
        specifications = {}
        for name, values in list(self.arrays.items()):
            values.flush()
            specifications[name] = {
                "file": f"{name}.npy",
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "sha256": self.hashers[name].hexdigest(),
            }
            del self.arrays[name]
            os.replace(self.temporary[name], self.directory / f"{name}.npy")
        manifest = {**metadata, "arrays": specifications}
        temporary = self.directory / f".manifest.{os.getpid()}.json"
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(temporary, self.directory / "manifest.json")
        self.marker.unlink(missing_ok=True)
        return manifest

    def abort(self):
        self.arrays.clear()
        for path in self.temporary.values():
            path.unlink(missing_ok=True)
        self.marker.unlink(missing_ok=True)


class ThreadedWriter:
    """Run a cache writer's ``write`` calls on a background thread.

    The forward pass only has to enqueue CPU arrays; the wrapped writer performs
    the memmap assignment and hashing while the next batch is computed.  A
    bounded queue keeps at most ``maxsize`` batches in flight.  Bookkeeping
    attributes such as ``cursor`` are proxied to the wrapped writer so existing
    progress reporting keeps working.

    Failure handling is queue-safe: enqueuing polls the worker with a timeout so
    a dead or failed worker can never block the producer forever; after a write
    error the worker keeps draining (discarding) queued items, and ``finalize``
    /``abort`` stop the worker before touching the underlying writer so its
    ``.building`` marker and temporary files are always cleaned up.
    """

    _POLL_SECONDS = 0.1

    def __init__(self, writer, maxsize: int = 4):
        self.writer = writer
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def __getattr__(self, name):
        return getattr(self.__dict__["writer"], name)

    def _drain(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=self._POLL_SECONDS)
            except queue.Empty:
                continue
            if item is None:
                break
            if self._error is not None:
                # Discard queued batches after a failed write so a full queue
                # can never block the producer waiting to enqueue more.
                continue
            args, kwargs = item
            try:
                self.writer.write(*args, **kwargs)
            except BaseException as error:
                self._error = error

    def _enqueue(self, item):
        while True:
            if self._error is not None:
                raise self._error
            if not self._thread.is_alive():
                raise RuntimeError("cache writer thread stopped unexpectedly")
            try:
                self._queue.put(item, timeout=self._POLL_SECONDS)
                return
            except queue.Full:
                continue

    def write(self, *args, **kwargs):
        self._enqueue((args, kwargs))

    def _shutdown(self):
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join()

    def finalize(self, metadata):
        try:
            if self._error is not None:
                raise self._error
            self._enqueue(None)
        except BaseException:
            self._shutdown()
            self.writer.abort()
            raise
        self._thread.join()
        if self._error is not None:
            self.writer.abort()
            raise self._error
        return self.writer.finalize(metadata)

    def abort(self):
        self._shutdown()
        self.writer.abort()


def _load_arrays(directory: Path, manifest) -> FrozenFeatureCache:
    arrays = {
        name: np.load(directory / specification["file"], mmap_mode="r")
        for name, specification in manifest["arrays"].items()
    }
    for name, values in arrays.items():
        specification = manifest["arrays"][name]
        if list(values.shape) != specification["shape"]:
            raise ValueError(f"cached {name} shape does not match manifest")
        if str(values.dtype) != specification["dtype"]:
            raise ValueError(f"cached {name} dtype does not match manifest")
    return FrozenFeatureCache(
        directory, arrays["features"], arrays["labels"],
        arrays["source_index"], arrays["event_number"], manifest)


def load_frozen_cache(
        study: StudyConfig, run: SeedRun, split: str) -> FrozenFeatureCache:
    checkpoint = study.checkpoint(run)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Parallel checkpoint: {checkpoint}")
    active_config = active_parallel_config(study, run)
    bundle = load_split_bundle(study.data["split_dir"], config=active_config)
    split_hash = bundle.summary["index_sha256"][split]
    directory = cache_directory(study, run, split, checkpoint, split_hash)
    if (directory / ".building").exists():
        raise RuntimeError(f"feature cache is still being built: {directory}")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing frozen feature cache: {directory}; run scripts/generate_cache.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "version": FEATURE_SCHEMA_VERSION,
        "study_name": study.cache_identity_name,
        "parallel_seed": run.seed,
        "parallel_output_name": run.output_name,
        "split": split,
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_index_sha256": split_hash,
        "top_k": int(active_config.top_k),
        "track_fields": list(active_config.track_fields),
        "jet_fields": list(active_config.jet_fields),
        "normalization": active_config.normalization,
        "kinematic_resampling": active_config.kinematic_resampling,
        "truth_vertex": active_config.truth_vertex,
        "storage_dtype": study.cache.get("dtype", "float32"),
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items() if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"frozen feature cache identity mismatch: {mismatches}")
    result = _load_arrays(directory, manifest)
    if sha256_array(np.asarray(result.source_index)) != manifest["source_index_sha256"]:
        raise ValueError("cached source_index hash mismatch")
    return result


@torch.no_grad()
def generate_frozen_cache(
        study: StudyConfig, run: SeedRun, split: str, device: torch.device,
        *, force: bool = False) -> FrozenFeatureCache:
    """Build (or load) only the structured-pool cache for one split."""
    frozen, _ = generate_frozen_and_graph_cache(
        study, run, split, device, force=force, include_graph=False)
    return frozen


def _frozen_metadata(
        study, run, split, checkpoint, split_hash, active_config, raw,
        feature_names, feature_groups):
    return {
        "version": FEATURE_SCHEMA_VERSION,
        "study_name": study.cache_identity_name,
        "experiment_config": str(study.path),
        "experiment_config_sha256": study.source_sha256,
        "experiment_markers": study.experiment_markers,
        "parallel_seed": run.seed,
        "parallel_output_name": run.output_name,
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_index_sha256": split_hash,
        "source_index_sha256": sha256_array(np.asarray(raw["source_index"])),
        "source_count": int(len(raw["source_index"])),
        "top_k": int(active_config.top_k),
        "track_fields": list(active_config.track_fields),
        "jet_fields": list(active_config.jet_fields),
        "normalization": active_config.normalization,
        "kinematic_resampling": active_config.kinematic_resampling,
        "truth_vertex": active_config.truth_vertex,
        "storage_dtype": study.cache.get("dtype", "float32"),
        "feature_names": feature_names,
        "groups": feature_groups,
        "recipes": {name: list(groups) for name, groups in FEATURE_RECIPES.items()},
    }


def _graph_metadata(
        version, study, run, split, checkpoint, split_hash, active_config, raw):
    return {
        "version": version,
        "study_name": study.cache_identity_name,
        "parallel_seed": run.seed,
        "parallel_output_name": run.output_name,
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_index_sha256": split_hash,
        "source_index_sha256": sha256_array(np.asarray(raw["source_index"])),
        "source_count": int(len(raw["source_index"])),
        "top_k": int(active_config.top_k),
        "storage_dtype": study.cache.get("graph_dtype", "float32"),
    }


def _threaded(writer, threaded):
    return ThreadedWriter(writer) if threaded else writer


@torch.no_grad()
def generate_frozen_and_graph_cache(
        study: StudyConfig, run: SeedRun, split: str, device: torch.device,
        *, force: bool = False, threaded: bool = True, progress: bool = True,
        include_frozen: bool = True, include_graph: bool | None = None):
    """Build the structured-pool and (when configured) pair-graph caches.

    Both frozen caches are derived from the same ``frozen_parallel_outputs``
    result, so a single loader and forward pass serves both, avoiding a
    duplicate model load, DataLoader traversal and forward pass.  CPU-side
    writes and hashing run on a :class:`ThreadedWriter` background thread so
    they overlap with the next batch's forward pass.

    ``include_graph=None`` builds the graph table only when a graph recipe is
    configured; pass an explicit boolean to force or forbid it.  Returns
    ``(frozen_cache, graph_cache)``; an entry is ``None`` when that cache is
    excluded.  Splits whose manifest already exists are not rebuilt unless
    ``force`` is set.  The produced files and manifests are identical to calling
    :func:`generate_frozen_cache` and :func:`generate_graph_cache` separately.
    """
    from src.parallel_refine.graph_cache import (
        GRAPH_CACHE_VERSION, _GraphWriter, graph_cache_directory,
        load_graph_cache)

    checkpoint = study.checkpoint(run)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Parallel checkpoint: {checkpoint}")
    active_config = active_parallel_config(study, run)
    bundle = load_split_bundle(active_config.split_dir, config=active_config)
    split_hash = bundle.summary["index_sha256"][split]

    if include_graph is None:
        include_graph = any(
            recipe in GRAPH_RECIPES for recipe in study.refiners["recipes"])
    frozen_directory = cache_directory(
        study, run, split, checkpoint, split_hash)
    need_frozen = (
        include_frozen
        and (force or not (frozen_directory / "manifest.json").is_file()))
    graph_directory = None
    need_graph = False
    if include_graph:
        graph_directory = graph_cache_directory(
            study, run, split, checkpoint, split_hash)
        need_graph = force or not (graph_directory / "manifest.json").is_file()

    if not need_frozen and not need_graph:
        return (load_frozen_cache(study, run, split) if include_frozen else None,
                load_graph_cache(study, run, split) if include_graph else None)

    loader, raw = create_loader(
        active_config, split, shuffle=False, progress=progress,
        batch_size=study.cache.get("batch_size", active_config.batch_size),
        fields=("X", "jet_X", "mask", "y", "source_index", "event_number"))
    model = build_parallel(checkpoint_config(checkpoint, active_config)).to(device)
    model.load_state_dict(torch.load(
        checkpoint, map_location=device, weights_only=True))
    model.eval()

    total = len(raw["y"])
    frozen_writer = None
    graph_writer = None
    feature_names = None
    feature_groups = None
    try:
        for raw_batch in loader:
            batch = {name: values.to(device) for name, values in raw_batch.items()}
            output = frozen_parallel_outputs(
                model, batch["X"], batch["jet_X"], batch["mask"])
            table = build_feature_table(output) if need_frozen else None
            if need_frozen and frozen_writer is None:
                frozen_writer = _threaded(_Writer(
                    frozen_directory, total, table.values.shape[-1],
                    study.cache.get("dtype", "float32")), threaded)
                feature_names = list(table.names)
                feature_groups = {
                    name: list(bounds) for name, bounds in table.groups.items()}
            elif need_frozen and (
                    tuple(feature_names) != table.names
                    or feature_groups != {
                        name: list(bounds)
                        for name, bounds in table.groups.items()}):
                raise ValueError("feature schema changed between batches")
            if need_graph and graph_writer is None:
                graph_writer = _threaded(_GraphWriter(
                    graph_directory, length=total,
                    tracks=batch["mask"].shape[1],
                    embedding_dim=output["track_embedding"].shape[-1],
                    dtype=study.cache.get("graph_dtype", "float32")), threaded)
            if need_frozen:
                frozen_writer.write(
                    table.values.detach().cpu().numpy(),
                    batch["y"].cpu().numpy(),
                    batch["source_index"].cpu().numpy(),
                    batch["event_number"].cpu().numpy())
                if progress:
                    print(f"  {run.output_name}/{split}: "
                          f"{frozen_writer.cursor:,}/{total:,}")
            if need_graph:
                graph_writer.write(
                    pair_probs=output["pair_probs"].cpu().numpy(),
                    track_mask=output["track_mask"].cpu().numpy(),
                    origin_probs=output["origin_probs"].cpu().numpy(),
                    track_embedding=output["track_embedding"].cpu().numpy(),
                    labels=batch["y"].cpu().numpy(),
                    source_index=batch["source_index"].cpu().numpy(),
                    event_number=batch["event_number"].cpu().numpy())
                if progress:
                    print(f"  graph {run.output_name}/{split}: "
                          f"{graph_writer.cursor:,}/{total:,}")
        if need_frozen and frozen_writer is None:
            raise ValueError(f"cannot cache empty split {split}")
        if need_graph and graph_writer is None:
            raise ValueError(f"cannot cache empty split {split}")
        if need_frozen:
            frozen_writer.finalize(_frozen_metadata(
                study, run, split, checkpoint, split_hash, active_config, raw,
                feature_names, feature_groups))
        if need_graph:
            graph_writer.finalize(_graph_metadata(
                GRAPH_CACHE_VERSION, study, run, split, checkpoint, split_hash,
                active_config, raw))
    except BaseException:
        if frozen_writer is not None:
            frozen_writer.abort()
        if graph_writer is not None:
            graph_writer.abort()
        raise
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (load_frozen_cache(study, run, split) if include_frozen else None,
            load_graph_cache(study, run, split) if include_graph else None)
