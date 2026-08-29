"""Checkpoint- and split-bound fixed feature tables for downstream refiners."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.parallel_refine.config import (
    FEATURE_RECIPES, SeedRun, StudyConfig, active_parallel_config)
from src.parallel_refine.data import create_loader
from src.parallel_refine.upstream import (
    build_parallel, checkpoint_config, frozen_parallel_outputs)
from src.parallel_refine.splits import load_split_bundle


FEATURE_SCHEMA_VERSION = "parallel_refine_structured_pool_v3"
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
        groups = FEATURE_RECIPES[recipe]
        indices = []
        for group in groups:
            start, stop = self.manifest["groups"][group]
            indices.extend(range(start, stop))
        return np.asarray(indices, dtype=np.int64)

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
        "study_name": study.study_name,
        "parallel_seed": run.seed,
        "parallel_output_name": run.output_name,
        "split": split,
        "checkpoint_sha256": sha256_file(checkpoint),
        "split_index_sha256": split_hash,
        "top_k": int(active_config.top_k),
        "track_fields": list(active_config.track_fields),
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
    checkpoint = study.checkpoint(run)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing Parallel checkpoint: {checkpoint}")
    active_config = active_parallel_config(study, run)
    bundle = load_split_bundle(active_config.split_dir, config=active_config)
    split_hash = bundle.summary["index_sha256"][split]
    directory = cache_directory(study, run, split, checkpoint, split_hash)
    if (directory / "manifest.json").is_file() and not force:
        return load_frozen_cache(study, run, split)

    loader, raw = create_loader(
        active_config, split, shuffle=False, progress=True,
        batch_size=study.cache.get("batch_size", active_config.batch_size),
        fields=("X", "mask", "y", "source_index", "event_number"))
    model_config = checkpoint_config(checkpoint, active_config)
    model = build_parallel(model_config).to(device)
    model.load_state_dict(torch.load(
        checkpoint, map_location=device, weights_only=True))
    model.eval()

    writer = None
    feature_names = None
    feature_groups = None
    try:
        for raw_batch in loader:
            batch = {name: values.to(device) for name, values in raw_batch.items()}
            output = frozen_parallel_outputs(model, batch["X"], batch["mask"])
            table = build_feature_table(output)
            if writer is None:
                writer = _Writer(
                    directory, len(raw["y"]), table.values.shape[-1],
                    study.cache.get("dtype", "float32"))
                feature_names = list(table.names)
                feature_groups = {
                    name: list(bounds) for name, bounds in table.groups.items()}
            elif tuple(feature_names) != table.names or feature_groups != {
                    name: list(bounds) for name, bounds in table.groups.items()}:
                raise ValueError("feature schema changed between batches")
            writer.write(
                table.values.detach().cpu().numpy(),
                batch["y"].cpu().numpy(),
                batch["source_index"].cpu().numpy(),
                batch["event_number"].cpu().numpy())
            print(f"  {run.output_name}/{split}: {writer.cursor:,}/{len(raw['y']):,}")
        if writer is None:
            raise ValueError(f"cannot cache empty split {split}")
        metadata = {
            "version": FEATURE_SCHEMA_VERSION,
            "study_name": study.study_name,
            "experiment_config": str(study.path),
            "experiment_config_sha256": study.source_sha256,
            "parallel_seed": run.seed,
            "downstream_seed": run.seed,
            "parallel_output_name": run.output_name,
            "split": split,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "split_index_sha256": split_hash,
            "source_index_sha256": sha256_array(np.asarray(raw["source_index"])),
            "source_count": int(len(raw["source_index"])),
            "top_k": int(active_config.top_k),
            "track_fields": list(active_config.track_fields),
            "storage_dtype": study.cache.get("dtype", "float32"),
            "feature_names": feature_names,
            "groups": feature_groups,
            "recipes": {name: list(groups) for name, groups in FEATURE_RECIPES.items()},
        }
        writer.finalize(metadata)
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return load_frozen_cache(study, run, split)
