"""Create and validate the event-disjoint A/B/Y split bundle."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from src.data import _read_fields, _require_fields, _row_count, _save_cache_atomic


SPLIT_VERSION = "parallel_refine_event_v1"
SPLIT_NAMES = ("a_train", "a_val", "b_train", "b_val", "y_test")


@dataclass(frozen=True)
class ParallelRefineSplits:
    arrays: dict[str, np.ndarray]
    summary: dict[str, Any]

    def __getattr__(self, name: str) -> np.ndarray:
        if name in self.arrays:
            return self.arrays[name]
        raise AttributeError(name)


def _hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _split_request(config) -> dict[str, Any]:
    stat = os.stat(config.train_file)
    return {
        "train_file": os.path.abspath(config.train_file),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "data_seed": int(config.data_seed),
        "flavour_to_label": {
            str(key): int(value) for key, value in config.flavour_to_label.items()
        },
        "sizes": {name: int(getattr(config, name)) for name in SPLIT_NAMES},
    }


def _reserve_natural(order, counts, requested, name):
    cumulative = np.cumsum(counts[order], dtype=np.int64)
    stop = int(np.searchsorted(cumulative, requested, side="left")) + 1
    if stop > len(order):
        available = int(cumulative[-1]) if len(cumulative) else 0
        raise ValueError(
            f"not enough jets for {name}: {requested} requested, "
            f"{available} available")
    return order[:stop], order[stop:]


def _balanced_targets(total: int, n_classes: int) -> np.ndarray:
    targets = np.full(n_classes, total // n_classes, dtype=np.int64)
    targets[:total % n_classes] += 1
    return targets


def _reserve_balanced(order, event_class_counts, targets, name):
    cumulative = np.cumsum(event_class_counts[order], axis=0, dtype=np.int64)
    ready = np.flatnonzero(np.all(cumulative >= targets, axis=1))
    if not len(ready):
        available = cumulative[-1].tolist() if len(cumulative) else []
        raise ValueError(
            f"not enough balanced jets for {name}: need {targets.tolist()}, "
            f"have {available}")
    stop = int(ready[0]) + 1
    return order[:stop], order[stop:]


def build_split_indices(config, flavours, event_numbers) -> ParallelRefineSplits:
    flavours = np.asarray(flavours)
    event_numbers = np.asarray(event_numbers)
    if (
            flavours.ndim != 1
            or event_numbers.ndim != 1
            or len(flavours) != len(event_numbers)):
        raise ValueError(
            "flavours/event_numbers must be equal-length one-dimensional arrays")

    valid = np.flatnonzero(np.isin(flavours, list(config.flavour_to_label)))
    valid_events = event_numbers[valid]
    unique_events, inverse, event_counts = np.unique(
        valid_events, return_inverse=True, return_counts=True)
    labels = np.asarray([
        config.flavour_to_label[int(value)] for value in flavours[valid]
    ], dtype=np.int64)
    event_class_counts = np.zeros(
        (len(unique_events), config.n_jet_classes), dtype=np.int64)
    np.add.at(event_class_counts, (inverse, labels), 1)

    rng = np.random.default_rng(config.data_seed)
    remaining = rng.permutation(len(unique_events))
    reserved: dict[str, np.ndarray] = {}
    reserved["y_test"], remaining = _reserve_natural(
        remaining, event_counts, config.y_test, "y_test")
    reserved["b_val"], remaining = _reserve_natural(
        remaining, event_counts, config.b_val, "b_val")
    reserved["b_train"], remaining = _reserve_balanced(
        remaining, event_class_counts,
        _balanced_targets(config.b_train, config.n_jet_classes), "b_train")
    reserved["a_val"], remaining = _reserve_natural(
        remaining, event_counts, config.a_val, "a_val")
    reserved["a_train"], remaining = _reserve_balanced(
        remaining, event_class_counts,
        _balanced_targets(config.a_train, config.n_jet_classes), "a_train")

    arrays: dict[str, np.ndarray] = {}
    for name in SPLIT_NAMES:
        candidate_mask = np.isin(inverse, reserved[name])
        candidates = valid[candidate_mask]
        if name.endswith("train"):
            candidate_labels = labels[candidate_mask]
            targets = _balanced_targets(
                int(getattr(config, name)), config.n_jet_classes)
            selected = np.concatenate([
                rng.choice(
                    candidates[candidate_labels == class_index],
                    size=int(targets[class_index]), replace=False)
                for class_index in range(config.n_jet_classes)
            ])
        else:
            selected = rng.choice(
                candidates, size=int(getattr(config, name)), replace=False)
        arrays[name] = np.sort(selected.astype(np.int64, copy=False))

    event_sets = {
        name: np.unique(event_numbers[indices])
        for name, indices in arrays.items()
    }
    overlaps = {}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1:]:
            jet_overlap = int(np.intersect1d(arrays[left], arrays[right]).size)
            event_overlap = int(
                np.intersect1d(event_sets[left], event_sets[right]).size)
            overlaps[f"{left}_{right}"] = {
                "jet_overlap": jet_overlap,
                "event_overlap": event_overlap,
            }
            if jet_overlap or event_overlap:
                raise RuntimeError(f"split overlap detected for {left}/{right}")

    class_counts = {
        name: {
            config.jet_class_names[class_index]: int(np.count_nonzero(
                flavours[indices] == flavour))
            for flavour, class_index in config.flavour_to_label.items()
        }
        for name, indices in arrays.items()
    }
    summary = {
        "split_version": SPLIT_VERSION,
        "data_seed": int(config.data_seed),
        "selected_jets": {name: int(len(value)) for name, value in arrays.items()},
        "unique_events": {name: int(len(value)) for name, value in event_sets.items()},
        "class_counts": class_counts,
        "index_sha256": {name: _hash(value) for name, value in arrays.items()},
        "event_sha256": {name: _hash(value) for name, value in event_sets.items()},
        "overlaps": overlaps,
    }
    return ParallelRefineSplits(arrays, summary)


def generate_split_bundle(config, *, force: bool = False) -> ParallelRefineSplits:
    output = Path(config.split_dir)
    index_path = output / "indices.npz"
    manifest_path = output / "split_manifest.json"
    if index_path.exists() and manifest_path.exists() and not force:
        return load_split_bundle(output, config=config)

    with h5py.File(config.train_file, "r") as handle:
        jets = handle["jets"]
        fields = ("HadronConeExclTruthLabelID", "eventNumber")
        _require_fields(jets, fields, "jets")
        values = _read_fields(jets, fields, 0, _row_count(jets, fields[0]))
    result = build_split_indices(config, values[fields[0]], values[fields[1]])
    result.summary["request"] = _split_request(config)
    output.mkdir(parents=True, exist_ok=True)
    _save_cache_atomic(str(index_path), result.arrays)
    temporary = output / ".split_manifest.json.tmp"
    temporary.write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(manifest_path)
    return result


def load_split_bundle(
        directory: str | Path, *, config=None) -> ParallelRefineSplits:
    directory = Path(directory)
    with np.load(directory / "indices.npz") as values:
        arrays = {name: values[name] for name in SPLIT_NAMES}
    summary = json.loads(
        (directory / "split_manifest.json").read_text(encoding="utf-8"))
    if summary.get("split_version") != SPLIT_VERSION:
        raise ValueError("split bundle version mismatch")
    if config is not None and summary.get("request") != _split_request(config):
        raise ValueError(
            "split bundle does not match the active data configuration; "
            "use a new split_dir or regenerate with --force")
    for name, values in arrays.items():
        if _hash(values) != summary["index_sha256"][name]:
            raise ValueError(f"split index hash mismatch for {name}")
    return ParallelRefineSplits(arrays, summary)
