"""Deterministic event-level train/validation/test splitting.

All jets sharing an ``eventNumber`` are reserved for exactly one split.  The
validation and test samples keep their natural flavour distribution, while the
training sample is balanced across configured jet classes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


SPLIT_VERSION = "event_number_v1"
JET_EVENT_FIELD = "eventNumber"


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    summary: dict[str, Any]


@dataclass
class DataSplits:
    train_loader: Any
    validation_loader: Any
    test_loader: Any
    train_data: dict[str, np.ndarray]
    validation_data: dict[str, np.ndarray]
    test_data: dict[str, np.ndarray]
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    summary: dict[str, Any]


def _hash_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _class_counts(indices, flavours, flavour_to_label, class_names):
    labels = np.array(
        [flavour_to_label[value] for value in flavours[indices]],
        dtype=np.int64)
    return {
        class_names[class_index]: int((labels == class_index).sum())
        for class_index in range(len(class_names))
    }


def _reserve_events(event_order, event_counts, requested_jets, split_name):
    if requested_jets <= 0:
        raise ValueError(f"{split_name} size must be positive")
    cumulative = np.cumsum(event_counts[event_order], dtype=np.int64)
    count = int(np.searchsorted(cumulative, requested_jets, side="left")) + 1
    if count > len(event_order):
        available = int(cumulative[-1]) if len(cumulative) else 0
        raise ValueError(
            f"Not enough valid jets for {split_name}: requested "
            f"{requested_jets:,}, available {available:,}")
    return event_order[:count], event_order[count:]


def split_indices_by_event(config, flavours, event_numbers) -> SplitIndices:
    """Return deterministic, pairwise event-disjoint split indices."""
    flavours = np.asarray(flavours)
    event_numbers = np.asarray(event_numbers)
    if flavours.ndim != 1 or event_numbers.ndim != 1:
        raise ValueError("flavours and event_numbers must be one-dimensional")
    if len(flavours) != len(event_numbers):
        raise ValueError("flavours and event_numbers must have equal length")

    valid_indices = np.where(np.isin(
        flavours, list(config.flavour_to_label.keys())))[0]
    valid_events = event_numbers[valid_indices]
    unique_events, event_inverse, event_counts = np.unique(
        valid_events, return_inverse=True, return_counts=True)

    rng = np.random.default_rng(config.data_seed)
    remaining_order = rng.permutation(len(unique_events))
    test_event_positions, remaining_order = _reserve_events(
        remaining_order, event_counts, config.n_test, "test")
    validation_event_positions, remaining_order = _reserve_events(
        remaining_order, event_counts, config.n_val, "validation")

    test_event_mask = np.isin(event_inverse, test_event_positions)
    validation_event_mask = np.isin(event_inverse, validation_event_positions)
    train_pool_mask = ~(test_event_mask | validation_event_mask)

    test_candidates = valid_indices[test_event_mask]
    validation_candidates = valid_indices[validation_event_mask]
    train_pool = valid_indices[train_pool_mask]

    test_indices = np.sort(rng.choice(
        test_candidates, size=config.n_test, replace=False))
    validation_indices = np.sort(rng.choice(
        validation_candidates, size=config.n_val, replace=False))

    pool_labels = np.array([
        config.flavour_to_label[value] for value in flavours[train_pool]
    ], dtype=np.int64)
    n_per_class = config.n_train // len(config.jet_class_names)
    train_parts = []
    for class_index in range(len(config.jet_class_names)):
        candidates = train_pool[pool_labels == class_index]
        if len(candidates) < n_per_class:
            raise ValueError(
                f"Not enough training jets for class "
                f"{config.jet_class_names[class_index]}: requested "
                f"{n_per_class:,}, available {len(candidates):,}")
        train_parts.append(rng.choice(
            candidates, size=n_per_class, replace=False))
    train_indices = np.sort(np.concatenate(train_parts))

    split_arrays = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    split_events = {
        name: np.unique(event_numbers[indices])
        for name, indices in split_arrays.items()
    }
    overlaps = {}
    for left, right in (
            ("train", "validation"), ("train", "test"),
            ("validation", "test")):
        jet_overlap = int(np.intersect1d(
            split_arrays[left], split_arrays[right]).size)
        event_overlap = int(np.intersect1d(
            split_events[left], split_events[right]).size)
        overlaps[f"{left}_{right}"] = {
            "jet_overlap": jet_overlap,
            "event_overlap": event_overlap,
        }
        if jet_overlap or event_overlap:
            raise RuntimeError(
                f"{left}/{right} split overlap detected: "
                f"{jet_overlap} jet(s), {event_overlap} event(s)")

    summary = {
        "split_version": SPLIT_VERSION,
        "event_field": JET_EVENT_FIELD,
        "data_seed": int(config.data_seed),
        "requested_jets": {
            "train": int(config.n_train),
            "validation": int(config.n_val),
            "test": int(config.n_test),
        },
        "selected_jets": {
            name: int(len(indices)) for name, indices in split_arrays.items()
        },
        "unique_events": {
            name: int(len(events)) for name, events in split_events.items()
        },
        "class_counts": {
            name: _class_counts(
                indices, flavours, config.flavour_to_label,
                config.jet_class_names)
            for name, indices in split_arrays.items()
        },
        "index_sha256": {
            name: _hash_array(indices)
            for name, indices in split_arrays.items()
        },
        "event_sha256": {
            name: _hash_array(events)
            for name, events in split_events.items()
        },
        "overlaps": overlaps,
    }
    return SplitIndices(
        train=train_indices,
        validation=validation_indices,
        test=test_indices,
        summary=summary)


def write_split_manifest(path, summary):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
