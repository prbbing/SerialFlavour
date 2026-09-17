#!/usr/bin/env python3
"""Materialize Experiment 2 splits with a shared non-A-train anchor.

The existing split generator is intentionally used once for the largest
Experiment 2 configuration.  This script then derives smaller, nested
A-train sets while copying the exact A-val/B/Y indices from that master
bundle.  The resulting directories use the normal ``indices.npz`` and
``split_manifest.json`` format, so no training code changes are required.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import _read_fields, _row_count, _save_cache_atomic
from src.parallel_refine.config import (
    active_parallel_config,
    load_study_config,
    write_json_atomic,
)
from src.parallel_refine.splits import (
    SPLIT_NAMES,
    SPLIT_VERSION,
    _hash,
    _split_request,
    generate_split_bundle,
    load_split_bundle,
)


DEFAULT_CONFIG_DIR = Path("configs/parallel_refine/experiments/experiment2")
TARGET_A_SIZES = (600_000, 800_000, 1_000_000, 1_500_000, 2_000_000, 3_000_000)


def _log(message: str) -> None:
    print(f"[experiment2-shared-splits] {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs-dir", type=Path, default=DEFAULT_CONFIG_DIR,
        help="directory containing experiment2_*.json files")
    parser.add_argument(
        "--force", action="store_true",
        help="regenerate the master split and replace derived bundles")
    parser.add_argument(
        "--a-order-seed", type=int, default=1729,
        help="seed used only to order jets within A-train strata")
    return parser.parse_args()


def _load_experiment_studies(config_dir: Path) -> dict[int, Any]:
    studies: dict[int, Any] = {}
    paths = sorted(config_dir.glob("experiment2_*.json"))
    if len(paths) != 12:
        raise ValueError(
            f"expected 12 Experiment 2 configs in {config_dir}, found {len(paths)}")
    for path in paths:
        study = load_study_config(path)
        a_train = int(study.data["sizes"]["a_train"])
        if a_train in studies:
            continue
        studies[a_train] = study
    missing = set(TARGET_A_SIZES) - set(studies)
    if missing:
        raise ValueError(f"missing A-train scales: {sorted(missing)}")
    return studies


def _read_source_metadata(config) -> dict[str, np.ndarray]:
    with h5py.File(config.train_file, "r") as handle:
        jets = handle["jets"]
        fields = (
            "HadronConeExclTruthLabelID", "eventNumber",
            "pt_btagJes", "eta_btagJes")
        return _read_fields(jets, fields, 0, _row_count(jets, fields[0]))


def _stratum_key(config, labels, pt, eta, index: int) -> tuple[int, int, int]:
    label = int(config.flavour_to_label[int(labels[index])])
    specification = config.kinematic_resampling
    pt_edges = np.asarray(specification["pt_bins"], dtype=np.float64)
    eta_edges = np.asarray(specification["eta_bins"], dtype=np.float64)
    pt_bin = int(np.clip(np.searchsorted(pt_edges, pt[index], side="right") - 1,
                         0, len(pt_edges) - 2))
    eta_bin = int(np.clip(np.searchsorted(eta_edges, eta[index], side="right") - 1,
                          0, len(eta_edges) - 2))
    return label, pt_bin, eta_bin


def _allocate_counts(capacities: dict[tuple[int, int, int], int], total: int) -> dict[tuple[int, int, int], int]:
    capacity_total = sum(capacities.values())
    if total < 0 or total > capacity_total:
        raise ValueError(f"cannot allocate {total} from {capacity_total} A-train jets")
    keys = sorted(capacities)
    ideal = {
        key: total * capacities[key] / capacity_total
        for key in keys
    }
    counts = {key: min(capacities[key], int(np.floor(ideal[key]))) for key in keys}
    remaining = total - sum(counts.values())
    while remaining:
        candidates = [key for key in keys if counts[key] < capacities[key]]
        candidates.sort(
            key=lambda key: (ideal[key] - counts[key], -capacities[key], key),
            reverse=True)
        if not candidates:
            raise RuntimeError("stratified A-train allocation ran out of capacity")
        for key in candidates[:remaining]:
            counts[key] += 1
        remaining = total - sum(counts.values())
    return counts


def _nested_a_indices(master_a: np.ndarray, config, metadata: dict[str, np.ndarray], sizes: tuple[int, ...], seed: int) -> dict[int, np.ndarray]:
    labels = metadata["HadronConeExclTruthLabelID"]
    pt = metadata["pt_btagJes"]
    eta = metadata["eta_btagJes"]
    rng = np.random.default_rng(seed)
    specification = config.kinematic_resampling
    pt_edges = np.asarray(specification["pt_bins"], dtype=np.float64)
    eta_edges = np.asarray(specification["eta_bins"], dtype=np.float64)
    groups_as_lists: dict[tuple[int, int, int], list[int]] = {}
    total = len(master_a)
    for position, index in enumerate(master_a, start=1):
        index = int(index)
        label = int(config.flavour_to_label[int(labels[index])])
        pt_bin = int(np.clip(np.searchsorted(pt_edges, pt[index], side="right") - 1,
                             0, len(pt_edges) - 2))
        eta_bin = int(np.clip(np.searchsorted(eta_edges, eta[index], side="right") - 1,
                              0, len(eta_edges) - 2))
        groups_as_lists.setdefault((label, pt_bin, eta_bin), []).append(index)
        if position % 500_000 == 0 or position == total:
            _log(f"stratifying A-train master pool: {position:,}/{total:,}")
    groups: dict[tuple[int, int, int], np.ndarray] = {
        key: np.asarray(members, dtype=np.int64)[rng.permutation(len(members))]
        for key, members in sorted(groups_as_lists.items())
    }
    capacities = {key: len(values) for key, values in groups.items()}
    result: dict[int, np.ndarray] = {}
    previous: np.ndarray | None = None
    for size in sorted(sizes):
        quotas = _allocate_counts(capacities, size)
        selected = np.concatenate([
            groups[key][:quotas[key]] for key in sorted(groups)
            if quotas[key]
        ])
        selected = np.sort(selected.astype(np.int64, copy=False))
        if previous is not None and not np.all(np.isin(previous, selected)):
            raise RuntimeError(f"A-train nesting failed at size {size}")
        result[size] = selected
        previous = selected
    return result


def _summary_for_arrays(arrays: dict[str, np.ndarray], config, metadata: dict[str, np.ndarray], master_summary: dict[str, Any]) -> dict[str, Any]:
    event_numbers = metadata["eventNumber"]
    flavours = metadata["HadronConeExclTruthLabelID"]
    event_sets = {
        name: np.unique(event_numbers[indices])
        for name, indices in arrays.items()
    }
    overlaps: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[left_index + 1:]:
            overlaps[f"{left}_{right}"] = {
                "jet_overlap": int(np.intersect1d(arrays[left], arrays[right]).size),
                "event_overlap": int(np.intersect1d(event_sets[left], event_sets[right]).size),
            }
            if any(overlaps[f"{left}_{right}"].values()):
                raise RuntimeError(f"split overlap detected for {left}/{right}")
    class_counts = {
        name: {
            config.jet_class_names[class_index]: int(np.count_nonzero(
                flavours[indices] == flavour))
            for flavour, class_index in config.flavour_to_label.items()
        }
        for name, indices in arrays.items()
    }
    summary = copy.deepcopy(master_summary)
    summary.update({
        "split_version": SPLIT_VERSION,
        "data_seed": int(config.data_seed),
        "selected_jets": {name: int(len(value)) for name, value in arrays.items()},
        "unique_events": {name: int(len(value)) for name, value in event_sets.items()},
        "class_counts": class_counts,
        "index_sha256": {name: _hash(value) for name, value in arrays.items()},
        "event_sha256": {name: _hash(value) for name, value in event_sets.items()},
        "overlaps": overlaps,
        "kinematic_resampling": config.kinematic_resampling,
        "request": _split_request(config),
    })
    return summary


def _write_or_verify_bundle(target_config, arrays, summary, anchor_arrays, force: bool) -> str:
    output = Path(target_config.split_dir)
    index_path = output / "indices.npz"
    manifest_path = output / "split_manifest.json"
    if index_path.exists() and manifest_path.exists() and not force:
        existing = load_split_bundle(output, config=target_config)
        for name in ("a_val", "b_train", "b_val", "y_test"):
            if not np.array_equal(existing.arrays[name], anchor_arrays[name]):
                raise ValueError(f"fixed split {name} differs in {output}; use --force")
        if not np.array_equal(existing.a_train, arrays["a_train"]):
            raise ValueError(f"A-train differs in {output}; use --force")
        return "verified"
    output.mkdir(parents=True, exist_ok=True)
    _save_cache_atomic(str(index_path), arrays)
    write_json_atomic(manifest_path, summary)
    load_split_bundle(output, config=target_config)
    return "written"


def main() -> None:
    args = _parse_args()
    started = time.monotonic()
    _log(f"loading Experiment 2 configs from {args.configs_dir}")
    studies = _load_experiment_studies(args.configs_dir)
    master_study = studies[max(TARGET_A_SIZES)]
    master_config = active_parallel_config(master_study, master_study.seeds[0])
    _log(f"generating/verifying A=3M master split: {master_config.split_dir}")
    master_bundle = generate_split_bundle(master_config, force=args.force)
    _log(
        "master split ready: "
        + ", ".join(f"{name}={len(values):,}" for name, values in master_bundle.arrays.items()))
    _log(f"reading source labels and kinematics: {master_config.train_file}")
    metadata = _read_source_metadata(master_config)
    _log("source metadata ready; deriving nested A-train subsets")
    nested = _nested_a_indices(
        master_bundle.a_train, master_config, metadata, TARGET_A_SIZES,
        args.a_order_seed)
    anchor_arrays = {name: np.asarray(master_bundle.arrays[name]) for name in SPLIT_NAMES}
    for size in TARGET_A_SIZES:
        target_study = studies[size]
        target_config = active_parallel_config(target_study, target_study.seeds[0])
        arrays = dict(anchor_arrays)
        arrays["a_train"] = nested[size]
        summary = _summary_for_arrays(arrays, target_config, metadata, master_bundle.summary)
        summary["shared_anchor"] = {
            "source_master_split": str(master_config.split_dir),
            "fixed_splits": ["a_val", "b_train", "b_val", "y_test"],
            "a_train_order_seed": args.a_order_seed,
        }
        _log(f"writing/verifying A={size:,} split bundle: {target_config.split_dir}")
        status = _write_or_verify_bundle(
            target_config, arrays, summary, anchor_arrays, args.force)
        _log(f"A={size:,}: {status}")
    _log(f"Experiment 2 shared split preparation complete in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
