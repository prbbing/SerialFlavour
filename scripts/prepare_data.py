#!/usr/bin/env python3
"""Create config-sized event-disjoint A/B/Y splits and optional processed caches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parallel_refine.config import (
    active_parallel_config, load_study_config, materialize_parallel_config,
    write_experiment_manifest, write_json_atomic)
from src.parallel_refine.data import default_cache_workers, load_processed_splits
from src.parallel_refine.splits import generate_split_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--build-processed-caches", action="store_true")
    parser.add_argument(
        "--processed-split", action="append",
        choices=("a_train", "a_val", "b_train", "b_val", "y_test"),
        help=("Build only this processed split; repeat for multiple splits. "
              "Requires --build-processed-caches. The default builds all splits."))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=None,
        help=("parallel processes for --build-processed-caches "
              "(default: min(cpu_count, 16))"))
    args = parser.parse_args(argv)
    if args.processed_split and not args.build_processed_caches:
        parser.error("--processed-split requires --build-processed-caches")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be a positive integer")
    workers = args.workers or default_cache_workers()
    study = load_study_config(args.config)
    print(f"experiment_manifest={write_experiment_manifest(study)}")
    run = study.seeds[0]
    resolved = materialize_parallel_config(study, run, stage="data")
    config = active_parallel_config(study, run, stage="data")
    bundle = generate_split_bundle(config, force=args.force)
    processed_splits = set(args.processed_split or bundle.arrays)
    processed = {}
    print(f"resolved_config={resolved}")
    print(f"split_dir={config.split_dir}")
    for name, indices in bundle.arrays.items():
        print(
            f"{name}: jets={len(indices):,} "
            f"events={bundle.summary['unique_events'][name]:,} "
            f"sha256={bundle.summary['index_sha256'][name]}")
    if args.build_processed_caches:
        requested = [
            name for name in bundle.arrays if name in processed_splits]
        summary = load_processed_splits(
            config, requested, force=args.force, progress=True,
            workers=workers)
        for name in requested:
            processed[name] = summary[name]
            print(
                f"  retained_after_track_selection="
                f"{processed[name]['retained_after_track_selection']:,}")
    write_json_atomic(study.data_directory / "data_preparation_manifest.json", {
        "stage": "data",
        "experiment_config": str(study.path),
        "experiment_config_sha256": study.source_sha256,
        "split_directory": str(Path(config.split_dir).resolve()),
        "split_manifest": bundle.summary,
        "processed": processed,
        "resolved_config": str(resolved.resolve()),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
