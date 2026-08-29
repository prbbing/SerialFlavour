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
    write_experiment_manifest)
from src.parallel_refine.data import load_processed_split
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
    args = parser.parse_args(argv)
    if args.processed_split and not args.build_processed_caches:
        parser.error("--processed-split requires --build-processed-caches")
    study = load_study_config(args.config)
    print(f"experiment_manifest={write_experiment_manifest(study)}")
    run = study.seeds[0]
    resolved = materialize_parallel_config(study, run, stage="data")
    config = active_parallel_config(study, run, stage="data")
    bundle = generate_split_bundle(config, force=args.force)
    processed_splits = set(args.processed_split or bundle.arrays)
    print(f"resolved_config={resolved}")
    print(f"split_dir={config.split_dir}")
    for name, indices in bundle.arrays.items():
        print(
            f"{name}: jets={len(indices):,} "
            f"events={bundle.summary['unique_events'][name]:,} "
            f"sha256={bundle.summary['index_sha256'][name]}")
        if args.build_processed_caches and name in processed_splits:
            processed = load_processed_split(
                config, name, force=args.force, progress=True)
            print(f"  retained_after_track_selection={len(processed['y']):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
