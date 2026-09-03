#!/usr/bin/env python3
"""Generate checkpoint-bound B/Y feature tables for every configured seed."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.parallel_refine.cache import generate_frozen_cache
from src.parallel_refine.config import (
    GRAPH_RECIPES, load_study_config, write_experiment_manifest)
from src.parallel_refine.graph_cache import generate_graph_cache


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--split", action="append")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    study = load_study_config(args.config)
    print(f"experiment_manifest={write_experiment_manifest(study)}")
    selected = study.selected_seeds(args.seed)
    splits = args.split or study.cache.get(
        "splits", ["b_train", "b_val", "y_test"])
    unknown = set(splits) - set(study.cache.get("splits", splits))
    if unknown:
        raise ValueError(f"split(s) not enabled by feature_cache.splits: {sorted(unknown)}")
    for run in selected:
        device = _device()
        print(f"seed={run.seed} device={device}")
        for split in splits:
            cache = generate_frozen_cache(
                study, run, split, device, force=args.force)
            print(
                f"cache seed={run.seed} split={split} rows={len(cache.labels):,} "
                f"features={cache.features.shape[1]} dir={cache.directory}")
            if any(recipe in GRAPH_RECIPES for recipe in study.refiners["recipes"]):
                graph = generate_graph_cache(
                    study, run, split, device, force=args.force)
                print(
                    f"graph_cache seed={run.seed} split={split} "
                    f"rows={len(graph.labels):,} tracks={graph.track_mask.shape[1]} "
                    f"embedding={graph.track_embedding.shape[-1]} dir={graph.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
