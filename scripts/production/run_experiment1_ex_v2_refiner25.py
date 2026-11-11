#!/usr/bin/env python3
"""Train refiners on a whole-event subset of the frozen 3M 100/0 Transformer."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parallel_refine.config import (
    GRAPH_RECIPES, active_parallel_config, load_study_config,
    write_experiment_manifest,
)


CONFIG = (ROOT / "configs/parallel_refine/experiments/experiment1_ex_v2"
          / "experiment1_ex_v2_p122k_n3m_a100_b000_refiner25.json")


def prepare_split(study):
    import numpy as np

    from src.parallel_refine.splits import _hash, _split_request, load_split_bundle

    config = active_parallel_config(study, study.seeds[0], stage="data")
    request = _split_request(config)
    selection = study.data["refiner_train_selection"]
    parent_path = Path(selection["parent_split_dir"])
    destination = Path(config.split_dir)
    if destination.resolve() == parent_path.resolve():
        raise ValueError("Derived split needs a separate directory")
    parent = load_split_bundle(parent_path)
    parent_request = {**request, "sizes": {**request["sizes"], "b_train": 0}}
    if parent.summary["request"] != parent_request:
        raise ValueError("Parent split does not match the upstream data settings")
    if not np.array_equal(parent.a_val, parent.b_val):
        raise ValueError("Expected the original shared validation split")

    with np.load(selection["file"], allow_pickle=False) as sample:
        indices = sample["jet_indices"]
        events = sample["event_numbers"]
        metadata = json.loads(sample["metadata_json"].item())
    if (_hash(indices) != selection["jet_index_sha256"]
            or _hash(events) != selection["event_sha256"]):
        raise ValueError("Refiner sample differs from the pinned event selection")
    if (len(indices) != config.b_train
            or not np.isin(indices, parent.a_train, assume_unique=True).all()):
        raise ValueError("Refiner train must be the configured subset of upstream train")
    source_hashes = parent.summary["index_sha256"]
    if (metadata["parent_index_sha256"]
            != {key: source_hashes[key] for key in ("a_train", "a_val", "y_test")}
            or metadata["validation_event_overlap"] != 0
            or metadata["y_test_event_overlap"] != 0):
        raise ValueError("Refiner selection provenance does not match the parent split")

    arrays = {**parent.arrays, "b_train": indices}
    summary = parent.summary
    summary["request"] = request
    summary["selected_jets"]["b_train"] = len(indices)
    summary["unique_events"]["b_train"] = len(events)
    summary["class_counts"]["b_train"] = {
        config.jet_class_names[label]: metadata["selected_flavour_counts"][str(flavour)]
        for flavour, label in config.flavour_to_label.items()
    }
    summary["index_sha256"]["b_train"] = selection["jet_index_sha256"]
    summary["event_sha256"]["b_train"] = selection["event_sha256"]
    # Since B is a subset of A, its overlap with the parent's holdouts is zero.
    summary["overlaps"]["a_train_b_train"] = {
        "jet_overlap": len(indices), "event_overlap": len(events),
    }
    summary["refiner_train_selection"] = {
        **selection, "parent": "a_train", "whole_events": True,
        "rng_seed": metadata["rng_seed"],
    }
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as temp:
            pending = Path(temp) / "bundle"
            pending.mkdir()
            np.savez_compressed(pending / "indices.npz", **arrays)
            (pending / "split_manifest.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            try:
                pending.rename(destination)
            except OSError:
                if not destination.is_dir():
                    raise
    published = load_split_bundle(destination, config=config)
    if published.summary != summary:
        raise ValueError(f"Existing derived split differs from the planned split: {destination}")
    print(f"Split: {destination} ({len(indices):,} jets, {len(events):,} events)", flush=True)


def commands(study, run, stage):
    common = ["--config", str(study.path), "--seed", str(run.seed)]
    if stage == "train":
        yield "b_cache", ["scripts/generate_cache.py", *common,
                          "--split", "b_train", "--split", "b_val"]
        for recipe in study.refiners["recipes"]:
            trainer = "train_graph_refiner.py" if recipe in GRAPH_RECIPES else "train_dnn.py"
            yield recipe, [f"scripts/{trainer}", *common, "--recipe", recipe, "--skip-complete"]
    if stage == "evaluate":
        yield "y_cache", ["scripts/generate_cache.py", *common, "--split", "y_test"]
        yield "evaluate", ["scripts/evaluate.py", *common, "--model", "parallel_dnn"]


def run_command(command, gpu, log_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    full = [sys.executable, *command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"GPU {gpu}: {shlex.join(full)} (log: {log_path})", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nCOMMAND: {shlex.join(full)}\n")
        log.flush()
        subprocess.run(full, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def preflight(study, runs, stage):
    parent_dir = Path(study.data["refiner_train_selection"]["parent_split_dir"])
    parent_manifest = json.loads((parent_dir / "split_manifest.json").read_text(encoding="utf-8"))
    for run in runs:
        checkpoint = study.checkpoint(run)
        saved_config = checkpoint.parent / "config.json"
        saved_split = checkpoint.parent / "split_manifest.json"
        for path in (checkpoint, saved_config, saved_split):
            if not path.is_file():
                raise FileNotFoundError(f"Missing upstream artifact: {path}")
        config = active_parallel_config(study, run)
        upstream = json.loads(saved_config.read_text(encoding="utf-8"))
        for key in ("train_file", "track_fields", "jet_fields", "top_k",
                    "truth_vertex", "normalization", "a_train"):
            if upstream.get(key) != getattr(config, key):
                raise ValueError(f"Upstream preprocessing differs for {key}: {saved_config}")
        if Path(upstream["split_dir"]) != parent_dir:
            raise ValueError(f"Checkpoint used a different upstream split: {saved_config}")
        upstream_manifest = json.loads(saved_split.read_text(encoding="utf-8"))
        if upstream_manifest["index_sha256"] != parent_manifest["index_sha256"]:
            raise ValueError(f"Checkpoint split indices differ from the parent split: {saved_split}")
        if stage == "evaluate":
            for recipe in study.refiners["recipes"]:
                kind = "graph_refiner" if recipe in GRAPH_RECIPES else "dnn"
                for seed in study.downstream_seeds:
                    directory = study.refiner_directory(run, recipe, seed)
                    for filename in (f"best_{kind}.pt", f"last_{kind}.pt"):
                        if not (directory / filename).is_file():
                            raise FileNotFoundError(f"Train refiner before evaluation: {directory / filename}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", default=None,
                        help="Physical GPU IDs. Defaults to CUDA_VISIBLE_DEVICES or GPU 0.")
    parser.add_argument("--seed", type=int, action="append", help="Upstream seed. Repeat to select multiple.")
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate"), default="train")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    gpus = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if any(not gpu.isdecimal() for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error("--gpus requires distinct numeric device IDs")

    study = load_study_config(CONFIG)
    runs = study.selected_seeds(args.seed)
    log_base = ROOT / "logs/parallel_refine/experiment1_ex_v2/refiner25"
    plan = [(run, list(commands(study, run, args.stage))) for run in runs]
    aggregate = (
        args.stage == "evaluate" and len(runs) == len(study.seeds)
    )
    aggregate_command = ["scripts/evaluate.py", "--config", str(study.path),
                         "--model", "parallel_dnn", "--aggregate-parallel-seeds"]
    processed_splits = ("b_train", "b_val") if args.stage == "train" else ("y_test",)
    processed_command = ["scripts/prepare_data.py", "--config", str(study.path),
                         "--build-processed-caches"]
    for split in processed_splits:
        processed_command.extend(("--processed-split", split))
    print(f"Stage: {args.stage} | Config: {study.path}", flush=True)
    print(f"Derived split: {study.data['split_dir']}", flush=True)
    print(f"Results: {study.output_directory}", flush=True)
    if args.dry_run and args.stage != "prepare":
        print(shlex.join([sys.executable, *processed_command]))
    for run, steps in plan:
        print(f"Upstream seed {run.seed}: {study.checkpoint(run)}", flush=True)
        if args.dry_run:
            for _, command in steps:
                print(shlex.join([sys.executable, *command]))
    if args.dry_run:
        if aggregate:
            print(shlex.join([sys.executable, *aggregate_command]))
        return 0
    if args.stage == "prepare":
        prepare_split(study)
        return 0

    import fcntl

    preflight(study, runs, args.stage)
    study.output_directory.mkdir(parents=True, exist_ok=True)
    with (study.output_directory / ".refiner25.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare_split(study)
        write_experiment_manifest(study)
        run_command(processed_command, gpus[0], log_base / f"{args.stage}_data.log")

        def run_on_gpu(slot):
            for run, steps in plan[slot::len(gpus)]:
                for label, command in steps:
                    log = log_base / f"seed{run.seed}" / f"{label}.log"
                    run_command(command, gpus[slot], log)

        slots = range(min(len(gpus), len(plan)))
        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            list(pool.map(run_on_gpu, slots))
        if aggregate:
            run_command(aggregate_command, gpus[0], log_base / "aggregate.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
