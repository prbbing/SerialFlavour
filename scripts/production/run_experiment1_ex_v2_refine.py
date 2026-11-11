#!/usr/bin/env python3
"""Supplement existing EX V2 runs with new refiners; never train upstream."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parallel_refine.config import (
    GRAPH_RECIPES, load_study_config, write_experiment_manifest,
)

RECIPES = ("F1_double", "F2_origin", "F2_vertex", "FG1_48d", "FG2_32d")
CONFIG_DIR = ROOT / "configs/parallel_refine/experiments/experiment1_ex_v2"

# Explicit parent-config inventory, grouped by total dataset size and A-train
# percentage. Default runs select 95/5 through 50/50; 100/0 is listed for
# completeness and is skipped when explicitly selected (there is no B-train).
PARENT_CONFIGS = {
    1: {
        100: "experiment1_ex_v2_p122k_n1m_a100_b000.json",
        95: "experiment1_ex_v2_p122k_n1m_a095_b005.json",
        90: "experiment1_ex_v2_p122k_n1m_a090_b010.json",
        85: "experiment1_ex_v2_p122k_n1m_a085_b015.json",
        80: "experiment1_ex_v2_p122k_n1m_a080_b020.json",
        75: "experiment1_ex_v2_p122k_n1m_a075_b025.json",
        70: "experiment1_ex_v2_p122k_n1m_a070_b030.json",
        65: "experiment1_ex_v2_p122k_n1m_a065_b035.json",
        60: "experiment1_ex_v2_p122k_n1m_a060_b040.json",
        55: "experiment1_ex_v2_p122k_n1m_a055_b045.json",
        50: "experiment1_ex_v2_p122k_n1m_a050_b050.json",
    },
    3: {
        100: "experiment1_ex_v2_p122k_n3m_a100_b000.json",
        95: "experiment1_ex_v2_p122k_n3m_a095_b005.json",
        90: "experiment1_ex_v2_p122k_n3m_a090_b010.json",
        85: "experiment1_ex_v2_p122k_n3m_a085_b015.json",
        80: "experiment1_ex_v2_p122k_n3m_a080_b020.json",
        75: "experiment1_ex_v2_p122k_n3m_a075_b025.json",
        70: "experiment1_ex_v2_p122k_n3m_a070_b030.json",
        65: "experiment1_ex_v2_p122k_n3m_a065_b035.json",
        60: "experiment1_ex_v2_p122k_n3m_a060_b040.json",
        55: "experiment1_ex_v2_p122k_n3m_a055_b045.json",
        50: "experiment1_ex_v2_p122k_n3m_a050_b050.json",
    },
    5: {
        100: "experiment1_ex_v2_p122k_n5m_a100_b000.json",
        95: "experiment1_ex_v2_p122k_n5m_a095_b005.json",
        90: "experiment1_ex_v2_p122k_n5m_a090_b010.json",
        85: "experiment1_ex_v2_p122k_n5m_a085_b015.json",
        80: "experiment1_ex_v2_p122k_n5m_a080_b020.json",
        75: "experiment1_ex_v2_p122k_n5m_a075_b025.json",
        70: "experiment1_ex_v2_p122k_n5m_a070_b030.json",
        65: "experiment1_ex_v2_p122k_n5m_a065_b035.json",
        60: "experiment1_ex_v2_p122k_n5m_a060_b040.json",
        55: "experiment1_ex_v2_p122k_n5m_a055_b045.json",
        50: "experiment1_ex_v2_p122k_n5m_a050_b050.json",
    },
}


def extension(parent_path, suffix):
    """Keep component references relative to the parent directory."""
    parent_path = Path(parent_path).resolve()
    parent = load_study_config(parent_path)
    if parent.data["sizes"]["b_train"] == 0:
        return None
    raw = json.loads(parent_path.read_text(encoding="utf-8"))
    raw["experiment"]["name"] = f"{parent.study_name}_{suffix}"
    raw["experiment"]["upstream_experiment_name"] = parent.upstream_experiment_name
    # Always isolate downstream results from the original run.
    raw["experiment"].pop("refiner_output_experiment_name", None)
    markers = raw["experiment"].setdefault("markers", {})
    markers["label"] = f"{parent.study_name}: supplementary refine inputs"
    markers["tags"] = list(dict.fromkeys(
        markers.get("tags", []) + ["supplementary-refine"]))
    markers.setdefault("variables", {})["refiner_recipes"] = ",".join(RECIPES)
    raw.setdefault("overrides", {}).setdefault("refiners", {})["recipes"] = list(RECIPES)
    path = parent_path.with_name(f"{raw['experiment']['name']}.json")
    if path == parent_path:
        raise ValueError("extension must have a new config path")
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != raw:
        raise ValueError(f"extension config differs; choose another --suffix: {path}")
    return parent, path, raw


def commands(path, seed, stage):
    common = ["--config", str(path), "--seed", str(seed)]
    if stage in {"train", "all"}:
        yield "b_cache", ["scripts/generate_cache.py", *common,
                          "--split", "b_train", "--split", "b_val"]
        for recipe in RECIPES:
            script = "train_graph_refiner.py" if recipe in GRAPH_RECIPES else "train_dnn.py"
            yield recipe, [f"scripts/{script}", *common,
                           "--recipe", recipe, "--skip-complete"]
    if stage in {"evaluate", "all"}:
        yield "y_cache", ["scripts/generate_cache.py", *common, "--split", "y_test"]
        yield "evaluate", ["scripts/evaluate.py", *common, "--model", "parallel_dnn"]


def preflight(study, runs, stage):
    """Require existing upstream and splits before any subprocess is launched."""
    for filename in ("indices.npz", "split_manifest.json"):
        path = Path(study.data["split_dir"]) / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing original split: {path}")
    for run in runs:
        for path in (study.checkpoint(run), study.parallel_directory(run) / "last.pt"):
            if not path.is_file():
                raise FileNotFoundError(f"missing completed upstream artifact: {path}")
        if stage == "evaluate":
            for recipe in RECIPES:
                marker = "last_graph_refiner.pt" if recipe in GRAPH_RECIPES else "last_dnn.pt"
                for seed in study.downstream_seeds:
                    path = study.refiner_directory(run, recipe, seed) / marker
                    if not path.is_file():
                        raise FileNotFoundError(f"train refiners before evaluation: {path}")


def run_command(cmd, gpu, log_path, python):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full = [python, *cmd]
    print(f"GPU {gpu}: {shlex.join(full)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nCOMMAND: {shlex.join(full)}\n")
        log.flush()
        subprocess.run(full, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-m", type=int, choices=(1, 3, 5), default=3)
    parser.add_argument("--allocation", type=int, action="append", choices=range(50, 101, 5),
                        help="A-train percent; repeat to select allocations. Default: 50..95.")
    parser.add_argument("--config", type=Path, action="append",
                        help="Explicit parent config; repeat. Overrides --total-m selection.")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--seed", type=int, action="append", help="Upstream seed; repeat.")
    parser.add_argument("--suffix", default="refine_inputs_v1")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="train",
                        help="Default train uses only B. evaluate/all explicitly enable locked Y.")
    parser.add_argument("--log-base", type=Path,
                        default=ROOT / "logs/parallel_refine/experiment1_ex_v2/refine_inputs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without writing configs/logs or requiring remote artifacts.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.suffix):
        parser.error("--suffix must contain only letters, digits, underscores and hyphens")
    if len(set(args.gpus)) != len(args.gpus) or any(not g.isdecimal() for g in args.gpus):
        parser.error("--gpus must be distinct numeric physical GPU IDs")
    if args.config and args.allocation:
        parser.error("use --config or --allocation, not both")
    catalog = PARENT_CONFIGS[args.total_m]
    allocations = args.allocation if args.allocation is not None else [
        allocation for allocation in catalog if allocation != 100]
    paths = args.config or [CONFIG_DIR / catalog[allocation] for allocation in allocations]
    paths = [p.resolve() for p in paths]
    if len(set(paths)) != len(paths):
        parser.error("parent configurations must be distinct")

    plans = []
    for parent_path in paths:
        plan = extension(parent_path, args.suffix)
        if plan is None:
            print(f"Skip {parent_path.name}: B-train is zero")
            continue
        parent, path, raw = plan
        runs = parent.selected_seeds(args.seed)
        print(f"Parent config: {parent_path}")
        print(f"Extension config: {path}")
        print(f"{parent.study_name} -> {raw['experiment']['name']}: "
              f"{len(runs)} upstream seeds x {len(parent.downstream_seeds)} downstream seeds "
              f"x {len(RECIPES)} recipes; stage={args.stage}")
        print(f"  Reuse upstream: {parent.parallel_output_directory}")
        print(f"  Results: {Path(raw['experiment']['output_root']) / raw['experiment']['name']}")
        plans.append((path, raw, runs))
        if args.dry_run:
            for run in runs:
                for label, cmd in commands(path, run.seed, args.stage):
                    print(f"  seed={run.seed} {label}: {shlex.join([args.python, *cmd])}")
    if args.dry_run:
        print("Dry run: no files written; checkpoint/split availability not checked.")
        return 0

    studies = []
    # Validate all dependencies before launching any training or evaluation.
    for path, raw, runs in plans:
        if not path.exists():
            with path.open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
        study = load_study_config(path)
        preflight(study, runs, args.stage)
        manifest = study.data_directory / "experiment_manifest.json"
        if manifest.exists():
            existing = json.loads(manifest.read_text(encoding="utf-8"))
            if existing.get("experiment_config_sha256") != study.source_sha256:
                raise ValueError(f"extension settings changed; choose another --suffix: {manifest}")
        studies.append((study, runs))
    for study, _ in studies:
        write_experiment_manifest(study)

    jobs = Queue()
    for study, runs in studies:
        for run in runs:
            jobs.put((study, run))

    def worker(gpu):
        failures = []
        while True:
            try:
                study, run = jobs.get_nowait()
            except Empty:
                return failures
            log_dir = args.log_base / study.study_name / f"seed{run.seed}"
            try:
                for label, cmd in commands(study.path, run.seed, args.stage):
                    run_command(cmd, gpu, log_dir / f"{label}.log", args.python)
            except Exception as error:
                failures.append((study.study_name, run.seed, str(error)))
                print(f"FAILED {study.study_name} seed={run.seed}: {error}; logs: {log_dir}",
                      file=sys.stderr, flush=True)
            finally:
                jobs.task_done()

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        results = list(pool.map(worker, args.gpus))
    failed = [failure for result in results for failure in result]
    if failed:
        print(f"Failed seed pipelines: {failed}", file=sys.stderr)
        return 1
    if args.stage in {"evaluate", "all"}:
        for study, runs in studies:
            if len(runs) != len(study.seeds):
                print(f"Skip cross-seed aggregate for {study.study_name}: subset of seeds selected")
                continue
            run_command(["scripts/evaluate.py", "--config", str(study.path),
                         "--model", "parallel_dnn", "--aggregate-parallel-seeds"],
                        args.gpus[0], args.log_base / study.study_name / "aggregate.log", args.python)
    print("Complete. Shared frozen/graph caches retained; no upstream training or cache pruning.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
