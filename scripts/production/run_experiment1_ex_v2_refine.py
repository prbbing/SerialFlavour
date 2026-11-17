#!/usr/bin/env python3
"""Supplement existing EX V2 runs with new refiners; never train upstream."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parallel_refine.cache_workspace import PRIVATE_CACHE_ENV, cache_workspace
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


def extension(parent_path, suffix, recipes=RECIPES):
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
    markers.setdefault("variables", {})["refiner_recipes"] = ",".join(recipes)
    raw.setdefault("overrides", {}).setdefault("refiners", {})["recipes"] = list(recipes)
    path = parent_path.with_name(f"{raw['experiment']['name']}.json")
    if path == parent_path:
        raise ValueError("extension must have a new config path")
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != raw:
        raise ValueError(f"extension config differs; choose another --suffix: {path}")
    return parent, path, raw


def command_groups(path, seed, stage, recipes, parallel_downstream_seeds=()):
    """Run each group to completion on one GPU before starting the next."""
    common = ["--config", str(path), "--seed", str(seed)]
    if stage in {"train", "all"}:
        yield [("b_cache", ["scripts/generate_cache.py", *common,
                            "--split", "b_train", "--split", "b_val"])]
        for recipe in recipes:
            script = "train_graph_refiner.py" if recipe in GRAPH_RECIPES else "train_dnn.py"
            base = [f"scripts/{script}", *common, "--recipe", recipe, "--skip-complete"]
            if parallel_downstream_seeds:
                yield [
                    (f"{recipe}.dnn_seed{downstream_seed}",
                     [*base, "--downstream-seed", str(downstream_seed)])
                    for downstream_seed in parallel_downstream_seeds
                ]
            else:
                yield [(recipe, base)]
    if stage in {"evaluate", "all"}:
        yield [("y_cache", ["scripts/generate_cache.py", *common, "--split", "y_test"])]
        yield [("evaluate", ["scripts/evaluate.py", *common, "--model", "parallel_dnn"])]


def commands(path, seed, stage):
    """Sequential command stream used by the original supplementary runs."""
    for group in command_groups(path, seed, stage, RECIPES):
        yield from group


def write_extension_config(path, raw):
    if not path.exists():
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")


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
            for recipe in study.refiners["recipes"]:
                marker = "last_graph_refiner.pt" if recipe in GRAPH_RECIPES else "last_dnn.pt"
                for seed in study.downstream_seeds:
                    path = study.refiner_directory(run, recipe, seed) / marker
                    if not path.is_file():
                        raise FileNotFoundError(f"train refiners before evaluation: {path}")


def stage_complete(study, run, stage, recipes):
    """Recognize finished upstream-seed stages from their published artifacts."""
    for recipe in recipes:
        if stage == "evaluate":
            recipe_dir = study.evaluation_directory(run, recipe, study.downstream_seeds[0]).parent
            if not (recipe_dir / "rejection_comparison_dnn_seed_mean.json").is_file():
                return False
        for seed in study.downstream_seeds:
            if stage == "train":
                directory = study.refiner_directory(run, recipe, seed)
                marker = "last_graph_refiner.pt" if recipe in GRAPH_RECIPES else "last_dnn.pt"
                required = ("run_manifest.json", marker)
            else:
                directory = study.evaluation_directory(run, recipe, seed)
                required = ("evaluation_manifest.json",)
            if not all((directory / file).is_file() for file in required):
                return False
    return True


def run_command(cmd, gpu, log_path, python, *, private_cache=False):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if private_cache:
        env[PRIVATE_CACHE_ENV] = "1"
    if "--downstream-seed" in cmd:
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full = [python, *cmd]
    print(f"GPU {gpu}: {shlex.join(full)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nCOMMAND: {shlex.join(full)}\n")
        log.flush()
        subprocess.run(full, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def run_group(group, gpu, log_dir, python, workers, execute):
    if len(group) == 1:
        label, cmd = group[0]
        execute(cmd, gpu, log_dir / f"{label}.log", python)
        return
    with ThreadPoolExecutor(max_workers=min(workers, len(group))) as pool:
        futures = [
            pool.submit(execute, cmd, gpu, log_dir / f"{label}.log", python)
            for label, cmd in group
        ]
        for future in futures:
            future.result()


@contextmanager
def lock_cache_workspace(study):
    """Prevent two launchers from writing or pruning one experiment's cache."""
    path = study.data_directory / "refiner-cache.lock"
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-m", type=int, choices=(1, 3, 5), nargs="+", default=[3],
                        help="One or more total dataset sizes in millions. Default: 3.")
    parser.add_argument("--allocation", type=int, action="append", choices=range(50, 101, 5),
                        help="A-train percent; repeat to select allocations. Default: 50..95.")
    parser.add_argument("--config", type=Path, action="append",
                        help="Explicit parent config; repeat. Overrides --total-m selection.")
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--seed", type=int, action="append", help="Upstream seed; repeat.")
    parser.add_argument("--recipe", choices=RECIPES, action="append",
                        help="Refiner recipe; repeat. Default: all supplementary recipes.")
    parser.add_argument("--downstream-workers", type=int, default=1,
                        help="Concurrent downstream seeds per GPU within one recipe. Default: 1.")
    parser.add_argument("--suffix", default="refine_inputs_v1")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="train",
                        help="Default train uses only B. evaluate/all explicitly enable locked Y.")
    parser.add_argument("--prune-after-aggregate", action="store_true",
                        help="Run one config at a time and release regenerable private caches before advancing.")
    parser.add_argument("--log-base", type=Path,
                        default=ROOT / "logs/parallel_refine/experiment1_ex_v2/refine_inputs")
    parser.add_argument("--python", default=sys.executable)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print plan without writing configs/logs or requiring remote artifacts.")
    mode.add_argument("--prepare-configs-only", action="store_true",
                      help="Write extension configs without checking or writing runtime artifacts.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.suffix):
        parser.error("--suffix must contain only letters, digits, underscores and hyphens")
    if any(not g.isdecimal() for g in args.gpus) or len({int(g) for g in args.gpus}) != len(args.gpus):
        parser.error("--gpus must be distinct numeric physical GPU IDs")
    args.gpus = [str(int(g)) for g in args.gpus]
    if args.downstream_workers < 1:
        parser.error("--downstream-workers must be positive")
    recipes = tuple(args.recipe) if args.recipe else RECIPES
    if len(set(recipes)) != len(recipes):
        parser.error("recipes must be distinct")
    if args.config and args.allocation:
        parser.error("use --config or --allocation, not both")
    allocations = args.allocation if args.allocation is not None else range(95, 49, -5)
    paths = args.config or [
        CONFIG_DIR / PARENT_CONFIGS[size][allocation]
        for size in args.total_m for allocation in allocations
    ]
    paths = [p.resolve() for p in paths]
    if len(set(paths)) != len(paths):
        parser.error("parent configurations must be distinct")

    stages = (("train", "evaluate")
              if args.prune_after_aggregate and args.stage == "all" else (args.stage,))
    plans = []
    for parent_path in paths:
        plan = extension(parent_path, args.suffix, recipes)
        if plan is None:
            print(f"Skip {parent_path.name}: B-train is zero")
            continue
        parent, path, raw = plan
        runs = parent.selected_seeds(args.seed)
        print(f"Parent config: {parent_path}")
        print(f"Extension config: {path}")
        print(f"{parent.study_name} -> {raw['experiment']['name']}: "
              f"{len(runs)} upstream seeds x {len(parent.downstream_seeds)} downstream seeds "
              f"x {len(recipes)} recipes; stage={args.stage}; "
              f"downstream workers per GPU={min(args.downstream_workers, len(parent.downstream_seeds))}")
        print(f"  Reuse upstream: {parent.parallel_output_directory}")
        print(f"  Results: {Path(raw['experiment']['output_root']) / raw['experiment']['name']}")
        plans.append((path, raw, runs))
        if args.dry_run:
            downstream = parent.downstream_seeds if args.downstream_workers > 1 else ()
            for stage in stages:
                for run in runs:
                    for group in command_groups(path, run.seed, stage, recipes, downstream):
                        for label, cmd in group:
                            print(f"  seed={run.seed} {label}: {shlex.join([args.python, *cmd])}")
            if args.prune_after_aggregate:
                if args.stage in {"evaluate", "all"} and len(runs) == len(parent.seeds):
                    print("  Aggregate all upstream seeds on success.")
                print("  Release private feature caches before advancing to the next configuration.")
    if args.dry_run:
        print("Dry run: no files written; checkpoint/split availability not checked.")
        return 0
    if args.prepare_configs_only:
        for path, raw, _ in plans:
            write_extension_config(path, raw)
        print(f"Prepared {len(plans)} extension configs.")
        return 0

    studies = []
    # Validate all dependencies before launching any training or evaluation.
    for path, raw, runs in plans:
        write_extension_config(path, raw)
        study = load_study_config(path)
        if not args.prune_after_aggregate:
            preflight(study, runs, args.stage)
        studies.append((study, runs))
    if not args.prune_after_aggregate:
        for study, _ in studies:
            write_experiment_manifest(study)

    execute = (partial(run_command, private_cache=True)
               if args.prune_after_aggregate else run_command)

    def report_failure(study, task, error, log_dir):
        print(f"FAILED {study.study_name} {task}: {error}; logs: {log_dir}",
              file=sys.stderr, flush=True)

    def run_jobs(batch, stage):
        jobs = Queue()
        for study, runs in batch:
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
                downstream = study.downstream_seeds if args.downstream_workers > 1 else ()
                try:
                    if args.prune_after_aggregate:
                        try:
                            preflight(study, [run], stage)
                        except Exception as error:
                            failures.append((study.study_name, run.seed))
                            report_failure(study, f"seed={run.seed} {stage} preflight", error, log_dir)
                            continue
                        if stage_complete(study, run, stage, recipes):
                            print(f"Skip completed {stage}: {study.study_name} seed={run.seed}", flush=True)
                            continue
                    seed_failed = False
                    for group in command_groups(study.path, run.seed, stage, recipes, downstream):
                        label = group[0][0]
                        if seed_failed and label == "y_cache":
                            break
                        try:
                            run_group(group, gpu, log_dir, args.python, args.downstream_workers, execute)
                        except Exception as error:
                            seed_failed = True
                            failures.append((study.study_name, run.seed))
                            report_failure(study, f"seed={run.seed} {label}", error, log_dir)
                            if label in {"b_cache", "y_cache"}:
                                break
                finally:
                    jobs.task_done()

        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            return [failure for result in pool.map(worker, args.gpus) for failure in result]

    def aggregate(study, runs):
        if args.stage == "train":
            return False
        if len(runs) != len(study.seeds):
            print(f"Skip cross-seed aggregate for {study.study_name}: subset of seeds selected")
            return False
        execute(["scripts/evaluate.py", "--config", str(study.path),
                 "--model", "parallel_dnn", "--aggregate-parallel-seeds"],
                args.gpus[0], args.log_base / study.study_name / "aggregate.log", args.python)
        return True

    failed_studies = set()
    if args.prune_after_aggregate:
        for study, runs in studies:
            log_dir = args.log_base / study.study_name
            try:
                study.data_directory.mkdir(parents=True, exist_ok=True)
                with lock_cache_workspace(study):
                    write_experiment_manifest(study)
                    failed_seeds = set()
                    for stage in stages:
                        print(f"{study.study_name}: {stage}", flush=True)
                        eligible = [run for run in runs if run.seed not in failed_seeds]
                        failed_seeds.update(seed for _, seed in run_jobs([(study, eligible)], stage))
                    if failed_seeds:
                        failed_studies.add(study.study_name)
                        print(f"Skip aggregate for {study.study_name}: failed seeds {sorted(failed_seeds)}",
                              file=sys.stderr, flush=True)
                    else:
                        try:
                            aggregate(study, runs)
                        except Exception as error:
                            failed_studies.add(study.study_name)
                            report_failure(study, "aggregate", error, log_dir)

                    roots = dict.fromkeys(study.cache[key] for key in ("root", "graph_root"))
                    for root in roots:
                        try:
                            directory = cache_workspace(root, study.study_name)
                            if directory.exists():
                                shutil.rmtree(directory)
                                print(f"Removed private feature cache: {directory}", flush=True)
                        except Exception as error:
                            failed_studies.add(study.study_name)
                            report_failure(study, f"cache cleanup under {root}", error, log_dir)
            except Exception as error:
                failed_studies.add(study.study_name)
                report_failure(study, "configuration setup", error, log_dir)
    else:
        failed_studies.update(name for name, _ in run_jobs(studies, args.stage))
        for study, runs in studies:
            if study.study_name in failed_studies:
                continue
            try:
                aggregate(study, runs)
            except Exception as error:
                failed_studies.add(study.study_name)
                report_failure(study, "aggregate", error, args.log_base / study.study_name)
    if failed_studies:
        print(f"Queue finished with failed configurations: {', '.join(sorted(failed_studies))}",
              file=sys.stderr, flush=True)
        return 1
    print("Complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
