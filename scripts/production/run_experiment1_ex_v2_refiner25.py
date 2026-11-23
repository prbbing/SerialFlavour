#!/usr/bin/env python3
"""Train refiners on a whole-event subset of a frozen 3M Transformer's training set."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parallel_refine.cache_workspace import PRIVATE_CACHE_ENV, cache_workspace
from src.parallel_refine.config import (
    GRAPH_RECIPES, active_parallel_config, load_study_config,
    write_experiment_manifest,
)


CONFIG = (ROOT / "configs/parallel_refine/experiments/experiment1_ex_v2"
          / "experiment1_ex_v2_p122k_n3m_a100_b000_refiner25.json")
TRAIN_SPLITS = ("b_train", "b_val")
EVALUATION_SPLITS = ("y_test",)
SWEEP_RECIPES = ("F3_embed_aux", "FG1_48d", "F1_embed", "F2_origin", "F2_vertex")


def sweep_configs():
    return [CONFIG.with_name(f"experiment1_ex_v2_p122k_n3m_a{a:03d}_b{100-a:03d}_refiner{100-a:02d}.json")
            for a in range(50, 100, 5)]


@dataclass(frozen=True)
class Task:
    name: str
    action: Callable
    after: tuple[str, ...] = ()
    gpu: bool = False
    cache_build: bool = False
    command: list[str] | None = None
    priority: int = 0


def read_jet_fields(jets, fields, indices):
    import numpy as np

    from src.data import _block_slices, _normalise_indices, _read_fields, _row_count

    n_rows = _row_count(jets, fields[0])
    indices = _normalise_indices(indices, n_rows)
    blocks = {field: [] for field in fields}
    for first, last, start, stop in _block_slices(indices, n_rows, 1, 65536):
        values = _read_fields(jets, fields, start, stop)
        local = indices[first:last] - start
        for field in fields:
            blocks[field].append(values[field][local])
    return tuple(np.concatenate(blocks[field]) for field in fields)


def select_whole_events(config, parent, rng_seed):
    import h5py
    import numpy as np

    from src.parallel_refine.splits import _hash

    event_field = "eventNumber"
    flavour_field = "HadronConeExclTruthLabelID"
    with h5py.File(config.train_file, "r") as handle:
        jets = handle["jets"]
        train_events, flavours = read_jet_fields(
            jets, (event_field, flavour_field), parent.a_train)
        events, inverse, counts = np.unique(
            train_events, return_inverse=True, return_counts=True)
        event_sets = {"a_train": events}
        for name in ("a_val", "y_test"):
            (rows,) = read_jet_fields(jets, (event_field,), parent.arrays[name])
            event_sets[name] = np.unique(rows)
    for name, event_ids in event_sets.items():
        if _hash(event_ids) != parent.summary["event_sha256"][name]:
            raise ValueError(f"Parent event hash mismatch for {name}")

    order = np.random.default_rng(rng_seed).permutation(events.size)
    cumulative = np.cumsum(counts[order])
    prefix = int(np.searchsorted(cumulative, config.b_train, side="right"))
    deficit = config.b_train - (int(cumulative[prefix - 1]) if prefix else 0)
    tail = order[prefix:]
    singletons = tail[counts[tail] == 1][:deficit]
    if len(singletons) != deficit:
        raise ValueError("Cannot reach the requested jet count using whole events")
    chosen = np.zeros(events.size, dtype=bool)
    chosen[order[:prefix]] = True
    chosen[singletons] = True
    mask = chosen[inverse]
    selected_events = events[chosen]
    selected_flavours = flavours[mask]
    metadata = {
        "rng_seed": rng_seed,
        "parent_index_sha256": {
            name: parent.summary["index_sha256"][name] for name in event_sets
        },
        "selected_flavour_counts": {
            str(flavour): int(np.count_nonzero(selected_flavours == flavour))
            for flavour in config.flavour_to_label
        },
        "validation_event_overlap": int(np.intersect1d(
            selected_events, event_sets["a_val"], assume_unique=True).size),
        "y_test_event_overlap": int(np.intersect1d(
            selected_events, event_sets["y_test"], assume_unique=True).size),
    }
    return parent.a_train[mask], selected_events, metadata


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
    parent_request = {**request, "sizes": {
        **request["sizes"], "b_train": selection.get("parent_b_train", 0),
    }}
    if parent.summary["request"] != parent_request:
        raise ValueError("Parent split does not match the upstream data settings")
    if not np.array_equal(parent.a_val, parent.b_val):
        raise ValueError("Expected the original shared validation split")

    if "file" in selection:
        with np.load(selection["file"], allow_pickle=False) as sample:
            indices = sample["jet_indices"]
            events = sample["event_numbers"]
            metadata = json.loads(sample["metadata_json"].item())
    else:
        indices, events, metadata = select_whole_events(
            config, parent, selection["rng_seed"])
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


def run_command(command, gpu, log_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu or "", PRIVATE_CACHE_ENV: "1"}
    if "--downstream-seed" in command:
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    full = [sys.executable, *command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    device = "CPU" if gpu is None else f"GPU {gpu}"
    print(f"{device}: {shlex.join(full)} (log: {log_path})", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\nCOMMAND: {shlex.join(full)}\n")
        log.flush()
        subprocess.run(full, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def refiner_checkpoints(study, run, recipe, seed):
    kind = "graph_refiner" if recipe in GRAPH_RECIPES else "dnn"
    directory = study.refiner_directory(run, recipe, seed)
    return directory / f"best_{kind}.pt", directory / f"last_{kind}.pt"


def sweep_has_pending_models(study, runs):
    return any(
        not all(path.is_file() for path in refiner_checkpoints(study, run, recipe, seed))
        for run in runs for recipe in SWEEP_RECIPES for seed in study.downstream_seeds
    )


def prune_feature_caches(study, run, splits, *, graph_only=False):
    keys = ("graph_root",) if graph_only else ("root", "graph_root")
    roots = dict.fromkeys(Path(study.cache[key]).resolve() for key in keys if key in study.cache)
    if graph_only:
        # If both families share a root, keep the split until its last consumer.
        roots.pop(Path(study.cache["root"]).resolve(), None)
    for root in roots:
        workspace = cache_workspace(root, study.study_name)
        for split in splits:
            directory = workspace / run.output_name / split
            if not directory.is_relative_to(workspace) or directory.resolve() != directory:
                raise ValueError(f"Expected a private feature-cache directory: {directory}")
            if directory.is_dir():
                shutil.rmtree(directory)
                print(f"Removed private feature cache: {directory}", flush=True)


def prepare_processed(study, command, log_path):
    prepare_split(study)
    write_experiment_manifest(study)
    run_command(command, None, log_path)


def processed_task(study, stage, log_base, splits):
    processed = ["scripts/prepare_data.py", "--config", str(study.path), "--build-processed-caches"]
    for split in splits:
        processed.extend(("--processed-split", split))
    return Task(f"{study.study_name}/prepare_{stage}",
                partial(prepare_processed, study, processed, log_base / f"{stage}_data.log"),
                command=processed)


def cache_task(study, run, splits, log_base, after, *, label, include_graph=True):
    build = ["scripts/generate_cache.py", "--config", str(study.path), "--seed", str(run.seed)]
    for split in splits:
        build.extend(("--split", split))
    if not include_graph:
        build.append("--no-graph")
    return Task(f"{study.study_name}/seed{run.seed}/{label}",
                partial(run_command, build, log_path=log_base / f"seed{run.seed}" / f"{label}.log"),
                after=after, gpu=True, cache_build=True, command=build)


def training_tasks(study, run, recipe, cache, log_base):
    trainer = "train_graph_refiner.py" if recipe in GRAPH_RECIPES else "train_dnn.py"
    common = ["--config", str(study.path), "--seed", str(run.seed)]
    for seed in study.downstream_seeds:
        best, last = refiner_checkpoints(study, run, recipe, seed)
        if best.is_file() and last.is_file():
            continue
        if last.is_file():
            raise FileNotFoundError(f"Completion marker exists but best checkpoint is missing: {best}")
        command = [f"scripts/{trainer}", *common, "--recipe", recipe,
                   "--downstream-seed", str(seed), "--skip-complete"]
        label = f"{recipe}_dnn_seed{seed}"
        yield Task(f"{study.study_name}/seed{run.seed}/{label}",
                   partial(run_command, command, log_path=log_base / f"seed{run.seed}" / f"{label}.log"),
                   after=(cache.name,), gpu=True, command=command)


def build_tasks(study, runs, stage, log_base, *, evaluation_recipes=()):
    splits = TRAIN_SPLITS if stage == "train" else EVALUATION_SPLITS
    common = ["--config", str(study.path)]
    recipe_args = [arg for recipe in evaluation_recipes for arg in ("--recipe", recipe)]
    prep = processed_task(study, stage, log_base, splits)
    tasks = []
    seed_results = []
    for run in runs:
        prefix = f"{study.study_name}/seed{run.seed}"
        seed_common = [*common, "--seed", str(run.seed)]
        seed_logs = log_base / f"seed{run.seed}"
        cache_label = "b_cache" if stage == "train" else "y_cache"
        cache = cache_task(study, run, splits, log_base, (prep.name,), label=cache_label)
        consumers = []
        if stage == "train":
            recipes = sorted(study.refiners["recipes"], key=lambda recipe: recipe not in GRAPH_RECIPES)
            for recipe in recipes:
                consumers.extend(training_tasks(study, run, recipe, cache, log_base))
            tasks.append(Task(f"{prefix}/prune_train",
                              partial(prune_feature_caches, study, run, splits),
                              after=tuple(task.name for task in consumers)))
        else:
            command = ["scripts/evaluate.py", *seed_common, "--model", "parallel_dnn", *recipe_args]
            consumer = Task(f"{prefix}/evaluate",
                            partial(run_command, command, log_path=seed_logs / "evaluate.log"),
                            after=(cache.name,), gpu=True, command=command)
            consumers.append(consumer)
            seed_results.append(consumer.name)
        if consumers:
            tasks.extend((cache, *consumers))

    if any(task.cache_build for task in tasks):
        tasks.insert(0, prep)
    if stage == "evaluate":
        after = tuple(seed_results)
        if len(runs) == len(study.seeds):
            command = ["scripts/evaluate.py", *common, "--model", "parallel_dnn",
                       "--aggregate-parallel-seeds", *recipe_args]
            aggregate = Task(f"{study.study_name}/aggregate",
                             partial(run_command, command, None, log_base / "aggregate.log"),
                             after=after, command=command)
            tasks.append(aggregate)
            after = (aggregate.name,)
        tasks.extend(Task(f"{study.study_name}/seed{run.seed}/prune_y",
                          partial(prune_feature_caches, study, run, splits), after=after)
                     for run in runs)
    return tasks


def build_sweep_tasks(selected, log_root):
    tasks = []
    for study, runs in selected:
        log_base = log_root / study.study_name
        prep = processed_task(study, "train", log_base, TRAIN_SPLITS)
        study_priorities = []
        for run in runs:
            prefix = f"{study.study_name}/seed{run.seed}"
            structured = cache_task(study, run, TRAIN_SPLITS, log_base, (prep.name,),
                                    label="b_cache_structured", include_graph=False)
            graph = cache_task(study, run, TRAIN_SPLITS, log_base, (prep.name,),
                               label="b_cache_graph")
            dnn_jobs = []
            graph_jobs = []
            for priority, recipe in enumerate(SWEEP_RECIPES):
                is_graph = recipe in GRAPH_RECIPES
                jobs = graph_jobs if is_graph else dnn_jobs
                cache = graph if is_graph else structured
                jobs.extend(replace(task, priority=priority)
                            for task in training_tasks(study, run, recipe, cache, log_base))
            consumers = dnn_jobs + graph_jobs
            tasks.extend(consumers)
            dependencies = tuple(task.name for task in consumers)
            if consumers:
                earliest = min(task.priority for task in consumers)
                study_priorities.append(earliest)
                if dnn_jobs:
                    tasks.append(replace(structured, priority=earliest))
                if graph_jobs:
                    if dnn_jobs:
                        graph = replace(graph, after=(structured.name,))
                        graph_cleanup = Task(
                            f"{prefix}/prune_graph",
                            partial(prune_feature_caches, study, run, TRAIN_SPLITS, graph_only=True),
                            after=tuple(task.name for task in graph_jobs),
                            priority=graph_jobs[0].priority)
                        tasks.append(graph_cleanup)
                        dependencies += (graph_cleanup.name,)
                    tasks.append(replace(graph, priority=graph_jobs[0].priority))
            tasks.append(Task(f"{prefix}/prune_train",
                              partial(prune_feature_caches, study, run, TRAIN_SPLITS),
                              after=dependencies))
        if study_priorities:
            tasks.append(replace(prep, priority=min(study_priorities)))
    return tasks


def run_tasks(tasks, gpus):
    # Prefer the highest-priority ready task, then training over another cache build.
    pending = {task.name: task for task in sorted(tasks, key=task_queue_order)}
    free_gpus = list(gpus)
    cpu_free = True
    active = {}
    outcomes = {}
    with ThreadPoolExecutor(max_workers=len(gpus) + 1) as pool:
        while pending or active:
            progressed = False
            for name, task in list(pending.items()):
                if any(dep not in outcomes for dep in task.after):
                    continue
                if any(not outcomes[dep] for dep in task.after):
                    print(f"Blocked by failed dependency: {name}", file=sys.stderr, flush=True)
                    outcomes[name] = False
                elif task.gpu and free_gpus:
                    gpu = free_gpus.pop(0)
                    active[pool.submit(task.action, gpu)] = (name, gpu)
                elif not task.gpu and cpu_free:
                    cpu_free = False
                    active[pool.submit(task.action)] = (name, None)
                else:
                    continue
                del pending[name]
                progressed = True
            if not active:
                if pending and not progressed:
                    raise RuntimeError(f"Unresolved task dependencies: {list(pending)}")
                continue
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in completed:
                name, gpu = active.pop(future)
                if gpu is None:
                    cpu_free = True
                else:
                    free_gpus.append(gpu)
                error = future.exception()
                outcomes[name] = error is None
                if error is not None:
                    print(f"Failed {name}: {error}", file=sys.stderr, flush=True)
    return all(outcomes.values())


def task_queue_order(task):
    return task.priority, task.cache_build


def preflight(study, runs, stage, *, evaluation_recipes=()):
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
            for recipe in evaluation_recipes or study.refiners["recipes"]:
                for seed in study.downstream_seeds:
                    for path in refiner_checkpoints(study, run, recipe, seed):
                        if not path.is_file():
                            raise FileNotFoundError(f"Train refiner before evaluation: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append",
                        help="Repeat to queue multiple experiments together or add to the allocation sweep.")
    parser.add_argument("--allocation-sweep", action="store_true",
                        help="Queue 3M 50/50 through 95/5 with cross-allocation recipe priorities.")
    parser.add_argument("--gpus", nargs="+", default=None,
                        help="Physical GPU IDs. Defaults to CUDA_VISIBLE_DEVICES or GPU 0.")
    parser.add_argument("--seed", type=int, action="append", help="Upstream seed. Repeat to select multiple.")
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate"), default="train")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    gpus = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    if any(not gpu.isdecimal() for gpu in gpus) or len(set(gpus)) != len(gpus):
        parser.error("--gpus requires distinct numeric device IDs")

    configs = args.config or []
    if args.allocation_sweep:
        configs = [*sweep_configs(), *configs]
    studies = [load_study_config(path) for path in (configs or [CONFIG])]
    if (len({study.study_name for study in studies}) != len(studies)
            or len({study.refiner_output_directory.resolve() for study in studies}) != len(studies)):
        parser.error("Each configuration must have a distinct study name and refiner output directory")
    if args.allocation_sweep:
        for study in studies:
            missing = set(SWEEP_RECIPES) - set(study.refiners["recipes"])
            if missing:
                parser.error(f"{study.path} is missing sweep recipes: {sorted(missing)}")
    selected = [(study, study.selected_seeds(args.seed)) for study in studies]
    if args.allocation_sweep and args.stage == "train":
        pending = []
        for study, runs in selected:
            if sweep_has_pending_models(study, runs):
                pending.append((study, runs))
            else:
                print(f"Skipping completed allocation: {study.study_name}", flush=True)
        selected = pending
        if not selected:
            return 0
    log_root = ROOT / "logs/parallel_refine/experiment1_ex_v2/refiner25"
    evaluation_recipes = SWEEP_RECIPES if args.allocation_sweep else ()

    def plan_tasks():
        if args.allocation_sweep and args.stage == "train":
            return build_sweep_tasks(selected, log_root)
        return [task for study, runs in selected
                for task in build_tasks(study, runs, args.stage, log_root / study.study_name,
                                        evaluation_recipes=evaluation_recipes)]

    for study, runs in selected:
        print(f"Stage: {args.stage} | Config: {study.path}", flush=True)
        print(f"Derived split: {study.data['split_dir']}", flush=True)
        print(f"Results: {study.refiner_output_directory}", flush=True)
        for run in runs:
            print(f"Upstream seed {run.seed}: {study.checkpoint(run)}", flush=True)
    if args.dry_run:
        if args.stage != "prepare":
            for task in sorted(plan_tasks(), key=task_queue_order):
                if task.command:
                    print(shlex.join([sys.executable, *task.command]))
                else:
                    print(task.name)
        return 0
    if args.stage == "prepare":
        for study in studies:
            prepare_split(study)
        return 0

    import fcntl

    for study, runs in selected:
        preflight(study, runs, args.stage, evaluation_recipes=evaluation_recipes)
    with ExitStack() as stack:
        directories = sorted(study.refiner_output_directory.resolve() for study, _ in selected)
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            lock_path = directory / ".refiner25.lock"
            lock = stack.enter_context(lock_path.open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SystemExit(f"Result directory is locked by another launcher: {lock_path}") from error
        return 0 if run_tasks(plan_tasks(), gpus) else 1


if __name__ == "__main__":
    raise SystemExit(main())
