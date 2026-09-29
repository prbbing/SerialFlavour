#!/usr/bin/env python3
"""Single Parallel-seed full lifecycle for Experiment 1 EX.

For exactly one Parallel seed of one experiment config, run the complete
downstream chain pinned to one GPU:

    stage 2 (Parallel training)
      -> stage 3 (B-train/B-val frozen + graph caches)
      -> stage 4 (DNN + graph-refiner training)
      -> stage 5 (Y-test frozen + graph caches)
      -> stage 6 (locked-Y evaluation)
      -> prune this seed's frozen/graph caches

Stages run sequentially on the single GPU.  Each stage is retried
(``--retries``) after removing only this seed's incomplete artifacts.  After
evaluation succeeds, prune deletes this seed's frozen/graph B and Y caches;
processed caches are shared across configs and are never deleted.

Run from anywhere; paths are resolved from the repository root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parallel_refine.cache import cache_directory
from src.parallel_refine.config import (
    FEATURE_RECIPES, GRAPH_RECIPES, active_parallel_config,
    load_study_config, write_experiment_manifest)
from src.parallel_refine.graph_cache import graph_cache_directory
from src.parallel_refine.splits import load_split_bundle


PYTHON = sys.executable
DEFAULT_GPUS = "0"
CACHE_SPLITS = ("b_train", "b_val", "y_test")

_LOG_PATH: Path | None = None


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
    print(line, flush=True)
    if _LOG_PATH is not None:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class StageError(RuntimeError):
    pass


def _env(gpu, *, training=False):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if training:
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
    return env


def _open_log(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path.open("w", encoding="utf-8")


def _subprocess_task(cmd, log_path, gpu, *, training=False):
    def run():
        handle = _open_log(log_path)
        try:
            process = subprocess.Popen(
                cmd, stdout=handle, stderr=subprocess.STDOUT,
                env=_env(gpu, training=training), cwd=str(ROOT))
            return process.wait()
        finally:
            handle.close()

    return run


def _run_stage(stage, build_tasks, retries, clean_partial):
    """Run every task of one stage in order, retrying the whole stage."""
    for attempt in range(retries + 1):
        results = []
        for label, runner in build_tasks():
            _log(f"{stage}: start {label}")
            try:
                rc = runner()
            except BaseException as error:  # noqa: BLE001 - reported as failure
                _log(f"{stage}: error {label}: {error}")
                rc = 1
            _log(f"{stage}: {'done' if rc == 0 else 'FAIL'} {label} rc={rc}")
            results.append((label, rc))
        failed = [label for label, rc in results if rc != 0]
        if not failed:
            return
        _log(f"{stage}: failed {sorted(failed)} (attempt {attempt + 1})")
        if attempt >= retries:
            raise StageError(f"{stage} failed: {sorted(failed)}")
        _log(f"{stage}: cleaning partial artifacts before retry")
        clean_partial()


def _selected_recipes(study):
    dnn = [r for r in study.refiners["recipes"] if r in FEATURE_RECIPES]
    graph = [r for r in study.refiners["recipes"] if r in GRAPH_RECIPES]
    return dnn, graph


def _parallel_cmd(study, run, patience, force=False):
    cmd = [
        PYTHON, "scripts/train_parallel.py", "--config", str(study.path),
        "--seed", str(run.seed), "--patience", str(patience),
    ]
    if not force:
        cmd.append("--skip-complete")
    return cmd


def _cache_cmd(study, run, splits, force=False):
    cmd = [
        PYTHON, "scripts/generate_cache.py", "--config", str(study.path),
        "--seed", str(run.seed),
    ]
    for split in splits:
        cmd += ["--split", split]
    if force:
        cmd.append("--force")
    return cmd


def _train_cmd(study, run, trainer, recipe, downstream_seed):
    script = (
        "scripts/train_dnn.py" if trainer == "dnn"
        else "scripts/train_graph_refiner.py")
    return [
        PYTHON, script, "--config", str(study.path), "--seed", str(run.seed),
        "--downstream-seed", str(downstream_seed), "--recipe", recipe,
        "--skip-complete",
    ]


def _eval_cmd(study, run, model):
    return [
        PYTHON, "scripts/evaluate.py", "--config", str(study.path),
        "--seed", str(run.seed), "--model", model,
    ]


def _train_unit(study, run, trainer, recipes, downstream_seeds, log_dir, gpu,
                tag=None):
    """Launch one child per downstream seed; each child walks its recipes."""

    def run_unit():
        env = _env(gpu, training=True)
        log_tag = tag or trainer
        children = []
        for downstream_seed in downstream_seeds:
            log_path = log_dir / "train" / (
                f"{run.output_name}_{log_tag}_d{downstream_seed}.log")
            handle = _open_log(log_path)
            # One child per downstream seed, recipes walked in sequence.
            script = (
                "scripts/train_dnn.py" if trainer == "dnn"
                else "scripts/train_graph_refiner.py")
            cmd = [
                PYTHON, script, "--config", str(study.path),
                "--seed", str(run.seed),
                "--downstream-seed", str(downstream_seed),
            ]
            for recipe in recipes:
                cmd += ["--recipe", recipe]
            cmd += ["--skip-complete"]
            process = subprocess.Popen(
                cmd, stdout=handle, stderr=subprocess.STDOUT, env=env,
                cwd=str(ROOT))
            children.append((handle, process, downstream_seed))
        rc = 0
        for handle, process, downstream_seed in children:
            if process.wait() != 0:
                _log(
                    f"train: FAIL {run.output_name} seed={run.seed} "
                    f"{trainer} downstream_seed={downstream_seed}")
                rc = 1
            handle.close()
        return rc

    return run_unit


def _stage_parallel(study, run, log_dir, gpu, patience, force=False):
    log_path = log_dir / "parallel" / f"seed{run.seed}.log"
    return [("parallel", _subprocess_task(
        _parallel_cmd(study, run, patience, force), log_path, gpu,
        training=True))]


def _stage_b(study, run, log_dir, gpu, force=False):
    log_path = log_dir / "b" / f"seed{run.seed}.log"
    return [("b_cache", _subprocess_task(
        _cache_cmd(study, run, ("b_train", "b_val"), force), log_path, gpu))]


def _stage_train(study, run, log_dir, gpu):
    dnn_recipes, graph_recipes = _selected_recipes(study)
    downstream_seeds = list(study.downstream_seeds)
    tasks = []
    for trainer, recipes in (("dnn", dnn_recipes), ("graph", graph_recipes)):
        if not recipes:
            continue
        tasks.append((
            f"train_{run.output_name}_{trainer}",
            _train_unit(study, run, trainer, recipes, downstream_seeds,
                        log_dir, gpu)))
    return tasks


def _stage_train_recipe(study, run, log_dir, gpu, recipe):
    """Train one recipe's replicas for every downstream seed in parallel.

    One GPU job fans out to one child process per downstream seed (each child
    trains this single recipe), mirroring the legacy per-seed fan-out but
    scoped to a single recipe.
    """
    if recipe in FEATURE_RECIPES:
        trainer = "dnn"
    elif recipe in GRAPH_RECIPES:
        trainer = "graph"
    else:
        raise ValueError(f"unknown refiner recipe: {recipe}")
    return [(f"train_{trainer}_{recipe}",
             _train_unit(study, run, trainer, [recipe],
                         list(study.downstream_seeds), log_dir, gpu,
                         tag=f"{trainer}_{recipe}"))]


def _stage_y(study, run, log_dir, gpu, force=False):
    log_path = log_dir / "y" / f"seed{run.seed}.log"
    return [("y_cache", _subprocess_task(
        _cache_cmd(study, run, ("y_test",), force), log_path, gpu))]


def _stage_eval(study, run, log_dir, gpu, model):
    log_path = log_dir / "eval" / f"seed{run.seed}.log"
    return [("eval", _subprocess_task(
        _eval_cmd(study, run, model), log_path, gpu))]


def _iter_cache_dirs(study, run, splits=CACHE_SPLITS):
    """Yield ``(directory, kind)`` for this seed's frozen and graph caches."""
    checkpoint = study.checkpoint(run)
    if not checkpoint.is_file():
        return
    active = active_parallel_config(study, run)
    bundle = load_split_bundle(active.split_dir, config=active)
    for split in splits:
        split_hash = bundle.summary["index_sha256"][split]
        yield cache_directory(study, run, split, checkpoint, split_hash), "frozen"
        yield graph_cache_directory(
            study, run, split, checkpoint, split_hash), "graph"


def _iter_refiner_dirs(study, run, recipes=None):
    dnn_recipes, graph_recipes = _selected_recipes(study)
    if recipes is not None:
        wanted = set(recipes)
        dnn_recipes = [r for r in dnn_recipes if r in wanted]
        graph_recipes = [r for r in graph_recipes if r in wanted]
    for recipe in dnn_recipes:
        for downstream_seed in study.downstream_seeds:
            yield (study.refiner_directory(run, recipe, downstream_seed),
                   "last_dnn.pt")
    for recipe in graph_recipes:
        for downstream_seed in study.downstream_seeds:
            yield (study.refiner_directory(run, recipe, downstream_seed),
                   "last_graph_refiner.pt")


def _iter_evaluation_dirs(study, run):
    dnn_recipes, graph_recipes = _selected_recipes(study)
    yield study.parallel_evaluation_directory(run), "evaluation_manifest.json"
    for recipe in [*dnn_recipes, *graph_recipes]:
        for downstream_seed in study.downstream_seeds:
            yield (study.evaluation_directory(run, recipe, downstream_seed),
                   "evaluation_manifest.json")


_FORCE_SPLITS = {"b": ("b_train", "b_val"), "y": ("y_test",)}


def _remove(directory: Path) -> None:
    if directory.exists():
        _log(f"force: removing {directory}")
        shutil.rmtree(directory, ignore_errors=True)


def _force_clean(study, run, stage, recipe=None) -> None:
    """Delete this seed's completed artifacts for ``stage`` so it recomputes."""
    if stage == "parallel":
        _remove(study.parallel_directory(run))
    elif stage in _FORCE_SPLITS:
        for directory, _ in _iter_cache_dirs(study, run, _FORCE_SPLITS[stage]):
            _remove(directory)
    elif stage == "recipe":
        for directory, _ in _iter_refiner_dirs(
                study, run, [recipe] if recipe else None):
            _remove(directory)
    elif stage == "eval":
        for directory, _ in _iter_evaluation_dirs(study, run):
            _remove(directory)


def _dir_bytes(directory: Path) -> int:
    total = 0
    for root, _, files in os.walk(directory):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _clean_partial(study, run) -> None:
    """Remove only this seed's incomplete artifacts so a stage can retry."""
    parallel_dir = study.parallel_directory(run)
    if parallel_dir.exists() and not (parallel_dir / "last.pt").is_file():
        _log(f"clean: incomplete parallel {parallel_dir}")
        shutil.rmtree(parallel_dir, ignore_errors=True)

    for directory, _ in _iter_cache_dirs(study, run):
        if directory.exists() and not (directory / "manifest.json").is_file():
            _log(f"clean: incomplete cache {directory}")
            shutil.rmtree(directory, ignore_errors=True)

    for directory, marker in _iter_refiner_dirs(study, run):
        if directory.exists() and not (directory / marker).is_file():
            _log(f"clean: incomplete refiner {directory}")
            shutil.rmtree(directory, ignore_errors=True)

    for directory, marker in _iter_evaluation_dirs(study, run):
        if directory.exists() and not (directory / marker).is_file():
            _log(f"clean: incomplete evaluation {directory}")
            shutil.rmtree(directory, ignore_errors=True)


def _prune(study, run) -> float:
    """Delete this seed's frozen/graph caches once evaluation has succeeded."""
    freed = 0
    for directory, _ in _iter_cache_dirs(study, run):
        if not directory.exists():
            continue
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _log(f"prune: skip unreadable manifest {manifest_path}")
            continue
        if manifest.get("study_name") != study.cache_identity_name:
            _log(f"prune: skip foreign cache {directory}")
            continue
        if (directory / ".building").exists():
            _log(f"prune: skip in-use cache {directory}")
            continue
        size = _dir_bytes(directory)
        shutil.rmtree(directory)
        freed += size
        _log(f"prune: removed {directory} ({size / 1024 ** 3:.2f} GiB)")
    return freed / 1024 ** 3


def _check_parallel_complete(study, run) -> None:
    marker = study.parallel_directory(run) / "last.pt"
    if not marker.is_file():
        raise StageError(f"stage 2 did not finish (missing {marker})")


def _print_plan(study, run, model, gpu, patience) -> None:
    _log(f"dry-run gpu={gpu} model={model}")
    _log(f"  stage-parallel: {' '.join(_parallel_cmd(study, run, patience))}")
    if study.data["sizes"]["b_train"] > 0:
        _log(f"  stage-b: {' '.join(_cache_cmd(study, run, ('b_train', 'b_val')))}")
        dnn_recipes, graph_recipes = _selected_recipes(study)
        for trainer, recipes in (("dnn", dnn_recipes), ("graph", graph_recipes)):
            for downstream_seed in study.downstream_seeds:
                for recipe in recipes:
                    _log(f"  stage-train: {' '.join(_train_cmd(study, run, trainer, recipe, downstream_seed))}")
    _log(f"  stage-y: {' '.join(_cache_cmd(study, run, ('y_test',)))}")
    _log(f"  stage-eval: {' '.join(_eval_cmd(study, run, model))}")
    _log("  prune: this seed's frozen/graph b_train, b_val, y_test caches")


def _print_stage_plan(study, run, model, gpu, args) -> None:
    stage, force = args.stage, args.force
    _log(f"dry-run stage={stage} gpu={gpu} model={model} force={force}")
    if stage == "parallel":
        _log(f"  {' '.join(_parallel_cmd(study, run, args.patience, force))}")
    elif stage == "b":
        _log(f"  {' '.join(_cache_cmd(study, run, ('b_train', 'b_val'), force))}")
    elif stage == "recipe":
        script = ("scripts/train_dnn.py" if args.recipe in FEATURE_RECIPES
                  else "scripts/train_graph_refiner.py")
        for downstream_seed in study.downstream_seeds:
            _log(f"  {PYTHON} {script} --config {study.path} --seed {run.seed} "
                 f"--downstream-seed {downstream_seed} --recipe {args.recipe} "
                 f"--skip-complete")
    elif stage == "y":
        _log(f"  {' '.join(_cache_cmd(study, run, ('y_test',), force))}")
    elif stage == "eval":
        _log(f"  {' '.join(_eval_cmd(study, run, model))}")
    else:
        _log("  prune: this seed's frozen/graph b_train, b_val, y_test caches")


def _parse_gpu(value: str) -> int:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != 1:
        raise ValueError("--gpus must list exactly one device id for one seed")
    return int(items[0])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpus", default=DEFAULT_GPUS,
                        help='one device id, e.g. "0"')
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--patience", type=int, default=100,
        help=("Parallel-training early-stopping patience in epochs, passed to "
              "scripts/train_parallel.py (default: 100)."))
    parser.add_argument("--list-seeds", action="store_true",
                        help="print the config's Parallel seed ids and exit")
    parser.add_argument("--clean-partial", action="store_true",
                        help="remove this seed's incomplete artifacts and exit")
    parser.add_argument("--keep-cache", action="store_true",
                        help="run through evaluation but leave the caches")
    parser.add_argument(
        "--stage", choices=("full", "parallel", "b", "recipe", "y", "eval", "prune"),
        default="full",
        help="Run one schedulable lifecycle stage instead of the full seed pipeline.")
    parser.add_argument("--recipe", default=None,
                        help="Required with --stage recipe.")
    parser.add_argument(
        "--force", action="store_true",
        help=("Recompute this seed's artifacts for the selected stage(s), "
              "deleting completed outputs instead of skipping them."))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    study = load_study_config(args.config)

    if args.list_seeds:
        for run in study.seeds:
            print(run.seed)
        return 0

    if args.seed is None:
        parser.error("--seed is required unless --list-seeds is used")
    if args.stage == "recipe" and args.recipe is None:
        parser.error("--recipe is required with --stage recipe")
    if args.stage != "recipe" and args.recipe is not None:
        parser.error("--recipe is valid only with --stage recipe")
    run = study.selected_seeds([args.seed])[0]
    gpu = _parse_gpu(args.gpus)
    log_dir = Path(args.log_dir) if args.log_dir else (
        ROOT / "logs/parallel_refine/experiment1_ex/seed" /
        f"{study.study_name}_seed{run.seed}")

    global _LOG_PATH
    _LOG_PATH = log_dir / "pipeline.log"

    if args.clean_partial:
        _log(f"clean-partial config={study.path} seed={run.seed}")
        _clean_partial(study, run)
        _log("clean-partial complete")
        return 0

    model = "parallel_dnn" if study.data["sizes"]["b_train"] > 0 else "parallel"

    _log(f"config={study.path}")
    _log(f"study={study.study_name} seed={run.seed} gpu={gpu} "
         f"model={model} log_dir={log_dir}")

    if args.dry_run:
        if args.stage == "full":
            _print_plan(study, run, model, gpu, args.patience)
        else:
            _print_stage_plan(study, run, model, gpu, args)
        _log("dry-run: no stages executed")
        return 0

    write_experiment_manifest(study)

    clean = lambda: _clean_partial(study, run)  # noqa: E731

    if args.stage != "full":
        if (args.stage in ("b", "recipe")
                and study.data["sizes"]["b_train"] == 0):
            _log(f"stage-{args.stage}: skipped (b_train=0 has no refiners)")
            return 0
        if args.stage == "prune":
            freed = _prune(study, run)
            _log(f"prune: freed {freed:.2f} GiB")
            return 0
        if args.force:
            _force_clean(study, run, args.stage, args.recipe)
        staged = {
            "parallel": lambda: _stage_parallel(
                study, run, log_dir, gpu, args.patience, args.force),
            "b": lambda: _stage_b(study, run, log_dir, gpu, args.force),
            "recipe": lambda: _stage_train_recipe(
                study, run, log_dir, gpu, args.recipe),
            "y": lambda: _stage_y(study, run, log_dir, gpu, args.force),
            "eval": lambda: _stage_eval(study, run, log_dir, gpu, model),
        }
        _run_stage(
            f"stage-{args.stage}", staged[args.stage], args.retries, clean)
        if args.stage == "parallel":
            _check_parallel_complete(study, run)
        _log(f"stage-{args.stage} complete")
        return 0

    if args.force:
        _force_clean(study, run, "parallel")
    _run_stage("stage-parallel",
               lambda: _stage_parallel(
                   study, run, log_dir, gpu, args.patience, args.force),
               args.retries, clean)
    _check_parallel_complete(study, run)

    if study.data["sizes"]["b_train"] > 0:
        if args.force:
            _force_clean(study, run, "b")
        _run_stage("stage-b",
                   lambda: _stage_b(study, run, log_dir, gpu, args.force),
                   args.retries, clean)
        if args.force:
            _force_clean(study, run, "recipe")
        _run_stage("stage-train",
                   lambda: _stage_train(study, run, log_dir, gpu),
                   args.retries, clean)

    if args.force:
        _force_clean(study, run, "y")
    _run_stage("stage-y",
               lambda: _stage_y(study, run, log_dir, gpu, args.force),
               args.retries, clean)
    if args.force:
        _force_clean(study, run, "eval")
    _run_stage("stage-eval",
               lambda: _stage_eval(study, run, log_dir, gpu, model),
               args.retries, clean)

    if args.keep_cache:
        _log("keep-cache: caches retained")
        return 0

    freed = _prune(study, run)
    _log(f"prune: freed {freed:.2f} GiB")
    _log("seed pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
