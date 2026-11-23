"""Check dependency scheduling and private feature-cache lifetime."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event

from scripts.production import run_experiment1_ex_v2_refiner25 as runner
from src.parallel_refine.cache_workspace import cache_workspace
from src.parallel_refine.config import load_study_config


def test_workers_take_ready_jobs_without_waiting_for_a_slow_gpu():
    both_running = Barrier(2)
    fast_started = Event()
    next_job_started = Event()
    assignments = {}

    def slow(gpu):
        assignments["slow"] = gpu
        both_running.wait(timeout=5)
        assert next_job_started.wait(timeout=5)

    def fast(gpu):
        assignments["fast"] = gpu
        both_running.wait(timeout=5)
        fast_started.set()

    def next_job(gpu):
        assert fast_started.is_set()
        assignments["next"] = gpu
        next_job_started.set()

    tasks = [runner.Task("next", next_job, gpu=True, priority=1),
             runner.Task("slow", slow, gpu=True), runner.Task("fast", fast, gpu=True)]
    assert runner.run_tasks(tasks, ["0", "1"])
    assert assignments["slow"] != assignments["fast"] == assignments["next"]


def test_failed_dependencies_block_cleanup_but_allow_other_tasks():
    completed = []

    def fail(_gpu):
        raise ValueError("training failed")

    tasks = [
        runner.Task("train_a", fail, gpu=True),
        runner.Task("clean_a", lambda: completed.append("a"), after=("train_a",)),
        runner.Task("train_b", lambda _gpu: None, gpu=True),
        runner.Task("clean_b", lambda: completed.append("b"), after=("train_b",)),
    ]
    assert not runner.run_tasks(tasks, ["0", "1"])
    assert completed == ["b"]


def test_training_skips_finished_models_and_only_prunes_private_splits(tmp_path):
    study = load_study_config(runner.CONFIG)
    assert study.output_directory != study.refiner_output_directory
    assert study.refiner_output_directory.name == runner.CONFIG.stem
    values = deepcopy(study.values)
    values["experiment"]["output_root"] = str(tmp_path / "results")
    values["feature_cache"]["root"] = str(tmp_path / "frozen")
    values["feature_cache"]["graph_root"] = str(tmp_path / "graph")
    study = replace(study, values=values)
    run = study.seeds[0]
    for recipe in study.refiners["recipes"]:
        if recipe == "FG1_48d":
            continue
        for seed in study.downstream_seeds:
            for path in runner.refiner_checkpoints(study, run, recipe, seed):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

    tasks = runner.build_tasks(study, (run,), "train", tmp_path / "logs")
    training = [task for task in tasks if task.command and "--recipe" in task.command]
    assert len(training) == 5
    assert all("FG1_48d" in task.command for task in training)
    cache = next(task for task in tasks if task.cache_build)
    assert all(task.after == (cache.name,) for task in training)
    cleanup = next(task for task in tasks if task.name.endswith("/prune_train"))
    assert cleanup.after == tuple(task.name for task in training)

    for key in ("root", "graph_root"):
        root = Path(study.cache[key])
        workspace = cache_workspace(root, study.study_name)
        for split in (*runner.TRAIN_SPLITS, "y_test"):
            (workspace / run.output_name / split).mkdir(parents=True)
            (root / run.output_name / split).mkdir(parents=True)
    cleanup.action()
    for key in ("root", "graph_root"):
        root = Path(study.cache[key])
        workspace = cache_workspace(root, study.study_name)
        for split in runner.TRAIN_SPLITS:
            assert not (workspace / run.output_name / split).exists()
            assert (root / run.output_name / split).is_dir()
        assert (workspace / run.output_name / "y_test").is_dir()


def test_evaluation_cleanup_waits_for_its_study_aggregate(tmp_path):
    study = load_study_config(runner.CONFIG)
    tasks = runner.build_tasks(study, study.seeds, "evaluate", tmp_path)
    aggregate = next(task for task in tasks if task.name.endswith("/aggregate"))
    evaluations = [task.name for task in tasks if task.name.endswith("/evaluate")]
    cleanups = [task for task in tasks if task.name.endswith("/prune_y")]
    assert aggregate.after == tuple(evaluations)
    assert len(cleanups) == len(evaluations) == len(study.seeds)
    assert all(task.after == (aggregate.name,) for task in cleanups)


def test_sweep_skips_a_finished_allocation_before_preflight_or_lock(tmp_path, monkeypatch, capsys):
    path = runner.sweep_configs()[5]
    study = load_study_config(path)
    values = deepcopy(study.values)
    values["experiment"]["output_root"] = str(tmp_path / "results")
    study = replace(study, values=values)
    run = study.seeds[0]
    for recipe in runner.SWEEP_RECIPES:
        for seed in study.downstream_seeds:
            for checkpoint in runner.refiner_checkpoints(study, run, recipe, seed):
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
    assert not runner.sweep_has_pending_models(study, (run,))
    monkeypatch.setattr(runner, "sweep_configs", lambda: [path])
    monkeypatch.setattr(runner, "load_study_config", lambda _path: study)
    assert runner.main(["--allocation-sweep", "--stage", "train", "--seed", str(run.seed)]) == 0
    assert "Skipping completed allocation" in capsys.readouterr().out
    assert not (study.refiner_output_directory / ".refiner25.lock").exists()


def test_allocation_sweep_prioritizes_recipes_with_only_cache_dependencies(tmp_path):
    selected = []
    for path in (runner.CONFIG, runner.sweep_configs()[5]):
        study = load_study_config(path)
        values = deepcopy(study.values)
        values["experiment"]["output_root"] = str(tmp_path / "results")
        study = replace(study, values=values)
        selected.append((study, study.seeds[:1]))

    tasks = runner.build_sweep_tasks(selected, tmp_path)
    by_name = {task.name: task for task in tasks}
    assert len(by_name) == len(tasks)
    assert sum(task.cache_build for task in tasks) == 4
    for index, recipe in enumerate(runner.SWEEP_RECIPES):
        trainers = [task for task in tasks if task.command and "--downstream-seed" in task.command
                    and recipe in task.command]
        assert len(trainers) == 10
        for task in trainers:
            assert task.priority == index
            assert len(task.after) == 1
            cache = by_name[task.after[0]]
            assert cache.cache_build
            assert ("--no-graph" in cache.command) == (recipe != "FG1_48d")
            if recipe == "FG1_48d":
                assert cache.after == (cache.name.removesuffix("graph") + "structured",)

    graph_cleanups = [task for task in tasks if task.name.endswith("/prune_graph")]
    assert len(graph_cleanups) == 2
    assert all(task.action.keywords["graph_only"] for task in graph_cleanups)
    assert all(len(task.after) == 5 for task in graph_cleanups)
    last_cleanups = [task for task in tasks if task.name.endswith("/prune_train")]
    assert len(last_cleanups) == 2
    assert all(len(task.after) == 26 and any(dep.endswith("/prune_graph") for dep in task.after)
               for task in last_cleanups)
