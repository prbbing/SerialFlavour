"""Supplementary runs preserve upstream identity and keep Y opt-in."""

import json
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
import torch

from scripts.production import run_experiment1_ex_v2_refine as runner
from src.parallel_refine.cache import FrozenFeatureCache, build_feature_table
from src.parallel_refine.config import load_study_config


def test_pair_statistics_ablation_changes_only_the_intended_features():
    embeddings = torch.ones(1, 2, 48)
    table = build_feature_table({
        "track_mask": torch.ones(1, 2, dtype=torch.bool),
        "flavour_probs": torch.full((1, 3), 1 / 3),
        "track_embedding": embeddings,
        "pool_attention": torch.full((1, 2, 1), 0.5),
        "pooled_embedding": embeddings.mean(dim=1),
        "origin_probs": torch.full((1, 2, 8), 1 / 8),
        "pair_probs": torch.full((1, 2, 2), 0.5),
    })
    cache = FrozenFeatureCache(
        directory=None, features=table.values.numpy(), labels=np.array([0]),
        source_index=np.array([10]), event_number=np.array([1]),
        manifest={"feature_names": table.names, "groups": table.groups},
    )
    statistics = ["pair_match_mean", "pair_match_max", "pair_match_sum"]
    pair_with_statistics = cache.recipe_names("DNN-pair-99")
    both_without_statistics = cache.recipe_names("F3_embed_aux_104")
    assert len(pair_with_statistics) == 99
    assert pair_with_statistics == cache.recipe_names("F2_vertex") + statistics
    assert len(both_without_statistics) == 104
    assert both_without_statistics + statistics == cache.recipe_names("F3_embed_aux")


@pytest.mark.parametrize("total_m", [1, 3, 5])
def test_explicit_catalog_matches_real_configs_and_default_plan(total_m, capsys):
    catalog = runner.PARENT_CONFIGS[total_m]
    assert set(catalog) == set(range(50, 101, 5))
    assert len(set(catalog.values())) == 11
    for allocation, filename in catalog.items():
        path = runner.CONFIG_DIR / filename
        study = load_study_config(path)
        variables = study.experiment_markers["variables"]
        assert variables["total_dataset_jets"] == total_m * 1_000_000
        assert variables["a_fraction_percent"] == allocation
    assert runner.main(["--total-m", str(total_m), "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert output.count("Parent config:") == 10
    assert output.count("Extension config:") == 10
    assert catalog[100] not in output
    for allocation, filename in catalog.items():
        if allocation != 100:
            assert f"Parent config: {runner.CONFIG_DIR / filename}" in output
    assert "y_test" not in output
    assert "scripts/train_parallel.py" not in output


@pytest.fixture
def parent(tmp_path):
    source = runner.CONFIG_DIR / "experiment1_ex_v2_p122k_n3m_a075_b025.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["components"] = {
        key: str((source.parent / ref).resolve()) for key, ref in raw["components"].items()}
    raw["experiment"]["output_root"] = str(tmp_path / "results")
    raw["overrides"]["data"]["split_dir"] = str(tmp_path / "splits")
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    study = load_study_config(path)
    Path(study.data["split_dir"]).mkdir()
    for name in ("indices.npz", "split_manifest.json"):
        (Path(study.data["split_dir"]) / name).touch()
    for run in study.seeds:
        study.parallel_directory(run).mkdir(parents=True)
        study.checkpoint(run).touch()
        (study.parallel_directory(run) / "last.pt").touch()
    return path


def test_extension_preserves_parent_and_isolates_outputs(parent):
    before = parent.read_bytes()
    original, path, raw = runner.extension(parent, "extra")
    path.write_text(json.dumps(raw), encoding="utf-8")
    study = load_study_config(path)
    assert study.data == original.data
    assert study.parallel == original.parallel
    assert study.parallel_output_directory == original.parallel_output_directory
    assert study.cache_identity_name == original.cache_identity_name
    assert study.refiner_output_directory != original.refiner_output_directory
    assert study.data_directory != original.data_directory
    assert study.refiners["recipes"] == list(runner.RECIPES)
    assert parent.read_bytes() == before
    runner.extension(parent, "extra")  # Identical extension is reusable.
    raw["experiment"]["name"] = "different"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="extension config differs"):
        runner.extension(parent, "extra")


def test_train_commands_do_not_touch_y_or_upstream():
    commands = list(runner.commands(Path("extension.json"), 2, "train"))
    assert len(commands) == 6
    for _, cmd in commands:
        assert "y_test" not in cmd
        assert "scripts/evaluate.py" not in cmd
        assert "scripts/train_parallel.py" not in cmd
    for _, cmd in commands[1:]:
        assert "--skip-complete" in cmd
    assert [label for label, _ in runner.commands(Path("x"), 2, "evaluate")] == [
        "y_cache", "evaluate"]


def test_dry_run_writes_nothing(parent, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    assert runner.main(["--config", str(parent), "--dry-run"]) == 0
    assert set(tmp_path.rglob("*")) == before


def test_training_resume_and_explicit_evaluation(parent, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "run_command", lambda cmd, *args: calls.append(cmd))
    args = ["--config", str(parent), "--gpus", "0", "1", "--log-base", str(tmp_path / "logs")]
    assert runner.main(args) == 0
    assert len(calls) == 30  # Five upstream seeds, six commands each.
    assert all("y_test" not in cmd for cmd in calls)
    _, path, _ = runner.extension(parent, "refine_inputs_v1")
    study = load_study_config(path)
    manifest = (study.data_directory / "experiment_manifest.json").read_bytes()
    assert runner.main(args) == 0
    assert (study.data_directory / "experiment_manifest.json").read_bytes() == manifest
    with pytest.raises(FileNotFoundError, match="train refiners before evaluation"):
        runner.main([*args, "--stage", "evaluate"])
    for run in study.seeds:
        for recipe in runner.RECIPES:
            marker = "last_graph_refiner.pt" if recipe in runner.GRAPH_RECIPES else "last_dnn.pt"
            for seed in study.downstream_seeds:
                folder = study.refiner_directory(run, recipe, seed)
                folder.mkdir(parents=True)
                (folder / marker).touch()
    calls.clear()
    assert runner.main([*args, "--stage", "evaluate"]) == 0
    assert len(calls) == 11  # Y cache/eval per seed, then one aggregate.
    assert "--aggregate-parallel-seeds" in calls[-1]


def test_missing_upstream_never_starts_training(parent, monkeypatch):
    study = load_study_config(parent)
    study.checkpoint(study.seeds[0]).unlink()
    calls = []
    monkeypatch.setattr(runner, "run_command", lambda *args: calls.append(args))
    with pytest.raises(FileNotFoundError, match="missing completed upstream"):
        runner.main(["--config", str(parent)])
    assert not calls


def test_failure_is_reported_and_other_seeds_continue(parent, monkeypatch):
    calls = []

    def fail_first(cmd, *args):
        calls.append(cmd)
        if "--seed" in cmd and cmd[cmd.index("--seed") + 1] == "1":
            raise RuntimeError("simulated stage failure")

    monkeypatch.setattr(runner, "run_command", fail_first)
    assert runner.main(["--config", str(parent), "--stage", "all"]) == 1
    assert any("5" in cmd for cmd in calls)
    assert not any("--aggregate-parallel-seeds" in cmd for cmd in calls)


def test_two_gpus_run_five_downstream_seeds_per_recipe(parent, monkeypatch):
    recipes = ("F1_double", "F2_origin", "F2_vertex", "FG1_48d")
    concurrent = {recipe: Barrier(10, timeout=5) for recipe in recipes}
    events = []

    def simulate(cmd, gpu, log, python):
        seed = int(cmd[cmd.index("--seed") + 1])
        if cmd[0] == "scripts/generate_cache.py":
            assert cmd[-4:] == ["--split", "b_train", "--split", "b_val"]
            events.append((gpu, seed, "cache", None))
            return
        recipe = cmd[cmd.index("--recipe") + 1]
        trainer = "train_graph_refiner.py" if recipe == "FG1_48d" else "train_dnn.py"
        assert cmd[0] == f"scripts/{trainer}"
        assert "--skip-complete" in cmd
        downstream = int(cmd[cmd.index("--downstream-seed") + 1])
        assert log.name == f"{recipe}.dnn_seed{downstream}.log"
        events.append((gpu, seed, recipe, downstream))
        concurrent[recipe].wait()

    monkeypatch.setattr(runner, "run_command", simulate)
    assert runner.main([
        "--config", str(parent), "--gpus", "0", "1", "--seed", "1", "--seed", "2",
        "--downstream-workers", "5",
        *(arg for recipe in recipes for arg in ("--recipe", recipe)),
    ]) == 0

    assert len(events) == 42
    assignments = {(gpu, seed) for gpu, seed, recipe, _ in events if recipe == "cache"}
    assert {gpu for gpu, _ in assignments} == {"0", "1"}
    assert {seed for _, seed in assignments} == {1, 2}
    for gpu, seed in assignments:
        stages = [(recipe, downstream) for card, upstream, recipe, downstream in events
                  if card == gpu and upstream == seed]
        assert stages[0] == ("cache", None)
        for start, recipe in enumerate(recipes):
            first = 1 + 5 * start
            assert sorted(stages[first:first + 5]) == [(recipe, seed) for seed in range(1, 6)]


@pytest.mark.parametrize("failed_stage", [None, "b_cache", "y_cache", "aggregate"])
def test_private_cache_lifetime(parent, tmp_path, monkeypatch, failed_stage):
    from src.parallel_refine.cache_workspace import (
        PRIVATE_CACHE_ENV, cache_workspace, feature_cache_path,
    )

    raw = json.loads(parent.read_text(encoding="utf-8"))
    roots = {"root": tmp_path / "frozen", "graph_root": tmp_path / "graph"}
    raw["overrides"]["feature_cache"] = {key: str(root) for key, root in roots.items()}
    parent.write_text(json.dumps(raw), encoding="utf-8")
    alternate = tmp_path / "parent2.json"
    raw["experiment"]["upstream_experiment_name"] = raw["experiment"]["name"]
    raw["experiment"]["name"] += "_other"
    alternate.write_text(json.dumps(raw), encoding="utf-8")

    first = load_study_config(parent).study_name + "_test"
    second = load_study_config(alternate).study_name + "_test"
    relative = Path("parallel_seed1/b_train/cache")
    shared = roots["root"] / relative
    shared.mkdir(parents=True)
    (shared / "manifest.json").touch()
    monkeypatch.setenv(PRIVATE_CACHE_ENV, "1")
    assert feature_cache_path(roots["root"], first, relative) == shared
    assert feature_cache_path(roots["graph_root"], first, relative) == (
        cache_workspace(roots["graph_root"], first) / relative
    )
    events = []

    def simulate(cmd, gpu, log, python, *, private_cache):
        assert private_cache
        name = Path(cmd[cmd.index("--config") + 1]).stem
        seed = int(cmd[cmd.index("--seed") + 1]) if "--seed" in cmd else None
        if name == second:
            assert all(not cache_workspace(root, first).exists() for root in roots.values())
        if "--aggregate-parallel-seeds" in cmd:
            assert all(cache_workspace(root, name).is_dir() for root in roots.values())
            events.append((name, "aggregate"))
            if failed_stage == "aggregate" and name == first:
                raise RuntimeError("aggregation failed")
        elif "y_test" in cmd:
            assert events.count((name, "train")) == (4 if name == first and failed_stage == "b_cache" else 5)
            events.append((name, f"y_cache:{seed}"))
            if name == first and seed == 1 and failed_stage == "y_cache":
                raise RuntimeError("Y cache failed")
        elif cmd[0] == "scripts/train_dnn.py":
            events.append((name, "train"))
            study = load_study_config(cmd[cmd.index("--config") + 1])
            run, = study.selected_seeds([seed])
            for downstream in study.downstream_seeds:
                directory = study.refiner_directory(run, "F2_origin", downstream)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "last_dnn.pt").touch()
                (directory / "run_manifest.json").touch()
        elif "b_train" in cmd:
            for root in roots.values():
                cache_workspace(root, name).mkdir(parents=True, exist_ok=True)
            if name == first and seed == 1 and failed_stage == "b_cache":
                raise RuntimeError("B cache failed")

    monkeypatch.setattr(runner, "run_command", simulate)
    args = ["--config", str(parent), "--config", str(alternate), "--suffix", "test",
            "--gpus", "0", "1", "--recipe", "F2_origin", "--stage", "all",
            "--prune-after-aggregate"]
    assert runner.main(args) == int(failed_stage is not None)
    expected_aggregates = [(first, "aggregate"), (second, "aggregate")]
    if failed_stage in {"b_cache", "y_cache"}:
        expected_aggregates.pop(0)
    assert [(name, action) for name, action in events if action == "aggregate"] == expected_aggregates
    for seed in range(2, 6):
        assert (first, f"y_cache:{seed}") in events
    assert ((first, "y_cache:1") in events) == (failed_stage != "b_cache")
    for root in roots.values():
        assert not cache_workspace(root, first).exists()
        assert not cache_workspace(root, second).exists()
    assert (shared / "manifest.json").is_file()
