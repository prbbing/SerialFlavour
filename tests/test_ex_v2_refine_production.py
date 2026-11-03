"""Supplementary runs preserve upstream identity and keep Y opt-in."""

import json
from pathlib import Path

import pytest

from scripts.production import run_experiment1_ex_v2_refine as runner
from src.parallel_refine.config import load_study_config


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
