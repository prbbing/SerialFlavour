"""Controlled plateau checks, including the three actual trainer loops."""

from types import SimpleNamespace

import torch

import json
from pathlib import Path

import numpy as np
import pytest

from src.lr_scheduler import (
    create_lr_scheduler, resolve_lr_scheduler, step_lr_scheduler)
from src.parallel_refine.config import (
    load_study_config, parallel_values, write_experiment_manifest)

def test_plateau_patience_floor_and_epoch_lr():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimiser = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler, _ = create_lr_scheduler(
        optimiser, {"patience": 4, "min_lr": 2.5e-4}, default_patience=4)
    rows = [step_lr_scheduler(optimiser, scheduler, 1.0) for _ in range(17)]
    assert [i + 1 for i, row in enumerate(rows) if row["lr_reduced"]] == [6, 11]
    assert rows[5] == {"lr": 1e-3, "lr_next": 5e-4, "lr_reduced": True}
    assert rows[6]["lr"] == 5e-4
    assert rows[-1]["lr_next"] == 2.5e-4


@pytest.mark.parametrize("options", [None, {"enabled": False}])
def test_fixed_lr_compatibility(options):
    optimiser = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
    scheduler, config = create_lr_scheduler(optimiser, options, default_patience=4)
    assert scheduler is None and config["enabled"] is False
    for _ in range(20):
        assert step_lr_scheduler(optimiser, scheduler, 1.0) == {
            "lr": 1e-3, "lr_next": 1e-3, "lr_reduced": False}


def test_improvement_resets_scheduler_patience():
    optimiser = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1e-3)
    scheduler, _ = create_lr_scheduler(optimiser, {"patience": 4}, default_patience=4)
    values = [1.0] * 5 + [0.8] * 6
    rows = [step_lr_scheduler(optimiser, scheduler, value) for value in values]
    assert [i + 1 for i, row in enumerate(rows) if row["lr_reduced"]] == [11]


@pytest.mark.parametrize("options", [
    {"patience": -1}, {"patience": True}, {"factor": 1}, {"factor": float("nan")},
    {"min_lr": 0.01}, {"threshold": -1}, {"enabled": "false"},
    {"type": "cosine"}, {"monitor": "y_test"}, {"cooldown": 0.5},
])
def test_invalid_scheduler_is_rejected(options):
    with pytest.raises(ValueError, match="lr_scheduler"):
        resolve_lr_scheduler(options, initial_lr=1e-3, default_patience=4)


def test_all_runtime_experiments_resolve_scheduler_defaults():
    for path in Path("configs/parallel_refine/experiments").rglob("*.json"):
        study = load_study_config(path)
        upstream = parallel_values(study, study.seeds[0], stage="parallel")
        assert upstream["lr_scheduler"]["patience"] == 8
        assert upstream["lr_scheduler"]["enabled"] is True
        for kind in ("dnn", "graph"):
            if kind in study.refiners:
                assert study.refiners[kind]["lr_scheduler"]["patience"] == 4


@pytest.mark.parametrize("stage", ["parallel", "dnn", "graph"])
def test_study_rejects_invalid_scheduler_before_training(tmp_path, stage):
    original = Path("configs/parallel_refine/experiments/default.json").resolve()
    source = json.loads(original.read_text())
    source["components"] = {
        kind: str((original.parent / ref).resolve())
        for kind, ref in source["components"].items()}
    if stage == "parallel":
        source["overrides"] = {"parallel": {"training": {"lr_scheduler": {"patience": -1}}}}
    else:
        source["overrides"] = {"refiners": {stage: {"lr_scheduler": {"patience": -1}}}}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="lr_scheduler.patience"):
        load_study_config(path)


def test_scheduler_change_cannot_reuse_an_existing_experiment_identity(tmp_path):
    original = Path("configs/parallel_refine/experiments/default_lr_decay.json").resolve()
    source = json.loads(original.read_text())
    source["components"] = {
        kind: str((original.parent / ref).resolve())
        for kind, ref in source["components"].items()}
    source["experiment"]["output_root"] = str(tmp_path / "results")
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(source))
    write_experiment_manifest(load_study_config(path))
    source["overrides"] = {"refiners": {"dnn": {"lr_scheduler": {"patience": 3}}}}
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError):
        write_experiment_manifest(load_study_config(path))


@pytest.mark.parametrize("kind", ["dnn", "graph"])
def test_downstream_trainer_steps_before_unchanged_early_stop(tmp_path, monkeypatch, kind):
    from scripts import train_dnn, train_graph_refiner
    module = train_dnn if kind == "dnn" else train_graph_refiner
    config = {
        "learning_rate": 1e-3, "weight_decay": 1e-4, "hidden_dims": [4],
        "dropout": 0.0, "epochs": 20, "batch_size": 6, "early_stopping_patience": 10,
        "lr_scheduler": {"patience": 4}, "tensorboard": {"enabled": False},
        "loss": {"name": "weighted_cross_entropy", "class_weights": [2, 2, 1],
                 "class_weights_source": "parallel.class_weights.jet_class_weights"}}
    run = SimpleNamespace(seed=1, output_name="parallel_seed1")
    output = tmp_path / kind
    study = SimpleNamespace(
        refiners={kind: config}, path=tmp_path / "experiment.json",
        source_sha256="test", experiment_markers={},
        refiner_directory=lambda *args: output,
        checkpoint=lambda *args: tmp_path / "upstream.pt")
    cache = SimpleNamespace(
        recipe_columns=lambda recipe: np.arange(2),
        recipe_names=lambda recipe: ["x", "z"], labels=np.array([0, 1, 2, 0, 1, 2]),
        source_index=np.arange(6), event_number=np.arange(6), manifest={},
        track_embedding=np.zeros((6, 2, 2)))
    monkeypatch.setattr(module, "load_frozen_cache", lambda *args: cache)
    monkeypatch.setattr(module, "fit_normalization", lambda *args: (np.zeros(2), np.ones(2)))
    monkeypatch.setattr(module, "probability_metrics", lambda *args: {})
    monkeypatch.setattr(module, "_device", lambda *args: torch.device("cpu"))
    features, labels = torch.randn(6, 2), torch.tensor(cache.labels, dtype=torch.long)
    def evaluate(model, loader, device, *, criterion, collect=True):
        result = {"loss": 1.0, "accuracy": 1 / 3}
        if collect:
            result.update(labels=cache.labels, probabilities=np.full((6, 3), 1 / 3))
        return result
    monkeypatch.setattr(module, "_evaluate", evaluate)
    if kind == "dnn":
        monkeypatch.setattr(module, "create_tabular_loader", lambda *args, **kwargs: [(features, labels)])
        recipe = "F1_embed"
    else:
        class GraphModel(torch.nn.Linear):
            def forward(self, context, *args):
                return super().forward(context)
        monkeypatch.setattr(module, "load_graph_cache", lambda *args: cache)
        monkeypatch.setattr(module, "resolve_graph_config", lambda requested, *args, **kwargs: requested)
        monkeypatch.setattr(module, "graph_node_values", lambda *args: np.zeros((2, 2)))
        monkeypatch.setattr(module, "save_graph_description", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "GraphDNNRefiner", lambda *args: GraphModel(2, 3))
        batch = {"context": features, "y": labels, "node_values": torch.zeros(6, 2, 2),
                 "pair_probs": torch.zeros(6, 2, 2), "track_mask": torch.ones(6, 2, dtype=torch.bool)}
        monkeypatch.setattr(module, "create_graph_loader", lambda *args, **kwargs: [batch])
        recipe = "FG2"
    module._train_one(study, run, recipe, 1, skip_complete=False)
    payload = json.loads((output / "training_history.json").read_text())
    rows = payload["epochs"]
    assert len(rows) == 11  # LR changes do not reset early-stopping patience.
    assert rows[5]["lr"] == 1e-3 and rows[5]["lr_next"] == 5e-4
    assert rows[6]["lr"] == 5e-4
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["training_summary"]["best_epoch"] == 1
    assert manifest["lr_scheduler"]["metric"] == "b_val.weighted_cross_entropy"
    assert payload["lr_scheduler"]["config"]["patience"] == 4


def test_parallel_trainer_monitors_jet_loss_and_preserves_best_checkpoint(tmp_path, monkeypatch):
    from scripts import train_parallel as module
    from src.parallel_refine.config import active_parallel_config
    study = load_study_config("configs/parallel_refine/experiments/default_lr_decay.json")
    run = study.seeds[0]
    config = active_parallel_config(study, run)
    config.gpu_ids, config.torch_compile, config.tensorboard_enabled = [-1], False, False
    config.epochs = 12
    output = tmp_path / "parallel"
    fake_study = SimpleNamespace(
        path=tmp_path / "experiment.json", experiment_name="test", experiment_markers={},
        source_sha256="test", selected_seeds=lambda *args: [run],
        parallel_directory=lambda *args: output)
    monkeypatch.setattr(module, "load_study_config", lambda *args: fake_study)
    monkeypatch.setattr(module, "write_experiment_manifest", lambda *args: tmp_path / "manifest.json")
    monkeypatch.setattr(module, "materialize_parallel_config", lambda *args, **kwargs: "test")
    monkeypatch.setattr(module, "active_parallel_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(module, "load_split_bundle", lambda *args, **kwargs: SimpleNamespace(summary={}))
    monkeypatch.setattr(module, "choose_device", lambda *args: torch.device("cpu"))
    batch = {"X": torch.randn(6, 2), "jet_X": torch.zeros(6, 2),
             "mask": torch.ones(6, 1, dtype=torch.bool), "y": torch.tensor([0, 1, 2, 0, 1, 2])}
    loader = [batch]
    class Loader(list):
        dataset = list(range(6))
    monkeypatch.setattr(module, "create_loader", lambda *args, **kwargs: (Loader(loader), None))
    class Model(torch.nn.Linear):
        def forward(self, x, *args):
            return {"jet_logits": super().forward(x)}
    monkeypatch.setattr(module, "build_parallel", lambda *args: Model(2, 3))
    def losses(output, batch, *args, **kwargs):
        jet = torch.nn.functional.cross_entropy(output["jet_logits"], batch["y"])
        return {"total": jet, "jet": jet, "origin": jet * 0, "pair": jet * 0}
    monkeypatch.setattr(module, "parallel_losses", losses)
    call = 0
    def evaluate(*args):
        nonlocal call
        call += 1
        # Total keeps improving; only jet CE should trigger decay.
        return {"total": 2.0 / call, "jet": 1.0, "origin": 0.0, "pair": 0.0, "jet_accuracy": 1 / 3}
    monkeypatch.setattr(module, "evaluate_loss", evaluate)
    monkeypatch.setattr(module, "_plot_training_history", lambda *args: None)
    assert module.main(["--config", "test", "--seed", "1"]) == 0
    payload = json.loads((output / "training_history.json").read_text())
    rows = payload["epochs"]
    assert rows[9]["lr"] == 1e-3 and rows[9]["lr_next"] == 5e-4
    assert rows[10]["lr"] == 5e-4
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["training_summary"]["best_jet_epoch"] == 1
    assert manifest["training_summary"]["best_total_epoch"] == 12
    assert manifest["lr_scheduler"]["metric"] == "a_val.jet_cross_entropy"
