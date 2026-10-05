"""Check inherited class weights, real trainer gradients and loss provenance."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.parallel_refine.config import load_study_config, write_experiment_manifest


def _experiment(tmp_path, **overrides):
    original = Path("configs/parallel_refine/experiments/default.json").resolve()
    source = json.loads(original.read_text())
    source["components"] = {
        kind: str((original.parent / reference).resolve())
        for kind, reference in source["components"].items()}
    source["experiment"]["output_root"] = str(tmp_path / "results")
    source["overrides"] = overrides
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(source))
    return load_study_config(path)


def test_downstream_weights_follow_upstream_override(tmp_path):
    study = _experiment(tmp_path, parallel={"class_weights": {"jet_class_weights": [3, 1, 2]}})
    for kind in ("dnn", "graph"):
        assert study.refiners[kind]["loss"] == {
            "name": "weighted_cross_entropy",
            "class_weights_source": "parallel.class_weights.jet_class_weights",
            "class_weights": [3, 1, 2]}


def test_weighted_loss_cannot_reuse_historical_unweighted_manifest(tmp_path):
    weighted = _experiment(tmp_path)
    historical_values = copy.deepcopy(weighted.values)
    for kind in ("dnn", "graph"):
        historical_values["refiners"][kind].pop("loss")
    historical = type(weighted)(weighted.path, historical_values, weighted.seeds)
    manifest = write_experiment_manifest(historical)
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="different configuration"):
        write_experiment_manifest(weighted)
    assert manifest.read_bytes() == before


@pytest.mark.parametrize("kind", ["dnn", "graph"])
def test_real_downstream_training_and_validation_use_weighted_ce(tmp_path, monkeypatch, kind):
    from scripts import train_dnn, train_graph_refiner
    module = train_dnn if kind == "dnn" else train_graph_refiner
    loss_metadata = {
        "name": "weighted_cross_entropy", "class_weights": [3, 1, 2],
        "class_weights_source": "parallel.class_weights.jet_class_weights"}
    config = {
        "learning_rate": 1e-3, "weight_decay": 1e-4, "hidden_dims": [4],
        "dropout": 0.0, "epochs": 1, "batch_size": 6,
        "early_stopping_patience": 10, "tensorboard": {"enabled": False},
        "loss": loss_metadata}
    output = tmp_path / kind
    run = SimpleNamespace(seed=1, output_name="parallel_seed1")
    study = SimpleNamespace(
        refiners={kind: config}, path=tmp_path / "experiment.json",
        source_sha256="test", experiment_markers={},
        refiner_directory=lambda *args: output,
        checkpoint=lambda *args: tmp_path / "upstream.pt")
    features = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., -1.], [-1., 2.], [-1., -1.]])
    labels = torch.tensor([0, 1, 2, 2, 2, 0])
    cache = SimpleNamespace(
        recipe_columns=lambda recipe: np.arange(2),
        recipe_names=lambda recipe: ["x", "z"], labels=labels.numpy(),
        source_index=np.arange(6), event_number=np.arange(6), manifest={},
        track_embedding=np.zeros((6, 2, 2)))
    monkeypatch.setattr(module, "load_frozen_cache", lambda *args: cache)
    monkeypatch.setattr(module, "fit_normalization", lambda *args: (np.zeros(2), np.ones(2)))
    monkeypatch.setattr(module, "probability_metrics", lambda *args: {})
    monkeypatch.setattr(module, "_device", lambda *args: torch.device("cpu"))

    class Model(torch.nn.Linear):
        def forward(self, context, *args):
            return super().forward(context)

    torch.manual_seed(42)
    model = Model(2, 3)
    reference = copy.deepcopy(model)
    weights = torch.tensor(loss_metadata["class_weights"], dtype=torch.float32)
    expected_train = torch.nn.functional.cross_entropy(reference(features), labels, weight=weights)
    ordinary_train = torch.nn.functional.cross_entropy(reference(features), labels)
    assert not torch.isclose(expected_train, ordinary_train)
    expected_train.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        parameter.register_hook(lambda gradient, name=name: gradients.update({name: gradient.clone()}))

    if kind == "dnn":
        monkeypatch.setattr(module, "create_tabular_loader", lambda *args, **kwargs: [(features, labels)])
        monkeypatch.setattr(module, "TabularDNN", lambda *args: model)
        recipe = "F1_embed"
    else:
        monkeypatch.setattr(module, "load_graph_cache", lambda *args: cache)
        monkeypatch.setattr(module, "resolve_graph_config", lambda requested, *args, **kwargs: requested)
        monkeypatch.setattr(module, "graph_node_values", lambda *args: np.zeros((2, 2)))
        monkeypatch.setattr(module, "save_graph_description", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "GraphDNNRefiner", lambda *args: model)
        batch = {"context": features, "y": labels, "node_values": torch.zeros(6, 2, 2),
                 "pair_probs": torch.zeros(6, 2, 2), "track_mask": torch.ones(6, 2, dtype=torch.bool)}
        monkeypatch.setattr(module, "create_graph_loader", lambda *args, **kwargs: [batch])
        recipe = "FG2"

    module._train_one(study, run, recipe, 1, skip_complete=False)
    for name, parameter in reference.named_parameters():
        torch.testing.assert_close(gradients[name], parameter.grad)
    with torch.no_grad():
        expected_val = torch.nn.functional.cross_entropy(model(features), labels, weight=weights).item()
    history = json.loads((output / "training_history.json").read_text())
    row = history["epochs"][0]
    assert row["train_cross_entropy"] == pytest.approx(expected_train.item())
    assert row["val_cross_entropy"] == pytest.approx(expected_val)
    assert row["saved_best"] is True
    assert history["loss"] == loss_metadata
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["loss"] == loss_metadata
    assert manifest["training_summary"]["best_validation_cross_entropy"] == pytest.approx(expected_val)
    assert manifest["lr_scheduler"]["metric"] == "b_val.weighted_cross_entropy"
    metrics = json.loads((output / "validation_metrics.json").read_text())
    assert metrics["weighted_cross_entropy"] == pytest.approx(expected_val)
    assert metrics["loss"] == loss_metadata
