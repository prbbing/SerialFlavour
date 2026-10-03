"""Protocol tests, plus an offline miniature run of all QM9 learning stages."""

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.qm9 import analyze
from data.qm9 import (BOND_CLASSES, allocate_splits, array_hash, collate_molecules, dense_pairs, distance_edges,
                      load_arrays, parse_exclusions, parse_xyz, processed_dir, save_arrays, subset_size)
from evaluate.qm9 import evaluate
from model.qm9 import TinySchNet
from pipeline.context import Context
from pipeline.download import download_verified
from pipeline.fit import fit
from pipeline.io import read_json, write_json
from pipeline.metrics import regression_metrics
from pipeline.readout import GraphSetReadout, TabularReadout, fit_standardization
from pipeline.runtime import seed_all
from pipeline import stages
from pipeline.units import enumerate_units, unit_filters
from refine.qm9 import cache, tabular_features, train as train_refiners
from training.qm9 import train as train_upstream


def smoke_config():
    return json.loads((ROOT / "config" / "qm9_gap_charge_bond.json").read_text())


def xyz(tag="gdb"):
    return f"""3
{tag} 42 1 2 3 1.5 2.5 -0.1 0.2 0.3 3.5 0.01 -10 -9 -8 -7 4.5
O 0 0 0 -0.4
H 1.0*^-1 0 0 0.2
H 0 1 0 0.2
1 2 3
O O
InChI=1S/H2O/h1H2 InChI=1S/H2O/h1H2
"""


@pytest.mark.parametrize("tag", ["gdb", "gdb9"])
def test_author_columns_numbers_and_invalid_xyz(tag):
    record = parse_xyz(xyz(tag), ["mu", "alpha", "r2", "cv"])
    assert record["id"] == 42
    np.testing.assert_array_equal(record["target"], [1.5, 2.5, 3.5, 4.5])
    np.testing.assert_allclose(record["pos"][1] - record["pos"][0], [.1, 0, 0], atol=1e-7)
    assert record["smiles"] == "O"
    np.testing.assert_allclose(record["charge"], [-0.4, 0.2, 0.2], atol=1e-6)
    assert {order for _, _, order in record["bonds"]} == {1}
    with pytest.raises(ValueError):
        parse_xyz(xyz(tag).replace("1.5", "nan"), ["mu"])
    with pytest.raises(ValueError):
        parse_xyz(xyz(tag).replace("H 0 1", "Xe 0 1"), ["mu"])


def test_exclusion_list_identity():
    text = "\n".join(["header"] * 9 + ["", "42 O InChI", "123 C InChI", "footer"])
    assert parse_exclusions(text, expected_count=2) == {42, 123}
    with pytest.raises(ValueError):
        parse_exclusions(text.replace("123", "42"), expected_count=2)


def test_dense_pairs_shape_and_classes():
    pair_index, pair_class = dense_pairs(4, [(0, 1, 1), (2, 3, 3)])
    assert pair_index.shape == (2, 6) and pair_class.shape == (6,)
    positions = {tuple(pair_index[:, k]): pair_class[k] for k in range(6)}
    assert positions[(0, 1)] == 1 and positions[(2, 3)] == 3 and positions[(0, 2)] == 0


def graph(record):
    edge_index, distance = distance_edges(record["pos"], 5.0)
    pair_index, pair_class = dense_pairs(len(record["z"]), record["bonds"])
    return {"z": torch.from_numpy(record["z"]), "edge_index": torch.from_numpy(edge_index),
            "distance": torch.from_numpy(distance), "charge": torch.from_numpy(record["charge"]),
            "pair_index": torch.from_numpy(pair_index), "pair_class": torch.from_numpy(pair_class),
            "target": torch.from_numpy(record["target"]), "id": record["id"]}


def test_schnet_symmetries_pairing_and_batch_isolation():
    config = smoke_config()
    record = parse_xyz(xyz(), ["mu", "alpha", "r2", "cv"])
    seed_all(1)
    single = TinySchNet(config["model"], ["mu"]).eval()
    seed_all(1)
    multi = TinySchNet(config["model"], ["mu", "alpha", "r2", "cv"]).eval()
    for name, value in single.state_dict().items():
        assert torch.equal(value, multi.state_dict()[name])
    batch = collate_molecules([graph(record), graph(record)])
    row, col = batch["edge_index"]
    assert torch.equal(batch["batch"][row], batch["batch"][col])
    with torch.inference_mode():
        expected = multi(collate_molecules([graph(record)]))["prediction"]
        transformed = copy.deepcopy(record)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        transformed["pos"] = record["pos"] @ rotation + np.array([2, 3, 4], dtype=np.float32)
        permutation = [2, 0, 1]
        transformed["pos"] = transformed["pos"][permutation]
        transformed["z"] = transformed["z"][permutation]
        actual = multi(collate_molecules([graph(transformed)]))["prediction"]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(multi(batch)["prediction"], expected.repeat(2, 1), atol=2e-6, rtol=2e-6)
    assert sum(parameter.numel() for parameter in single.parameters()) == 62977
    assert sum(parameter.numel() for parameter in multi.parameters()) == 81796


def test_local_heads_and_readout_capacity():
    config = smoke_config()
    record = parse_xyz(xyz(), ["gap"])
    model = TinySchNet(config["model"], ["gap"], ["charge", "bond"]).eval()
    batch = collate_molecules([graph(record)])
    with torch.inference_mode():
        output = model(batch)
    assert output["charge_prediction"].shape == (3,)
    assert output["bond_logits"].shape == (3, len(BOND_CLASSES))
    # Tabular R0 and R2 must share the same input width and parameter count.
    zeros = np.zeros((5, 64), dtype=np.float32)
    atom_offsets = np.arange(0, 16, 3, dtype=np.int64)
    fake = {"ids": np.arange(5), "embedding": zeros, "charge_prediction": np.zeros(15, dtype=np.float32),
            "atom_offsets": atom_offsets, "pair_index": np.zeros((2, 15), dtype=np.int64),
            "bond_probs": np.zeros((15, 5), dtype=np.float32), "pair_offsets": atom_offsets}
    r0, r2 = tabular_features(fake, "R0"), tabular_features(fake, "R2")
    assert r0.shape == r2.shape == (5, 64 + len(BOND_CLASSES) + 5)
    assert sum(p.numel() for p in TabularReadout(r0.shape[1], [64, 32]).parameters()) == 6465 + (r0.shape[1] - 67) * 64
    model = GraphSetReadout(65, 32, [32], layers=1, use_edges=True)
    set_batch = {"features": torch.randn(9, 65), "z": torch.ones(9, dtype=torch.long),
                 "batch": torch.repeat_interleave(torch.arange(3), 3),
                 "pair_index": torch.randint(0, 9, (2, 12)), "bond_probs": torch.rand(12, 5),
                 "target": torch.zeros(3, 1)}
    assert model(set_batch).shape == (3, 1)


def test_verified_download_size_and_checksum(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"QM9 fixture")
    md5 = hashlib.md5(source.read_bytes()).hexdigest()
    destination = tmp_path / "download.bin"
    download_verified(source.as_uri(), destination, md5)
    assert destination.read_bytes() == source.read_bytes()
    destination.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        download_verified(source.as_uri(), destination, md5)
    with pytest.raises(ValueError, match="limit"):
        download_verified(source.as_uri(), tmp_path / "oversized.bin", md5, max_bytes=2)


def test_stage_completion_requires_intact_artifacts(tmp_path, monkeypatch):
    artifact = tmp_path / "data.txt"
    calls = []
    def adapter(context):
        calls.append(1)
        artifact.write_text("valid")
        return [artifact]
    monkeypatch.setattr(stages.importlib, "import_module", lambda name: SimpleNamespace(run=adapter))
    context = SimpleNamespace(output_dir=tmp_path, identity="fixture")
    stages.execute(context, "download")
    stages.execute(context, "download")
    assert len(calls) == 1
    artifact.write_text("tampered")
    with pytest.raises(RuntimeError, match="requires"):
        stages.execute(context, "prepare")


def test_regression_metric_units():
    metrics = regression_metrics([1, 2, 3], [1, 3, 2])
    assert metrics["mae"] == pytest.approx(2 / 3)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["r_squared"] == pytest.approx(0)
    with pytest.raises(ValueError):
        regression_metrics([1], [np.nan])


def fixture_context(tmp_path):
    config = smoke_config()
    config.update(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "results"), experiment="fixture")
    config["data"]["sizes"] = {"a_train": 8, "a_val": 4, "b_train": 8, "b_val": 4, "y_test": 4}
    config["model"].update(hidden_channels=8, num_interactions=1, num_gaussians=4, head_hidden=[8], bond_classes=len(BOND_CLASSES))
    config["upstream"].update(epochs=1, batch_size=4)
    config["refiner"].update(epochs=1, batch_size=4, hidden=[8], set_hidden=8, set_layers=1, set_epochs=1,
                             dropout=0.0, clip_grad=1.0, early_stopping_patience=0, recipes=["R0", "R1", "R2", "R3", "R4"])
    context = Context(config, ROOT, tmp_path / "fixture.json")
    context.initialize()
    rng = np.random.default_rng(6)
    n = sum(config["data"]["sizes"].values())
    z, edges, distances, offsets, edge_offsets = [], [], [], [0], [0]
    charge, pair_index, pair_class, pair_offsets = [], [], [], [0]
    coordinates_list = []
    for index in range(n):
        coordinates = np.array([[0, 0, 0], [1 + index / n, 0, 0], [0, 1, 0]], dtype=np.float32)
        coordinates_list.append(coordinates)
        edge, distance = distance_edges(coordinates, 5)
        local_pairs, local_classes = dense_pairs(3, [(0, 1, 1), (0, 2, 1)])
        z.extend([8, 1, 1])
        edges.append(edge + offsets[-1])
        distances.append(distance)
        charge.extend(rng.normal(size=3))
        pair_index.append(local_pairs)
        pair_class.append(local_classes)
        offsets.append(offsets[-1] + 3)
        edge_offsets.append(edge_offsets[-1] + len(distance))
        pair_offsets.append(pair_offsets[-1] + local_pairs.shape[1])
    directory = processed_dir(context)
    save_arrays(directory / "dataset.npz", z=np.array(z), edge_index=np.concatenate(edges, axis=1),
        distance=np.concatenate(distances), atom_offsets=np.array(offsets), edge_offsets=np.array(edge_offsets),
        charge=np.array(charge, dtype=np.float32), pair_index=np.concatenate(pair_index, axis=1),
        pair_class=np.concatenate(pair_class), pair_offsets=np.array(pair_offsets),
        ids=np.arange(1, n + 1), target=rng.uniform(.1, 1.0, size=(n, 1)).astype(np.float32))
    splits, cursor = {}, 0
    for name, size in config["data"]["sizes"].items():
        splits[name] = np.arange(cursor, cursor + size)
        cursor += size
    save_arrays(directory / "splits.npz", **splits)
    return context


def test_offline_training_freezing_refinement_evaluation(tmp_path):
    context = fixture_context(tmp_path)
    train_upstream(context)
    originals = {variant: hashlib.sha256((context.output_dir / "upstream" / variant / "seed1" / "best.pt").read_bytes()).hexdigest()
                 for variant in context.config["upstream"]["variants"]}
    cache(context)
    train_refiners(context)
    evaluate(context)
    analyze(context)
    for variant, digest in originals.items():
        assert hashlib.sha256((context.output_dir / "upstream" / variant / "seed1" / "best.pt").read_bytes()).hexdigest() == digest
        for split in ("b_train", "b_val", "y_test"):
            manifest = read_json(context.output_dir / "cache" / variant / "seed1" / f"{split}.json")
            assert manifest["freeze_verified"] and not manifest["auxiliary_truth_cached"]
    metrics = read_json(context.output_dir / "evaluation" / "metrics.json")
    assert {row["method"] for row in metrics["main"]} == {
        "ST-native", "ST-R0", "ST-R3", "MT-native", "MT-R0", "MT-R1", "MT-R2", "MT-R3", "MT-R4"}
    assert all(row["n_test"] == 4 and np.isfinite(row["mae"]) for row in metrics["main"])
    assert metrics["auxiliary"] and metrics["auxiliary"][0]["bond_order_accuracy"] is not None
    summary = read_json(context.output_dir / "analysis" / "summary.json")
    assert all(row["sample_sd"] is None for row in summary["methods"])
    raw = load_arrays(context)
    expected = raw["target"][:8, :1].mean(axis=0)
    upstream = read_json(context.output_dir / "upstream" / "multi_task" / "seed1" / "training_manifest.json")
    np.testing.assert_allclose(upstream["target_mean"], expected, atol=1e-7)
    assert upstream["local_tasks"] == ["charge", "bond"]


def test_subset_size_and_shared_split_allocation():
    shared = {"shared_validation": True, "validation_size": 4,
              "sizes": {"a_train": 3, "b_train": 2, "y_test": 5}}
    assert subset_size(shared) == 14
    splits = allocate_splits(14, shared, np.arange(14))
    assert set(splits) == {"a_train", "b_train", "a_val", "b_val", "y_test"}
    assert np.array_equal(splits["a_val"], splits["b_val"]) and len(splits["a_val"]) == 4
    unique = np.concatenate([splits["a_train"], splits["b_train"], splits["y_test"], splits["a_val"]])
    assert len(np.unique(unique)) == 14
    plain = {"sizes": {"a_train": 2, "a_val": 1, "b_train": 2, "b_val": 1, "y_test": 2}}
    assert subset_size(plain) == 8
    assert not np.array_equal(allocate_splits(8, plain, np.arange(8))["a_val"],
                              allocate_splits(8, plain, np.arange(8))["b_val"])


def test_early_stopping(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
    settings = {"epochs": 50, "learning_rate": 0.01, "weight_decay": 0.0,
                "selection_mode": "min", "early_stopping_patience": 3}
    loader = [{"x": torch.zeros(4, 1), "y": torch.zeros(4, 1)}]

    def loss(current, batch):
        return (current(batch["x"]) - batch["y"]).square().mean(), len(batch["x"])

    def score(current, current_loader, device):
        return 1.0

    fit(model, loader, loader, loss, score, tmp_path / "es", settings, {}, torch.device("cpu"))
    history = read_json(tmp_path / "es" / "history.json")
    assert 3 <= len(history) < 50
    manifest = read_json(tmp_path / "es" / "training_manifest.json")
    assert manifest["early_stopped"] is True and manifest["best_epoch"] == 1


def test_readout_dropout_and_units():
    tabular = TabularReadout(10, [8], dropout=0.2)
    assert any(isinstance(module, torch.nn.Dropout) for module in tabular.modules())
    graph = GraphSetReadout(4, 8, [8], layers=1, use_edges=True, dropout=0.2)
    assert any(isinstance(module, torch.nn.Dropout) for module in graph.modules())
    assert unit_filters("refine:multi_task:1:R4:2") == {"variant": "multi_task", "upstream_seed": 1,
                                                        "recipe": "R4", "downstream_seed": 2}


def test_unit_enumeration(tmp_path):
    context = fixture_context(tmp_path)
    units = enumerate_units(context)
    assert "prepare" in units and "evaluate" in units and "analyze" in units
    assert "upstream:multi_task:1" in units and "cache:multi_task:1" in units
    assert "refine:multi_task:1:R4:1" in units
    assert not any(unit.startswith("refine:single_task") and "R4" in unit for unit in units)
    assert "refine:single_task:1:R0:1" in units
