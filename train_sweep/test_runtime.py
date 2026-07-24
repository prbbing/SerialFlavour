"""Focused tests for the standalone multi-model sweep runtime."""
import argparse
from copy import deepcopy
from io import StringIO
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.training import train_epoch, validate_epoch
from train_sweep.__main__ import _parse_gpu, main
from train_sweep.runtime import (
    RunState,
    _apply_gpu_override,
    _build_runs,
    _log_checkpoint_summary,
    _log_epoch,
    _log_model_setup,
    _log_startup,
    _save_epoch,
    _visible_gpu_inventory,
    _write_history,
    _write_manifest,
    normalise_gpu_request,
    select_device,
    train_epoch_sweep,
    validate_epoch_sweep,
    validate_sweep_configs,
)


class TinyOriginModel(torch.nn.Module):
    def __init__(self, features=4, hidden=6, origins=3, jets=3):
        super().__init__()
        self.encoder = torch.nn.Linear(features, hidden)
        self.origin_head = torch.nn.Linear(hidden, origins)
        self.jet_head = torch.nn.Linear(hidden, jets)

    def forward(self, x, mask):
        hidden = torch.tanh(self.encoder(x))
        masked = hidden * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return {
            "jet_logits": self.jet_head(pooled),
            "origin_logits": self.origin_head(hidden),
        }


def _config():
    return SimpleNamespace(
        n_origin_classes=3,
        lambda_jet=1.0,
        lambda_origin=0.7,
        lambda_vertex=0.0,
        lambda_pair=0.0,
        fit_lxy=True,
        fit_dz=True,
    )


def _loader():
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(6, 5, 4, generator=generator)
    mask = torch.ones(6, 5, dtype=torch.bool)
    mask[1, -1] = False
    mask[4, -2:] = False
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    origin = torch.randint(0, 3, (6, 5), generator=generator)
    origin[~mask] = -1
    lxy = torch.zeros(6, 2)
    dz = torch.zeros(6, 2)
    valid = torch.zeros(6, 2, dtype=torch.bool)
    empty_pair = torch.empty(6, 0)
    return DataLoader(TensorDataset(
        x, mask, y, origin, lxy, dz, valid, empty_pair),
        batch_size=2, shuffle=False)


def _state(name, model):
    return RunState(
        name=name,
        config_path=f"{name}.json",
        config=_config(),
        config_dict={},
        output_dir="",
        model=model,
        optimiser=torch.optim.Adam(model.parameters(), lr=1e-3),
        criterion_jet=torch.nn.CrossEntropyLoss(),
        criterion_origin=torch.nn.CrossEntropyLoss(ignore_index=-1),
    )


def test_two_models_update_from_one_shared_loader_on_cpu():
    torch.manual_seed(3)
    run_a = _state("a", TinyOriginModel())
    torch.manual_seed(4)
    run_b = _state("b", TinyOriginModel())
    before_a = [parameter.detach().clone() for parameter in run_a.model.parameters()]
    before_b = [parameter.detach().clone() for parameter in run_b.model.parameters()]

    metrics = train_epoch_sweep(
        [run_a, run_b], _loader(), torch.device("cpu"), max_concurrent=2)

    assert set(metrics) == {"a", "b"}
    assert any(not torch.equal(old, new)
               for old, new in zip(before_a, run_a.model.parameters()))
    assert any(not torch.equal(old, new)
               for old, new in zip(before_b, run_b.model.parameters()))
    assert all(torch.isfinite(torch.tensor(value))
               for result in metrics.values() for value in result.values())


def test_training_log_keeps_legacy_epoch_format_without_console_prefix(capsys):
    config = _config()
    config.epochs = 7
    config.calibrate_vertex_fit = False
    config.vertex_leg_names = ["b_vertex", "c_vertex"]
    run = _state("model_a", TinyOriginModel())
    run.config = config
    run.log_handle = StringIO()
    train_metrics = {
        "loss": 1.0, "jet": 0.4, "origin": 0.5, "vertex": 0.1,
        "lxy": 0.06, "dz": 0.04, "refine": 0.0, "vtx_weight": 0.0,
    }
    val_metrics = {
        "loss": 0.9, "jet": 0.3, "origin": 0.5, "vertex": 0.1,
        "lxy": 0.06, "dz": 0.04, "acc": 0.75, "origin_acc": 0.60,
    }

    _log_epoch(run, 2, train_metrics, val_metrics, {}, {}, 3.21)

    expected = (
        "Epoch 02/7  loss=1.0000 (jet=0.4000 origin=0.5000 "
        "vtx=0.1000(Lxy=0.0600,dz=0.0400))  val_loss=0.9000 "
        "(jet=0.3000 origin=0.5000 vtx=0.1000(Lxy=0.0600,dz=0.0400))  "
        "val_acc=0.7500  origin_acc=0.6000  epoch_seconds=3.21\n")
    assert run.log_handle.getvalue() == expected
    assert capsys.readouterr().out == f"[model_a] {expected}"


def test_middle_epochs_are_compact_and_last_five_are_detailed():
    config = _config()
    config.epochs = 20
    config.calibrate_vertex_fit = False
    config.vertex_leg_names = ["b_vertex", "c_vertex"]
    run = _state("model_a", TinyOriginModel())
    run.config = config
    run.log_handle = StringIO()
    train_metrics = {
        "loss": 1.0, "jet": 0.4, "origin": 0.5, "vertex": 0.1,
        "lxy": 0.06, "dz": 0.04, "refine": 0.0, "vtx_weight": 0.0,
    }
    val_metrics = {
        "loss": 0.9, "jet": 0.3, "origin": 0.5, "vertex": 0.1,
        "lxy": 0.06, "dz": 0.04, "acc": 0.75, "origin_acc": 0.60,
    }

    _log_epoch(run, 6, train_metrics, val_metrics, {}, {}, 12.34)
    middle_line = run.log_handle.getvalue()
    assert middle_line == (
        "Epoch 06/20  loss=1.0000  val_loss=0.9000  "
        "val_acc=0.7500  origin_acc=0.6000  epoch_seconds=12.34\n")
    assert "jet=" not in middle_line

    run.log_handle = StringIO()
    _log_epoch(run, 16, train_metrics, val_metrics, {}, {}, 5.67)
    final_line = run.log_handle.getvalue()
    assert final_line.startswith("Epoch 16/20  loss=1.0000 (jet=0.4000")
    assert "vtx=0.1000(Lxy=0.0600,dz=0.0400)" in final_line
    assert "epoch_seconds=5.67" in final_line


def test_single_model_cpu_path_matches_existing_epoch_functions():
    torch.manual_seed(12)
    reference = TinyOriginModel()
    sweep_model = deepcopy(reference)
    config = _config()
    loader = _loader()
    jet_criterion = torch.nn.CrossEntropyLoss()
    origin_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    reference_optimiser = torch.optim.Adam(reference.parameters(), lr=1e-3)
    reference_train = train_epoch(
        reference, loader, reference_optimiser,
        jet_criterion, origin_criterion, config.n_origin_classes,
        config.lambda_jet, config.lambda_origin, config.lambda_vertex,
        config.lambda_pair, config.fit_lxy, config.fit_dz,
        torch.device("cpu"))

    run = _state("sweep", sweep_model)
    sweep_train = train_epoch_sweep(
        [run], loader, torch.device("cpu"), max_concurrent=1)["sweep"]
    for reference_value, key in zip(
            reference_train,
            ("loss", "jet", "origin", "vertex", "lxy", "dz",
             "refine", "vtx_weight")):
        assert sweep_train[key] == pytest.approx(reference_value, abs=1e-7)
    for reference_parameter, sweep_parameter in zip(
            reference.parameters(), sweep_model.parameters()):
        torch.testing.assert_close(reference_parameter, sweep_parameter)

    reference_val = validate_epoch(
        reference, loader, jet_criterion, origin_criterion,
        config.n_origin_classes, config.lambda_jet, config.lambda_origin,
        config.lambda_vertex, config.lambda_pair, config.fit_lxy,
        config.fit_dz, torch.device("cpu"))
    sweep_val, sweep_arrays = validate_epoch_sweep(
        [run], loader, torch.device("cpu"), max_concurrent=1)["sweep"]
    for reference_value, key in zip(
            reference_val[:-1],
            ("loss", "jet", "origin", "vertex", "lxy", "dz",
             "acc", "origin_acc")):
        assert sweep_val[key] == pytest.approx(reference_value, abs=1e-7)
    for key in ("all_preds", "all_true", "all_probs",
                "origin_preds", "origin_true"):
        torch.testing.assert_close(
            torch.from_numpy(sweep_arrays[key]),
            torch.from_numpy(reference_val[-1][key]))


def test_config_validation_rejects_incompatible_data_and_pair_settings():
    config, raw = load_config(
        "configs/1M_training_jets/vertex_ablation/a5_no_vertex_loss.json")
    changed = deepcopy(config)
    changed.batch_size += 1
    with pytest.raises(ValueError, match="batch_size"):
        validate_sweep_configs([
            ("a.json", config, raw),
            ("b.json", changed, deepcopy(raw)),
        ])

    parallel = deepcopy(config)
    parallel.model_type = "parallel_origin_vertex_jet"
    parallel.use_pair_target = False
    with pytest.raises(ValueError, match="requires use_pair_target=true"):
        validate_sweep_configs([("parallel.json", parallel, deepcopy(raw))])


def test_config_validation_rejects_multi_gpu_and_duplicate_names():
    config, raw = load_config(
        "configs/1M_training_jets/vertex_ablation/a5_no_vertex_loss.json")
    multi_gpu = deepcopy(config)
    multi_gpu.gpu_ids = [0, 1]
    with pytest.raises(ValueError, match="DataParallel"):
        validate_sweep_configs([("multi.json", multi_gpu, deepcopy(raw))])
    with pytest.raises(ValueError, match="stems must be unique"):
        validate_sweep_configs([
            ("one/same.json", config, raw),
            ("two/same.json", deepcopy(config), deepcopy(raw)),
        ])


def test_gpu_argument_accepts_auto_and_visible_indices():
    assert normalise_gpu_request(None) is None
    assert _parse_gpu("auto") == "auto"
    assert _parse_gpu("0") == 0
    assert _parse_gpu("2") == 2
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_gpu("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_gpu("gpu0")


def test_cli_forwards_manual_gpu(monkeypatch):
    received = {}

    def fake_run_sweep(configs, **kwargs):
        received["configs"] = configs
        received.update(kwargs)

    monkeypatch.setattr("train_sweep.__main__.run_sweep", fake_run_sweep)
    assert main([
        "--config", "a.json", "--config", "b.json", "--gpu", "2",
        "--max-concurrent", "1", "--seed", "9",
    ]) == 0
    assert received == {
        "configs": ["a.json", "b.json"],
        "max_concurrent": 1,
        "seed": 9,
        "gpu": 2,
    }


def test_gpu_override_replaces_inconsistent_and_multi_gpu_configs():
    config, raw = load_config(
        "configs/1M_training_jets/vertex_ablation/a5_no_vertex_loss.json")
    first = deepcopy(config)
    second = deepcopy(config)
    first.gpu_ids = [0, 1]
    second.gpu_ids = [3]
    entries = [
        ("one.json", first, deepcopy(raw)),
        ("two.json", second, deepcopy(raw)),
    ]

    validate_sweep_configs(entries, gpu_override=True)
    selection = {"visible_index": 2}
    _apply_gpu_override(entries, selection)

    for _, effective, effective_raw in entries:
        assert effective.gpu_ids == [2]
        assert effective_raw["gpu_ids"] == [2]


def test_auto_gpu_uses_most_free_visible_device_and_lowest_tie(
        monkeypatch):
    inventory = [
        {"physical_index": 0, "uuid": "GPU-0", "name": "A10-0",
         "memory_total_mib": 24000, "memory_used_mib": 23000,
         "memory_free_mib": 1000},
        {"physical_index": 1, "uuid": "GPU-1", "name": "A10-1",
         "memory_total_mib": 24000, "memory_used_mib": 4000,
         "memory_free_mib": 20000},
        {"physical_index": 2, "uuid": "GPU-2", "name": "A10-2",
         "memory_total_mib": 24000, "memory_used_mib": 4000,
         "memory_free_mib": 20000},
    ]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        "train_sweep.runtime._query_gpu_inventory", lambda: inventory)
    config = SimpleNamespace(gpu_ids=[-1])

    device, selection = select_device(config, "auto")

    assert device == torch.device("cuda:0")
    assert selection["mode"] == "auto"
    assert selection["visible_index"] == 0
    assert selection["physical_index"] == 2
    assert selection["uuid"] == "GPU-2"
    assert selection["memory_free_mib"] == 20000
    assert selection["cuda_visible_devices"] == "2,1"


def test_manual_gpu_uses_pytorch_visible_index(monkeypatch):
    inventory = [
        {"physical_index": 3, "uuid": "GPU-3", "name": "A10-3",
         "memory_total_mib": 24000, "memory_used_mib": 3000,
         "memory_free_mib": 21000},
        {"physical_index": 7, "uuid": "GPU-7", "name": "A10-7",
         "memory_total_mib": 24000, "memory_used_mib": 5000,
         "memory_free_mib": 19000},
    ]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        "train_sweep.runtime._query_gpu_inventory", lambda: inventory)

    device, selection = select_device(SimpleNamespace(gpu_ids=[-1]), 1)

    assert device == torch.device("cuda:1")
    assert selection["mode"] == "manual"
    assert selection["visible_index"] == 1
    assert selection["physical_index"] == 7
    assert selection["name"] == "A10-7"


def test_visible_gpu_mapping_supports_uuid_order():
    inventory = [
        {"physical_index": 0, "uuid": "GPU-aaa", "name": "first"},
        {"physical_index": 1, "uuid": "GPU-bbb", "name": "second"},
    ]
    visible = _visible_gpu_inventory(
        inventory, 2, visible_devices="GPU-bbb,GPU-aaa")
    assert [(row["visible_index"], row["physical_index"])
            for row in visible] == [(0, 1), (1, 0)]


def test_explicit_gpu_errors_before_training_when_unavailable_or_invalid(
        monkeypatch):
    config = SimpleNamespace(gpu_ids=[-1])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device, selection = select_device(config, None)
    assert device == torch.device("cpu")
    assert selection["mode"] == "config"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        select_device(config, 0)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="out of range"):
        select_device(config, 1)

    def failed_query():
        raise RuntimeError("Unable to query GPUs with nvidia-smi; use --gpu N")

    monkeypatch.setattr(
        "train_sweep.runtime._query_gpu_inventory", failed_query)
    with pytest.raises(RuntimeError, match="use --gpu N"):
        select_device(config, "auto")


def test_run_outputs_keep_effective_config_checkpoints_history_and_manifest(
        tmp_path):
    config, raw = load_config(
        "configs/1M_training_jets/vertex_ablation/a5_no_vertex_loss.json")
    config = deepcopy(config)
    raw = deepcopy(raw)
    config.plot_dir = str(tmp_path / "run")
    config.tensorboard_log_dir = None
    raw["train_plot_dir"] = config.plot_dir
    raw["tensorboard_log_dir"] = None
    train_data = {
        "origin": np.array([[0, 1, 2, -1], [2, 1, 0, -1]], dtype=np.int64),
    }
    entries = [("sample.json", config, raw)]
    _apply_gpu_override(entries, {"visible_index": 2})
    runs = _build_runs(
        entries, train_data, torch.device("cpu"), "20260101_000000", seed=7)
    run = runs[0]
    try:
        with open(run.output_dir + "config.json", encoding="utf-8") as handle:
            effective = json.load(handle)
        assert effective["num_workers"] == 0
        assert effective["train_plot_dir"] == run.output_dir
        assert effective["gpu_ids"] == [2]

        _log_startup(run, [0], "  cached data loaded\n")
        _log_model_setup(run, torch.device("cpu"), 2, 1)

        for key in run.history:
            run.history[key].append(0.0)
        _save_epoch(run, 1, {"jet": 0.4, "loss": 0.5})
        _log_checkpoint_summary(run)
        _write_history(run)
        manifest = {
            "status": "complete",
            "ordered_configs": [run.config_path],
            "runs": {run.name: run.output_dir},
            "num_workers": 0,
            "gpu_selection": {
                "mode": "manual", "visible_index": 2,
                "physical_index": 7,
            },
        }
        _write_manifest(manifest, runs)

        for filename in (
            "best_jet.pt", "best_total.pt", "last.pt",
            "training_history.json", "training_history.csv",
            "gradient_diagnostics.csv", "sweep_manifest.json",
        ):
            assert (tmp_path / "run_20260101_000000" / filename).is_file()

        run.log_handle.flush()
        log_text = (tmp_path / "run_20260101_000000" / "training_log.md").read_text(
            encoding="utf-8")
        assert log_text.startswith("Device: [0]  |  DataParallel: False\n")
        assert "[sample]" not in log_text
        assert "Loading data...\n  cached data loaded\n" in log_text
        assert "Origin class weights:\n" in log_text
        assert "Saved best_jet.pt (epoch=1, val_jet_loss=0.400000)\n" in log_text
        assert "Checkpoint summary:\n" in log_text
        assert "  last.pt: epoch=" in log_text
        manifest_text = json.loads(
            (tmp_path / "run_20260101_000000" / "sweep_manifest.json").read_text(
                encoding="utf-8"))
        assert manifest_text["gpu_selection"]["visible_index"] == 2
        assert manifest_text["gpu_selection"]["physical_index"] == 7
    finally:
        for state in runs:
            state.close()


def test_duplicate_output_directories_fail_before_creation(tmp_path):
    config, raw = load_config(
        "configs/1M_training_jets/vertex_ablation/a5_no_vertex_loss.json")
    first = deepcopy(config)
    second = deepcopy(config)
    first.plot_dir = second.plot_dir = str(tmp_path / "same")
    first.tensorboard_log_dir = second.tensorboard_log_dir = None
    train_data = {"origin": np.array([[0, 1, 2]], dtype=np.int64)}

    with pytest.raises(ValueError, match="output directories must be unique"):
        _build_runs([
            ("one.json", first, deepcopy(raw)),
            ("two.json", second, deepcopy(raw)),
        ], train_data, torch.device("cpu"), "20260101_000000", seed=7)
    assert not (tmp_path / "same_20260101_000000").exists()
