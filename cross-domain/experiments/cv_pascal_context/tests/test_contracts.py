"""Offline engineering contracts; no real data, weights, downloads or training."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest
import torch

DOMAIN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DOMAIN_ROOT))
from pipeline.context import Context
from pipeline.units import enumerate_units, unit_filters
from experiments.cv_pascal_context import data, model, refine, source, training, evaluate


def full_config():
    return json.loads((Path(__file__).resolve().parents[1] / "config/cluster_full.json").read_text())


def context(tmp_path):
    config = full_config()
    config.update(data_root=str(tmp_path / "data"), output_root=str(tmp_path / "out"), log_root=str(tmp_path / "logs"))
    config["data"]["raw_root"] = str(tmp_path / "raw")
    return Context(config, DOMAIN_ROOT, tmp_path / "config.json")


def test_complete_matrix_filters_and_epoch_zero(tmp_path):
    ctx = context(tmp_path)
    units = enumerate_units(ctx)
    assert len(units) == len(set(units)) == 113
    assert len(evaluate.expected_keys(ctx)) == 105
    assert sum(u.startswith("refine:") for u in units) == 100
    assert unit_filters("refine:multi_task:11:aux_hidden:101") == {"variant": "multi_task", "upstream_seed": 11, "recipe": "aux_hidden", "downstream_seed": 101}
    assert ctx.config["refiner"]["include_initial_checkpoint"]
    assert not ctx.output_dir.exists()


def test_pinned_author_source_no_network(monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network forbidden")))
    assert source.verify_source()["commit"] == source.COMMIT
    author = source.author_module("models.mti_net")
    assert author.MTINet.__module__.startswith(source.PREFIX)


def test_actual_full_author_models_and_shared_initialization():
    # This builds and executes the actual vendored architecture with randomly
    # initialized weights. It does not verify pretrained checkpoint loading.
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        states = {}
        counts = {}
        for variant in model.VARIANTS:
            torch.manual_seed(42)
            network = model.AuthorNetwork(variant).eval()
            states[variant] = (model.state_digest(network.network.backbone.state_dict()), model.state_digest((network.network.decoder if variant == "single_task" else network.network.heads["semseg"]).state_dict()))
            counts[variant] = sum(p.numel() for p in network.parameters())
            with torch.inference_mode():
                output = network(torch.randn(1, 3, 64, 64))
            assert output["semseg"].shape == (1, 21, 64, 64)
            reconstructed = model.native_logits({"native_lowres": output["native_lowres"], "output_size": torch.tensor([[64, 64]])})
            assert output["native_lowres"].shape == (1, 21, 16, 16)
            assert torch.equal(reconstructed, output["semseg"])
            assert [value.shape[1] for value in output["h"]] == [36, 72, 144, 288]
            if variant != "single_task":
                assert set(model.TASK_OUTPUTS) <= set(output)
                assert len(output["deep_supervision"]) == 4
                assert [v.shape[1] for v in output["aux_hidden"]] == [72, 144, 288, 576]
                assert [v.shape[1] for v in output["aux_prediction"]] == [24]*4
            del network, output
        assert len(set(states.values())) == 1
        assert counts["main_only_matched"] == counts["multi_task"] > counts["single_task"]
    finally:
        torch.set_num_threads(old_threads)


def payload():
    return {"h": [torch.randn(2*c, 16//(2**i), 16//(2**i)) for i, c in enumerate(model.CHANNELS)],
            "aux_hidden": [torch.randn(4*c, 16//(2**i), 16//(2**i)) for i, c in enumerate(model.CHANNELS)],
            "aux_prediction": [torch.rand(24, 16//(2**i), 16//(2**i)) for i, c in enumerate(model.CHANNELS)],
            "native_lowres": torch.randn(21, 16, 16), "output_size": torch.tensor([64, 64]),
            "semseg": torch.randint(0, 21, (64, 64))}


def test_native_epoch_zero_and_matched_capacity_without_aux_truth():
    sample = payload()
    expected_params = None
    for recipe in data.RECIPES:
        network = model.SpatialResidual(width=8).eval()
        values = refine.features(sample, recipe, sample if recipe == "aux_shuffle" else None)
        batch = {"features": [value[None] for value in values], "native_lowres": sample["native_lowres"][None], "output_size": sample["output_size"][None]}
        assert [v.shape[1] for v in batch["features"]] == [108, 216, 432, 864]
        with torch.inference_mode():
            assert torch.equal(network(batch), model.native_logits(batch))
        count = sum(p.numel() for p in network.parameters())
        expected_params = count if expected_params is None else expected_params
        assert count == expected_params
    assert "auxiliary_truth" not in sample


def test_shuffle_fixed_derangement_and_partition_size():
    first = refine.derangement(498, 19)
    assert first == refine.derangement(498, 19)
    assert first != refine.derangement(498, 20)
    assert sorted(first) == list(range(498))
    assert all(i != donor for i, donor in enumerate(first))
    with pytest.raises(ValueError):
        refine.derangement(1, 0)


def test_group_split_preserves_duplicates_and_exact_counts():
    train = [f"t{i}" for i in range(8)]
    val = ["v0", "v1"]
    groups = {i: i for i in train+val}
    groups["t1"] = "t0"
    result = data.grouped_split(train, val, groups, dict(a_train=4, a_val=2, b_train=1, b_val=1), 4)
    assert [len(result[k]) for k in data.SPLITS] == [4, 2, 1, 1, 2]
    assert next(k for k in result if "t0" in result[k]) == next(k for k in result if "t1" in result[k])
    assert result == data.grouped_split(train, val, groups, dict(a_train=4, a_val=2, b_train=1, b_val=1), 4)
    groups["v0"] = "t0"
    with pytest.raises(ValueError, match="locked Y"):
        data.grouped_split(train, val, groups, dict(a_train=4, a_val=2, b_train=1, b_val=1), 4)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "dir\\escape"])
def test_safe_tar_path_rejection(tmp_path, name):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as stream:
        info = tarfile.TarInfo(name)
        info.size = 1
        stream.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        data.safe_extract(archive, tmp_path / "extract", 100, 100)


def test_safe_tar_links_and_size_limits(tmp_path):
    for label, kind, size in (("link", tarfile.SYMTYPE, 0), ("large", tarfile.REGTYPE, 11)):
        archive = tmp_path / f"{label}.tar"
        with tarfile.open(archive, "w") as stream:
            info = tarfile.TarInfo(label)
            info.type, info.size, info.linkname = kind, size, "outside"
            stream.addfile(info, io.BytesIO(b"x"*size) if size else None)
        with pytest.raises(ValueError):
            data.safe_extract(archive, tmp_path / label, 10, 10)


def test_auxiliary_labels_forbidden_in_B_and_Y(tmp_path):
    ctx = context(tmp_path)
    for split in ("b_train", "b_val", "y_test"):
        with pytest.raises(ValueError, match="restricted to A_train"):
            data.PascalSamples(ctx, split, auxiliary=True)


def test_all_void_aux_loss_is_finite_and_semantic_label_guard():
    logits = torch.randn(2, 7, 8, 8, requires_grad=True)
    loss = training.task_loss(logits, torch.full((2, 8, 8), 255), "human_parts")
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert logits.grad is not None
    with pytest.raises(ValueError):
        training.semantic_loss(torch.randn(1, 21, 2, 2), torch.full((1, 2, 2), 59))


def synthetic_raw(ctx, bad_label=None, duplicate_y=False):
    import numpy as np
    from PIL import Image
    raw = data.raw_dir(ctx)
    ctx.config["data"].update(shared_validation=False, unused_train_count=0, expected_train=4, expected_val=2, counts={"a_train": 1, "a_val": 1, "b_train": 1, "b_val": 1})
    split_dir = raw / "ImageSets/Context"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("t0\nt1\nt2\nt3\n")
    (split_dir / "val.txt").write_text("v0\nv1\n")
    for index, image_id in enumerate(["t0", "t1", "t2", "t3", "v0", "v1"]):
        image = np.full((8, 8, 3), index*30, dtype=np.uint8)
        paths = {"image": raw / f"JPEGImages/{image_id}.jpg", "semseg": raw / f"semseg/VOC12/{image_id}.png", "context": raw / f"pascal-context/trainval/{image_id}.mat", "human_parts": raw / f"human_parts/{image_id}.mat", "normals": raw / f"normals_distill/{image_id}.png", "sal": raw / f"sal_distill/{image_id}.png"}
        for task, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if task == "image":
                Image.fromarray(image).save(path)
            elif task == "semseg":
                Image.fromarray(np.full((8, 8), bad_label if bad_label is not None else index, dtype=np.uint8)).save(path)
            else:
                path.write_bytes(b"fixture-not-decoded-by-audit")
    if duplicate_y:
        (raw / "JPEGImages/v0.jpg").write_bytes((raw / "JPEGImages/t0.jpg").read_bytes())


def test_prepare_synthetic_schema_and_disjointness(tmp_path):
    ctx = context(tmp_path)
    synthetic_raw(ctx)
    paths = data.prepare(ctx)
    manifest = json.loads(paths[0].read_text())
    assert len(set(sum(manifest["splits"].values(), []))) == 6
    assert manifest["auxiliary_truth_downstream"] is False
    assert len(data.data_identity(ctx)) == 64


@pytest.mark.parametrize("bad_label,duplicate", [(59, False), (None, True)])
def test_prepare_rejects_59_class_and_train_Y_duplicates(tmp_path, bad_label, duplicate):
    ctx = context(tmp_path)
    synthetic_raw(ctx, bad_label=bad_label, duplicate_y=duplicate)
    with pytest.raises(ValueError):
        data.prepare(ctx)


def test_required_pretrained_hash_and_no_implicit_fetch(tmp_path):
    ctx = context(tmp_path)
    ctx.config["model"]["pretrained"]["path"] = str(tmp_path / "missing.pth")
    with pytest.raises(FileNotFoundError):
        model.pretrained_identity(ctx)
    path = tmp_path / "missing.pth"
    torch.save({"anything": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="sha256"):
        model.pretrained_identity(ctx)


def test_Y_scoring_is_not_reached_when_checkpoint_matrix_missing(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    ctx.config.pop("workflow", None)
    monkeypatch.setattr(evaluate, "metrics", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Y accessed prematurely")))
    with pytest.raises(FileNotFoundError):
        evaluate.evaluate(ctx)


def test_pool_marker_checks_identity_and_artifacts(tmp_path):
    specification = importlib.util.spec_from_file_location("pascal_pool_contract", Path(__file__).resolve().parents[1] / "scripts/run_cluster.py")
    pool = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(pool)
    ctx = context(tmp_path)
    from pipeline.io import artifact_record, write_json
    from pipeline.worker import marker_path
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("complete")
    marker = marker_path(ctx, "prepare")
    write_json(marker, {"status": "complete", "identity": ctx.identity, "artifacts": [artifact_record(artifact)]})
    assert pool.unit_complete(ctx, "prepare")
    artifact.write_text("changed")
    assert not pool.unit_complete(ctx, "prepare")


def test_exact_lowres_cache_disk_readout_and_main_metrics(tmp_path):
    from torch.utils.data import DataLoader
    from pipeline.io import sha256_file
    sample = payload()
    sample.update(id="synthetic_image", split="b_val")
    path = tmp_path / "sample.pt"
    torch.save(sample, path)
    rows = [{"path": path.name, "id": sample["id"], "sha256": sha256_file(path)}]
    dataset = refine.DiskCached(tmp_path, rows, "aux_prediction", 17)
    loader = DataLoader(dataset, batch_size=1)
    batch = next(iter(loader))
    assert "native" not in batch and "final_predictions" not in batch
    assert batch["native_lowres"].shape == (1, 21, 16, 16)
    expected = model.native_logits(batch)
    network = model.SpatialResidual(width=8).eval()
    with torch.inference_mode():
        assert torch.equal(network(batch), expected)
    metrics = evaluate.metrics(evaluate.Native(), loader, "cpu")
    assert metrics["images"] == 1 and metrics["valid_pixels"] == 64*64
    assert sum(map(sum, metrics["confusion_matrix"])) == 64*64


def test_storage_plan_counts_only_exact_lowres_native_and_required_features(tmp_path):
    ctx = context(tmp_path)
    plan = refine.storage_plan(ctx)
    assert plan["cache_splits"] == ["b_train", "b_val"]
    assert plan["full_cache_bytes"] == 5*plan["peak_cache_bytes"]
    assert plan["planned_output_bytes"] == plan["peak_cache_bytes"]+plan["peak_spool_bytes"]
    assert 37_000_000_000 < plan["planned_output_bytes"] < 39_000_000_000
    assert plan["prediction_uint8_uncompressed_bytes"] == 0
    assert not ctx.output_dir.exists()
