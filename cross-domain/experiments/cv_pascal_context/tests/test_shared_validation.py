"""Intentional shared validation, while train and final Y stay isolated."""
import copy
import json
from pathlib import Path
import sys

import pytest

DOMAIN_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(DOMAIN_ROOT))
from pipeline.io import read_json, write_json
from experiments.cv_pascal_context import data, workflow
from experiments.cv_pascal_context.tests.test_contracts import context, synthetic_raw


def test_grouped_shared_validation_is_one_partition_with_two_roles():
    train = [f"t{i}" for i in range(8)]
    final = ["v0", "v1"]
    groups = {i: i for i in train + final}
    groups["t1"] = "t0"
    counts = dict(a_train=4, a_val=2, b_train=2, b_val=2)
    splits = data.grouped_split(train, final, groups, counts, 4, shared_validation=True)
    assert splits == data.grouped_split(train, final, groups, counts, 4, shared_validation=True)
    assert splits["a_val"] == splits["b_val"]
    unique = ["a_train", "a_val", "b_train", "y_test"]
    assert [len(splits[k]) for k in unique] == [4, 2, 2, 2]
    assert set().union(*(set(splits[k]) for k in unique)) == set(train + final)
    for index, left in enumerate(unique):
        for right in unique[index+1:]:
            assert not {groups[i] for i in splits[left]} & {groups[i] for i in splits[right]}
    # No partially shared or differently ordered validation partition is created.
    assert splits["a_val"] is not splits["b_val"]
    groups["v0"] = "t0"
    with pytest.raises(ValueError, match="locked Y"):
        data.grouped_split(train, final, groups, counts, 4, shared_validation=True)


@pytest.mark.parametrize("counts", [
    dict(a_train=4, a_val=2, b_train=2, b_val=1),
    dict(a_train=4, a_val=0, b_train=2, b_val=0),
    dict(a_train=4, a_val=True, b_train=2, b_val=True),
    dict(a_train=4, a_val=2, b_train=2),
])
def test_invalid_shared_counts_fail(counts):
    with pytest.raises(ValueError):
        data.partition_counts(counts, shared_validation=True)


def test_prepare_shared_val_manifest_and_both_loader_roles(tmp_path):
    ctx = context(tmp_path)
    synthetic_raw(ctx)
    ctx.config["data"].update(shared_validation=True, counts=dict(a_train=1, a_val=2, b_train=1, b_val=2))
    artifacts = data.prepare(ctx)
    manifest = read_json(artifacts[0])
    assert manifest["shared_validation"] is True
    assert manifest["unique_train_images"] == 4
    assert manifest["splits"]["a_val"] == manifest["splits"]["b_val"]
    assert data.PascalSamples(ctx, "a_val").ids == data.PascalSamples(ctx, "b_val").ids
    assert len(data.data_identity(ctx)) == 64
    unique = [manifest["splits"][s] for s in ("a_train", "a_val", "b_train", "y_test")]
    assert len(set(sum(unique, []))) == sum(map(len, unique)) == 6
    manifest["splits"]["b_val"].reverse()
    write_json(artifacts[0], manifest)
    with pytest.raises(ValueError, match="identical ordered"):
        data.data_manifest(ctx)


def test_default_and_three_variant_configs_count_shared_ids_once(tmp_path):
    ctx = context(tmp_path)
    for filename in ("cluster_full.json", "cluster_three_variant.json"):
        cfg = json.loads((Path(__file__).resolve().parents[1]/"config"/filename).read_text())
        cfg.update(output_root=str(tmp_path/"out"), data_root=str(tmp_path/"data"))
        ctx.config = cfg
        assert "v4_sharedval500" in cfg["experiment"]
        assert cfg["data"]["shared_validation"] is True
        assert cfg["data"]["counts"] == dict(a_train=3000, a_val=500, b_train=1000, b_val=500)
        plan = workflow.plan(ctx)
        assert plan["shared_validation"] is True
        assert plan["unique_official_train_images"] == 4998
        assert plan["unique_active_train_images"] == 4500
        assert plan["unused_train_images"] == 498
        # B cache has exactly B_train + shared val; A_val is not duplicated.
        expected = plan["cache_bytes_per_upstream"]["multi_task"]
        old = copy.deepcopy(cfg)
        old["data"].update(shared_validation=False, unused_train_count=0, counts=dict(a_train=3000, a_val=500, b_train=1000, b_val=498))
        ctx.config = old
        previous = workflow.plan(ctx)["cache_bytes_per_upstream"]["multi_task"]
        assert abs(expected - previous*1500/1498) < 2
        ctx.config = cfg
        cfg["data"]["counts"]["b_val"] = 499
        with pytest.raises(ValueError, match="equal"):
            workflow.plan(ctx)


def test_val_reduction_preserves_A_B_Y_membership_and_duplicate_groups():
    train = [f"t{i}" for i in range(10)]
    final = ["v0", "v1"]
    groups = {i: i for i in train+final}
    groups["t1"] = "t0"
    old = data.grouped_split(train, final, groups, dict(a_train=4, a_val=4, b_train=2, b_val=4), 2718, True)
    new = data.grouped_split(train, final, groups, dict(a_train=4, a_val=2, b_train=2, b_val=2), 2718, True, 2)
    assert all(new[k] == old[k] for k in ("a_train", "b_train", "y_test"))
    assert new["a_val"] == new["b_val"] and len(new["a_val"]) == 2
    assert set(new["a_val"]) | set(new["unused_train"]) == set(old["a_val"])
    assert not {groups[i] for i in new["a_val"]} & {groups[i] for i in new["unused_train"]}


def test_prepare_records_unused_ids_and_forbids_loading_them(tmp_path):
    ctx = context(tmp_path)
    synthetic_raw(ctx)
    ctx.config["data"].update(shared_validation=True, unused_train_count=1,
                              counts=dict(a_train=1, a_val=1, b_train=1, b_val=1))
    manifest_path, = data.prepare(ctx)
    manifest = read_json(manifest_path)
    assert manifest["unique_train_images"] == 3
    assert manifest["unused_train_count"] == len(manifest["unused_train_ids"]) == 1
    used = set().union(*(set(ids) for ids in manifest["splits"].values()))
    assert not used & set(manifest["unused_train_ids"])
    assert len(data.data_identity(ctx)) == 64
    with pytest.raises(ValueError):
        data.PascalSamples(ctx, "unused_train")
    manifest["unused_train_ids"] = list(manifest["splits"]["a_train"])
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="unused images"):
        data.data_manifest(ctx)
