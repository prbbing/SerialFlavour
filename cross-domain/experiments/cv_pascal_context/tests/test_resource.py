"""Resource lifecycle engineering checks without real data/weights."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader
from pipeline.io import read_json, write_json, sha256_file
from experiments.cv_pascal_context import data, evaluate, model, refine, workflow
from experiments.cv_pascal_context import stream_evaluate as streamed
from experiments.cv_pascal_context.tests.test_contracts import context, payload


def test_default_mt_and_optional_three_variant_matrix(tmp_path):
    from pipeline.units import enumerate_units
    ctx = context(tmp_path)
    assert ctx.config["upstream"]["variants"] == ["multi_task"]
    assert ctx.config["cache"]["splits"] == ["b_train","b_val"]
    assert ctx.config["evaluation"]["save_predictions"] is False
    assert workflow.plan(ctx)["refine_units"] == 100
    assert len(enumerate_units(ctx)) == 113
    other = json.loads((Path(__file__).resolve().parents[1]/"config/cluster_three_variant.json").read_text())
    assert other["experiment"] != ctx.config["experiment"]
    ctx.config = other
    assert len(enumerate_units(ctx)) == 258
    assert len(evaluate.expected_keys(ctx)) == 240
    assert workflow.plan(ctx)["peak_cache_bytes"] < workflow.plan(ctx)["full_cache_bytes"]


def test_plan_import_is_read_only_without_torch(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = f"""import sys
sys.path.insert(0,{str(root)!r})
from experiments.cv_pascal_context import workflow
from pipeline.context import load_context
ctx=load_context({str(Path(__file__).resolve().parents[1]/'config/cluster_full.json')!r},{str(root)!r})
workflow.plan(ctx)
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable,"-B","-c",script],check=True)


def test_stream_batch_features_and_derangement_exact_cached_semantics(tmp_path):
    ctx = context(tmp_path)
    samples = [payload() for _ in range(4)]
    output = {key:[torch.stack([sample[key][i] for sample in samples]) for i in range(4)] for key in ("h","aux_hidden","aux_prediction")}
    donors = {key:[torch.stack([samples[(j+1)%4][key][i] for j in range(4)]) for i in range(4)] for key in ("aux_prediction",)}
    for recipe in data.RECIPES:
        actual = streamed.batch_features(output,recipe,donors if recipe == "aux_shuffle" else None)
        expected = [refine.features(samples[j],recipe,samples[(j+1)%4] if recipe == "aux_shuffle" else None) for j in range(4)]
        assert all(torch.equal(value,torch.stack([sample[i] for sample in expected])) for i,value in enumerate(actual))
    for ds in ctx.config["refiner"]["seeds"]:
        permutation = streamed.shuffle_permutation(ctx,11,ds,4)
        assert permutation == refine.derangement(4,ctx.config["refiner"]["shuffle_seed"]+11*1009+ds*10007+3)
        assert all(i != j for i,j in enumerate(permutation))


@pytest.mark.parametrize("save_predictions",[False,True])
def test_streamed_complete_seed_equals_cached_metrics_and_spools_only_aux(tmp_path,monkeypatch,save_predictions):
    ctx = context(tmp_path)
    ctx.config["runtime"]["device"] = "cpu"
    ctx.config["loader"]["workers"] = 0
    ctx.config["refiner"]["seeds"] = [101,202]
    ctx.config["refiner"]["width"] = 4
    ctx.config["evaluation"].update(batch_size=2,save_predictions=save_predictions,prediction_shard_images=3)
    ids = [f"image{i}" for i in range(4)]
    samples = [payload() for _ in ids]
    for sample,image_id in zip(samples,ids):
        sample.update(id=image_id,split="y_test")
        sample["semseg"][0,0] = 255
    y_loader = DataLoader([{"image":torch.full((3,64,64),float(index)),"id":image_id,"semseg":sample["semseg"]} for index,(sample,image_id) in enumerate(zip(samples,ids))],batch_size=2)
    class Frozen(torch.nn.Module):
        calls = 0
        def forward(self,image):
            self.calls += 1
            chosen = [samples[int(index)] for index in image[:,0,0,0]]
            output = {key:[torch.stack([p[key][i] for p in chosen]) for i in range(4)] for key in ("h","aux_hidden","aux_prediction")}
            output["native_lowres"] = torch.stack([p["native_lowres"] for p in chosen])
            output["output_size"] = torch.stack([p["output_size"] for p in chosen])
            output["semseg"] = model.native_logits(output)
            return output
    upstream = Frozen().eval().requires_grad_(False)
    keys = [(r,ds) for r in data.RECIPES for ds in ctx.config["refiner"]["seeds"]]
    networks = {}
    for key in keys:
        network = model.SpatialResidual(width=4).eval()
        torch.nn.init.normal_(network.fusion[-1].weight,std=.04)
        networks[key] = network
    updir = ctx.output_dir/"upstream/multi_task/seed11"
    updir.mkdir(parents=True)
    (updir/"best.pt").write_bytes(b"synthetic checkpoint")
    cdir = refine.cache_dir(ctx,"multi_task",11)
    cdir.mkdir(parents=True)
    write_json(cdir/"manifest.json",{"records":{"b_train":[],"b_val":[]}})
    for recipe,ds in keys:
        directory = refine.readout_dir(ctx,"multi_task",11,recipe,ds)
        directory.mkdir(parents=True)
        (directory/"best.pt").write_bytes(b"synthetic readout checkpoint")
    monkeypatch.setattr(workflow,"unit_complete",lambda *args:True)
    monkeypatch.setattr(streamed,"configure",lambda runtime:"cpu")
    monkeypatch.setattr(streamed,"load_cache",lambda *args:None)
    monkeypatch.setattr(streamed,"load_upstream",lambda *args,**kwargs:(upstream,{}))
    monkeypatch.setattr(streamed,"load_readout",lambda context,v,s,r,ds,device:copy.deepcopy(networks[(r,ds)]))
    monkeypatch.setattr(streamed,"data_manifest",lambda context:{"splits":{"y_test":ids}})
    monkeypatch.setattr(streamed,"data_identity",lambda context:"synthetic_identity")
    monkeypatch.setattr(streamed,"make_loader",lambda *args,**kwargs:y_loader)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        paths = streamed.evaluate_seed(ctx,"multi_task",11)
        value = read_json(paths[0])
        assert value["y_passes"] == 2 and len(value["results"]) == 9
        assert upstream.calls == 4  # two batches x two passes, independent of readout count
        assert bool(value["prediction_artifacts"]) == save_predictions
        directory = tmp_path/"cached_y"
        directory.mkdir()
        rows = []
        for sample in samples:
            path = directory/f"{sample['id']}.pt"
            torch.save(sample,path)
            rows.append({"path":path.name,"id":sample["id"],"sha256":sha256_file(path)})
        for row in value["results"]:
            recipe,ds = row["recipe"],row["downstream_seed"]
            dataset = refine.DiskCached(directory,rows,"embedding_matched" if ds is None else recipe,0 if ds is None else ctx.config["refiner"]["shuffle_seed"]+11*1009+ds*10007+3)
            current = evaluate.Native() if ds is None else networks[(recipe,ds)]
            expected = evaluate.metrics(current,DataLoader(dataset,batch_size=2),"cpu")
            for metric in ("miou","pixel_accuracy","confusion_matrix","valid_pixels","class_iou"):
                assert row[metric] == expected[metric]
        spool = read_json(value["spool_manifest"])
        assert [r["id"] for r in spool["records"]] == ids
        for record in spool["records"]:
            sample = torch.load(record["path"],weights_only=True)
            assert set(sample) == {"id","aux_prediction"}
        cleanup = workflow.cleanup_records(ctx,"multi_task",11)
        assert len(cleanup) == 4
        assert all(Path(r["path"]).is_file() for r in cleanup)
    finally:
        torch.set_num_threads(old_threads)


def test_b_cache_uint8_no_y_or_aux_truth(tmp_path,monkeypatch):
    ctx = context(tmp_path)
    ctx.config["runtime"]["device"] = "cpu"
    ctx.config["cache"]["check_free_space"] = False
    ctx.config["upstream"]["seeds"] = [11]
    sample = payload()
    class Frozen(torch.nn.Module):
        def forward(self,image):
            output = {key:[v[None] for v in sample[key]] for key in ("h","aux_hidden","aux_prediction")}
            output["native_lowres"] = sample["native_lowres"][None]
            output["semseg"] = model.native_logits({"native_lowres":output["native_lowres"],"output_size":sample["output_size"][None]})
            return output
    upstream = Frozen().eval().requires_grad_(False)
    manifest = {"splits":{"b_train":["train"],"b_val":["val"],"y_test":["never_cached"]}}
    splits_seen = []
    def loader(context,split,*args,**kwargs):
        splits_seen.append(split)
        return [{"image":torch.zeros(1,3,64,64),"semseg":sample["semseg"][None],"id":manifest["splits"][split]}]
    monkeypatch.setattr(refine,"configure",lambda *args:"cpu")
    monkeypatch.setattr(refine,"load_upstream",lambda *args,**kwargs:(upstream,{}))
    monkeypatch.setattr(refine,"data_manifest",lambda context:manifest)
    monkeypatch.setattr(refine,"data_identity",lambda context:"synthetic")
    monkeypatch.setattr(refine,"make_loader",loader)
    updir = ctx.output_dir/"upstream/multi_task/seed11"
    updir.mkdir(parents=True)
    (updir/"best.pt").write_bytes(b"synthetic")
    paths = refine.cache(ctx)
    assert splits_seen == ["b_train","b_val"]
    cached = read_json(paths[-1])
    assert set(cached["records"]) == {"b_train","b_val"}
    assert cached["auxiliary_truth_cached"] is False
    for path in paths[:-1]:
        value = torch.load(path,weights_only=True)
        assert value["semseg"].dtype == torch.uint8
        assert set(value) == {"id","split","h","aux_hidden","aux_prediction","native_lowres","output_size","semseg"}

def test_resource_Y_loader_blocked_until_all_B_units_valid(tmp_path,monkeypatch):
    ctx = context(tmp_path)
    monkeypatch.setattr(workflow,"unit_complete",lambda *args:False)
    monkeypatch.setattr(streamed,"make_loader",lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError("Y accessed prematurely")))
    with pytest.raises(ValueError,match="complete B readout matrix"):
        streamed.evaluate_seed(ctx,"multi_task",11)


def test_retaining_cache_reports_lifetime_storage_as_peak(tmp_path):
    ctx = context(tmp_path)
    before = workflow.plan(ctx)
    ctx.config["workflow"]["release_cache"] = False
    retained = workflow.plan(ctx)
    assert retained["peak_cache_bytes"] == retained["total_cache_bytes"]
    assert retained["peak_spool_bytes"] == retained["total_spool_bytes"]
    assert retained["peak_output_bytes"] == 5*before["peak_output_bytes"]

def test_seed_receipt_includes_stable_global_lineage_only(tmp_path):
    ctx = context(tmp_path)
    stable = ["run_manifest.json","resolved_config.json","prepared/manifest.json",
              "prepared/acquisition.json","prepared/source.json"]
    for name in stable:
        write_json(ctx.output_dir/name,{"fixture":name})
    for name in ("evaluation.json","summary.json","lifecycle/complete.json"):
        write_json(ctx.output_dir/name,{"changes_between_seeds":True})
    write_json(workflow.seed_evaluation_path(ctx,"multi_task",11),{})
    paths = set(workflow.seed_artifacts(ctx,"multi_task",11))
    assert all(ctx.output_dir/name in paths for name in stable)
    assert all(ctx.output_dir/name not in paths for name in ("evaluation.json","summary.json","lifecycle/complete.json"))
    # Historical/synthetic fixtures without optional acquisition/runtime outputs
    # can still enumerate the seed-specific durable files.
    (ctx.output_dir/"prepared/source.json").unlink()
    assert ctx.output_dir/"prepared/source.json" not in workflow.seed_artifacts(ctx,"multi_task",11)


def test_download_snapshots_pinned_source_without_network(tmp_path,monkeypatch):
    ctx = context(tmp_path)
    split_dir = data.raw_dir(ctx)/"ImageSets/Context"
    split_dir.mkdir(parents=True)
    (split_dir/"train.txt").write_text("synthetic_fixture")
    monkeypatch.setattr(data,"bounded_download",lambda *a,**k:(_ for _ in ()).throw(AssertionError("network forbidden")))
    artifacts = data.download(ctx)
    snapshot = read_json(ctx.output_dir/"prepared/source.json")
    assert ctx.output_dir/"prepared/source.json" in artifacts
    assert snapshot["source"] == data.verify_source()
    assert snapshot["source_lock"]["commit"] == snapshot["source"]["commit"]
