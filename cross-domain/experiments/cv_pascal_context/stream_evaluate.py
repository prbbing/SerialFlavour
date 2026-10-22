"""Two-pass Y streaming; never persist Y H, hidden, or native logits."""
import math
from pathlib import Path
import numpy as np
import torch
from pipeline.fit import save_checkpoint
from pipeline.io import artifact_record, read_json, sha256_file, write_json
from pipeline.runtime import configure, to_device
from experiments.cv_pascal_context.data import applicable_recipes, data_identity, data_manifest, make_loader
from experiments.cv_pascal_context.model import CHANNELS, native_logits, state_digest
from experiments.cv_pascal_context.refine import cache_dir, load_cache, load_readout, readout_dir, derangement
from experiments.cv_pascal_context.source import verify_source
from experiments.cv_pascal_context.training import load_upstream, upstream_dir


class Accumulator:
    """Same dataset-level 21-class confusion semantics as cached evaluation."""
    def __init__(self, expected_ids, prediction_dir=None, shard_images=32):
        self.expected_ids = list(expected_ids)
        self.confusion = torch.zeros(21,21,dtype=torch.int64)
        self.seen, self.pending, self.pending_ids, self.records, self.artifacts = [], [], [], [], []
        self.prediction_dir = Path(prediction_dir) if prediction_dir else None
        self.shard_images = shard_images

    def update(self, logits, target, ids):
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite streamed evaluation logits")
        prediction = logits.argmax(1)
        target = target.long()
        valid = (target >= 0) & (target < 21)
        self.confusion += torch.bincount((21*target[valid]+prediction[valid]).cpu(),minlength=441).reshape(21,21)
        for index, image_id in enumerate(ids):
            self.seen.append(image_id)
            if self.prediction_dir:
                self.pending.append(prediction[index].cpu().numpy().astype(np.uint8))
                self.pending_ids.append(image_id)
                if len(self.pending) >= self.shard_images:
                    self.publish()

    def publish(self):
        if not self.pending:
            return
        self.prediction_dir.mkdir(parents=True,exist_ok=True)
        path = self.prediction_dir/f"shard{len(self.records):05d}.npz"
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,predictions=np.stack(self.pending),ids=np.array(self.pending_ids))
        temporary.replace(path)
        self.records.append({"path":str(path),"sha256":sha256_file(path),"ids":list(self.pending_ids),"images":len(self.pending_ids)})
        self.artifacts.append(path)
        self.pending.clear()
        self.pending_ids.clear()

    def finish(self):
        if self.seen != self.expected_ids or len(self.seen) != len(set(self.seen)):
            raise ValueError("streamed evaluation coverage/order mismatch")
        if not self.confusion.sum():
            raise ValueError("no valid evaluation pixels")
        if self.prediction_dir:
            self.publish()
            self.artifacts.append(write_json(self.prediction_dir/"manifest.json",{"images":len(self.seen),"storage":"compressed uint8 argmax; fixed input resolution","records":self.records}))
        confusion = self.confusion
        intersection = confusion.diag().double()
        union = confusion.sum(0)+confusion.sum(1)-confusion.diag()
        return {"miou":(intersection/union.clamp_min(1)).mean().item(),"pixel_accuracy":(intersection.sum()/confusion.sum()).item(),
                "class_iou":(intersection/union.clamp_min(1)).tolist(),"classes_with_union":int((union>0).sum()),
                "confusion_matrix":confusion.tolist(),"valid_pixels":int(confusion.sum()),"images":len(self.seen),
                "miou_definition":"mean over all 21 VOC classes including background; void=255 excluded"}


def batch_features(output, recipe, donor=None):
    result = []
    for index, channels in enumerate(CHANNELS):
        h = output["h"][index]
        extra = h if recipe == "embedding_matched" else output["aux_hidden"][index] if recipe == "aux_hidden" else (donor if donor is not None else output)["aux_prediction"][index]
        if extra.shape[1] > 4*channels:
            raise ValueError("recipe exceeds matched channel budget")
        repeated = extra.repeat(1,math.ceil(4*channels/extra.shape[1]),1,1)[:,:4*channels]
        result.append(torch.cat((h,repeated),1))
    return result


def shuffle_permutation(context, upstream_seed, downstream_seed, count):
    return derangement(count,context.config["refiner"]["shuffle_seed"]+upstream_seed*1009+downstream_seed*10007+3)


def evaluate_seed(context, variant, seed):
    from experiments.cv_pascal_context.workflow import seed_units, unit_complete, seed_evaluation_path
    if variant not in context.config["upstream"]["variants"] or seed not in context.config["upstream"]["seeds"]:
        raise ValueError("evaluation seed outside fixed matrix")
    # Validate every required B unit and all checkpoint provenance before Y loader.
    if any(not unit_complete(context,unit) for unit in seed_units(context,variant,seed)):
        raise ValueError("seed Y evaluation requires verified upstream/cache and complete B readout matrix")
    load_cache(context,variant,seed,"b_train")
    load_cache(context,variant,seed,"b_val")
    device = configure(context.config["runtime"])
    upstream, _ = load_upstream(context,variant,seed,device,frozen=True)
    before = state_digest(upstream.state_dict())
    keys = [(recipe,ds) for recipe in applicable_recipes(context,variant) for ds in context.config["refiner"]["seeds"]]
    # Keep readouts on CPU, transfer one at a time, retaining only the current
    # upstream batch on GPU. Models/provenance are all checked before Y access.
    models = {key:load_readout(context,variant,seed,*key,"cpu") for key in keys}
    y_ids = data_manifest(context)["splits"]["y_test"]
    settings = context.config["evaluation"]
    all_keys = [("native",None),*keys]
    accumulators = {}
    for recipe,ds in all_keys:
        path = context.output_dir/"predictions"/variant/f"seed{seed}"/recipe/("native" if ds is None else f"seed{ds}")
        accumulators[(recipe,ds)] = Accumulator(y_ids,path if settings["save_predictions"] else None,settings["prediction_shard_images"])
    shuffle_keys = [key for key in keys if key[0] == "aux_shuffle"]
    spool_dir = context.output_dir/"scratch"/variant/f"seed{seed}"/"y_aux_prediction"
    spool_rows = []
    spool_manifest = None
    passes = (False,True) if shuffle_keys else (False,)
    permutations = {ds:shuffle_permutation(context,seed,ds,len(y_ids)) for _,ds in shuffle_keys}
    with torch.inference_mode():
        for shuffle_pass in passes:
            seen = []
            loader = make_loader(context,"y_test",seed,settings["batch_size"],auxiliary=False)
            offset = 0
            for batch in loader:
                batch = to_device(batch,device)
                ids = list(batch["id"])
                seen.extend(ids)
                output = upstream(batch["image"])
                output["output_size"] = torch.tensor(batch["image"].shape[-2:],device=device,dtype=torch.long).expand(len(ids),2)
                if not shuffle_pass:
                    reconstructed = native_logits(output)
                    if not torch.equal(reconstructed,output["semseg"]):
                        raise AssertionError("streamed native reconstruction differs from author forward")
                    accumulators[("native",None)].update(reconstructed,batch["semseg"],ids)
                    if shuffle_keys:
                        for index,image_id in enumerate(ids):
                            path = spool_dir/f"{image_id}.pt"
                            # Only the small four-scale auxiliary prediction is
                            # spooled; H, aux hidden and native logits stay transient.
                            save_checkpoint(path,{"id":image_id,"aux_prediction":[v[index].detach().cpu().clone() for v in output["aux_prediction"]]})
                            spool_rows.append({"id":image_id,**artifact_record(path)})
                current_keys = shuffle_keys if shuffle_pass else [key for key in keys if key not in shuffle_keys]
                for key in current_keys:
                    recipe,ds = key
                    donor = None
                    if shuffle_pass:
                        donor_samples = []
                        for position in range(offset,offset+len(ids)):
                            row = spool_rows[permutations[ds][position]]
                            path = Path(row["path"])
                            if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
                                raise ValueError("Y donor spool checksum mismatch")
                            sample = torch.load(path,map_location="cpu",weights_only=True)
                            if sample["id"] != row["id"]:
                                raise ValueError("Y donor identity mismatch")
                            donor_samples.append(sample)
                        donor = {"aux_prediction":[torch.stack([s["aux_prediction"][scale] for s in donor_samples]).to(device) for scale in range(4)]}
                    inputs = {"features":batch_features(output,recipe,donor),"native_lowres":output["native_lowres"],"output_size":output["output_size"]}
                    model = models[key].to(device).eval()
                    accumulators[key].update(model(inputs),batch["semseg"],ids)
                    model.to("cpu")
                    del inputs, donor
                offset += len(ids)
                del output, batch
            if seen != y_ids:
                raise ValueError("Y source order/coverage mismatch")
            if shuffle_keys and not shuffle_pass:
                if [row["id"] for row in spool_rows] != y_ids:
                    raise ValueError("Y donor spool coverage mismatch")
                spool_manifest = write_json(spool_dir.parent/"spool_manifest.json",{"identity":context.identity,"variant":variant,"upstream_seed":seed,
                          "split":"y_test","content":"aux_prediction only; no H/hidden/native/auxiliary truth","records":spool_rows})
    if state_digest(upstream.state_dict()) != before or upstream.training or any(p.requires_grad for p in upstream.parameters()):
        raise AssertionError("frozen upstream state changed during streaming")
    checkpoint_hash = sha256_file(upstream_dir(context,variant,seed)/"best.pt")
    cache_hash = sha256_file(cache_dir(context,variant,seed)/"manifest.json")
    rows, artifacts = [], []
    for recipe,ds in all_keys:
        accumulator = accumulators[(recipe,ds)]
        row = {"variant":variant,"upstream_seed":seed,"recipe":recipe,"downstream_seed":ds,
               "upstream_checkpoint_sha256":checkpoint_hash,"cache_manifest_sha256":cache_hash,
               "readout_checkpoint_sha256":None if ds is None else sha256_file(readout_dir(context,variant,seed,recipe,ds)/"best.pt"),**accumulator.finish()}
        rows.append(row)
        artifacts.extend(accumulator.artifacts)
    path = write_json(seed_evaluation_path(context,variant,seed),{"identity":context.identity,"data_identity":data_identity(context),"source":verify_source(),
              "variant":variant,"upstream_seed":seed,"matrix_complete":True,"y_passes":len(passes),"results":rows,
              "prediction_artifacts":[str(p) for p in artifacts],"spool_manifest":str(spool_manifest) if spool_manifest else None,
              "policy":"all fixed recipes/downstream seeds validated before Y; no Y-dependent selection; spool cleanup only after shared driver receipt"})
    return [path,*artifacts,*([spool_manifest] if spool_manifest else [])]
