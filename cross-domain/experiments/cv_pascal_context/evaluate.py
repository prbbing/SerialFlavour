"""Full locked-Y matrix, 21-class dataset mIoU and bounded prediction shards."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from pipeline.io import read_json, write_json, sha256_file
from pipeline.runtime import configure, to_device
from experiments.cv_pascal_context.data import applicable_recipes, data_identity
from experiments.cv_pascal_context.source import verify_source
from experiments.cv_pascal_context.model import native_logits


class Native(nn.Module):
    def forward(self, batch):
        return native_logits(batch)


def metrics(model, loader, device, prediction_dir=None, shard_images=32):
    confusion = torch.zeros(21, 21, dtype=torch.int64)
    model.eval()
    pending, ids, files, records, seen, images = [], [], [], [], set(), 0
    def publish():
        if not pending:
            return
        directory = Path(prediction_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"shard{len(records):05d}.npz"
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, predictions=np.stack(pending), ids=np.array(ids))
        temporary.replace(path)
        records.append({"path": str(path), "sha256": sha256_file(path), "ids": list(ids), "images": len(ids)})
        files.append(path)
        pending.clear()
        ids.clear()
    with torch.inference_mode():
        for batch in loader:
            batch = to_device(batch, device)
            output = model(batch["image"]) if "image" in batch else model(batch)
            logits = output["semseg"] if isinstance(output, dict) else output
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite evaluation logits")
            target, prediction = batch["semseg"], logits.argmax(1)
            valid = (target >= 0) & (target < 21)
            confusion += torch.bincount((21*target[valid]+prediction[valid]).cpu(), minlength=441).reshape(21, 21)
            for index, image_id in enumerate(batch["id"]):
                if image_id in seen:
                    raise ValueError("duplicate evaluation sample")
                seen.add(image_id)
                images += 1
                if prediction_dir is not None:
                    pending.append(prediction[index].cpu().numpy().astype(np.uint8))
                    ids.append(image_id)
                    if len(pending) >= shard_images:
                        publish()
    if not confusion.sum():
        raise ValueError("no valid evaluation pixels")
    if images != len(loader.dataset):
        raise ValueError("evaluation sample coverage incomplete")
    if prediction_dir is not None:
        publish()
        files.append(write_json(Path(prediction_dir) / "manifest.json", {"images": images, "storage": "compressed uint8 21-class argmax; fixed input resolution", "records": records}))
    intersection = confusion.diag().double()
    union = confusion.sum(0)+confusion.sum(1)-confusion.diag()
    iou = intersection/union.clamp_min(1)
    return {"miou": iou.mean().item(), "pixel_accuracy": (intersection.sum()/confusion.sum()).item(), "class_iou": iou.tolist(), "classes_with_union": int((union > 0).sum()), "confusion_matrix": confusion.tolist(), "valid_pixels": int(confusion.sum()), "images": images, "miou_definition": "mean over all 21 VOC classes including background; void=255 excluded", "_artifacts": files}


def score(model, loader, device):
    return metrics(model, loader, device)["miou"]


def expected_keys(context):
    keys = set()
    for variant in context.config["upstream"]["variants"]:
        for us in context.config["upstream"]["seeds"]:
            keys.add((variant, us, "native", None))
            keys.update((variant, us, recipe, ds) for recipe in applicable_recipes(context, variant) for ds in context.config["refiner"]["seeds"])
    return keys


def assert_complete(context):
    """Check every provenance before any Y metric is computed."""
    from experiments.cv_pascal_context.training import upstream_dir
    from experiments.cv_pascal_context.refine import load_cache, load_readout
    for variant in context.config["upstream"]["variants"]:
        for us in context.config["upstream"]["seeds"]:
            checkpoint = torch.load(upstream_dir(context, variant, us) / "best.pt", map_location="cpu", weights_only=True)
            metadata = checkpoint["metadata"]
            if any(metadata.get(k) != v for k, v in {"identity": context.identity, "data_identity": data_identity(context), "source": verify_source(), "variant": variant, "seed": us}.items()):
                raise ValueError("incomplete or stale upstream evaluation matrix")
            for split in ("b_train", "b_val", "y_test"):
                load_cache(context, variant, us, split)
            for recipe in applicable_recipes(context, variant):
                for ds in context.config["refiner"]["seeds"]:
                    # load_readout validates the exact checkpoint and cache hashes.
                    model = load_readout(context, variant, us, recipe, ds, "cpu")
                    del model
            del checkpoint


def evaluate(context):
    if context.config.get("workflow", {}).get("mode") == "seed_lifecycle":
        from experiments.cv_pascal_context.workflow import aggregate
        return aggregate(context)
    from experiments.cv_pascal_context.refine import cache_loader, load_readout, readout_dir, cache_dir
    from experiments.cv_pascal_context.training import upstream_dir
    verify_source()
    assert_complete(context)
    device = configure(context.config["runtime"])
    artifacts, rows = [], []
    settings = context.config["evaluation"]
    for variant, us, recipe, ds in sorted(expected_keys(context), key=lambda value: (value[0], value[1], value[2], -1 if value[3] is None else value[3])):
        native = recipe == "native"
        model = Native().to(device) if native else load_readout(context, variant, us, recipe, ds, device)
        loader = cache_loader(context, variant, us, "y_test", "embedding_matched" if native else recipe, 0 if native else ds, settings["batch_size"])
        prediction_dir = context.output_dir / "predictions" / variant / f"seed{us}" / recipe / ("native" if native else f"seed{ds}")
        value = metrics(model, loader, device, prediction_dir if settings["save_predictions"] else None, settings["prediction_shard_images"])
        artifacts.extend(value.pop("_artifacts"))
        row = {"variant": variant, "upstream_seed": us, "recipe": recipe, "downstream_seed": ds, "upstream_checkpoint_sha256": sha256_file(upstream_dir(context, variant, us) / "best.pt"), "cache_manifest_sha256": sha256_file(cache_dir(context, variant, us) / "manifest.json"), "readout_checkpoint_sha256": None if native else sha256_file(readout_dir(context, variant, us, recipe, ds) / "best.pt"), **value}
        rows.append(row)
        del model, loader
    actual = [(r["variant"], r["upstream_seed"], r["recipe"], r["downstream_seed"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected_keys(context):
        raise ValueError("evaluation matrix missing or duplicated cells")
    artifacts.append(write_json(context.output_dir / "evaluation.json", {"identity": context.identity, "data_identity": data_identity(context), "source": verify_source(), "split": "y_test", "selection": "A_val/B_val only; Y metrics computed only after all required checkpoints validated", "matrix_complete": True, "results": rows}))
    return artifacts
