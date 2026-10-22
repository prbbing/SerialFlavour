"""Frozen per-image cache and B-only matched-capacity spatial residuals."""
import math
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from pipeline.fit import fit, save_checkpoint
from pipeline.io import read_json, write_json, sha256_file
from pipeline.runtime import configure, seed_all, to_device
from experiments.cv_pascal_context.data import make_loader, data_identity, data_manifest, applicable_recipes
from experiments.cv_pascal_context.model import SpatialResidual, CHANNELS, state_digest, native_logits
from experiments.cv_pascal_context.training import load_upstream, upstream_dir, semantic_loss
from experiments.cv_pascal_context.source import verify_source


def cache_dir(context, variant, seed):
    return context.output_dir / "cache" / variant / f"seed{seed}"


def readout_dir(context, variant, upstream_seed, recipe, downstream_seed):
    return context.output_dir / "refine" / variant / f"seed{upstream_seed}" / recipe / f"seed{downstream_seed}"


def clone_sample(value, index):
    if isinstance(value, torch.Tensor):
        return value[index].detach().cpu().clone()
    if isinstance(value, list):
        return [clone_sample(v, index) for v in value]
    if isinstance(value, dict):
        return {k: clone_sample(v, index) for k, v in value.items()}
    raise TypeError(type(value))


def all_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return all(all_finite(v) for v in (value.values() if isinstance(value, dict) else value))


def cache(context):
    device = configure(context.config["runtime"])
    artifacts = []
    manifest = data_manifest(context)
    for variant in context.config["upstream"]["variants"]:
        if context.filters.get("variant") not in (None, variant):
            continue
        for seed in context.config["upstream"]["seeds"]:
            if context.filters.get("seed") not in (None, seed):
                continue
            model, _ = load_upstream(context, variant, seed, device, frozen=True)
            before = state_digest(model.state_dict())
            directory = cache_dir(context, variant, seed)
            splits = context.config["cache"].get("splits", ["b_train", "b_val", "y_test"])
            required = estimated_cache_bytes(context, variant, sum(len(manifest["splits"][split]) for split in splits))
            if context.config["cache"].get("check_free_space", True) and shutil.disk_usage(context.output_dir).free < required + context.config["cache"]["free_space_reserve_bytes"]:
                raise OSError(f"frozen cache needs about {required/1e9:.1f} GB for this upstream, plus reserve; insufficient free output space")
            records = {}
            with torch.inference_mode():
                for split in splits:
                    records[split] = []
                    loader = make_loader(context, split, seed, context.config["cache"]["batch_size"], auxiliary=False)
                    for batch in loader:
                        output = model(to_device(batch["image"], device))
                        size = torch.tensor(batch["image"].shape[-2:], device=device, dtype=torch.long).expand(len(batch["id"]), 2)
                        values = {"h": output["h"], "native_lowres": output["native_lowres"],
                                  "output_size": size, "semseg": batch["semseg"].to(torch.uint8)}
                        reconstructed = native_logits(values)
                        if not torch.equal(reconstructed, output["semseg"]):
                            raise AssertionError("pre-upsample native cache does not exactly reproduce author output")
                        if variant != "single_task":
                            values.update(aux_prediction=output["aux_prediction"], aux_hidden=output["aux_hidden"])
                        if not all_finite(values):
                            raise FloatingPointError("non-finite frozen cache tensor")
                        for index, image_id in enumerate(batch["id"]):
                            path = directory / split / f"{image_id}.pt"
                            # A sample slice is cloned to avoid serializing full
                            # batch storage; no entire-split tensors live in RAM.
                            payload = clone_sample(values, index)
                            payload["id"] = image_id
                            payload["split"] = split
                            save_checkpoint(path, payload)
                            artifacts.append(path)
                            records[split].append({"id": image_id, "path": str(path.relative_to(directory)), "sha256": sha256_file(path), "bytes": path.stat().st_size})
                    if [r["id"] for r in records[split]] != manifest["splits"][split]:
                        raise ValueError("cache split coverage/order mismatch")
            after = state_digest(model.state_dict())
            if before != after or model.training or any(p.requires_grad for p in model.parameters()):
                raise AssertionError("frozen upstream parameters or BN state changed")
            artifacts.append(write_json(directory / "manifest.json", {"identity": context.identity, "data_identity": data_identity(context), "source": verify_source(), "checkpoint_sha256": sha256_file(upstream_dir(context, variant, seed) / "best.pt"), "variant": variant, "upstream_seed": seed, "records": records, "storage": "one-image-per-file; FP32 features/native and lossless uint8 semantic target; RGB output size; no duplicated final/deep outputs", "state_sha256_before": before, "state_sha256_after": after, "upstream_frozen": True, "upstream_eval": True, "auxiliary_truth_cached": False, "truth_masks_on_predictions": False, "H": "all four native backbone and semantic-distilled scale maps; no spatial pooling", "cache_bytes": sum(r["bytes"] for rows in records.values() for r in rows)}))
            del model
    return artifacts


def load_cache(context, variant, seed, split):
    if split not in ("b_train", "b_val", "y_test"):
        raise ValueError("unknown frozen-cache partition")
    directory = cache_dir(context, variant, seed)
    manifest = read_json(directory / "manifest.json")
    expected = {"identity": context.identity, "data_identity": data_identity(context), "source": verify_source(), "variant": variant, "upstream_seed": seed, "checkpoint_sha256": sha256_file(upstream_dir(context, variant, seed) / "best.pt")}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("cache provenance mismatch")
    if manifest.get("auxiliary_truth_cached") is not False or manifest.get("truth_masks_on_predictions") is not False or manifest["state_sha256_before"] != manifest["state_sha256_after"]:
        raise ValueError("cache freeze/truth contract violated")
    rows = manifest["records"][split]
    if [r["id"] for r in rows] != data_manifest(context)["splits"][split]:
        raise ValueError("cache coverage differs from prepared split")
    return directory, rows


def repeat_channels(tensor, channels):
    if tensor.shape[0] > channels:
        raise ValueError("recipe exceeds matched channel budget")
    return tensor.repeat(math.ceil(channels/tensor.shape[0]), 1, 1)[:channels]


def features(payload, recipe, donor=None):
    h = payload["h"]
    result = []
    for scale, channels in enumerate(CHANNELS):
        if recipe == "embedding_matched":
            extra = h[scale]
        elif recipe == "aux_hidden":
            extra = payload["aux_hidden"][scale]
        elif recipe in ("aux_prediction", "aux_shuffle"):
            extra = (donor if donor is not None else payload)["aux_prediction"][scale]
        else:
            raise ValueError(f"unknown recipe: {recipe}")
        # Repetition fills every active input slot with real spatial features;
        # all recipes have identical trainable parameter count and receptive field.
        result.append(torch.cat((h[scale], repeat_channels(extra, 4*channels)), 0))
    return result


def derangement(count, seed):
    if count < 2:
        raise ValueError("shuffle requires at least two examples in each partition")
    order = list(range(count))
    random.Random(seed).shuffle(order)
    permutation = list(range(count))
    for position, value in enumerate(order):
        permutation[value] = order[(position+1) % count]
    return permutation


class DiskCached(Dataset):
    def __init__(self, directory, rows, recipe, shuffle_seed):
        self.directory, self.rows, self.recipe = Path(directory), rows, recipe
        self.verified = set()
        self.permutation = derangement(len(rows), shuffle_seed) if recipe == "aux_shuffle" else None

    def __len__(self):
        return len(self.rows)

    def payload(self, index):
        row = self.rows[index]
        path = (self.directory / row["path"]).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError("cache path traversal")
        if index not in self.verified:
            if sha256_file(path) != row["sha256"]:
                raise ValueError("cache sample checksum mismatch")
            self.verified.add(index)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["id"] != row["id"]:
            raise ValueError("cache sample identity mismatch")
        return payload

    def __getitem__(self, index):
        payload = self.payload(index)
        donor = self.payload(self.permutation[index]) if self.permutation is not None else None
        return {"features": features(payload, self.recipe, donor), "native_lowres": payload["native_lowres"],
                "output_size": payload["output_size"], "semseg": payload["semseg"].long(), "id": payload["id"]}


def cache_loader(context, variant, us, split, recipe, ds, batch_size, shuffle=False):
    directory, rows = load_cache(context, variant, us, split)
    # A deterministic derangement stays inside each split and seed; its donor
    # is independent of target/GT, preserves intact spatial predictions, and
    # avoids accidentally pairing a sample with itself.
    split_seed = context.config["refiner"]["shuffle_seed"] + us*1009 + ds*10007 + {"b_train": 1, "b_val": 2, "y_test": 3}[split]
    dataset = DiskCached(directory, rows, recipe, split_seed)
    workers = context.config.get("loader", {}).get("workers", 0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      generator=torch.Generator().manual_seed(ds), pin_memory=context.config["runtime"]["device"].startswith("cuda"), persistent_workers=workers > 0)


def train(context):
    from experiments.cv_pascal_context.evaluate import score
    device = configure(context.config["runtime"])
    settings = context.config["refiner"]
    if not settings.get("include_initial_checkpoint"):
        raise ValueError("native epoch-0 checkpoint must be eligible for B_val selection")
    artifacts = []
    for variant in context.config["upstream"]["variants"]:
        if context.filters.get("variant") not in (None, variant):
            continue
        for us in context.config["upstream"]["seeds"]:
            if context.filters.get("upstream_seed") not in (None, us):
                continue
            for recipe in applicable_recipes(context, variant):
                if context.filters.get("recipe") not in (None, recipe):
                    continue
                for ds in settings["seeds"]:
                    if context.filters.get("downstream_seed") not in (None, ds):
                        continue
                    seed_all(ds)
                    model = SpatialResidual(settings["width"])
                    training = cache_loader(context, variant, us, "b_train", recipe, ds, settings["batch_size"], True)
                    validation = cache_loader(context, variant, us, "b_val", recipe, ds, settings["batch_size"])
                    first = next(iter(validation))
                    model.eval()
                    with torch.inference_mode():
                        error = float((model(first)-native_logits(first)).abs().max())
                    if error != 0:
                        raise AssertionError(f"native residual initialization mismatch: {error}")
                    checkpoint_hash = sha256_file(upstream_dir(context, variant, us) / "best.pt")
                    cache_hash = sha256_file(cache_dir(context, variant, us) / "manifest.json")
                    metadata = {"identity": context.identity, "data_identity": data_identity(context), "source": verify_source(), "variant": variant, "upstream_seed": us, "recipe": recipe, "downstream_seed": ds, "checkpoint_sha256": checkpoint_hash, "cache_manifest_sha256": cache_hash, "training_split": "b_train", "selection_split": "b_val", "supervision": "21-class semantic only", "upstream_frozen": True, "native_initialization_max_error": error, "initial_residual_state_sha256": state_digest(model.state_dict()), "auxiliary_truth_input": False, "readout_parameters": sum(p.numel() for p in model.parameters()), "spatial_scope": "four complete native scale maps, convolution 3x3 per scale and 3x3 fusion", "shuffle_partition_seed": settings["shuffle_seed"]+us*1009+ds*10007}
                    def loss(current, batch):
                        return semantic_loss(current(batch), batch["semseg"]), len(batch["semseg"])
                    artifacts.extend(fit(model, training, validation, loss, score, readout_dir(context, variant, us, recipe, ds), settings, metadata, device))
                    if checkpoint_hash != sha256_file(upstream_dir(context, variant, us) / "best.pt") or cache_hash != sha256_file(cache_dir(context, variant, us) / "manifest.json"):
                        raise AssertionError("frozen prerequisite changed during B fitting")
                    del model, training, validation
    return artifacts


def load_readout(context, variant, us, recipe, ds, device):
    checkpoint = torch.load(readout_dir(context, variant, us, recipe, ds) / "best.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    expected = {"identity": context.identity, "data_identity": data_identity(context), "source": verify_source(), "variant": variant, "upstream_seed": us, "recipe": recipe, "downstream_seed": ds, "checkpoint_sha256": sha256_file(upstream_dir(context, variant, us) / "best.pt"), "cache_manifest_sha256": sha256_file(cache_dir(context, variant, us) / "manifest.json")}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("B readout checkpoint provenance mismatch")
    model = SpatialResidual(context.config["refiner"]["width"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def estimated_cache_bytes(context, variant, images):
    height, width = context.config["data"]["image_size"]
    if height % 32 or width % 32:
        raise ValueError("author four-scale MTI propagation requires image dimensions divisible by 32")
    area = height*width
    scales = [(height//(4*2**i))*(width//(4*2**i)) for i in range(4)]
    channel_area = sum(c*a for c, a in zip(CHANNELS, scales))
    # FP32 complete H and exact pre-upsample native logits; uint8 target/int64 size.
    # No full-resolution native or unused final auxiliary/deep output copies.
    sample = 4*(2*channel_area + 21*scales[0]) + area + 16
    if variant != "single_task":
        sample += 4*(4*channel_area + 24*sum(scales))
    return math.ceil(sample*images*1.05)  # serialization/metadata allowance


def storage_plan(context):
    from experiments.cv_pascal_context.workflow import plan
    return plan(context)
