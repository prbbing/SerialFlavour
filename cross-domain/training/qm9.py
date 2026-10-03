"""Paired single-task/multitask training on A with property and local heads."""

import numpy as np
import torch
from torch.nn import functional as F

from data.qm9 import (BOND_CLASSES, UNITS, gather_values, load_arrays, local_tasks, make_loader,
                      processed_dir, split_indices, tasks)
from model.qm9 import TinySchNet
from pipeline.fit import fit
from pipeline.io import sha256_file
from pipeline.readout import fit_standardization
from pipeline.runtime import configure, seed_all, to_device


def upstream_dir(context, variant, seed):
    return context.output_dir / "upstream" / variant / f"seed{seed}"


def variant_tasks(context, variant):
    property_names = tasks(context)
    if variant == "single_task":
        return property_names[:1], []
    if variant != "multi_task":
        raise ValueError(f"unknown upstream variant: {variant}")
    return property_names, local_tasks(context)


def load_upstream(context, variant, seed, device, frozen=False):
    checkpoint = torch.load(upstream_dir(context, variant, seed) / "best.pt", map_location=device, weights_only=True)
    metadata = checkpoint["metadata"]
    property_tasks = metadata.get("property_tasks", metadata.get("task_names"))
    model = TinySchNet(context.config["model"], property_tasks, metadata.get("local_tasks", [])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if frozen:
        model.requires_grad_(False)
    return model, metadata


def train(context):
    settings = context.config["upstream"]
    filters = getattr(context, "filters", {}) or {}
    seeds = [value for value in settings["seeds"] if filters.get("seed") in (None, value)]
    variants = [value for value in settings["variants"] if filters.get("variant") in (None, value)]
    device = configure(context.config["runtime"])
    arrays = load_arrays(context)
    a_train = split_indices(context, "a_train")
    property_names = tasks(context)
    target = arrays["target"][a_train]
    charge_values = gather_values(arrays, a_train, "atom_offsets", "charge").reshape(-1, 1)
    pair_values = gather_values(arrays, a_train, "pair_offsets", "pair_class")
    class_counts = np.bincount(pair_values, minlength=len(BOND_CLASSES)).astype(np.float64)
    class_weights = class_counts.sum() / (len(BOND_CLASSES) * np.maximum(class_counts, 1.0))
    artifacts = []
    for seed in seeds:
        for variant in variants:
            property_tasks, local = variant_tasks(context, variant)
            seed_all(seed)
            mean, std = fit_standardization(target[:, :len(property_tasks)])
            charge_mean, charge_std = fit_standardization(charge_values) if charge_values.size else (None, None)
            mean_tensor, std_tensor = torch.tensor(mean, device=device), torch.tensor(std, device=device)
            charge_mean_tensor = torch.tensor(charge_mean, device=device) if charge_mean is not None else None
            charge_std_tensor = torch.tensor(charge_std, device=device) if charge_std is not None else None
            bond_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            model = TinySchNet(context.config["model"], property_tasks, local)

            def loss(current, batch):
                output = current(batch)
                normalized = (batch["target"][:, :len(property_tasks)] - mean_tensor) / std_tensor
                errors = (output["prediction"] - normalized).square().mean(dim=0)
                objective = errors[0]
                if len(property_tasks) > 1:
                    objective = objective + settings["auxiliary_weight"] * errors[1:].mean()
                if "charge" in local:
                    q = (batch["charge"] - charge_mean_tensor) / charge_std_tensor
                    objective = objective + settings.get("charge_weight", 1.0) * (output["charge_prediction"] - q).square().mean()
                if "bond" in local:
                    objective = objective + settings.get("bond_weight", 1.0) * F.cross_entropy(
                        output["bond_logits"], batch["pair_class"], weight=bond_weights)
                return objective, len(batch["target"])

            def score(current, loader, current_device):
                absolute, count = 0.0, 0
                for batch in loader:
                    batch = to_device(batch, current_device)
                    prediction = current(batch)["main_prediction"][:, 0] * std_tensor[0] + mean_tensor[0]
                    absolute += float((prediction - batch["target"][:, 0]).abs().sum())
                    count += len(prediction)
                return absolute / count

            metadata = {"variant": variant, "seed": seed,
                        "task_names": property_tasks, "property_tasks": property_tasks, "local_tasks": local,
                        "target_mean": mean.tolist(), "target_std": std.tolist(),
                        "charge_mean": None if charge_mean is None else charge_mean.tolist(),
                        "charge_std": None if charge_std is None else charge_std.tolist(),
                        "bond_class_weights": class_weights.tolist() if pair_values.size else None,
                        "units": {name: UNITS[name] for name in property_tasks},
                        "normalization_split": "a_train", "selection_split": "a_val",
                        "selection_metric": "main_mae_physical_units", "model": context.config["model"],
                        "data_sha256": sha256_file(processed_dir(context) / "dataset.npz"),
                        "splits_sha256": sha256_file(processed_dir(context) / "splits.npz"),
                        "config_sha256": context.config_hash, "pretrained": False}
            artifacts.extend(fit(model,
                make_loader(context, "a_train", seed, settings["batch_size"], shuffle=True),
                make_loader(context, "a_val", seed, settings["batch_size"]), loss, score,
                upstream_dir(context, variant, seed), settings, metadata, device))
    return artifacts
