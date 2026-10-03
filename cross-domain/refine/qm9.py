"""Frozen graph features and main-task-only readouts for the gap experiment."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.qm9 import BOND_CLASSES, array_hash, load_arrays, make_loader, processed_dir, save_arrays, split_indices, tasks
from pipeline.fit import fit
from pipeline.io import read_json, sha256_file, write_json
from pipeline.readout import GraphSetReadout, TabularReadout, fit_standardization
from pipeline.runtime import configure, seed_all, to_device
from training.qm9 import load_upstream, upstream_dir, variant_tasks

TABULAR_RECIPES = ("R0", "R2")
SET_RECIPES = ("R3",)
GNN_RECIPES = ("R1", "R4", "R4-shuffle")
SUMMARY_DIM = len(BOND_CLASSES) + 5


def _select(values, wanted):
    return [value for value in values if wanted in (None, value)]


def cache_dir(context, variant, seed):
    return context.output_dir / "cache" / variant / f"seed{seed}"


def has_local(context, variant):
    return bool(variant_tasks(context, variant)[1])


def cache_fields(context, variant):
    fields = ["ids", "embedding", "target", "main_prediction", "atom_embedding", "atom_z"]
    if has_local(context, variant):
        fields += ["charge_prediction", "pair_index", "bond_probs"]
    return fields


def cache_identity(context, variant, seed, split):
    arrays = load_arrays(context)
    return {"identity": context.identity, "variant": variant, "upstream_seed": seed, "split": split,
            "checkpoint_sha256": sha256_file(upstream_dir(context, variant, seed) / "best.pt"),
            "data_sha256": sha256_file(processed_dir(context) / "dataset.npz"),
            "splits_sha256": sha256_file(processed_dir(context) / "splits.npz"),
            "fields": cache_fields(context, variant),
            "ids_sha256": array_hash(arrays["ids"][split_indices(context, split)])}


def cache(context):
    device = configure(context.config["runtime"])
    filters = getattr(context, "filters", {}) or {}
    seeds = _select(context.config["upstream"]["seeds"], filters.get("seed"))
    variants = _select(context.config["upstream"]["variants"], filters.get("variant"))
    artifacts = []
    for seed in seeds:
        for variant in variants:
            model, metadata = load_upstream(context, variant, seed, device, frozen=True)
            before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            mean = np.array(metadata["target_mean"], dtype=np.float32)
            std = np.array(metadata["target_std"], dtype=np.float32)
            local = metadata.get("local_tasks", [])
            directory = cache_dir(context, variant, seed)
            directory.mkdir(parents=True, exist_ok=True)
            for split in ("b_train", "b_val", "y_test"):
                collected = {key: [] for key in cache_fields(context, variant)}
                atom_offsets, pair_offsets = [0], [0]
                with torch.inference_mode():
                    for batch in make_loader(context, split, seed, context.config["upstream"]["batch_size"]):
                        batch = to_device(batch, device)
                        output = model(batch)
                        count = len(batch["ids"])
                        atom_base = atom_offsets[-1]
                        collected["ids"].append(batch["ids"].cpu().numpy())
                        collected["embedding"].append(output["embedding"].cpu().numpy())
                        collected["target"].append(batch["target"][:, :1].cpu().numpy())
                        collected["main_prediction"].append((output["main_prediction"].cpu().numpy() * std + mean))
                        collected["atom_embedding"].append(output["atomic_embedding"].cpu().numpy())
                        collected["atom_z"].append(batch["z"].cpu().numpy())
                        atom_counts = torch.bincount(batch["batch"], minlength=count)
                        for value in atom_counts.tolist():
                            atom_offsets.append(atom_offsets[-1] + value)
                        if "charge" in local:
                            collected["charge_prediction"].append(output["charge_prediction"].cpu().numpy())
                        if "bond" in local:
                            collected["pair_index"].append(batch["pair_index"].cpu().numpy() + atom_base)
                            collected["bond_probs"].append(torch.softmax(output["bond_logits"], dim=-1).cpu().numpy())
                            pair_counts = torch.bincount(batch["pair_batch"], minlength=count)
                            for value in pair_counts.tolist():
                                pair_offsets.append(pair_offsets[-1] + value)
                arrays = {}
                for key, values in collected.items():
                    arrays[key] = np.concatenate(values, axis=0) if key != "pair_index" else np.concatenate(values, axis=1)
                arrays["atom_offsets"] = np.array(atom_offsets, dtype=np.int64)
                if "pair_index" in arrays:
                    arrays["pair_offsets"] = np.array(pair_offsets, dtype=np.int64)
                if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
                    raise RuntimeError("upstream was not fully frozen during feature extraction")
                unchanged = all(torch.equal(before[name], value.detach().cpu()) for name, value in model.state_dict().items())
                if not unchanged:
                    raise RuntimeError("upstream state changed during frozen inference")
                path = save_arrays(directory / f"{split}.npz", **arrays)
                write_json(directory / f"{split}.json", {**cache_identity(context, variant, seed, split),
                    "cache_sha256": sha256_file(path), "count": len(arrays["ids"]),
                    "atom_count": int(len(arrays["atom_z"])), "embedding_width": arrays["embedding"].shape[1],
                    "local_tasks": local, "auxiliary_cached": bool(local), "auxiliary_truth_cached": False,
                    "freeze_verified": unchanged, "dtype": "float32"})
                artifacts.extend([path, directory / f"{split}.json"])
    return artifacts


def load_cache(context, variant, seed, split):
    directory = cache_dir(context, variant, seed)
    manifest = read_json(directory / f"{split}.json")
    expected = cache_identity(context, variant, seed, split)
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen cache source identity mismatch")
    path = directory / f"{split}.npz"
    if manifest["cache_sha256"] != sha256_file(path):
        raise ValueError("frozen cache content checksum mismatch")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    if array_hash(arrays["ids"]) != expected["ids_sha256"] or len(arrays["ids"]) != manifest["count"]:
        raise ValueError("frozen cache molecule identity/order mismatch")
    if set(arrays) != set(expected["fields"]) | {"atom_offsets"} | ({"pair_offsets"} if "pair_index" in expected["fields"] else set()):
        raise ValueError("unexpected frozen cache fields")
    for name in ("embedding", "main_prediction", "atom_embedding", "target"):
        if arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError(f"non-finite or non-FP32 cache field: {name}")
    return arrays


def molecule_summary(arrays):
    """Per-molecule charge statistics and predicted bond-class fractions."""
    offsets = arrays["atom_offsets"]
    charge = arrays["charge_prediction"]
    probs = arrays["bond_probs"]
    pair_offsets = arrays["pair_offsets"]
    rows = []
    for index in range(len(arrays["ids"])):
        q = charge[offsets[index]:offsets[index + 1]]
        summary = [q.mean(), q.std(), q.min(), q.max(), q.sum()]
        classes = probs[pair_offsets[index]:pair_offsets[index + 1]].argmax(axis=1)
        counts = np.bincount(classes, minlength=len(BOND_CLASSES)).astype(np.float64)
        summary.extend((counts / max(len(classes), 1)).tolist())
        rows.append(summary)
    return np.asarray(rows, dtype=np.float32)


def tabular_features(arrays, recipe):
    if recipe == "R0":
        padding = np.zeros((len(arrays["ids"]), SUMMARY_DIM), dtype=np.float32)
        return np.concatenate([arrays["embedding"], padding], axis=1)
    if recipe == "R2":
        return np.concatenate([arrays["embedding"], molecule_summary(arrays)], axis=1)
    raise ValueError(f"unknown tabular recipe: {recipe}")


def set_features(arrays, recipe):
    if recipe == "R1":
        return arrays["charge_prediction"][:, None]
    if recipe in ("R4", "R4-shuffle"):
        return np.concatenate([arrays["atom_embedding"], arrays["charge_prediction"][:, None]], axis=1)
    if recipe == "R3":
        return arrays["atom_embedding"]
    raise ValueError(f"unknown set recipe: {recipe}")


def set_chunks(arrays, recipe, feature_mean, feature_std, target_mean, target_std, batch_size):
    """Molecule-chunked mini-batches over concatenated atoms and pairs."""
    features = (set_features(arrays, recipe) - feature_mean) / feature_std
    target = (arrays["target"] - target_mean) / target_std
    atom_offsets = arrays["atom_offsets"]
    z = arrays["atom_z"]
    use_edges = recipe in GNN_RECIPES
    if use_edges:
        pair_offsets = arrays["pair_offsets"]
        pair_index = arrays["pair_index"]
        bond = arrays["bond_probs"]
        if recipe == "R4-shuffle":
            bond = bond[np.random.default_rng(0).permutation(len(bond))]
    chunks, start, total = [], 0, len(arrays["ids"])
    while start < total:
        end = min(start + batch_size, total)
        a0, a1 = int(atom_offsets[start]), int(atom_offsets[end])
        counts = np.diff(atom_offsets[start:end + 1])
        chunk = {"features": torch.from_numpy(features[a0:a1].astype(np.float32)),
                 "z": torch.from_numpy(z[a0:a1]),
                 "batch": torch.from_numpy(np.repeat(np.arange(end - start), counts)),
                 "target": torch.from_numpy(target[start:end].astype(np.float32))}
        if use_edges:
            p0, p1 = int(pair_offsets[start]), int(pair_offsets[end])
            chunk["pair_index"] = torch.from_numpy(pair_index[:, p0:p1].astype(np.int64) - a0)
            chunk["bond_probs"] = torch.from_numpy(bond[p0:p1].astype(np.float32))
        chunks.append(chunk)
        start = end
    return chunks


def refiner_dir(context, variant, upstream_seed, recipe, seed):
    return context.output_dir / "refiners" / variant / f"seed{upstream_seed}" / recipe / f"seed{seed}"


class TabularData(Dataset):
    def __init__(self, values, target, metadata):
        self.features = torch.from_numpy((values - np.array(metadata["feature_mean"], dtype=np.float32)) / np.array(metadata["feature_std"], dtype=np.float32))
        self.target = torch.from_numpy((target - np.array(metadata["target_mean"], dtype=np.float32)) / np.array(metadata["target_std"], dtype=np.float32))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return {"features": self.features[index], "target": self.target[index]}


def build_metadata(context, variant, upstream_seed, recipe, seed, train_arrays, **extra):
    target_mean, target_std = fit_standardization(train_arrays["target"])
    metadata = {"variant": variant, "recipe": recipe, "upstream_seed": upstream_seed,
                "downstream_seed": seed, "kind": "tabular" if recipe in TABULAR_RECIPES else "set",
                "target_name": tasks(context)[0], "target_mean": target_mean.tolist(), "target_std": target_std.tolist(),
                "normalization_split": "b_train", "selection_split": "b_val",
                "selection_metric": "main_mae_physical_units",
                "source_checkpoint_sha256": sha256_file(upstream_dir(context, variant, upstream_seed) / "best.pt"),
                "b_train_cache_sha256": sha256_file(cache_dir(context, variant, upstream_seed) / "b_train.npz"),
                "b_val_cache_sha256": sha256_file(cache_dir(context, variant, upstream_seed) / "b_val.npz"),
                "supervision": "main_task_truth_only", "identity": context.identity}
    metadata.update(extra)
    return metadata


def train(context):
    settings = context.config["refiner"]
    filters = getattr(context, "filters", {}) or {}
    upstream_seeds = _select(context.config["upstream"]["seeds"], filters.get("upstream_seed", filters.get("seed")))
    variants = _select(context.config["upstream"]["variants"], filters.get("variant"))
    recipes = _select(settings["recipes"], filters.get("recipe"))
    downstream_seeds = _select(settings["seeds"], filters.get("downstream_seed"))
    dropout = float(settings.get("dropout", 0.0) or 0.0)
    device = configure(context.config["runtime"])
    artifacts = []
    for upstream_seed in upstream_seeds:
        for variant in variants:
            local = has_local(context, variant)
            train_cache = load_cache(context, variant, upstream_seed, "b_train")
            val_cache = load_cache(context, variant, upstream_seed, "b_val")
            for recipe in recipes:
                if recipe in ("R1", "R2", "R4", "R4-shuffle") and not local:
                    continue
                for seed in downstream_seeds:
                    seed_all(seed)
                    if recipe in TABULAR_RECIPES:
                        x_train, x_val = tabular_features(train_cache, recipe), tabular_features(val_cache, recipe)
                        feature_mean, feature_std = fit_standardization(x_train)
                        metadata = build_metadata(context, variant, upstream_seed, recipe, seed, train_cache,
                                                  inputs=x_train.shape[1], hidden=settings["hidden"], dropout=dropout,
                                                  feature_mean=feature_mean.tolist(), feature_std=feature_std.tolist())
                        model = TabularReadout(x_train.shape[1], settings["hidden"], dropout=dropout)
                        train_loader = DataLoader(TabularData(x_train, train_cache["target"], metadata), batch_size=settings["batch_size"],
                                                  shuffle=True, generator=torch.Generator().manual_seed(seed),
                                                  num_workers=context.config["runtime"]["num_workers"])
                        val_loader = DataLoader(TabularData(x_val, val_cache["target"], metadata), batch_size=settings["batch_size"],
                                                num_workers=context.config["runtime"]["num_workers"])
                    else:
                        train_features = set_features(train_cache, recipe)
                        feature_mean, feature_std = fit_standardization(train_features)
                        target_mean, target_std = fit_standardization(train_cache["target"])
                        metadata = build_metadata(context, variant, upstream_seed, recipe, seed, train_cache,
                                                  node_inputs=train_features.shape[1], node_hidden=settings["set_hidden"],
                                                  decoder_hidden=settings["hidden"], layers=settings["set_layers"],
                                                  use_edges=recipe in GNN_RECIPES, use_atom_type=recipe == "R1",
                                                  dropout=dropout, feature_mean=feature_mean.tolist(), feature_std=feature_std.tolist())
                        model = GraphSetReadout(train_features.shape[1], settings["set_hidden"], settings["hidden"],
                                                layers=settings["set_layers"], use_edges=recipe in GNN_RECIPES,
                                                use_atom_type=recipe == "R1", dropout=dropout)
                        train_loader = set_chunks(train_cache, recipe, feature_mean, feature_std,
                                                  target_mean, target_std, settings["batch_size"])
                        val_loader = set_chunks(val_cache, recipe, feature_mean, feature_std,
                                                target_mean, target_std, settings["batch_size"])
                    target_std = float(metadata["target_std"][0])
                    recipe_settings = dict(settings)
                    if recipe not in TABULAR_RECIPES:
                        recipe_settings["epochs"] = settings.get("set_epochs", settings["epochs"])
                        recipe_settings["learning_rate"] = settings.get("set_learning_rate", settings["learning_rate"])
                        recipe_settings["clip_grad"] = settings.get("clip_grad", 1.0)

                    def loss(current, batch):
                        return (current(batch) - batch["target"]).square().mean(), len(batch["target"])

                    def score(current, loader, current_device):
                        absolute, count = 0.0, 0
                        for batch in loader:
                            batch = to_device(batch, current_device)
                            absolute += float((current(batch) - batch["target"]).abs().sum()) * target_std
                            count += len(batch["target"])
                        return absolute / count

                    artifacts.extend(fit(model, train_loader, val_loader, loss, score,
                        refiner_dir(context, variant, upstream_seed, recipe, seed), recipe_settings, metadata, device))
    return artifacts


def predict_refiner(context, variant, upstream_seed, recipe, seed, arrays, device):
    directory = refiner_dir(context, variant, upstream_seed, recipe, seed)
    checkpoint = torch.load(directory / "best.pt", map_location=device, weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["identity"] != context.identity or metadata["source_checkpoint_sha256"] != sha256_file(upstream_dir(context, variant, upstream_seed) / "best.pt"):
        raise ValueError("refiner source identity mismatch")
    for split in ("b_train", "b_val"):
        if metadata[f"{split}_cache_sha256"] != sha256_file(cache_dir(context, variant, upstream_seed) / f"{split}.npz"):
            raise ValueError("refiner training-cache identity mismatch")
    feature_mean = np.array(metadata["feature_mean"], dtype=np.float32)
    feature_std = np.array(metadata["feature_std"], dtype=np.float32)
    target_mean = np.array(metadata["target_mean"], dtype=np.float32)
    target_std = np.array(metadata["target_std"], dtype=np.float32)
    dropout = float(metadata.get("dropout", 0.0) or 0.0)
    if recipe in TABULAR_RECIPES:
        model = TabularReadout(metadata["inputs"], metadata["hidden"], dropout=dropout).to(device)
        values = (tabular_features(arrays, recipe) - feature_mean) / feature_std
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            prediction = model(to_device({"features": torch.from_numpy(values.astype(np.float32))}, device)).cpu().numpy()
    else:
        model = GraphSetReadout(metadata["node_inputs"], metadata["node_hidden"], metadata["decoder_hidden"],
                                layers=metadata["layers"], use_edges=metadata["use_edges"],
                                use_atom_type=metadata["use_atom_type"], dropout=dropout).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        chunks = set_chunks(arrays, recipe, feature_mean, feature_std, target_mean, target_std,
                            context.config["refiner"]["batch_size"])
        outputs = []
        with torch.inference_mode():
            for chunk in chunks:
                outputs.append(model(to_device(chunk, device)).cpu().numpy())
        prediction = np.concatenate(outputs, axis=0)
    return prediction * target_std + target_mean
