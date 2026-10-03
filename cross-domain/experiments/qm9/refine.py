"""Frozen graph features and main-task-only readouts for the gap experiment."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.qm9.data import BOND_CLASSES, applicable_recipes, array_hash, load_arrays, make_loader, processed_dir, save_arrays, split_indices, tasks
from experiments.qm9.model import make_head
from pipeline.fit import fit
from pipeline.io import read_json, sha256_file, write_json
from pipeline.readout import GraphSetReadout, TabularReadout, fit_standardization
from pipeline.runtime import configure, seed_all, to_device
from experiments.qm9.training import load_upstream, upstream_dir, variant_tasks

TABULAR_RECIPES = ("R0", "R2", "R0-native", "R2-native")
NATIVE_RECIPES = ("R0-native", "R2-native")
SET_RECIPES = ("R3",)
GNN_RECIPES = ("R1", "R3-graph", "R4", "R4-nocharge", "R4-existence", "R4-uniform", "R4-shuffle")
SUMMARY_DIM = len(BOND_CLASSES) + 5


def _select(values, wanted):
    return [value for value in values if wanted in (None, value)]


def cache_dir(context, variant, seed):
    return context.output_dir / "cache" / variant / f"seed{seed}"


def has_local(context, variant):
    return bool(variant_tasks(context, variant)[1])


def cache_fields(context, variant):
    fields = ["ids", "embedding", "target", "main_prediction", "atom_embedding", "atom_z", "pair_index"]
    if has_local(context, variant):
        fields += ["charge_prediction", "bond_probs"]
    return fields


def cache_identity(context, variant, seed, split):
    arrays = load_arrays(context)
    return {"identity": context.identity, "variant": variant, "upstream_seed": seed, "split": split,
            "checkpoint_sha256": sha256_file(upstream_dir(context, variant, seed) / "best.pt"),
            "data_sha256": sha256_file(processed_dir(context) / "dataset.npz"),
            "splits_sha256": sha256_file(processed_dir(context) / "splits.npz"),
            "fields": cache_fields(context, variant),
            "charge_prediction_unit": "elementary_charge" if has_local(context, variant) else None,
            "ids_sha256": array_hash(arrays["ids"][split_indices(context, split)].astype(np.int64))}


def physical_charge(normalized_prediction, metadata):
    """Undo A-train target scaling exactly once, before caching/diagnostics."""
    mean = np.asarray(metadata["charge_mean"], dtype=np.float32)
    std = np.asarray(metadata["charge_std"], dtype=np.float32)
    if mean.shape != (1,) or std.shape != (1,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or std[0] <= 0:
        raise ValueError("invalid upstream charge normalization")
    return np.asarray(normalized_prediction, dtype=np.float32) * std[0] + mean[0]


def validate_cache_arrays(arrays):
    floats = ("embedding", "main_prediction", "atom_embedding", "target", "charge_prediction", "bond_probs")
    for name in floats:
        if name in arrays and (arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all()):
            raise ValueError(f"non-finite or non-FP32 cache field: {name}")
    offsets, pairs = arrays["atom_offsets"], arrays["pair_index"]
    pair_offsets = arrays["pair_offsets"]
    n_atoms = len(arrays["atom_z"])
    if pairs.ndim != 2 or pairs.shape[0] != 2 or offsets.ndim != 1 or pair_offsets.ndim != 1:
        raise ValueError("invalid frozen atom/pair shapes")
    n_molecules = len(arrays["ids"])
    if (arrays["embedding"].ndim != 2 or arrays["embedding"].shape[0] != n_molecules
            or arrays["atom_embedding"].ndim != 2 or arrays["atom_embedding"].shape[0] != n_atoms
            or arrays["target"].shape != (n_molecules, 1)
            or arrays["main_prediction"].shape != (n_molecules, 1)):
        raise ValueError("invalid frozen feature/target shapes")
    if (len(offsets) != len(arrays["ids"]) + 1 or offsets[0] != 0 or offsets[-1] != n_atoms
            or np.any(np.diff(offsets) <= 0) or len(pair_offsets) != len(offsets)
            or pair_offsets[0] != 0 or pair_offsets[-1] != pairs.shape[1]
            or np.any(np.diff(pair_offsets) < 0) or pairs.shape[0] != 2
            or np.any(pairs < 0) or np.any(pairs >= n_atoms)):
        raise ValueError("invalid frozen atom/pair offsets or indices")
    atoms = np.repeat(np.arange(len(arrays["ids"])), np.diff(offsets))
    pair_molecules = np.repeat(np.arange(len(arrays["ids"])), np.diff(pair_offsets))
    if not np.array_equal(atoms[pairs[0]], pair_molecules) or not np.array_equal(atoms[pairs[1]], pair_molecules):
        raise ValueError("frozen pair crosses molecule boundaries")
    if "charge_prediction" in arrays and arrays["charge_prediction"].shape != (n_atoms,):
        raise ValueError("invalid charge prediction shape")
    if "bond_probs" in arrays:
        probs = arrays["bond_probs"]
        if (probs.shape != (pairs.shape[1], len(BOND_CLASSES)) or np.any(probs < 0)
                or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)):
            raise ValueError("invalid predicted bond probabilities")


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
                        collected["main_prediction"].append((output["main_prediction"].cpu().numpy() * std[:1] + mean[:1]))
                        collected["atom_embedding"].append(output["atomic_embedding"].cpu().numpy())
                        collected["atom_z"].append(batch["z"].cpu().numpy())
                        atom_counts = torch.bincount(batch["batch"], minlength=count)
                        for value in atom_counts.tolist():
                            atom_offsets.append(atom_offsets[-1] + value)
                        if "charge" in local:
                            collected["charge_prediction"].append(physical_charge(output["charge_prediction"].cpu().numpy(), metadata))
                        # Pair indices contain topology-free all-pair endpoints, not labels.
                        collected["pair_index"].append(batch["pair_index"].cpu().numpy() + atom_base)
                        pair_counts = torch.bincount(batch["pair_batch"], minlength=count)
                        for value in pair_counts.tolist():
                            pair_offsets.append(pair_offsets[-1] + value)
                        if "bond" in local:
                            collected["bond_probs"].append(torch.softmax(output["bond_logits"], dim=-1).cpu().numpy())
                arrays = {}
                for key, values in collected.items():
                    arrays[key] = np.concatenate(values, axis=0) if key != "pair_index" else np.concatenate(values, axis=1)
                arrays["atom_offsets"] = np.array(atom_offsets, dtype=np.int64)
                arrays["pair_offsets"] = np.array(pair_offsets, dtype=np.int64)
                validate_cache_arrays(arrays)
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
    if manifest.get("freeze_verified") is not True or manifest.get("auxiliary_truth_cached") is not False:
        raise ValueError("frozen cache protocol flags are invalid")
    path = directory / f"{split}.npz"
    if manifest["cache_sha256"] != sha256_file(path):
        raise ValueError("frozen cache content checksum mismatch")
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    if array_hash(arrays["ids"]) != expected["ids_sha256"] or len(arrays["ids"]) != manifest["count"]:
        raise ValueError("frozen cache molecule identity/order mismatch")
    if set(arrays) != set(expected["fields"]) | {"atom_offsets", "pair_offsets"}:
        raise ValueError("unexpected frozen cache fields")
    validate_cache_arrays(arrays)
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
        pair_probs = probs[pair_offsets[index]:pair_offsets[index + 1]]
        summary.extend(pair_probs.mean(axis=0).tolist() if len(pair_probs) else [0.0] * len(BOND_CLASSES))
        rows.append(summary)
    return np.asarray(rows, dtype=np.float32)


def tabular_features(arrays, recipe):
    if recipe in ("R0", "R0-native"):
        padding = np.zeros((len(arrays["ids"]), SUMMARY_DIM), dtype=np.float32)
        return np.concatenate([arrays["embedding"], padding], axis=1)
    if recipe in ("R2", "R2-native"):
        return np.concatenate([arrays["embedding"], molecule_summary(arrays)], axis=1)
    raise ValueError(f"unknown tabular recipe: {recipe}")


def set_features(arrays, recipe):
    if recipe == "R1":
        return arrays["charge_prediction"][:, None]
    if recipe in ("R4", "R4-shuffle", "R4-existence", "R4-uniform"):
        return np.concatenate([arrays["atom_embedding"], arrays["charge_prediction"][:, None]], axis=1)
    if recipe in ("R3-graph", "R4-nocharge"):
        return np.concatenate([arrays["atom_embedding"], np.zeros((len(arrays["atom_z"]), 1), dtype=np.float32)], axis=1)
    if recipe == "R3":
        return arrays["atom_embedding"]
    raise ValueError(f"unknown set recipe: {recipe}")


def graph_probabilities(arrays, recipe):
    if recipe in ("R3-graph", "R4-uniform"):
        probs = np.zeros((arrays["pair_index"].shape[1], len(BOND_CLASSES)), dtype=np.float32)
        probs[:, 1:] = 1.0 / (len(BOND_CLASSES) - 1)
        return probs
    probs = arrays["bond_probs"]
    if recipe == "R4-existence":
        probs = probs.copy()
        probs[:, 1:] = (1.0 - probs[:, :1]) / (len(BOND_CLASSES) - 1)
    elif recipe == "R4-shuffle":
        probs = probs.copy()
        # Preserve each molecule's class distribution and avoid inter-molecule mixing.
        for index, molecule_id in enumerate(arrays["ids"]):
            lo, hi = arrays["pair_offsets"][index:index + 2]
            rng = np.random.default_rng(int(molecule_id) ^ 2026)
            probs[lo:hi] = probs[lo:hi][rng.permutation(hi - lo)]
    return probs


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
        bond = graph_probabilities(arrays, recipe)
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


class NativeHeadReadout(torch.nn.Module):
    """Native activation/widths, with identical zero slots for g and g+aux arms."""

    def __init__(self, inputs, hidden):
        super().__init__()
        self.network = make_head(inputs, hidden, 1)

    def forward(self, batch):
        return self.network(batch["features"])


def initialize_native_head(readout, source_head, feature_mean, feature_std,
                           source_mean, source_std, target_mean, target_std):
    """Absorb B-feature and A-to-B target scaling, preserving physical predictions."""
    source = [layer for layer in source_head if isinstance(layer, torch.nn.Linear)]
    destination = [layer for layer in readout.network if isinstance(layer, torch.nn.Linear)]
    if len(source) != len(destination):
        raise ValueError("native readout architecture differs from upstream head")
    with torch.no_grad():
        for index, (old, new) in enumerate(zip(source, destination)):
            if index == 0:
                width = old.in_features
                weights = old.weight.detach().cpu()
                new.weight.zero_()
                new.weight[:, :width].copy_(weights * torch.as_tensor(feature_std[:width]))
                new.bias.copy_(old.bias.detach().cpu() + weights @ torch.as_tensor(feature_mean[:width]))
            else:
                new.weight.copy_(old.weight.detach().cpu())
                new.bias.copy_(old.bias.detach().cpu())
        scale = float(source_std[0]) / float(target_std[0])
        destination[-1].weight.mul_(scale)
        destination[-1].bias.mul_(scale).add_((float(source_mean[0]) - float(target_mean[0])) / float(target_std[0]))


def verify_native_predictions(model, loader, arrays, metadata, device):
    model.to(device).eval()
    outputs = []
    with torch.inference_mode():
        for batch in loader:
            outputs.append(model(to_device(batch, device)).cpu().numpy())
    predictions = np.concatenate(outputs) * np.asarray(metadata["target_std"], dtype=np.float32) + np.asarray(metadata["target_mean"], dtype=np.float32)
    if not np.allclose(predictions, arrays["main_prediction"], atol=3e-5, rtol=3e-5):
        raise RuntimeError("native-initialized readout does not reproduce frozen native predictions")
    return float(np.abs(predictions - arrays["main_prediction"]).max())


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
            train_cache = load_cache(context, variant, upstream_seed, "b_train")
            val_cache = load_cache(context, variant, upstream_seed, "b_val")
            for recipe in recipes:
                if recipe not in applicable_recipes(context, variant):
                    continue
                for seed in downstream_seeds:
                    seed_all(seed)
                    if recipe in TABULAR_RECIPES:
                        x_train, x_val = tabular_features(train_cache, recipe), tabular_features(val_cache, recipe)
                        feature_mean, feature_std = fit_standardization(x_train)
                        native = recipe in NATIVE_RECIPES
                        hidden = context.config["model"]["head_hidden"] if native else settings["hidden"]
                        metadata = build_metadata(context, variant, upstream_seed, recipe, seed, train_cache,
                                                  inputs=x_train.shape[1], hidden=hidden, dropout=0.0 if native else dropout,
                                                  initialization="native_head" if native else "random",
                                                  feature_mean=feature_mean.tolist(), feature_std=feature_std.tolist())
                        if native:
                            model = NativeHeadReadout(x_train.shape[1], hidden)
                            source_model, source_metadata = load_upstream(context, variant, upstream_seed, device, frozen=True)
                            initialize_native_head(model, source_model.heads[tasks(context)[0]], feature_mean, feature_std,
                                                   source_metadata["target_mean"], source_metadata["target_std"],
                                                   metadata["target_mean"], metadata["target_std"])
                            del source_model
                        else:
                            model = TabularReadout(x_train.shape[1], hidden, dropout=dropout)
                        train_loader = DataLoader(TabularData(x_train, train_cache["target"], metadata), batch_size=settings["batch_size"],
                                                  shuffle=True, generator=torch.Generator().manual_seed(seed),
                                                  num_workers=context.config["runtime"]["num_workers"])
                        val_loader = DataLoader(TabularData(x_val, val_cache["target"], metadata), batch_size=settings["batch_size"],
                                                num_workers=context.config["runtime"]["num_workers"])
                        if native:
                            metadata["initial_native_max_abs_error"] = verify_native_predictions(model, val_loader, val_cache, metadata, device)
                    else:
                        train_features = set_features(train_cache, recipe)
                        feature_mean, feature_std = fit_standardization(train_features)
                        target_mean, target_std = fit_standardization(train_cache["target"])
                        metadata = build_metadata(context, variant, upstream_seed, recipe, seed, train_cache,
                                                  node_inputs=train_features.shape[1], node_hidden=settings["set_hidden"],
                                                  decoder_hidden=settings["hidden"], layers=settings["set_layers"],
                                                  use_edges=recipe in GNN_RECIPES, use_atom_type=recipe == "R1",
                                                  edge_features="typed_bond_probabilities" if recipe in GNN_RECIPES else None,
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
                    if recipe in NATIVE_RECIPES:
                        recipe_settings.update(dropout=0.0, include_initial_checkpoint=True,
                                               learning_rate=settings.get("native_learning_rate", 0.0003),
                                               weight_decay=settings.get("native_weight_decay", 0.00001))
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
        model = (NativeHeadReadout(metadata["inputs"], metadata["hidden"]) if recipe in NATIVE_RECIPES
                 else TabularReadout(metadata["inputs"], metadata["hidden"], dropout=dropout)).to(device)
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
