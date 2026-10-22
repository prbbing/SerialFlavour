"""New-cache contract and main-label-only residual readout training hooks."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from experiments.qm9 import data as legacy_data
from experiments.qm9 import refine as legacy_cache
from experiments.qm9.training import upstream_dir
from experiments.qm9_native_graph.data import nested_rows
from experiments.qm9_native_graph.identity import bind_sources, verify_sources
from experiments.qm9_native_graph.model import NativeResidualGraphReadout
from experiments.qm9_native_graph.plan import RECIPE_BY_NAME, all_configs
from experiments.qm9_native_graph.training import load_upstream
from pipeline.fit import fit
from pipeline.io import read_json, sha256_file, write_json
from pipeline.readout import fit_standardization
from pipeline.runtime import configure, seed_all, to_device


def cache_dir(context, variant, seed):
    return context.output_dir / "native_graph_cache" / variant / f"seed{seed}"


def _hidden(head, value):
    # Legacy make_head is Sequential, ending in the supervised output Linear.
    return head[:-1](value)


def cache(context):
    binding = bind_sources(context)
    # These legacy caches are produced under this NEW experiment only. No old
    # manifest is rewritten/imported, and no historical completion marker is read.
    artifacts = [binding, *legacy_cache.cache(context)]
    device = configure(context.config["runtime"])
    prepared = legacy_data.load_arrays(context)
    id_to_row = {int(molecule_id): row for row, molecule_id in enumerate(prepared["ids"])}
    for variant in context.config["upstream"]["variants"]:
        for seed in context.config["upstream"]["seeds"]:
            filters = getattr(context, "filters", {}) or {}
            if filters.get("variant") not in (None, variant) or filters.get("seed") not in (None, seed):
                continue
            model, metadata = load_upstream(context, variant, seed, device)
            width = int(context.config["model"]["head_hidden"][-1])
            for split in ("b_train", "b_val", "y_test"):
                arrays = legacy_cache.load_cache(context, variant, seed, split)
                positions, qparts, bparts = [], [], []
                # Work molecule-by-molecule rather than placing the full pool on GPU.
                with torch.inference_mode():
                    for i, molecule_id in enumerate(arrays["ids"]):
                        index = id_to_row[int(molecule_id)]
                        lo, hi = prepared["atom_offsets"][index:index + 2]
                        alo, ahi = arrays["atom_offsets"][i:i + 2]
                        plo, phi = arrays["pair_offsets"][i:i + 2]
                        positions.append(prepared["pos"][lo:hi])
                        if not np.array_equal(prepared["z"][lo:hi], arrays["atom_z"][alo:ahi]):
                            raise ValueError("geometry and frozen atomic order differ")
                        h = torch.from_numpy(arrays["atom_embedding"][alo:ahi]).to(device)
                        g = torch.from_numpy(arrays["embedding"][i:i + 1]).to(device)
                        if variant == "multi_task":
                            qparts.append(_hidden(model.charge_head, torch.cat([h, g.expand(len(h), -1)], -1)).cpu().numpy())
                            pair = arrays["pair_index"][:, plo:phi] - alo
                            pair = torch.as_tensor(pair, dtype=torch.long, device=device)
                            left, right = h[pair[0]], h[pair[1]]
                            pair_input = torch.cat([left + right, (left - right).abs(), g.expand(len(left), -1)], -1)
                            bparts.append(_hidden(model.bond_head, pair_input).cpu().numpy())
                        else:
                            qparts.append(np.zeros((len(h), width), dtype=np.float32))
                            bparts.append(np.zeros((phi - plo, width), dtype=np.float32))
                extra = {"pos": np.concatenate(positions).astype(np.float32),
                         "charge_hidden": np.concatenate(qparts).astype(np.float32),
                         "bond_hidden": np.concatenate(bparts).astype(np.float32)}
                directory = cache_dir(context, variant, seed)
                path = legacy_data.save_arrays(directory / f"{split}.npz", **extra)
                manifest = directory / f"{split}.json"
                write_json(manifest, {"identity": context.identity, "legacy_source_sha256": verify_sources(context),
                    "legacy_cache_sha256": sha256_file(legacy_cache.cache_dir(context, variant, seed) / f"{split}.npz"),
                    "legacy_manifest_sha256": sha256_file(legacy_cache.cache_dir(context, variant, seed) / f"{split}.json"),
                    "extra_sha256": sha256_file(path), "ids_sha256": legacy_data.array_hash(arrays["ids"]),
                    "checkpoint_sha256": sha256_file(upstream_dir(context, variant, seed) / "best.pt"),
                    "charge_hidden_width": width, "bond_hidden_width": width,
                    "native_prediction_unit": "eV", "target_unit": "eV", "geometry_unit": "angstrom",
                    "charge_prediction_unit": "elementary_charge", "auxiliary_truth_cached": False,
                    "auxiliary_hidden": "frozen auxiliary head penultimate activation, no labels"})
                artifacts.extend([path, manifest])
    return artifacts


def load_cache(context, variant, seed, split):
    arrays = legacy_cache.load_cache(context, variant, seed, split)
    directory = cache_dir(context, variant, seed)
    manifest = read_json(directory / f"{split}.json")
    expected = {"identity": context.identity, "legacy_source_sha256": verify_sources(context),
        "legacy_cache_sha256": sha256_file(legacy_cache.cache_dir(context, variant, seed) / f"{split}.npz"),
        "legacy_manifest_sha256": sha256_file(legacy_cache.cache_dir(context, variant, seed) / f"{split}.json"),
        "checkpoint_sha256": sha256_file(upstream_dir(context, variant, seed) / "best.pt"),
        "extra_sha256": sha256_file(directory / f"{split}.npz"),
        "ids_sha256": legacy_data.array_hash(arrays["ids"]), "auxiliary_truth_cached": False,
        "native_prediction_unit": "eV", "target_unit": "eV", "geometry_unit": "angstrom",
        "charge_prediction_unit": "elementary_charge"}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("native graph cache dependency/units identity mismatch")
    with np.load(directory / f"{split}.npz", allow_pickle=False) as stored:
        extra = {name: stored[name] for name in stored.files}
    shapes = {"pos": (len(arrays["atom_z"]), 3),
              "charge_hidden": (len(arrays["atom_z"]), manifest["charge_hidden_width"]),
              "bond_hidden": (arrays["pair_index"].shape[1], manifest["bond_hidden_width"])}
    if set(extra) != set(shapes) or any(extra[k].shape != shape or extra[k].dtype != np.float32
            or not np.isfinite(extra[k]).all() for k, shape in shapes.items()):
        raise ValueError("invalid extra graph cache fields/shapes/dtypes")
    return {**arrays, **extra}


class FrozenMolecules(Dataset):
    def __init__(self, arrays, rows):
        self.arrays, self.rows = arrays, np.asarray(rows, dtype=np.int64)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return int(self.rows[index])


class Collate:
    def __init__(self, arrays, recipe, shuffle_seed):
        self.arrays, self.recipe, self.shuffle_seed = arrays, RECIPE_BY_NAME[recipe], shuffle_seed

    def __call__(self, rows):
        a, r = self.arrays, self.recipe
        fields = {name: [] for name in ("atom_embedding", "atom_z", "pos", "charge", "charge_hidden",
                                         "bond_probs", "bond_hidden", "pair_index", "batch")}
        offset = 0
        for molecule, row in enumerate(rows):
            lo, hi = a["atom_offsets"][row:row + 2]
            plo, phi = a["pair_offsets"][row:row + 2]
            for name in ("atom_embedding", "atom_z", "pos"):
                fields[name].append(a[name][lo:hi])
            fields["pair_index"].append(a["pair_index"][:, plo:phi] - lo + offset)
            fields["batch"].append(np.full(hi - lo, molecule, dtype=np.int64))
            fields["charge"].append(a["charge_prediction"][lo:hi, None] if r.charge else np.zeros((hi - lo, 1), np.float32))
            fields["charge_hidden"].append(a["charge_hidden"][lo:hi] if r.hidden else np.zeros_like(a["charge_hidden"][lo:hi]))
            fields["bond_hidden"].append(a["bond_hidden"][plo:phi] if r.hidden else np.zeros_like(a["bond_hidden"][plo:phi]))
            probs = np.zeros((phi - plo, 5), dtype=np.float32)
            if r.bond_mode in ("typed", "existence", "shuffle"):
                probs = a["bond_probs"][plo:phi].copy()
                if r.bond_mode == "existence":
                    probs[:, 1:] = (1.0 - probs[:, :1]) / 4
                elif r.bond_mode == "shuffle":
                    rng = np.random.default_rng(int(a["ids"][row]) ^ int(self.shuffle_seed))
                    probs = probs[rng.permutation(len(probs))]
            elif r.bond_mode == "uniform":
                probs.fill(0.2)
            fields["bond_probs"].append(probs)
            offset += hi - lo
        batch = {key: torch.from_numpy(np.concatenate(values, axis=1 if key == "pair_index" else 0))
                 for key, values in fields.items()}
        batch.update({key: torch.from_numpy(a[key][rows]) for key in ("embedding", "target", "main_prediction")})
        return batch


def loader(context, arrays, rows, recipe, seed, shuffle=False):
    return DataLoader(FrozenMolecules(arrays, rows), batch_size=context.config["refiner"]["batch_size"],
        shuffle=shuffle, generator=torch.Generator().manual_seed(seed), num_workers=0,
        collate_fn=Collate(arrays, recipe, context.config["graph"]["shuffle_seed"]))


def refiner_dir(context, item):
    return (context.output_dir / "refiners" / item["variant"] / f"seed{item['upstream_seed']}"
            / f"b{item['b_train_size']}" / item["recipe"] / f"seed{item['downstream_seed']}")


def source_identity(context, item):
    v, s = item["variant"], item["upstream_seed"]
    return {"identity": context.identity, "legacy_source_sha256": verify_sources(context),
        "source_checkpoint_sha256": sha256_file(upstream_dir(context, v, s) / "best.pt"),
        "nested_b_sha256": sha256_file(context.output_dir / "data" / "nested_b.json"),
        "cache_manifests_sha256": {split: sha256_file(cache_dir(context, v, s) / f"{split}.json")
                                   for split in ("b_train", "b_val")},
        "cache_arrays_sha256": {split: {
            "legacy": sha256_file(legacy_cache.cache_dir(context, v, s) / f"{split}.npz"),
            "extra": sha256_file(cache_dir(context, v, s) / f"{split}.npz")}
            for split in ("b_train", "b_val")}, **item}


def selected_items(context):
    filters = getattr(context, "filters", {}) or {}
    if filters and any(key in filters for key in ("recipe", "downstream_seed")) and "b_train_size" not in filters:
        raise ValueError("filtered readout execution requires an explicit b_train_size; generic unit scheduler is unsupported")
    for item in all_configs(context.config):
        if all(filters.get(key) in (None, value) for key, value in item.items()):
            yield item


def train(context):
    artifacts = [bind_sources(context)]
    device = configure(context.config["runtime"])
    for item in selected_items(context):
        variant, us, recipe, ds = item["variant"], item["upstream_seed"], item["recipe"], item["downstream_seed"]
        arrays = load_cache(context, variant, us, "b_train")
        validation = load_cache(context, variant, us, "b_val")
        rows = nested_rows(context, item["b_train_size"], arrays["ids"])
        upstream, source = load_upstream(context, variant, us, "cpu")
        atom_rows = np.concatenate([np.arange(*arrays["atom_offsets"][i:i + 2]) for i in rows])
        metadata = {**source_identity(context, item), "supervision": "gap_truth_only",
            "normalization_split": "nested_b_train", "selection_split": "b_val", "target_unit": "eV",
            "source_target_mean": source["target_mean"][:1], "source_target_std": source["target_std"][:1],
            "train_ids_sha256": legacy_data.array_hash(arrays["ids"][rows]),
            "charge_hidden_width": arrays["charge_hidden"].shape[1],
            "bond_hidden_width": arrays["bond_hidden"].shape[1]}
        for key, values in (("g", arrays["embedding"][rows]), ("h", arrays["atom_embedding"][atom_rows]),
                            ("target", arrays["target"][rows])):
            mean, std = fit_standardization(values)
            metadata[f"{key}_mean"], metadata[f"{key}_std"] = mean.tolist(), std.tolist()
        seed_all(ds)
        model = NativeResidualGraphReadout(upstream.heads["gap"], metadata, context.config["graph"]).to(device)
        train_loader = loader(context, arrays, rows, recipe, ds, shuffle=True)
        val_loader = loader(context, validation, np.arange(len(validation["ids"])), recipe, ds)
        # Same-backend reference avoids treating CPU/CUDA cache roundoff as a failure.
        max_error, cache_error = 0.0, 0.0
        reference_head = upstream.heads["gap"].to(device).eval()
        with torch.inference_mode():
            for batch in val_loader:
                batch = to_device(batch, device)
                ref = reference_head(batch["embedding"]) * float(source["target_std"][0]) + float(source["target_mean"][0])
                pred = model.physical(batch)
                max_error = max(max_error, float((ref - pred).abs().max()))
                cache_error = max(cache_error, float((ref - batch["main_prediction"]).abs().max()))
                if not torch.allclose(pred, ref, atol=3e-5, rtol=3e-5):
                    raise ValueError("epoch 0 fails same-device native equivalence")
        metadata.update({"epoch0_native_max_abs_eV": max_error, "same_device_native_vs_cache_max_abs_eV": cache_error})

        def loss(current, batch):
            target = (batch["target"] - current.target_mean) / current.target_std
            return (current(batch) - target).square().mean(), len(target)

        def score(current, batches, current_device):
            absolute, count = 0.0, 0
            for batch in batches:
                batch = to_device(batch, current_device)
                absolute += float((current.physical(batch) - batch["target"]).abs().sum())
                count += len(batch["target"])
            return absolute / count

        artifacts.extend(fit(model, train_loader, val_loader, loss, score, refiner_dir(context, item),
                             context.config["refiner"], metadata, device))
    return artifacts


def predict_refiner(context, item, arrays, device):
    checkpoint = torch.load(refiner_dir(context, item) / "best.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if any(metadata.get(key) != value for key, value in source_identity(context, item).items()):
        raise ValueError("readout checkpoint identity mismatch")
    upstream, _ = load_upstream(context, item["variant"], item["upstream_seed"], "cpu")
    model = NativeResidualGraphReadout(upstream.heads["gap"], metadata, context.config["graph"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    values = []
    with torch.inference_mode():
        for batch in loader(context, arrays, np.arange(len(arrays["ids"])), item["recipe"], item["downstream_seed"]):
            values.append(model.physical(to_device(batch, device)).cpu().numpy())
    return np.concatenate(values), checkpoint["epoch"]
