"""Main-task evaluation and local-auxiliary diagnostics on the locked test set."""

import csv
import numpy as np

from data.qm9 import BOND_CLASSES, UNITS, array_hash, gather_values, load_arrays, split_indices, tasks
from pipeline.io import read_json, sha256_file, write_json
from pipeline.metrics import regression_metrics
from pipeline.runtime import configure
from refine.qm9 import TABULAR_RECIPES, has_local, load_cache, predict_refiner, refiner_dir
from training.qm9 import upstream_dir


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def prediction_rows(ids, truth, prediction):
    return [{"molecule_id": int(index), "truth": float(y), "prediction": float(p), "error": float(p - y)}
            for index, y, p in zip(ids, truth.reshape(-1), prediction.reshape(-1))]


def auxiliary_diagnostics(arrays, indices, cached):
    charge_truth = gather_values(arrays, indices, "atom_offsets", "charge")
    charge_error = float(np.abs(cached["charge_prediction"] - charge_truth).mean())
    bond_truth = gather_values(arrays, indices, "pair_offsets", "pair_class")
    bond_prediction = cached["bond_probs"].argmax(axis=1)
    existence_accuracy = float(((bond_prediction > 0) == (bond_truth > 0)).mean())
    bonded = bond_truth > 0
    order_accuracy = float((bond_prediction[bonded] == bond_truth[bonded]).mean()) if bonded.any() else None
    return {"charge_mae": charge_error, "bond_existence_accuracy": existence_accuracy,
            "bond_order_accuracy": order_accuracy, "bonded_pairs": int(bonded.sum())}


def evaluate(context):
    device = configure(context.config["runtime"])
    directory = context.output_dir / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    artifacts, rows, auxiliary_rows = [], [], []
    arrays = load_arrays(context)
    indices = split_indices(context, "y_test")
    ids = arrays["ids"][indices]
    names = tasks(context)
    for upstream_seed in context.config["upstream"]["seeds"]:
        for variant in context.config["upstream"]["variants"]:
            cached = load_cache(context, variant, upstream_seed, "y_test")
            if not np.array_equal(ids, cached["ids"]):
                raise ValueError("Y-test cache IDs differ from the locked split")
            prefix = "ST" if variant == "single_task" else "MT"
            upstream_manifest = read_json(upstream_dir(context, variant, upstream_seed) / "training_manifest.json")
            source_hash = sha256_file(upstream_dir(context, variant, upstream_seed) / "best.pt")

            def add_result(method, prediction, downstream_seed, downstream_parameters, checkpoint_hash):
                record = {"method": method, "upstream_seed": upstream_seed, "downstream_seed": downstream_seed,
                          "n_test": len(ids), "target": names[0], "unit": UNITS[names[0]],
                          **regression_metrics(cached["target"], prediction),
                          "upstream_parameters": upstream_manifest["parameters"],
                          "downstream_parameters": downstream_parameters,
                          "checkpoint_sha256": checkpoint_hash}
                rows.append(record)
                suffix = f"u{upstream_seed}" + (f"_d{downstream_seed}" if downstream_seed is not None else "")
                artifacts.append(write_csv(directory / f"{method}_{suffix}_predictions.csv",
                                           prediction_rows(ids, cached["target"], prediction)))

            add_result(f"{prefix}-native", cached["main_prediction"], None, 0, source_hash)
            if has_local(context, variant):
                auxiliary_rows.append({"upstream_seed": upstream_seed, **auxiliary_diagnostics(arrays, indices, cached)})
            for recipe in context.config["refiner"]["recipes"]:
                if recipe in ("R1", "R2", "R4", "R4-shuffle") and not has_local(context, variant):
                    continue
                for downstream_seed in context.config["refiner"]["seeds"]:
                    refiner_path = refiner_dir(context, variant, upstream_seed, recipe, downstream_seed)
                    manifest = read_json(refiner_path / "training_manifest.json")
                    prediction = predict_refiner(context, variant, upstream_seed, recipe, downstream_seed, cached, device)
                    add_result(f"{prefix}-{recipe}", prediction, downstream_seed, manifest["parameters"],
                               sha256_file(refiner_path / "best.pt"))
    metrics_path = directory / "metrics.json"
    write_json(metrics_path, {"identity": context.identity, "test_ids_sha256": array_hash(ids),
                              "n_test": len(ids), "main": rows, "auxiliary": auxiliary_rows})
    artifacts.extend([metrics_path, write_csv(directory / "metrics.csv", rows)])
    if auxiliary_rows:
        artifacts.append(write_csv(directory / "auxiliary_metrics.csv", auxiliary_rows))
    return artifacts
