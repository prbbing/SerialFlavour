"""Main-task evaluation and local-auxiliary diagnostics on the locked test set."""

import csv
import numpy as np

from data.qm9 import BOND_CLASSES, UNITS, applicable_recipes, array_hash, gather_values, load_arrays, split_indices, tasks
from pipeline.io import read_json, sha256_file, write_json
from pipeline.metrics import regression_metrics
from pipeline.runtime import configure
from refine.qm9 import has_local, load_cache, predict_refiner, refiner_dir
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


def binary_auc(truth, scores):
    """Rank AUC with average ranks for exact ties; no extra dependency."""
    truth = np.asarray(truth, dtype=bool)
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    starts = np.r_[0, ends[:-1] + 1]
    positive_counts = np.add.reduceat(truth[order].astype(np.float64), starts)
    rank_sum = float(np.dot((starts + ends + 2) / 2, positive_counts))
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def auxiliary_diagnostics(arrays, indices, cached):
    charge_truth = gather_values(arrays, indices, "atom_offsets", "charge")
    charge_error = float(np.abs(cached["charge_prediction"] - charge_truth).mean())
    bond_truth = gather_values(arrays, indices, "pair_offsets", "pair_class")
    bond_prediction = cached["bond_probs"].argmax(axis=1)
    bonded = bond_truth > 0
    existence_probability = 1.0 - cached["bond_probs"][:, 0]
    predicted_bonded = existence_probability >= 0.5
    tp = int((predicted_bonded & bonded).sum())
    fp = int((predicted_bonded & ~bonded).sum())
    fn = int((~predicted_bonded & bonded).sum())
    tn = int((~predicted_bonded & ~bonded).sum())
    existence_accuracy = float((predicted_bonded == bonded).mean())
    order_accuracy = float((bond_prediction[bonded] == bond_truth[bonded]).mean()) if bonded.any() else None
    confusion = np.zeros((len(BOND_CLASSES), len(BOND_CLASSES)), dtype=np.int64)
    np.add.at(confusion, (bond_truth[bonded], bond_prediction[bonded]), 1)
    return {"charge_mae": charge_error, "charge_unit": "elementary_charge",
            "bond_existence_accuracy": existence_accuracy,
            "bond_existence_auc": binary_auc(bonded, existence_probability),
            "bond_existence_precision": tp / (tp + fp) if tp + fp else None,
            "bond_existence_recall": tp / (tp + fn) if tp + fn else None,
            "bond_existence_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "bond_existence_tp": tp, "bond_existence_fp": fp, "bond_existence_fn": fn, "bond_existence_tn": tn,
            "bond_multiclass_existence_accuracy": float(((bond_prediction > 0) == bonded).mean()),
            "bond_order_accuracy": order_accuracy, "bonded_pairs": int(bonded.sum()),
            "bond_order_confusion": confusion.tolist()}


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

            def add_result(method, prediction, downstream_seed, downstream_parameters, checkpoint_hash, training_manifest):
                record = {"method": method, "upstream_seed": upstream_seed, "downstream_seed": downstream_seed,
                          "n_test": len(ids), "target": names[0], "unit": UNITS[names[0]],
                          **regression_metrics(cached["target"], prediction),
                          "upstream_parameters": upstream_manifest["parameters"],
                          "downstream_parameters": downstream_parameters,
                          "best_epoch": training_manifest.get("best_epoch"),
                          "initial_validation_metric": training_manifest.get("initial_validation_metric"),
                          "checkpoint_sha256": checkpoint_hash}
                rows.append(record)
                suffix = f"u{upstream_seed}" + (f"_d{downstream_seed}" if downstream_seed is not None else "")
                artifacts.append(write_csv(directory / f"{method}_{suffix}_predictions.csv",
                                           prediction_rows(ids, cached["target"], prediction)))

            add_result(f"{prefix}-native", cached["main_prediction"], None, 0, source_hash, upstream_manifest)
            if has_local(context, variant):
                auxiliary_rows.append({"upstream_seed": upstream_seed, **auxiliary_diagnostics(arrays, indices, cached)})
            for recipe in applicable_recipes(context, variant):
                for downstream_seed in context.config["refiner"]["seeds"]:
                    refiner_path = refiner_dir(context, variant, upstream_seed, recipe, downstream_seed)
                    manifest = read_json(refiner_path / "training_manifest.json")
                    prediction = predict_refiner(context, variant, upstream_seed, recipe, downstream_seed, cached, device)
                    add_result(f"{prefix}-{recipe}", prediction, downstream_seed, manifest["parameters"],
                               sha256_file(refiner_path / "best.pt"), manifest)
    metrics_path = directory / "metrics.json"
    write_json(metrics_path, {"identity": context.identity, "test_ids_sha256": array_hash(ids),
                              "n_test": len(ids), "main": rows, "auxiliary": auxiliary_rows})
    artifacts.extend([metrics_path, write_csv(directory / "metrics.csv", rows)])
    if auxiliary_rows:
        artifacts.append(write_csv(directory / "auxiliary_metrics.csv", auxiliary_rows))
    return artifacts
