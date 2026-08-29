"""Streaming truth diagnostics and plots for frozen Parallel auxiliary heads."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from src.parallel_refine.config import active_parallel_config, write_json_atomic
from src.parallel_refine.data import create_loader
from src.parallel_refine.upstream import (
    build_parallel, checkpoint_config, frozen_parallel_outputs)


AUXILIARY_EVALUATION_VERSION = "parallel_refine_auxiliary_y_v1"
PAIR_BIN_EDGES = np.linspace(0.0, 1.0, 101, dtype=np.float64)


def _pyplot():
    """Import pyplot only for direct-Parallel evaluation/plot generation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _json_ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator / denominator)


def _origin_metrics(confusion, negative_log_likelihood):
    confusion = np.asarray(confusion, dtype=np.int64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    normalized = np.divide(
        confusion, support[:, None],
        out=np.zeros_like(confusion, dtype=np.float64),
        where=support[:, None] != 0)
    precision = np.divide(
        true_positive, predicted,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted != 0)
    recall = np.divide(
        true_positive, support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=support != 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(precision), where=(precision + recall) != 0)
    total = int(support.sum())
    return {
        "track_definition": "mask=true and origin truth >= 0",
        "confusion_matrix_orientation": "rows=truth, columns=prediction",
        "n_tracks": total,
        "accuracy": _json_ratio(int(true_positive.sum()), total),
        "cross_entropy": _json_ratio(negative_log_likelihood, total),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_row_normalized": normalized.tolist(),
        "support": support.astype(int).tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }


def _histogram_quantile(counts, probability):
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return None
    target = probability * total
    index = min(int(np.searchsorted(np.cumsum(counts), target, side="left")),
                len(counts) - 1)
    return float(0.5 * (PAIR_BIN_EDGES[index] + PAIR_BIN_EDGES[index + 1]))


def _score_summary(count, total, square_total, above_half, histogram):
    if count == 0:
        return {
            "n": 0, "mean": None, "std": None,
            "q25_histogram": None, "median_histogram": None,
            "q75_histogram": None, "fraction_ge_0p5": None,
        }
    mean = total / count
    variance = max(square_total / count - mean * mean, 0.0)
    return {
        "n": int(count),
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "q25_histogram": _histogram_quantile(histogram, 0.25),
        "median_histogram": _histogram_quantile(histogram, 0.50),
        "q75_histogram": _histogram_quantile(histogram, 0.75),
        "fraction_ge_0p5": float(above_half / count),
    }


def _histogram_auc(match_counts, other_counts):
    positives = int(np.sum(match_counts))
    negatives = int(np.sum(other_counts))
    if positives == 0 or negatives == 0:
        return None
    other_below = np.cumsum(other_counts) - other_counts
    favourable = np.sum(match_counts * (other_below + 0.5 * other_counts))
    return float(favourable / (positives * negatives))


class AuxiliaryAccumulator:
    def __init__(self, n_origin_classes, n_jet_classes):
        self.origin_confusion = np.zeros(
            (n_origin_classes, n_origin_classes), dtype=np.int64)
        self.origin_nll = 0.0
        self.pair_histograms = np.zeros((2, len(PAIR_BIN_EDGES) - 1), dtype=np.int64)
        self.pair_by_jet = np.zeros(
            (n_jet_classes, 2, len(PAIR_BIN_EDGES) - 1), dtype=np.int64)
        self.pair_count = np.zeros(2, dtype=np.int64)
        self.pair_sum = np.zeros(2, dtype=np.float64)
        self.pair_square_sum = np.zeros(2, dtype=np.float64)
        self.pair_above_half = np.zeros(2, dtype=np.int64)
        self.pair_bce_sum = 0.0

    def update_origin(self, logits, truth, mask):
        logits = logits.detach().cpu()
        truth = np.asarray(truth, dtype=np.int64)
        mask = np.asarray(mask, dtype=bool)
        if truth.shape != tuple(logits.shape[:2]) or mask.shape != truth.shape:
            raise ValueError("origin logits, truth, and mask shapes do not align")
        valid = mask & (truth >= 0)
        prediction = logits.argmax(dim=-1).numpy()
        np.add.at(self.origin_confusion, (truth[valid], prediction[valid]), 1)
        log_probability = torch.log_softmax(logits, dim=-1).numpy()
        rows, columns = np.nonzero(valid)
        self.origin_nll -= float(log_probability[
            rows, columns, truth[rows, columns]].sum(dtype=np.float64))

    def update_pair(self, logits, truth, mask, jet_labels):
        logits = logits.detach().cpu().numpy().astype(np.float64, copy=False)
        truth = np.asarray(truth, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        jet_labels = np.asarray(jet_labels)
        if truth.shape != logits.shape or mask.shape != logits.shape[:2]:
            raise ValueError("pair logits, truth, and track mask shapes do not align")
        if jet_labels.shape != (logits.shape[0],):
            raise ValueError("pair jet labels must have one entry per jet")
        tracks = mask.shape[1]
        off_diagonal = ~np.eye(tracks, dtype=bool)[None, :, :]
        valid = (
            mask[:, :, None] & mask[:, None, :] & off_diagonal
            & (truth >= 0))
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))

        for category, target in enumerate((1.0, 0.0)):
            selected = valid & (truth == target)
            values = probability[selected]
            selected_logits = logits[selected]
            self.pair_count[category] += len(values)
            self.pair_sum[category] += values.sum(dtype=np.float64)
            self.pair_square_sum[category] += np.square(values).sum(dtype=np.float64)
            self.pair_above_half[category] += np.count_nonzero(values >= 0.5)
            self.pair_histograms[category] += np.histogram(
                values, bins=PAIR_BIN_EDGES)[0]
            self.pair_bce_sum += np.logaddexp(0.0, selected_logits).sum()
            self.pair_bce_sum -= target * selected_logits.sum()

            for jet_class in range(self.pair_by_jet.shape[0]):
                class_selected = selected & (jet_labels == jet_class)[:, None, None]
                self.pair_by_jet[jet_class, category] += np.histogram(
                    probability[class_selected], bins=PAIR_BIN_EDGES)[0]

    def metrics(self, origin_class_names, jet_class_names):
        origin = _origin_metrics(self.origin_confusion, self.origin_nll)
        origin["class_names"] = list(origin_class_names)
        pair_total = int(self.pair_count.sum())
        pair = {
            "pair_definition": (
                "valid ordered track pairs excluding diagonal self-pairs, "
                "matching src.losses.pair_vertex_loss; match=truth_pair 1, "
                "other=truth_pair 0"),
            "n_pairs": pair_total,
            "includes_diagonal": False,
            "score": "sigmoid(pair_logit)",
            "roc_definition": "match is positive; other is negative",
            "histogram_edges": PAIR_BIN_EDGES.tolist(),
            "match": _score_summary(
                self.pair_count[0], self.pair_sum[0], self.pair_square_sum[0],
                self.pair_above_half[0], self.pair_histograms[0]),
            "other": _score_summary(
                self.pair_count[1], self.pair_sum[1], self.pair_square_sum[1],
                self.pair_above_half[1], self.pair_histograms[1]),
            "binary_cross_entropy": _json_ratio(self.pair_bce_sum, pair_total),
            "auc_histogram_100_bins": _histogram_auc(
                self.pair_histograms[0], self.pair_histograms[1]),
            "histogram_counts": {
                "match": self.pair_histograms[0].astype(int).tolist(),
                "other": self.pair_histograms[1].astype(int).tolist(),
            },
            "by_jet_class_histogram_counts": {
                name: {
                    "match": self.pair_by_jet[index, 0].astype(int).tolist(),
                    "other": self.pair_by_jet[index, 1].astype(int).tolist(),
                }
                for index, name in enumerate(jet_class_names)
            },
        }
        return {"track_origin": origin, "track_pair": pair}


def _plot_origin(metrics, output, class_names):
    plt = _pyplot()
    matrix = np.asarray(metrics["confusion_matrix_row_normalized"], dtype=float)
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("Prediction")
    axis.set_ylabel("Truth")
    axis.set_title("Parallel track-origin confusion matrix (row-normalized)")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                      color="white" if value >= 0.5 else "black", fontsize=8)
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _plot_pair(metrics, output, jet_class_names):
    plt = _pyplot()
    edges = np.asarray(metrics["histogram_edges"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    match = np.asarray(metrics["histogram_counts"]["match"], dtype=np.float64)
    other = np.asarray(metrics["histogram_counts"]["other"], dtype=np.float64)
    width = np.diff(edges)
    match_density = match / max(match.sum(), 1) / width
    other_density = other / max(other.sum(), 1) / width

    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].step(centers, match_density, where="mid", linewidth=1.7,
                 label=f"match (n={int(match.sum()):,})")
    axes[0].step(centers, other_density, where="mid", linewidth=1.7,
                 label=f"other (n={int(other.sum()):,})")
    axes[0].set_xlabel("Predicted p(match)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Track-pair score comparison")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    by_jet = metrics["by_jet_class_histogram_counts"]
    colours = plt.get_cmap("tab10")
    for index, class_name in enumerate(jet_class_names):
        class_match = np.asarray(by_jet[class_name]["match"], dtype=np.float64)
        class_other = np.asarray(by_jet[class_name]["other"], dtype=np.float64)
        colour = colours(index)
        if class_match.sum():
            axes[1].step(
                centers, class_match / class_match.sum() / width, where="mid",
                color=colour, linestyle="-", linewidth=1.4,
                label=f"{class_name} match")
        if class_other.sum():
            axes[1].step(
                centers, class_other / class_other.sum() / width, where="mid",
                color=colour, linestyle="--", linewidth=1.4,
                label=f"{class_name} other")
    axes[1].set_xlabel("Predicted p(match)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Scores by jet flavour")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    true_positive = np.cumsum(match[::-1]) / max(match.sum(), 1)
    false_positive = np.cumsum(other[::-1]) / max(other.sum(), 1)
    minimum = 1.0 / max(other.sum(), 1)
    auc = metrics["auc_histogram_100_bins"]
    label = "ROC" if auc is None else f"histogram AUC={auc:.4f}"
    axes[2].plot(true_positive, np.clip(false_positive, minimum, None), label=label)
    axes[2].set_xlabel("Match efficiency")
    axes[2].set_ylabel("Other-pair rate")
    axes[2].set_yscale("log")
    axes[2].set_title("Match vs other ROC")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend()

    figure.suptitle("Parallel track-pair evaluation (Y test)", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def _write_origin_csv(path, metrics, class_names):
    counts = metrics["confusion_matrix"]
    normalized = metrics["confusion_matrix_row_normalized"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["truth", "prediction", "count", "row_fraction"])
        for row, truth_name in enumerate(class_names):
            for column, prediction_name in enumerate(class_names):
                writer.writerow([
                    truth_name, prediction_name, counts[row][column],
                    normalized[row][column]])


def _write_pair_csv(path, metrics):
    edges = metrics["histogram_edges"]
    groups = {"all": metrics["histogram_counts"]}
    groups.update(metrics["by_jet_class_histogram_counts"])
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "jet_class", "bin_left", "bin_right", "match_count", "other_count"])
        for jet_class, histograms in groups.items():
            match = histograms["match"]
            other = histograms["other"]
            for index in range(len(match)):
                writer.writerow([
                    jet_class, edges[index], edges[index + 1],
                    match[index], other[index]])


@torch.no_grad()
def evaluate_parallel_auxiliary(study, run, cache, directory, device):
    active_config = active_parallel_config(study, run)
    loader, raw = create_loader(
        active_config, "y_test", shuffle=False, progress=True,
        batch_size=study.cache.get("batch_size", active_config.batch_size),
        fields=(
            "X", "jet_X", "mask", "y", "origin", "truth_pair",
            "source_index"))
    if not np.array_equal(np.asarray(raw["source_index"]), np.asarray(cache.source_index)):
        raise ValueError("auxiliary Y source_index does not match frozen feature cache")

    checkpoint = study.checkpoint(run)
    model_config = checkpoint_config(checkpoint, active_config)
    model = build_parallel(model_config).to(device)
    model.load_state_dict(torch.load(
        checkpoint, map_location=device, weights_only=True))
    model.eval()
    accumulator = AuxiliaryAccumulator(
        active_config.n_origin_classes, active_config.n_jet_classes)

    processed = 0
    for batch in loader:
        values = batch["X"].to(device)
        jet_values = batch["jet_X"].to(device)
        mask = batch["mask"].to(device)
        output = frozen_parallel_outputs(model, values, jet_values, mask)
        accumulator.update_origin(
            output["origin_logits"], batch["origin"].numpy(),
            batch["mask"].numpy())
        accumulator.update_pair(
            output["pair_logits"], batch["truth_pair"].numpy(),
            batch["mask"].numpy(), batch["y"].numpy())
        processed += len(batch["y"])
        print(f"  auxiliary {run.output_name}/y_test: {processed:,}/{len(raw['y']):,}")
        del output, values, jet_values, mask

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metrics = accumulator.metrics(
        active_config.origin_class_names, active_config.jet_class_names)
    directory = Path(directory) / "auxiliary_tasks"
    directory.mkdir(parents=True, exist_ok=True)
    _plot_origin(
        metrics["track_origin"], directory / "origin_confusion_matrix.png",
        active_config.origin_class_names)
    _write_origin_csv(
        directory / "origin_confusion_matrix.csv", metrics["track_origin"],
        active_config.origin_class_names)
    _plot_pair(
        metrics["track_pair"], directory / "pair_score_comparison.png",
        active_config.jet_class_names)
    _write_pair_csv(
        directory / "pair_score_histogram.csv", metrics["track_pair"])
    write_json_atomic(directory / "auxiliary_metrics.json", {
        "version": AUXILIARY_EVALUATION_VERSION,
        "split": "y_test",
        "n_jets": int(len(raw["y"])),
        "experiment_name": study.experiment_name,
        "experiment_config": str(study.path),
        "experiment_config_sha256": study.source_sha256,
        "experiment_markers": study.experiment_markers,
        "parallel_seed": run.seed,
        "parallel_checkpoint": str(checkpoint.resolve()),
        "parallel_checkpoint_sha256": cache.manifest["checkpoint_sha256"],
        "feature_cache_source_index_sha256": cache.manifest["source_index_sha256"],
        "metrics": metrics,
    })
    return {
        **metrics,
        "artifacts": {
            "origin_confusion_matrix_png": "auxiliary_tasks/origin_confusion_matrix.png",
            "origin_confusion_matrix_csv": "auxiliary_tasks/origin_confusion_matrix.csv",
            "pair_score_comparison_png": "auxiliary_tasks/pair_score_comparison.png",
            "pair_score_histogram_csv": "auxiliary_tasks/pair_score_histogram.csv",
            "metrics_json": "auxiliary_tasks/auxiliary_metrics.json",
        },
    }
