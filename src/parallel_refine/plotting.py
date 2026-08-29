"""Plots for the Parallel training and locked Y-evaluation workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np


JET_CLASS_NAMES = ("b-jet", "c-jet", "light-jet")
JET_COLOURS = {"b-jet": "#1f77b4", "c-jet": "#ff7f0e", "light-jet": "#2ca02c"}


def _pyplot():
    """Load a non-interactive backend only when plots are requested."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_input_variables(features, mask, labels, field_names, output_directory):
    """Plot valid track-feature distributions split by jet flavour."""
    plt = _pyplot()
    output_directory = Path(output_directory)
    flat_features = np.asarray(features).reshape(-1, len(field_names))
    valid = np.asarray(mask).ravel().astype(bool)
    repeated_labels = np.repeat(np.asarray(labels), np.asarray(mask).shape[1])
    columns = min(len(field_names), 4)
    rows = (len(field_names) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    figure.suptitle("Parallel input variables by jet flavour", fontweight="bold")
    for index, name in enumerate(field_names):
        values = flat_features[:, index]
        finite_valid = values[valid & np.isfinite(values)]
        clip = np.percentile(np.abs(finite_valid), 99) if len(finite_valid) else 1.0
        clip = max(float(clip), 1e-12)
        for class_index, class_name in enumerate(JET_CLASS_NAMES):
            selected = valid & (repeated_labels == class_index)
            axes[index].hist(values[selected], bins=80, range=(-clip, clip),
                             histtype="step", density=True, linewidth=1.5,
                             color=JET_COLOURS[class_name], label=class_name)
        axes[index].set_title(name, fontsize=8)
        axes[index].set_xlabel(name, fontsize=7)
        axes[index].set_ylabel("Density", fontsize=7)
    axes[0].legend(fontsize=7)
    for axis in axes[len(field_names):]:
        axis.set_visible(False)
    figure.tight_layout()
    figure.savefig(output_directory / "input_variables.png", dpi=150,
                   bbox_inches="tight")
    plt.close(figure)


def plot_training_history(history, output_directory):
    """Plot Parallel total, jet, origin, and pair losses from saved history."""
    if not history:
        return
    plt = _pyplot()
    output_directory = Path(output_directory)
    epochs = np.arange(1, len(history) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    figure.suptitle("Parallel training history", fontweight="bold")
    for prefix, colour, label in (("train", "#1f77b4", "train"),
                                  ("val", "#d62728", "validation")):
        for key, style, component in (("total", "-", "total"),
                                      ("jet", "--", "jet CE"),
                                      ("origin", ":", "origin CE"),
                                      ("pair", "-.", "pair BCE")):
            values = [row.get(f"{prefix}_{key}") for row in history]
            if all(value is not None for value in values):
                loss_axis.plot(epochs, values, color=colour, linestyle=style,
                               label=f"{label} {component}")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_yscale("log")
    loss_axis.legend(fontsize=7)
    loss_axis.grid(True, which="both", linestyle="--", alpha=0.3)
    for prefix, colour, label in (("train", "#1f77b4", "train"),
                                  ("val", "#d62728", "validation")):
        values = [row.get(f"{prefix}_jet_accuracy") for row in history]
        if all(value is not None for value in values):
            accuracy_axis.plot(epochs, values, color=colour, label=label)
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Jet accuracy")
    accuracy_axis.set_ylim(0, 1)
    accuracy_axis.legend(fontsize=8)
    accuracy_axis.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_directory / "training_history.png", dpi=150,
                   bbox_inches="tight")
    plt.close(figure)


def _roc_curve(labels, scores):
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order].astype(bool)
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    true_positive = np.r_[0, np.cumsum(labels)] / positives
    false_positive = np.r_[0, np.cumsum(~labels)] / negatives
    return false_positive, true_positive


def _plot_discriminant(plt, probabilities, labels, signal, weights, output, stem):
    background = [index for index in range(3) if index != signal]
    denominator = probabilities[:, background] @ np.asarray(weights)
    score = np.log(np.clip(probabilities[:, signal], 1e-12, None)
                   / np.clip(denominator, 1e-12, None))
    signal_name = JET_CLASS_NAMES[signal]
    figure, (distribution_axis, roc_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    figure.suptitle(f"{signal_name} discriminant on locked Y", fontweight="bold")
    finite = np.isfinite(score)
    limit = max(float(np.percentile(np.abs(score[finite]), 99)), 1e-12)
    for index, name in enumerate(JET_CLASS_NAMES):
        distribution_axis.hist(score[finite & (labels == index)], bins=80,
                               range=(-limit, limit), density=True,
                               histtype="step", linewidth=1.5,
                               color=JET_COLOURS[name], label=name)
    distribution_axis.set_xlabel("log(signal probability / weighted background)")
    distribution_axis.set_ylabel("Density")
    distribution_axis.legend(fontsize=7)
    for index in background:
        selected = (labels == signal) | (labels == index)
        curve = _roc_curve(labels[selected] == signal, score[selected])
        if curve is not None:
            false_positive, true_positive = curve
            roc_axis.plot(true_positive, false_positive, linewidth=1.5,
                          color=JET_COLOURS[JET_CLASS_NAMES[index]],
                          label=f"vs {JET_CLASS_NAMES[index]}")
    roc_axis.set_xlabel(f"{signal_name} efficiency")
    roc_axis.set_ylabel("Background rate")
    roc_axis.set_yscale("log")
    roc_axis.set_ylim(bottom=1e-4, top=1.0)
    roc_axis.legend(fontsize=8)
    roc_axis.grid(True, which="both", linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(Path(output) / f"{stem}_discriminant_roc.png", dpi=150,
                   bbox_inches="tight")
    plt.close(figure)


def plot_jet_evaluation(labels, probabilities, output_directory):
    """Plot class probabilities and b/c discriminant ROC curves on locked Y."""
    plt = _pyplot()
    output_directory = Path(output_directory)
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    figure.suptitle("Jet output probabilities on locked Y", fontweight="bold")
    for predicted, axis in enumerate(axes):
        for truth, name in enumerate(JET_CLASS_NAMES):
            axis.hist(probabilities[labels == truth, predicted], bins=50,
                      range=(0, 1), density=True, histtype="step", linewidth=1.5,
                      color=JET_COLOURS[name], label=name)
        axis.set_title(f"P({JET_CLASS_NAMES[predicted]})")
        axis.set_xlabel("Probability")
        axis.set_ylabel("Density")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_directory / "output_probabilities.png", dpi=150,
                   bbox_inches="tight")
    plt.close(figure)
    _plot_discriminant(plt, probabilities, labels, 0, (0.2, 0.8), output_directory,
                       "b")
    _plot_discriminant(plt, probabilities, labels, 1, (0.3, 0.7), output_directory,
                       "c")
