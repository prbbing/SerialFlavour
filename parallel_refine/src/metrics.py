"""Jet-flavour metrics and auditable prediction artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _binary_auc(binary, score):
    """Return tie-aware Mann-Whitney AUC without a scikit-learn dependency."""
    binary = np.asarray(binary, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positives = int(np.count_nonzero(binary == 1))
    negatives = int(np.count_nonzero(binary == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both signal and background samples")

    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        stop = start + 1
        while stop < len(score) and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = ranks[binary == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2)
        / (positives * negatives))


def b_discriminant(probabilities):
    denominator = 0.2 * probabilities[:, 1] + 0.8 * probabilities[:, 2]
    return np.log(
        np.clip(probabilities[:, 0], 1e-12, None)
        / np.clip(denominator, 1e-12, None))


def c_discriminant(probabilities):
    denominator = 0.3 * probabilities[:, 0] + 0.7 * probabilities[:, 2]
    return np.log(
        np.clip(probabilities[:, 1], 1e-12, None)
        / np.clip(denominator, 1e-12, None))


def _binary_metrics(y, score, signal, background, efficiency, prefix):
    selected = np.isin(y, (signal, background))
    binary = (y[selected] == signal).astype(np.int64)
    values = score[selected]
    signal_values = values[binary == 1]
    threshold = float(np.quantile(signal_values, 1.0 - efficiency))
    passed = int(np.count_nonzero(values[binary == 0] >= threshold))
    background_count = int(np.count_nonzero(binary == 0))
    return {
        f"{prefix}_auc": _binary_auc(binary, values),
        f"{prefix}_rejection": (
            None if passed == 0 else float(background_count / passed)),
        f"{prefix}_zero_background_pass": passed == 0,
        f"{prefix}_background_pass": passed,
        f"{prefix}_background_total": background_count,
    }


def probability_metrics(y, probabilities):
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape != (len(y), 3):
        raise ValueError("probabilities must have shape (n_jets, 3)")
    if np.any((y < 0) | (y > 2)):
        raise ValueError("labels must be in {0, 1, 2}")
    probabilities = np.clip(probabilities, 1e-12, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    result = {
        "accuracy": float(np.mean(y == probabilities.argmax(1))),
        "cross_entropy": float(-np.mean(np.log(probabilities[np.arange(len(y)), y]))),
    }
    result.update(_binary_metrics(
        y, b_discriminant(probabilities), 0, 1, 0.70, "b_vs_c_at_b70"))
    result.update(_binary_metrics(
        y, b_discriminant(probabilities), 0, 2, 0.70, "b_vs_light_at_b70"))
    result.update(_binary_metrics(
        y, c_discriminant(probabilities), 1, 0, 0.30, "c_vs_b_at_c30"))
    result.update(_binary_metrics(
        y, c_discriminant(probabilities), 1, 2, 0.30, "c_vs_light_at_c30"))
    return result


def write_prediction_result(
        directory, *, model_name, split, y, probabilities,
        source_index, event_number, metadata=None, auxiliary_metrics=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(
        directory / "test_predictions.npz", y=y, probabilities=probabilities,
        source_index=source_index, event_number=event_number)
    payload = {
        "model": model_name,
        "split": split,
        "n_jets": int(len(y)),
        "class_counts": {
            "b-jet": int(np.count_nonzero(y == 0)),
            "c-jet": int(np.count_nonzero(y == 1)),
            "light-jet": int(np.count_nonzero(y == 2)),
        },
        "metric_definition": {
            "b_discriminant": "log(p_b / (0.2*p_c + 0.8*p_light))",
            "b_working_point": 0.70,
            "c_discriminant": "log(p_c / (0.3*p_b + 0.7*p_light))",
            "c_working_point": 0.30,
            "zero_background_pass": "rejection stored as null finite-sample lower bound",
        },
        "metrics": probability_metrics(y, probabilities),
        "metadata": metadata or {},
    }
    if auxiliary_metrics is not None:
        payload["auxiliary_tasks"] = auxiliary_metrics
    (directory / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return payload
