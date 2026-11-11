"""Standalone rejection scans for locked-Y upstream evaluations."""

import hashlib
import json

import numpy as np

from src.parallel_refine.plotting import (
    _JET_CLASS_NAMES, _REJECTION_SPECS, _finite_or_none, _rejection_curve,
)


def build_scan(directory):
    """Build a standalone scan from a completed upstream Y evaluation."""
    prediction = directory / "test_predictions.npz"
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (directory / "evaluation_manifest.json").read_text(encoding="utf-8"))
    with np.load(prediction, allow_pickle=False) as data:
        y = data["y"]
        probabilities = data["probabilities"]
        if (probabilities.shape != (len(y), 3)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(1), 1, atol=1e-5)
                or np.any(~np.isin(y, [0, 1, 2]))):
            raise ValueError(f"Invalid predictions: {prediction}")
        identity = {
            key: hashlib.sha256(np.ascontiguousarray(data[key]).tobytes()).hexdigest()
            for key in ("y", "source_index", "event_number")
        }
    if metrics["n_jets"] != len(y) or manifest["split"] != "y_test":
        raise ValueError(f"Prediction/metadata mismatch: {directory}")
    payload = {
        "schema_version": "parallel_rejection_scan_v1",
        "model": "parallel",
        "study_name": manifest["study_name"],
        "parallel_seed": manifest["parallel_seed"],
        "split": "y_test",
        "n_jets": len(y),
        "class_counts": metrics["class_counts"],
        "definition": {
            "signal_threshold": "Y-test signal-score quantile at each target efficiency; score >= threshold.",
            "quantile_method": "numpy.quantile default linear interpolation",
            "rejection": "background_total / background_pass",
            "zero_background_pass": "parallel_rejection is null rather than capped",
            "seed_aggregation": "none; one upstream seed per file",
            "b_discriminant": "log(p_b / (0.2*p_c + 0.8*p_light))",
            "c_discriminant": "log(p_c / (0.3*p_b + 0.7*p_light))",
            "probability_clip_min": 1e-12,
            "efficiency_grids": {
                "b-jet": {"start": 0.60, "stop": 1.00, "step": 0.005, "points": 81},
                "c-jet": {"start": 0.10, "stop": 0.40, "step": 0.005, "points": 61},
            },
        },
        "provenance": {
            "prediction_file": str(prediction.resolve()),
            "prediction_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
            "evaluation_manifest": str((directory / "evaluation_manifest.json").resolve()),
            "parallel_checkpoint": manifest["parallel_checkpoint"],
            "experiment_config_sha256": manifest["experiment_config_sha256"],
            "split_index_sha256": manifest["feature_cache"]["split_index_sha256"],
            "sample_array_sha256": identity,
            "scan_implementation": "src.parallel_refine.plotting._rejection_curve",
        },
        "curves": [],
    }
    for signal, backgrounds, efficiencies, working_point, discriminant in _REJECTION_SPECS:
        for background in backgrounds:
            curve = _rejection_curve(
                y, probabilities, signal, background, efficiencies, discriminant)
            for index, efficiency in enumerate(efficiencies):
                payload["curves"].append({
                    "signal": _JET_CLASS_NAMES[signal],
                    "background": _JET_CLASS_NAMES[background],
                    "target_signal_efficiency": float(efficiency),
                    "parallel_actual_signal_efficiency": _finite_or_none(curve["actual_signal_efficiency"][index]),
                    "parallel_threshold": _finite_or_none(curve["threshold"][index]),
                    "parallel_rejection": _finite_or_none(curve["rejection"][index]),
                    "parallel_background_pass": int(curve["background_pass"][index]),
                    "parallel_background_total": curve["background_total"],
                    "parallel_zero_background_pass": bool(curve["background_pass"][index] == 0),
                })
            wp_index = int(np.flatnonzero(np.isclose(efficiencies, working_point))[0])
            prefix = ("b_vs_" + ("c" if background == 1 else "light") + "_at_b70"
                      if signal == 0 else
                      "c_vs_" + ("b" if background == 0 else "light") + "_at_c30")
            for suffix, value in (
                    ("background_pass", int(curve["background_pass"][wp_index])),
                    ("background_total", curve["background_total"]),
                    ("rejection", _finite_or_none(curve["rejection"][wp_index]))):
                if metrics["metrics"][prefix + "_" + suffix] != value:
                    raise ValueError(f"Working-point mismatch: {directory}: {prefix}_{suffix}")
    if len(payload["curves"]) != 284:
        raise ValueError(f"Incomplete scan: {directory}")
    return payload
