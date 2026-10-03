"""Regression metrics in the caller's physical target units."""

import numpy as np


def regression_metrics(truth, prediction):
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if truth.shape != prediction.shape or not truth.size or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("metrics require matching, nonempty, finite arrays")
    residual = prediction - truth
    denominator = np.square(truth - truth.mean()).sum()
    return {"mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.square(residual).mean())),
            "r_squared": float(1 - np.square(residual).sum() / denominator) if denominator > 0 else None}
