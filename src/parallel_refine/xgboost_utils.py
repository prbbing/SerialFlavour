"""Stable XGBoost Booster persistence and prediction helpers.

The sklearn wrapper is convenient for fitting with early stopping, but some
XGBoost/sklearn version combinations fail in ``XGBClassifier.save_model``
while querying the optional sklearn ``_estimator_type`` mixin.  The trained
Booster itself is complete and versioned JSON, so persist and evaluate that
object directly.
"""

from __future__ import annotations

import numpy as np


def booster_probabilities(booster, features, *, iteration_range):
    """Return three-class probabilities through a CPU DMatrix.

    This deliberately uses ``Booster.predict`` rather than sklearn
    ``predict_proba``.  With a CUDA Booster and NumPy cache, XGBoost transfers
    the DMatrix through its normal prediction path instead of first trying a
    device-mismatched inplace prediction.
    """
    import xgboost

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("XGBoost features must be a two-dimensional array")
    probabilities = np.asarray(booster.predict(
        xgboost.DMatrix(values), iteration_range=iteration_range),
        dtype=np.float32)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError(
            "expected XGBoost multi:softprob output with shape (n_jets, 3), "
            f"got {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("XGBoost returned non-finite probabilities")
    return probabilities
