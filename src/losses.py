"""Loss helpers for the Parallel jet/origin/pair objectives."""

import numpy as np
import torch
import torch.nn.functional as F


def compute_origin_class_weights(train_data, n_origin_classes, device):
    """Return capped inverse-frequency weights for valid origin labels."""
    labels = train_data["origin"].ravel()
    labels = labels[labels >= 0]
    counts = np.bincount(labels, minlength=n_origin_classes).astype(np.float64)
    inverse = 1.0 / np.maximum(counts, 1)
    inverse /= inverse.mean()
    inverse = np.clip(inverse, None, 20.0 * np.median(inverse))
    return torch.tensor(inverse, dtype=torch.float32, device=device)


def pair_vertex_loss(pair_logits, pair_target, mask):
    """Binary cross-entropy over valid, non-ignored track pairs."""
    valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
    selected = valid_pair & (pair_target >= 0)
    if not selected.any():
        return pair_logits.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(pair_logits[selected], pair_target[selected])
