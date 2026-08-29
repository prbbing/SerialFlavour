"""Loss helpers for the Parallel jet/origin/pair objectives."""

import torch
import torch.nn.functional as F


def classification_class_weights(weights, n_classes, device):
    """Materialise a fixed, configuration-owned classification weight vector."""
    if len(weights) != n_classes:
        raise ValueError(
            f"expected {n_classes} class weights, received {len(weights)}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def pair_vertex_loss(pair_logits, pair_target, mask):
    """Binary cross-entropy over valid, non-self, non-ignored track pairs."""
    valid_pair = mask.unsqueeze(2) & mask.unsqueeze(1)
    tracks = mask.shape[1]
    off_diagonal = ~torch.eye(
        tracks, dtype=torch.bool, device=mask.device).unsqueeze(0)
    selected = valid_pair & off_diagonal & (pair_target >= 0)
    if not selected.any():
        return pair_logits.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(pair_logits[selected], pair_target[selected])
