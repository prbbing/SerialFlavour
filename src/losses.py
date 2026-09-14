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


def pair_vertex_loss_dense(pair_logits, pair_target, mask):
    """Equivalent pair BCE written without boolean indexing.

    Computing the mean over a dense, masked tensor avoids the data-dependent
    gather that both stalls eager execution and forces graph breaks under
    ``torch.compile``. Mathematically this is the same mean as
    :func:`pair_vertex_loss`; only the order of the floating-point sum changes.
    """
    valid = (
        mask.unsqueeze(2) & mask.unsqueeze(1)
        & ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device).unsqueeze(0)
        & (pair_target >= 0))
    per_pair = F.binary_cross_entropy_with_logits(
        pair_logits, pair_target.clamp(min=0), reduction="none")
    return (per_pair * valid).sum() / valid.sum().clamp(min=1)
