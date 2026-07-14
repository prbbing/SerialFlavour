"""
Loss functions: origin class weights, vertex-fit loss, pair-vertex loss.
"""
import numpy as np
import torch
import torch.nn.functional as F


# ===========================================================================
# compute_origin_class_weights — inverse-frequency class weights for
# the track-origin cross-entropy loss.  Rare classes (Fake, From tau)
# would be drowned out by Primary/Pileup without re-weighting.
# Weights are clipped at 20× median to avoid extreme values.
# ===========================================================================
def compute_origin_class_weights(train_data, n_origin_classes, device):
    _origin_flat = train_data["origin"].ravel()         # (N*K,)
    _origin_flat = _origin_flat[_origin_flat >= 0]       # drop padding (-1)
    _counts = np.bincount(_origin_flat, minlength=n_origin_classes).astype(np.float64)
    _counts = np.maximum(_counts, 1)                     # avoid divide-by-zero
    _inv_freq = 1.0 / _counts                            # 1 / count
    _inv_freq = _inv_freq / _inv_freq.mean()             # normalise to mean = 1
    _inv_freq = np.clip(_inv_freq, None, 20.0 * np.median(_inv_freq))  # cap
    return torch.tensor(_inv_freq, dtype=torch.float32).to(device)


# ===========================================================================
# vertex_loss_fn — smooth-L1 loss over fitted vs true vertex coordinates.
#
# Works in log1p-compressed space to handle large dynamic range of Lxy / dz.
# For dz, the sign is preserved via sign(log1p(|dz|)) * sign(z).
# Only active fit coordinates (fit_lxy / fit_dz) contribute.
#
# model: staged_origin_vertex_jet
# ===========================================================================
def vertex_loss_fn(lxy_pred, dz_pred, vtx_lxy, vtx_dz, vtx_valid,
                   fit_lxy=True, fit_dz=True, return_components=False):
    total      = lxy_pred.new_tensor(0.0)
    lxy_total  = lxy_pred.new_tensor(0.0)
    dz_total   = lxy_pred.new_tensor(0.0)
    for leg in range(lxy_pred.shape[-1]):
        v = vtx_valid[:, leg]
        if not v.any():
            continue
        if fit_lxy:
            lp = torch.log1p(lxy_pred[v, leg].clamp(min=0))
            lt = torch.log1p(vtx_lxy[v, leg].clamp(min=0))
            loss = F.smooth_l1_loss(lp, lt)
            total     = total + loss
            lxy_total = lxy_total + loss
        if fit_dz:
            zp = torch.log1p(dz_pred[v, leg].abs()) * dz_pred[v, leg].sign()
            zt = torch.log1p(vtx_dz[v, leg].abs())  * vtx_dz[v, leg].sign()
            loss = F.smooth_l1_loss(zp, zt)
            total    = total + loss
            dz_total = dz_total + loss
    if return_components:
        return total, lxy_total, dz_total
    return total


# ===========================================================================
# pair_vertex_loss — binary cross-entropy for track-pair compatibility.
#
# For each pair of valid tracks (i, j): does pair_logits[i,j] indicate
# the same vertex?  target = 1 (same), 0 (different), -1 (ignore).
#
# model: parallel_origin_vertex_jet
# ===========================================================================
def pair_vertex_loss(pair_logits, pair_target, mask):
    B, K, _ = pair_logits.shape
    valid_pair   = mask.unsqueeze(2) & mask.unsqueeze(1)    # (B, K, K)
    valid_target = valid_pair & (pair_target >= 0)           # non-ignore
    if not valid_target.any():
        return pair_logits.new_tensor(0.0)
    return F.binary_cross_entropy_with_logits(
        pair_logits[valid_target],
        pair_target[valid_target],
    )
