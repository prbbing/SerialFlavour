"""
Training and validation loops with model-dispatch for loss/metrics.

The core learning loop (run_training) delegates to model-format-agnostic
train_epoch / validate_epoch.  These detect the model's output keys at
runtime — "lxy_pred" triggers the staged-vertex-fit loss branch;
"pair_logits" triggers the pair-compatibility loss branch.  This design
lets the same training loop drive both staged_origin_vertex_jet and
parallel_origin_vertex_jet without modification.
"""
import torch
import numpy as np
import os
import csv
from torch.utils.tensorboard import SummaryWriter

from .losses import vertex_loss_fn, pair_vertex_loss


# ===========================================================================
# train_epoch — one full pass over the training set.
# ===========================================================================
def train_epoch(model, dataloader, optimiser, criterion_jet, criterion_origin,
                n_origin_classes, lambda_jet, lambda_origin, lambda_vertex,
                fit_lxy, fit_dz, device):
    model.train()
    tot_loss = tot_jet = tot_origin = tot_vertex = 0.0
    tot_lxy_vtx = tot_dz_vtx = 0.0
    tot_refine = tot_vtx_weight = 0.0
    n_total = 0

    for X_b, mask_b, y_b, origin_b, lxy_b, dz_b, vvalid_b, pair_b in dataloader:
        # move batch to target device (CPU or GPU)
        X_b, mask_b, y_b   = X_b.to(device), mask_b.to(device), y_b.to(device)
        origin_b           = origin_b.to(device)
        lxy_b              = lxy_b.to(device)
        dz_b               = dz_b.to(device)
        vvalid_b           = vvalid_b.to(device)
        pair_b             = pair_b.to(device)

        optimiser.zero_grad()
        out = model(X_b, mask_b)

        # ── core losses (shared by all models) ──────────────────────────
        jet_loss    = criterion_jet(out["jet_logits"], y_b)
        origin_loss = criterion_origin(
            out["origin_logits"].reshape(-1, n_origin_classes),
            origin_b.reshape(-1))

        # ── model-specific vertex / pair loss ───────────────────────────
        # model: staged_origin_vertex_jet — Lxy/dz vertex-fit loss
        if "lxy_pred" in out:
            vtx_loss, lxy_vtx_loss, dz_vtx_loss = vertex_loss_fn(
                out["lxy_pred"], out["dz_pred"],
                lxy_b, dz_b, vvalid_b,
                fit_lxy=fit_lxy, fit_dz=fit_dz,
                return_components=True)
            tot_lxy_vtx += lxy_vtx_loss.item() * len(y_b)
            tot_dz_vtx  += dz_vtx_loss.item()  * len(y_b)
        # model: parallel_origin_vertex_jet — pair-vertexing BCE loss
        elif "pair_logits" in out:
            vtx_loss = pair_vertex_loss(out["pair_logits"], pair_b, mask_b)
        else:
            vtx_loss = out["jet_logits"].new_tensor(0.0)

        # ── weighted sum & step ────────────────────────────────────────
        loss = lambda_jet * jet_loss + lambda_origin * origin_loss + lambda_vertex * vtx_loss
        loss.backward()
        optimiser.step()

        # track running totals (weighted by batch size)
        tot_loss   += loss.item()        * len(y_b)
        tot_jet    += jet_loss.item()    * len(y_b)
        tot_origin += origin_loss.item() * len(y_b)
        tot_vertex += vtx_loss.item()    * len(y_b)
        n_total    += len(y_b)

        # per-batch refine / vtx_weight diagnostics (valid tracks only)
        if "refine" in out and "vtx_weight" in out:
            valid_3d = mask_b.unsqueeze(-1).expand_as(out["refine"])
            tot_refine      += out["refine"][valid_3d].float().mean().item()     * len(y_b)
            tot_vtx_weight  += out["vtx_weight"][valid_3d].float().mean().item() * len(y_b)

    return (tot_loss / n_total, tot_jet / n_total,
            tot_origin / n_total, tot_vertex / n_total,
            tot_lxy_vtx / max(n_total, 1), tot_dz_vtx / max(n_total, 1),
            tot_refine / max(n_total, 1), tot_vtx_weight / max(n_total, 1))


# ===========================================================================
# validate_epoch — one full pass over the validation set (no gradients).
# Collects predictions for later evaluation & plotting.
# ===========================================================================
@torch.no_grad()
def validate_epoch(model, dataloader, criterion_jet, criterion_origin,
                   n_origin_classes, lambda_jet, lambda_origin, lambda_vertex,
                   fit_lxy, fit_dz, device):
    model.eval()
    val_loss, val_jet_loss, val_origin_loss, val_vertex_loss = 0.0, 0.0, 0.0, 0.0
    val_lxy_vtx_loss, val_dz_vtx_loss = 0.0, 0.0
    correct, origin_correct, origin_total, n_total = 0, 0, 0, 0
    all_preds, all_true, all_probs = [], [], []
    all_origin_preds, all_origin_true = [], []

    # model: staged_origin_vertex_jet — Lxy/dz metrics buffers
    all_lxy_pred, all_lxy_true, all_dz_pred, all_dz_true, all_vtx_valid = [], [], [], [], []
    all_vtx_weight, all_origin_full, all_mask_full = [], [], []
    all_leg_origin_probs, all_gate, all_refine = [], [], []
    # model: parallel_origin_vertex_jet — pair metrics buffers
    all_pair_logits, all_pair_target, all_pair_mask = [], [], []

    for X_b, mask_b, y_b, origin_b, lxy_b, dz_b, vvalid_b, pair_b in dataloader:
        # move to device
        X_b, mask_b, y_b = X_b.to(device), mask_b.to(device), y_b.to(device)
        origin_b   = origin_b.to(device)
        lxy_b      = lxy_b.to(device)
        dz_b       = dz_b.to(device)
        vvalid_b   = vvalid_b.to(device)
        pair_b     = pair_b.to(device)

        out = model(X_b, mask_b)

        # ── core losses ─────────────────────────────────────────────────
        jet_loss    = criterion_jet(out["jet_logits"], y_b)
        origin_loss = criterion_origin(
            out["origin_logits"].reshape(-1, n_origin_classes),
            origin_b.reshape(-1))

        # ── model-specific vertex / pair loss ───────────────────────────
        if "lxy_pred" in out:
            # model: staged_origin_vertex_jet
            vtx_loss, lxy_vtx_loss, dz_vtx_loss = vertex_loss_fn(
                out["lxy_pred"], out["dz_pred"],
                lxy_b, dz_b, vvalid_b,
                fit_lxy=fit_lxy, fit_dz=fit_dz,
                return_components=True)
            val_lxy_vtx_loss += lxy_vtx_loss.item() * len(y_b)
            val_dz_vtx_loss  += dz_vtx_loss.item()  * len(y_b)
        elif "pair_logits" in out:
            # model: parallel_origin_vertex_jet
            vtx_loss = pair_vertex_loss(out["pair_logits"], pair_b, mask_b)
        else:
            vtx_loss = out["jet_logits"].new_tensor(0.0)

        loss = lambda_jet * jet_loss + lambda_origin * origin_loss + lambda_vertex * vtx_loss
        val_loss        += loss.item()        * len(y_b)
        val_jet_loss    += jet_loss.item()    * len(y_b)
        val_origin_loss += origin_loss.item() * len(y_b)
        val_vertex_loss += vtx_loss.item()    * len(y_b)

        # ── jet classification predictions ──────────────────────────────
        preds = out["jet_logits"].argmax(dim=1)
        correct += (preds == y_b).sum().item()
        all_preds.append(preds.cpu())
        all_true.append(y_b.cpu())
        all_probs.append(torch.softmax(out["jet_logits"], dim=1).cpu())

        # ── track-origin predictions (valid tracks only) ────────────────
        origin_preds = out["origin_logits"].argmax(dim=-1)
        origin_mask  = origin_b >= 0   # true tracks only
        origin_correct += ((origin_preds == origin_b) & origin_mask).sum().item()
        origin_total   += origin_mask.sum().item()
        all_origin_preds.append(origin_preds[origin_mask].cpu())
        all_origin_true.append(origin_b[origin_mask].cpu())

        # ── model-specific metric collection ────────────────────────────
        if "lxy_pred" in out:
            # model: staged_origin_vertex_jet
            all_lxy_pred.append(out["lxy_pred"].cpu())
            all_lxy_true.append(lxy_b.cpu())
            all_dz_pred.append(out["dz_pred"].cpu())
            all_dz_true.append(dz_b.cpu())
            all_vtx_valid.append(vvalid_b.cpu())
        if "vtx_weight" in out:
            # model: staged_origin_vertex_jet — track-to-vertex assignment
            all_vtx_weight.append(out["vtx_weight"].cpu())
            all_origin_full.append(origin_b.cpu())
            all_mask_full.append(mask_b.cpu())
        if "leg_origin_probs" in out:
            all_leg_origin_probs.append(out["leg_origin_probs"].cpu())
            all_gate.append(out["gate"].cpu())
            all_refine.append(out["refine"].cpu())
        if "pair_logits" in out:
            # model: parallel_origin_vertex_jet
            all_pair_logits.append(out["pair_logits"].cpu())
            all_pair_target.append(pair_b.cpu())
            all_pair_mask.append(mask_b.cpu())

        n_total += len(y_b)

    # ── aggregate metrics ────────────────────────────────────────────────
    val_loss          /= n_total
    val_jet_loss      /= n_total
    val_origin_loss   /= n_total
    val_vertex_loss   /= n_total
    val_lxy_vtx_loss  /= n_total
    val_dz_vtx_loss   /= n_total
    val_acc          = correct / n_total
    val_origin_acc   = origin_correct / max(origin_total, 1)

    pred_arrays = {
        "all_preds":       torch.cat(all_preds).numpy(),
        "all_true":        torch.cat(all_true).numpy(),
        "all_probs":       torch.cat(all_probs).numpy(),
        "origin_preds":    torch.cat(all_origin_preds).numpy(),
        "origin_true":     torch.cat(all_origin_true).numpy(),
    }

    # ── attach model-specific metrics to pred_arrays ────────────────────
    # model: staged_origin_vertex_jet
    if all_lxy_pred:
        pred_arrays["lxy_pred"]  = torch.cat(all_lxy_pred).numpy()
        pred_arrays["lxy_true"]  = torch.cat(all_lxy_true).numpy()
        pred_arrays["dz_pred"]   = torch.cat(all_dz_pred).numpy()
        pred_arrays["dz_true"]   = torch.cat(all_dz_true).numpy()
        pred_arrays["vtx_valid"] = torch.cat(all_vtx_valid).numpy().astype(bool)
    if all_vtx_weight:
        pred_arrays["vtx_weight"]  = torch.cat(all_vtx_weight).numpy()
        pred_arrays["origin_full"] = torch.cat(all_origin_full).numpy()
        pred_arrays["mask_full"]   = torch.cat(all_mask_full).numpy().astype(bool)
    if all_leg_origin_probs:
        pred_arrays["leg_origin_probs"] = torch.cat(all_leg_origin_probs).numpy()
        pred_arrays["gate"]             = torch.cat(all_gate).numpy()
        pred_arrays["refine"]           = torch.cat(all_refine).numpy()
    # model: parallel_origin_vertex_jet
    if all_pair_logits:
        pred_arrays["pair_logits"] = torch.cat(all_pair_logits).numpy()
        pred_arrays["pair_target"] = torch.cat(all_pair_target).numpy()
        pred_arrays["pair_mask"]   = torch.cat(all_pair_mask).numpy().astype(bool)

    return (val_loss, val_jet_loss, val_origin_loss, val_vertex_loss,
            val_lxy_vtx_loss, val_dz_vtx_loss,
            val_acc, val_origin_acc, pred_arrays)


# ===========================================================================
# _measure_task_gradients — per-task gradient norms & cosine conflicts.
# ===========================================================================
def _measure_task_gradients(model, X_b, mask_b, y_b, origin_b,
                             lxy_b, dz_b, vvalid_b, pair_b,
                             criterion_jet, criterion_origin,
                             n_origin_classes, fit_lxy, fit_dz):
    """Run 3 independent backward passes on a single forward graph and
    record per-parameter-group gradient L2 norms, plus pairwise cosine
    similarity on the shared encoder (encoder1 for staged, encoder
    for parallel).

    Returns a dict of measured values or {} for unsupported models.
    """
    _m = model.module if hasattr(model, "module") else model

    out = _m(X_b, mask_b)

    jet_loss = criterion_jet(out["jet_logits"], y_b)
    origin_loss = criterion_origin(
        out["origin_logits"].reshape(-1, n_origin_classes), origin_b.reshape(-1))

    if "lxy_pred" in out:
        from .losses import vertex_loss_fn
        vtx_loss = vertex_loss_fn(out["lxy_pred"], out["dz_pred"],
                                  lxy_b, dz_b, vvalid_b,
                                  fit_lxy=fit_lxy, fit_dz=fit_dz)
    elif "pair_logits" in out:
        from .losses import pair_vertex_loss
        vtx_loss = pair_vertex_loss(out["pair_logits"], pair_b, mask_b)
    else:
        return {}

    # ── identify parameter groups ──────────────────────────────────────
    # staged model groups
    has_staged = hasattr(_m, "encoder1")
    if has_staged:
        enc1_params   = list(_m.input_proj1.parameters()) + list(_m.encoder1.parameters())
        enc2_params   = (list(_m.input_proj2.parameters()) + list(_m.encoder2.parameters())
                         if hasattr(_m, "input_proj2") else [])
        enc3_params   = list(_m.input_proj3.parameters()) + list(_m.encoder3.parameters())
        origin_head_p = list(_m.origin_head.parameters())
        jet_head_p    = list(_m.jet_head.parameters())
        if hasattr(_m, "vertex_weight_head"):
            vtxw_head_p = list(_m.vertex_weight_head.parameters())
        elif hasattr(_m, "vertex_delta_head_lxy"):
            vtxw_head_p = (list(_m.vertex_delta_head_lxy.parameters()) +
                           list(_m.vertex_delta_head_dz.parameters()))
        elif hasattr(_m, "residual_mlp"):
            vtxw_head_p = list(_m.residual_mlp.parameters())
        else:
            vtxw_head_p = []
    else:
        # parallel model
        enc1_params   = list(_m.init_net.parameters()) + list(_m.encoder.parameters())
        enc2_params   = []
        enc3_params   = []
        origin_head_p = list(_m.origin_head.parameters())
        jet_head_p    = list(_m.jet_head.parameters())
        vtxw_head_p   = list(_m.pair_head.parameters())

    def _grad_norm(params):
        sq = 0.0
        for p in params:
            if p.grad is not None:
                sq += p.grad.data.norm(2).item() ** 2
        return sq ** 0.5

    def _flatten_grad(params):
        g = []
        for p in params:
            if p.grad is not None:
                g.append(p.grad.data.flatten())
        return torch.cat(g) if g else torch.zeros(1, device=X_b.device)

    def _cos(g1, g2):
        n1 = g1.norm(2)
        n2 = g2.norm(2)
        if n1 < 1e-12 or n2 < 1e-12:
            return 0.0
        return (g1 @ g2).item() / (n1.item() * n2.item())

    results = {}

    # ── measure task-specific gradients ────────────────────────────────
    for _ in range(2):
        _m.zero_grad()
    jet_loss.backward(retain_graph=True)
    results["grad_norm_shared_encoder_jet"]     = _grad_norm(enc1_params)
    if enc3_params:
        results["grad_norm_jet_encoder"] = _grad_norm(enc3_params)
    results["grad_norm_head_jet"]     = _grad_norm(jet_head_p)
    g_enc1_jet = _flatten_grad(enc1_params)

    for _ in range(2):
        _m.zero_grad()
    origin_loss.backward(retain_graph=True)
    results["grad_norm_shared_encoder_origin"]     = _grad_norm(enc1_params)
    results["grad_norm_head_origin"]     = _grad_norm(origin_head_p)
    g_enc1_origin = _flatten_grad(enc1_params)

    for _ in range(2):
        _m.zero_grad()
    vtx_loss.backward(retain_graph=False)
    results["grad_norm_shared_encoder_vertex"]     = _grad_norm(enc1_params)
    if enc2_params:
        results["grad_norm_vertex_encoder"] = _grad_norm(enc2_params)
    results["grad_norm_head_vtxw"]       = _grad_norm(vtxw_head_p)
    g_enc1_vtx = _flatten_grad(enc1_params)

    # ── cosine similarities ────────────────────────────────────────────
    results["grad_cos_origin_vertex"] = _cos(g_enc1_origin, g_enc1_vtx)
    results["grad_cos_origin_jet"]    = _cos(g_enc1_origin, g_enc1_jet)
    results["grad_cos_vertex_jet"]    = _cos(g_enc1_vtx, g_enc1_jet)

    for _ in range(2):
        _m.zero_grad()
    return results


# ===========================================================================
# _compute_epoch_refine_vtx_stats — per-leg match/other means from val data.
# ===========================================================================
def _compute_epoch_refine_vtx_stats(pred_arrays, config):
    refine      = pred_arrays["refine"]       # (N, K, L)
    vtx_weight  = pred_arrays["vtx_weight"]   # (N, K, L)
    origin_full = pred_arrays["origin_full"]  # (N, K)
    mask_full   = pred_arrays["mask_full"]    # (N, K)
    all_true    = pred_arrays["all_true"]     # (N,)

    origin_class_names = config.origin_class_names

    stats = {}
    for leg in range(config.n_vertex_legs):
        leg_name  = config.vertex_leg_names[leg]
        owner_cls = config.leg_owner_cls[leg_name]
        leg_origin_names = config.vertex_legs[leg_name]
        if isinstance(leg_origin_names, str):
            leg_origin_names = [leg_origin_names]
        leg_origin_ids = [origin_class_names.index(c) for c in leg_origin_names]

        owner_mask = (all_true == owner_cls)
        if not owner_mask.any():
            continue

        jet_vtx_w = vtx_weight[owner_mask, :, leg]
        jet_ref   = refine[owner_mask, :, leg]
        jet_orig  = origin_full[owner_mask]
        jet_mask  = mask_full[owner_mask]

        match_mask = np.isin(jet_orig, leg_origin_ids) & jet_mask
        other_mask = ~np.isin(jet_orig, leg_origin_ids) & jet_mask & (jet_orig >= 0)

        s = leg_name.replace("_vertex", "")
        if match_mask.any():
            stats[f"val_{s}_refine_match_mean"]      = jet_ref[match_mask].mean()
            stats[f"val_{s}_vtx_weight_match_mean"]  = jet_vtx_w[match_mask].mean()
        if other_mask.any():
            stats[f"val_{s}_refine_other_mean"]      = jet_ref[other_mask].mean()
            stats[f"val_{s}_vtx_weight_other_mean"]  = jet_vtx_w[other_mask].mean()

    return stats


# ===========================================================================
# _compute_vertex_metrics — per-leg Lxy/dz MAE and Pearson r from val data.
# ===========================================================================
def _compute_vertex_metrics(pred_arrays, config):
    lxy_pred  = pred_arrays["lxy_pred"]   # (N, L)
    lxy_true  = pred_arrays["lxy_true"]   # (N, L)
    dz_pred   = pred_arrays["dz_pred"]    # (N, L)
    dz_true   = pred_arrays["dz_true"]    # (N, L)
    vtx_valid = pred_arrays["vtx_valid"]  # (N, L)

    stats = {}
    for leg in range(config.n_vertex_legs):
        leg_name = config.vertex_leg_names[leg]
        s = leg_name.replace("_vertex", "")
        v = vtx_valid[:, leg]
        if not v.any():
            continue

        if config.fit_lxy:
            pred = lxy_pred[v, leg]
            true = lxy_true[v, leg]
            stats[f"val_{s}_lxy_mae"] = float(np.mean(np.abs(pred - true)))
            if len(pred) >= 2:
                r = np.corrcoef(pred, true)[0, 1]
                stats[f"val_{s}_lxy_pearson"] = float(r) if np.isfinite(r) else 0.0

        if config.fit_dz:
            pred = dz_pred[v, leg]
            true = dz_true[v, leg]
            stats[f"val_{s}_dz_mae"] = float(np.mean(np.abs(pred - true)))
            if len(pred) >= 2:
                r = np.corrcoef(pred, true)[0, 1]
                stats[f"val_{s}_dz_pearson"] = float(r) if np.isfinite(r) else 0.0

    return stats


# ===========================================================================
# run_training — full training loop over *epochs*.
# Prints per-epoch metrics and (if supported) vertex calibration scales.
# ===========================================================================
def run_training(model, train_loader, val_loader, optimiser,
                 criterion_jet, criterion_origin,
                 config, device, epochs, n_origin_classes, n_y_train,
                 vertex_leg_names, calibrate_vertex_fit):
    history = {
        "train_loss": [], "train_jet_loss": [], "train_origin_loss": [],
        "train_vertex_loss": [], "train_lxy_loss": [], "train_dz_loss": [],
        "val_loss": [], "val_jet_loss": [], "val_origin_loss": [],
        "val_vertex_loss": [], "val_lxy_loss": [], "val_dz_loss": [],
        "val_acc": [], "val_origin_acc": [],
        # refine / vtx_weight diagnostics (train: overall mean; val: match/other per leg)
        "train_refine_mean": [], "train_vtx_weight_mean": [],
        "val_b_refine_match_mean": [], "val_b_refine_other_mean": [],
        "val_b_vtx_weight_match_mean": [], "val_b_vtx_weight_other_mean": [],
        "val_c_refine_match_mean": [], "val_c_refine_other_mean": [],
        "val_c_vtx_weight_match_mean": [], "val_c_vtx_weight_other_mean": [],
        # gradient diagnostics
        "grad_norm_shared_encoder_jet": [], "grad_norm_shared_encoder_origin": [], "grad_norm_shared_encoder_vertex": [],
        "grad_norm_vertex_encoder": [], "grad_norm_jet_encoder": [],
        "grad_norm_head_jet": [], "grad_norm_head_origin": [], "grad_norm_head_vtxw": [],
        "grad_cos_origin_vertex": [], "grad_cos_origin_jet": [], "grad_cos_vertex_jet": [],
        # vertex reconstruction metrics (Lxy/dz MAE & Pearson r)
        "val_b_lxy_mae": [], "val_c_lxy_mae": [],
        "val_b_dz_mae": [], "val_c_dz_mae": [],
        "val_b_lxy_pearson": [], "val_c_lxy_pearson": [],
        "val_b_dz_pearson": [], "val_c_dz_pearson": [],
    }

    _tb_dir = getattr(config, "tensorboard_log_dir", None)
    if _tb_dir:
        _tb_dir = os.path.join(_tb_dir.rstrip("/"),
                               os.path.basename(config.plot_dir.rstrip("/")))
    writer = SummaryWriter(_tb_dir) if _tb_dir else None

    best_jet_loss = float("inf")
    best_total_loss = float("inf")
    best_jet_epoch = None
    best_total_epoch = None
    best_jet_path = os.path.join(config.plot_dir, "best_jet.pt")
    best_total_path = os.path.join(config.plot_dir, "best_total.pt")
    last_path = os.path.join(config.plot_dir, "last.pt")
    checkpoint_interval = config.checkpoint_interval

    # grab one fixed batch for gradient diagnostics
    _diag_batch = next(iter(val_loader))
    _diag_X, _diag_mask, _diag_y, _diag_orig, _diag_lxy, _diag_dz, _diag_vv, _diag_pair = (
        t.to(device) for t in _diag_batch)

    for epoch in range(1, epochs + 1):
        (train_loss, train_jet_loss, train_origin_loss, train_vertex_loss,
         train_lxy_loss, train_dz_loss,
         train_refine, train_vtxw) = train_epoch(
            model, train_loader, optimiser, criterion_jet, criterion_origin,
            n_origin_classes, config.lambda_jet, config.lambda_origin,
            config.lambda_vertex, config.fit_lxy, config.fit_dz, device)

        # ── gradient diagnostics (every epoch) ─────────────────────────
        _grad_stats = _measure_task_gradients(
                model, _diag_X, _diag_mask, _diag_y, _diag_orig,
                _diag_lxy, _diag_dz, _diag_vv, _diag_pair,
                criterion_jet, criterion_origin,
                n_origin_classes, config.fit_lxy, config.fit_dz)

        _expected_grad_keys = [
            "grad_norm_shared_encoder_jet", "grad_norm_shared_encoder_origin", "grad_norm_shared_encoder_vertex",
            "grad_norm_vertex_encoder", "grad_norm_jet_encoder",
            "grad_norm_head_jet", "grad_norm_head_origin", "grad_norm_head_vtxw",
            "grad_cos_origin_vertex", "grad_cos_origin_jet", "grad_cos_vertex_jet",
        ]
        for _k in _expected_grad_keys:
            history[_k].append(_grad_stats.get(_k, float("nan")))

        (val_loss, val_jet_loss, val_origin_loss, val_vertex_loss,
         val_lxy_loss, val_dz_loss,
         val_acc, val_origin_acc, pred_arrays) = validate_epoch(
            model, val_loader, criterion_jet, criterion_origin,
            n_origin_classes, config.lambda_jet, config.lambda_origin,
            config.lambda_vertex, config.fit_lxy, config.fit_dz, device)

        # record metrics
        history["train_loss"].append(train_loss)
        history["train_jet_loss"].append(train_jet_loss)
        history["train_origin_loss"].append(train_origin_loss)
        history["train_vertex_loss"].append(train_vertex_loss)
        history["train_lxy_loss"].append(train_lxy_loss)
        history["train_dz_loss"].append(train_dz_loss)
        history["val_loss"].append(val_loss)
        history["val_jet_loss"].append(val_jet_loss)
        history["val_origin_loss"].append(val_origin_loss)
        history["val_vertex_loss"].append(val_vertex_loss)
        history["val_lxy_loss"].append(val_lxy_loss)
        history["val_dz_loss"].append(val_dz_loss)
        history["val_acc"].append(val_acc)
        history["val_origin_acc"].append(val_origin_acc)

        if val_jet_loss < best_jet_loss:
            best_jet_loss = val_jet_loss
            best_jet_epoch = epoch
            torch.save(model.state_dict(), best_jet_path)
            print(f"Saved best_jet.pt (epoch={epoch}, val_jet_loss={val_jet_loss:.6f})")

        if val_loss < best_total_loss:
            best_total_loss = val_loss
            best_total_epoch = epoch
            torch.save(model.state_dict(), best_total_path)
            print(f"Saved best_total.pt (epoch={epoch}, val_loss={val_loss:.6f})")

        if epoch % checkpoint_interval == 0:
            epoch_path = os.path.join(
                config.plot_dir, f"epoch_{epoch}.pt")
            torch.save(model.state_dict(), epoch_path)
            print(f"Saved {os.path.basename(epoch_path)}")

        # Overwrite on every completed epoch; at normal completion this is
        # the final-epoch state.
        torch.save(model.state_dict(), last_path)

        history["train_refine_mean"].append(train_refine)
        history["train_vtx_weight_mean"].append(train_vtxw)

        # val per-leg match/other means — always fill all 8 keys per epoch
        _vtx_stats = {}
        _expected_val_keys = [
            "val_b_refine_match_mean", "val_b_refine_other_mean",
            "val_b_vtx_weight_match_mean", "val_b_vtx_weight_other_mean",
            "val_c_refine_match_mean", "val_c_refine_other_mean",
            "val_c_vtx_weight_match_mean", "val_c_vtx_weight_other_mean",
        ]
        if "refine" in pred_arrays:
            _vtx_stats = _compute_epoch_refine_vtx_stats(pred_arrays, config)
            for _k in _expected_val_keys:
                history[_k].append(_vtx_stats.get(_k, 0.0))
        else:
            for _k in _expected_val_keys:
                history[_k].append(0.0)

        # vertex reconstruction metrics (staged model only)
        _vtx_metrics = {}
        _expected_vtx_keys = [
            "val_b_lxy_mae", "val_c_lxy_mae",
            "val_b_dz_mae", "val_c_dz_mae",
            "val_b_lxy_pearson", "val_c_lxy_pearson",
            "val_b_dz_pearson", "val_c_dz_pearson",
        ]
        if "lxy_pred" in pred_arrays:
            _vtx_metrics = _compute_vertex_metrics(pred_arrays, config)
        for _k in _expected_vtx_keys:
            history[_k].append(_vtx_metrics.get(_k, 0.0))

        # optional: log vertex calibration scale values
        _calib_str = ""
        if calibrate_vertex_fit and hasattr(model, "calibration_scales"):
            _scales = model.calibration_scales()
            _parts = []
            for _coord, _vals in _scales.items():
                _per_leg = ", ".join(
                    f"{_n}={_v:.3f}" for _n, _v in zip(vertex_leg_names, _vals))
                _parts.append(f"{_coord}_scale=[{_per_leg}]")
            if _parts:
                _calib_str = "  " + "  ".join(_parts)

        print(f"Epoch {epoch:02d}/{epochs}  "
              f"loss={train_loss:.4f} (jet={train_jet_loss:.4f} "
              f"origin={train_origin_loss:.4f} vtx={train_vertex_loss:.4f}(Lxy={train_lxy_loss:.4f},dz={train_dz_loss:.4f}))  "
              f"val_loss={val_loss:.4f} (jet={val_jet_loss:.4f} "
              f"origin={val_origin_loss:.4f} vtx={val_vertex_loss:.4f}(Lxy={val_lxy_loss:.4f},dz={val_dz_loss:.4f}))  "
              f"val_acc={val_acc:.4f}  origin_acc={val_origin_acc:.4f}{_calib_str}")
        if _vtx_stats:
            print(f"         refine train={train_refine:.4f}  "
                  f"val b(m={_vtx_stats.get('val_b_refine_match_mean', 0):.4f} "
                  f"o={_vtx_stats.get('val_b_refine_other_mean', 0):.4f})  "
                  f"c(m={_vtx_stats.get('val_c_refine_match_mean', 0):.4f} "
                  f"o={_vtx_stats.get('val_c_refine_other_mean', 0):.4f})  "
                  f"vtx_w b(m={_vtx_stats.get('val_b_vtx_weight_match_mean', 0):.4f} "
                  f"o={_vtx_stats.get('val_b_vtx_weight_other_mean', 0):.4f})  "
                  f"c(m={_vtx_stats.get('val_c_vtx_weight_match_mean', 0):.4f} "
                  f"o={_vtx_stats.get('val_c_vtx_weight_other_mean', 0):.4f})")
        if _vtx_metrics:
            _vtx_parts = []
            for _lname in config.vertex_leg_names:
                _s = _lname.replace("_vertex", "")
                _sub = []
                if config.fit_lxy:
                    _sub.append(f"Lxy MAE={_vtx_metrics.get(f'val_{_s}_lxy_mae', 0):.4f} r={_vtx_metrics.get(f'val_{_s}_lxy_pearson', 0):.3f}")
                if config.fit_dz:
                    _sub.append(f"dz MAE={_vtx_metrics.get(f'val_{_s}_dz_mae', 0):.4f} r={_vtx_metrics.get(f'val_{_s}_dz_pearson', 0):.3f}")
                if _sub:
                    _vtx_parts.append(f"{_s}: " + "  ".join(_sub))
            if _vtx_parts:
                print(f"    vtx metrics  " + " | ".join(_vtx_parts))
        if _grad_stats:
            print(f"    grad  shared: jet={_grad_stats.get('grad_norm_shared_encoder_jet', 0):.4f}  "
                  f"origin={_grad_stats.get('grad_norm_shared_encoder_origin', 0):.4f}  "
                  f"vertex={_grad_stats.get('grad_norm_shared_encoder_vertex', 0):.4f}")
            if "grad_norm_vertex_encoder" in _grad_stats:
                print(f"          vertex_enc={_grad_stats['grad_norm_vertex_encoder']:.4f}  "
                      f"jet_enc={_grad_stats.get('grad_norm_jet_encoder', 0):.4f}")
            print(f"          heads: jet={_grad_stats.get('grad_norm_head_jet', 0):.4f}  "
                  f"origin={_grad_stats.get('grad_norm_head_origin', 0):.4f}  "
                  f"vtxw={_grad_stats.get('grad_norm_head_vtxw', 0):.4f}")
            print(f"          cos: orig-vtx={_grad_stats.get('grad_cos_origin_vertex', 0):.3f}  "
                  f"orig-jet={_grad_stats.get('grad_cos_origin_jet', 0):.3f}  "
                  f"vtx-jet={_grad_stats.get('grad_cos_vertex_jet', 0):.3f}")

        if writer:
            for key, values in history.items():
                v = values[-1]
                if v == v:  # skip NaN
                    writer.add_scalar(key, v, epoch)
            writer.flush()

    if writer:
        writer.close()

    print("Checkpoint summary:")
    print(f"  best_jet.pt: epoch={best_jet_epoch}, val_jet_loss={best_jet_loss:.6f}")
    print(f"  best_total.pt: epoch={best_total_epoch}, val_loss={best_total_loss:.6f}")
    print(f"  last.pt: epoch={epochs}")
    print(f"  periodic checkpoints: every {checkpoint_interval} epoch(s)")

    # ── export gradient diagnostics as CSV ────────────────────────────
    _grad_keys = [k for k in history if k.startswith("grad_")]
    if any(len(history[k]) > 0 for k in _grad_keys):
        _csv_path = os.path.join(config.plot_dir, "gradient_diagnostics.csv")
        _all_keys = ["epoch"] + _grad_keys
        _n = len(history[_grad_keys[0]])
        with open(_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(_all_keys)
            for i in range(_n):
                w.writerow([i + 1] + [history[k][i] for k in _grad_keys])
        print(f"Exported gradient_diagnostics.csv")

    return history, pred_arrays
