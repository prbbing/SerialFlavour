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

from .losses import vertex_loss_fn, pair_vertex_loss


# ===========================================================================
# train_epoch — one full pass over the training set.
# ===========================================================================
def train_epoch(model, dataloader, optimiser, criterion_jet, criterion_origin,
                n_origin_classes, lambda_jet, lambda_origin, lambda_vertex,
                fit_lxy, fit_dz, device):
    model.train()
    tot_loss = tot_jet = tot_origin = tot_vertex = 0.0
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
            vtx_loss = vertex_loss_fn(out["lxy_pred"], out["dz_pred"],
                                      lxy_b, dz_b, vvalid_b,
                                      fit_lxy=fit_lxy, fit_dz=fit_dz)
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

    return (tot_loss / n_total, tot_jet / n_total,
            tot_origin / n_total, tot_vertex / n_total)


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
            vtx_loss = vertex_loss_fn(out["lxy_pred"], out["dz_pred"],
                                      lxy_b, dz_b, vvalid_b,
                                      fit_lxy=fit_lxy, fit_dz=fit_dz)
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
    val_loss        /= n_total
    val_jet_loss    /= n_total
    val_origin_loss /= n_total
    val_vertex_loss /= n_total
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
            val_acc, val_origin_acc, pred_arrays)


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
        "train_vertex_loss": [],
        "val_loss": [], "val_jet_loss": [], "val_origin_loss": [],
        "val_vertex_loss": [], "val_acc": [], "val_origin_acc": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_jet_loss, train_origin_loss, train_vertex_loss = train_epoch(
            model, train_loader, optimiser, criterion_jet, criterion_origin,
            n_origin_classes, config.lambda_jet, config.lambda_origin,
            config.lambda_vertex, config.fit_lxy, config.fit_dz, device)

        (val_loss, val_jet_loss, val_origin_loss, val_vertex_loss,
         val_acc, val_origin_acc, pred_arrays) = validate_epoch(
            model, val_loader, criterion_jet, criterion_origin,
            n_origin_classes, config.lambda_jet, config.lambda_origin,
            config.lambda_vertex, config.fit_lxy, config.fit_dz, device)

        # record metrics
        history["train_loss"].append(train_loss)
        history["train_jet_loss"].append(train_jet_loss)
        history["train_origin_loss"].append(train_origin_loss)
        history["train_vertex_loss"].append(train_vertex_loss)
        history["val_loss"].append(val_loss)
        history["val_jet_loss"].append(val_jet_loss)
        history["val_origin_loss"].append(val_origin_loss)
        history["val_vertex_loss"].append(val_vertex_loss)
        history["val_acc"].append(val_acc)
        history["val_origin_acc"].append(val_origin_acc)

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
              f"origin={train_origin_loss:.4f} vtx={train_vertex_loss:.4f})  "
              f"val_loss={val_loss:.4f} (jet={val_jet_loss:.4f} "
              f"origin={val_origin_loss:.4f} vtx={val_vertex_loss:.4f})  "
              f"val_acc={val_acc:.4f}  origin_acc={val_origin_acc:.4f}{_calib_str}")

    return history, pred_arrays
