"""
Evaluation plots: input distributions, training curves, confusion matrices,
vertex-fit validation, pair-vertexing evaluation, output probabilities,
b-/c-tagging discriminant and ROC curves, track-to-vertex assignment.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


# ===========================================================================
# plot_input_variables — per-feature histograms coloured by jet flavour.
# Shows how the track-feature distributions differ across b/c/light jets.
# ===========================================================================
def plot_input_variables(X_train, mask_train, y_train, track_fields, top_k,
                         jet_class_names, colours, plot_dir):
    tracks_flat = X_train.reshape(-1, len(track_fields))     # (N*K, F)
    labels_rep  = np.repeat(y_train, top_k)                  # per-track label
    nonzero     = mask_train.ravel()                         # valid entries

    n_cols = min(len(track_fields), 4)
    n_rows = (len(track_fields) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).ravel()
    fig.suptitle("Input variables by jet flavour (training sample)", fontweight="bold")
    for fi, fld in enumerate(track_fields):
        col = tracks_flat[:, fi]
        valid_col = col[nonzero]
        clip = np.percentile(np.abs(valid_col), 99) if valid_col.size else 1.0
        for cls_idx, name in enumerate(jet_class_names):
            m = (labels_rep == cls_idx) & nonzero
            axes[fi].hist(col[m], bins=80, range=(-clip, clip),
                          histtype="step", label=name, color=colours[name],
                          linewidth=1.5, density=True)
        axes[fi].set_title(fld, fontsize=8)
        axes[fi].set_xlabel(fld, fontsize=7)
        axes[fi].set_ylabel("Density", fontsize=7)
        if fi == 0:
            axes[fi].legend(fontsize=7)
    for ax in axes[len(track_fields):]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.savefig(plot_dir + "input_variables.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved input_variables.png")


# ===========================================================================
# plot_training_summary — loss curves, accuracy curves, jet confusion matrix.
# ===========================================================================
def plot_training_summary(history, all_true, all_preds, jet_class_names,
                          plot_dir, epochs, model_type="staged_origin_vertex_jet"):
    _model_titles = {
        "staged_origin_vertex_jet":   "Staged origin → vertex → jet",
        "parallel_origin_vertex_jet": "Parallel origin + vertex + jet",
    }
    _title = _model_titles.get(model_type, model_type) + " — training summary"

    n_jet_classes = len(jet_class_names)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(_title, fontweight="bold")
    ep = range(1, epochs + 1)

    # left: loss curves
    axes[0].plot(ep, history["train_loss"],        label="train (total)")
    axes[0].plot(ep, history["train_jet_loss"],    label="train jet CE",    linestyle="--")
    axes[0].plot(ep, history["train_origin_loss"], label="train origin CE", linestyle=":")
    axes[0].plot(ep, history["train_vertex_loss"], label="train vertex",    linestyle="-.")
    axes[0].plot(ep, history["val_loss"],          label="val (total)")
    axes[0].plot(ep, history["val_jet_loss"],      label="val jet CE",      linestyle="--")
    axes[0].plot(ep, history["val_origin_loss"],   label="val origin CE",   linestyle=":")
    axes[0].plot(ep, history["val_vertex_loss"],   label="val vertex",      linestyle="-.")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=6)

    # middle: validation accuracy
    axes[1].plot(ep, history["val_acc"],        label="jet accuracy")
    axes[1].plot(ep, history["val_origin_acc"], label="origin accuracy (Stage 1)")
    axes[1].set_title("Validation accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8)

    # right: row-normalised jet confusion matrix
    cm = confusion_matrix(all_true, all_preds, normalize="true")
    im = axes[2].imshow(cm, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_xticks(range(n_jet_classes))
    axes[2].set_yticks(range(n_jet_classes))
    axes[2].set_xticklabels(jet_class_names, rotation=45, ha="right", fontsize=8)
    axes[2].set_yticklabels(jet_class_names, fontsize=8)
    axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
    axes[2].set_title("Jet confusion matrix (normalised)")
    for i in range(n_jet_classes):
        for j in range(n_jet_classes):
            axes[2].text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                         color="white" if cm[i,j] > 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=axes[2])
    plt.tight_layout()
    plt.savefig(plot_dir + "training_summary.png", dpi=150, bbox_inches="tight")

    axes[0].set_yscale("log")
    axes[0].set_title("Loss (log scale)")
    try:
        plt.savefig(plot_dir + "training_summary_log.png", dpi=150, bbox_inches="tight")
    except ValueError:
        axes[0].set_yscale("symlog", linthresh=1e-4)
        plt.savefig(plot_dir + "training_summary_log.png", dpi=150, bbox_inches="tight")
        axes[0].set_title("Loss (symlog scale)")
    plt.close(fig)
    print("Saved training_summary.png  training_summary_log.png")


# ===========================================================================
# plot_origin_confusion_matrix — Stage 1 track-origin classification.
# ===========================================================================
def plot_origin_confusion_matrix(origin_true, origin_preds, origin_class_names,
                                 plot_dir):
    n_origin_classes = len(origin_class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Track-origin confusion matrix -- Stage 1 (normalised)",
                 fontweight="bold")
    cm_o = confusion_matrix(origin_true, origin_preds, normalize="true",
                            labels=list(range(n_origin_classes)))
    im = ax.imshow(cm_o, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_origin_classes))
    ax.set_yticks(range(n_origin_classes))
    ax.set_xticklabels(origin_class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(origin_class_names, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(n_origin_classes):
        for j in range(n_origin_classes):
            ax.text(j, i, f"{cm_o[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm_o[i,j] > 0.5 else "black", fontsize=7)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(plot_dir + "origin_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved origin_confusion_matrix.png")


# ===========================================================================
# plot_vertex_fit — predicted vs true Lxy/dz per vertex leg per jet flavour.
#
# Left panel: truth (solid) vs predicted (dashed) for jets that own the leg.
# Right panel: fitted-value distributions for jets without a matching truth
#   hadron — diagnostic of whether the network suppresses spurious fits.
#
# model: staged_origin_vertex_jet
# ===========================================================================
def plot_vertex_fit(lxy_pred, lxy_true, dz_pred, dz_true, vtx_valid, all_true,
                    config, plot_dir):
    fit_lxy      = config.fit_lxy
    fit_dz       = config.fit_dz
    vertex_legs  = config.vertex_legs
    vertex_leg_names = config.vertex_leg_names
    n_vertex_legs    = config.n_vertex_legs
    vertex_targets   = config.vertex_targets
    jet_class_names  = config.jet_class_names
    colours          = config.colours

    # auto-generate leg titles from config
    _leg_titles = {
        lname: (f"{lname.replace('_', '-')} (tracks: "
                f"{', '.join(vertex_legs[lname]) if isinstance(vertex_legs[lname], list) else vertex_legs[lname]})")
        for lname in vertex_leg_names
    }

    for leg in range(n_vertex_legs):
        leg_name   = vertex_leg_names[leg]
        owner_cls  = config.leg_owner_cls[leg_name]
        owner_name = jet_class_names[owner_cls]
        # "has truth" = truth-valid, Lxy>0, and jet flavour matches the leg owner
        has_truth  = vtx_valid[:, leg] & (lxy_true[:, leg] > 0) & (all_true == owner_cls)
        no_truth   = ~has_truth

        # -- Lxy comparison --
        if fit_lxy:
            fig, (ax_cmp, ax_dist) = plt.subplots(1, 2, figsize=(13, 5.5))
            fig.suptitle(
                f"Differentiable origin-gated vertex fit — {_leg_titles[leg_name]}  "
                r"— $L_{xy}$ (test sample, by true jet flavour)", fontweight="bold")

            if has_truth.any():
                _clip = max(np.percentile(lxy_true[has_truth, leg], 99),
                            np.percentile(lxy_pred[has_truth, leg], 99), 1e-3)
                _bins = np.linspace(0, _clip, 61)
                ax_cmp.hist(lxy_true[has_truth, leg], bins=_bins, histtype="step",
                            color=colours[owner_name], linestyle="-", linewidth=1.5,
                            density=True, label=f"{owner_name} (truth)")
                ax_cmp.hist(lxy_pred[has_truth, leg], bins=_bins, histtype="step",
                            color=colours[owner_name], linestyle="--", linewidth=1.5,
                            density=True, label=f"{owner_name} (pred)")
                ax_cmp.set_xlim(0, _clip)
            else:
                ax_cmp.text(0.5, 0.5, "no jets with a matching truth hadron",
                            ha="center", va="center", transform=ax_cmp.transAxes,
                            fontsize=9, color="grey")
            ax_cmp.set_xlabel(r"$L_{xy}$ [mm]"); ax_cmp.set_ylabel("Density")
            ax_cmp.set_title(r"Truth (solid) vs. predicted (dashed) — truth hadron exists")
            ax_cmp.legend(fontsize=6); ax_cmp.grid(True, linestyle="--", alpha=0.3)

            if no_truth.any():
                _lxy_clip = np.percentile(lxy_pred[no_truth, leg], 99)
                for cls_idx, cls_name in enumerate(jet_class_names):
                    m = no_truth & (all_true == cls_idx)
                    if not m.any():
                        continue
                    ax_dist.hist(lxy_pred[m, leg], bins=60, range=(0, _lxy_clip),
                                 histtype="step", color=colours[cls_name],
                                 label=cls_name, linewidth=1.5, density=True)
            else:
                ax_dist.text(0.5, 0.5, "every jet has a matching truth hadron",
                             ha="center", va="center", transform=ax_dist.transAxes,
                             fontsize=9, color="grey")
            ax_dist.set_xlabel(r"Predicted $L_{xy}$ [mm]")
            ax_dist.set_ylabel("Density")
            ax_dist.set_title(r"Fitted-value distribution (no truth hadron)")
            ax_dist.legend(fontsize=7); ax_dist.grid(True, linestyle="--", alpha=0.3)

            plt.tight_layout()
            plt.savefig(plot_dir + f"vertex_fit_{leg_name}.png", dpi=150,
                        bbox_inches="tight")
            plt.close(fig)
            print(f"Saved vertex_fit_{leg_name}.png")

        # -- dz comparison (same structure, signed axis) --
        if fit_dz:
            fig_dz, axes_dz = plt.subplots(1, 2, figsize=(13, 5.5))
            fig_dz.suptitle(
                f"Vertex dz — {_leg_titles[leg_name]}  "
                r"— $d_z$ (test sample, by true jet flavour)", fontweight="bold")
            ax_dz_cmp, ax_dz_dist = axes_dz

            if has_truth.any():
                _dz_all  = np.concatenate([dz_true[has_truth, leg],
                                           dz_pred[has_truth, leg]])
                _dz_edge = np.percentile(np.abs(_dz_all[np.isfinite(_dz_all)]), 99)
                _dz_bins = np.linspace(-_dz_edge, _dz_edge, 61)
                ax_dz_cmp.hist(dz_true[has_truth, leg], bins=_dz_bins, histtype="step",
                               color=colours[owner_name], linestyle="-", linewidth=1.5,
                               density=True, label=f"{owner_name} (truth)")
                ax_dz_cmp.hist(dz_pred[has_truth, leg], bins=_dz_bins, histtype="step",
                               color=colours[owner_name], linestyle="--", linewidth=1.5,
                               density=True, label=f"{owner_name} (pred)")
            else:
                ax_dz_cmp.text(0.5, 0.5, "no jets with a matching truth hadron",
                               ha="center", va="center", transform=ax_dz_cmp.transAxes,
                               fontsize=9, color="grey")
            ax_dz_cmp.set_xlabel(r"$d_z$ [mm]"); ax_dz_cmp.set_ylabel("Density")
            ax_dz_cmp.set_title(r"Truth (solid) vs. predicted (dashed) — truth hadron exists")
            ax_dz_cmp.legend(fontsize=6); ax_dz_cmp.grid(True, linestyle="--", alpha=0.3)

            if no_truth.any():
                _dz_clip = np.percentile(np.abs(dz_pred[no_truth, leg]), 99)
                for cls_idx, cls_name in enumerate(jet_class_names):
                    m = no_truth & (all_true == cls_idx)
                    if not m.any():
                        continue
                    ax_dz_dist.hist(dz_pred[m, leg], bins=60,
                                    range=(-_dz_clip, _dz_clip),
                                    histtype="step", color=colours[cls_name],
                                    label=cls_name, linewidth=1.5, density=True)
            else:
                ax_dz_dist.text(0.5, 0.5, "every jet has a matching truth hadron",
                                ha="center", va="center",
                                transform=ax_dz_dist.transAxes,
                                fontsize=9, color="grey")
            ax_dz_dist.set_xlabel(r"Predicted $d_z$ [mm]")
            ax_dz_dist.set_ylabel("Density")
            ax_dz_dist.set_title(r"Fitted-value distribution (no truth hadron)")
            ax_dz_dist.legend(fontsize=7); ax_dz_dist.grid(True, linestyle="--", alpha=0.3)

            plt.tight_layout()
            plt.savefig(plot_dir + f"vertex_fit_{leg_name}_dz.png", dpi=150,
                        bbox_inches="tight")
            plt.close(fig_dz)
            print(f"Saved vertex_fit_{leg_name}_dz.png")


# ===========================================================================
# plot_output_probabilities — P(cls) histogram per true jet flavour.
# ===========================================================================
def plot_output_probabilities(all_probs, all_true, jet_class_names, colours,
                              plot_dir):
    n_jet_classes = len(jet_class_names)
    fig, axes = plt.subplots(1, n_jet_classes, figsize=(5 * n_jet_classes, 4))
    fig.suptitle("Jet output probabilities by true flavour (test sample)",
                 fontweight="bold")
    axes = np.atleast_1d(axes)
    for cls_idx, cls_name in enumerate(jet_class_names):
        ax = axes[cls_idx]
        for true_idx, true_name in enumerate(jet_class_names):
            ax.hist(all_probs[all_true == true_idx, cls_idx], bins=50, range=(0, 1),
                    histtype="step", label=true_name, color=colours[true_name],
                    linewidth=1.5, density=True)
        ax.set_title(f"P({cls_name})"); ax.set_xlabel("Probability")
        ax.set_ylabel("Density"); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(plot_dir + "output_probs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved output_probs.png")


# ===========================================================================
# plot_discriminant_roc — b-tagging discriminant log(p_b / Σ w_i p_i) and
# per-background ROC curves.
# ===========================================================================
def plot_discriminant_roc(all_probs, all_true, jet_class_names,
                          disc_bkg_weights, colours, plot_dir):
    n_jet_classes = len(jet_class_names)
    b_idx = jet_class_names.index("b-jet")
    bkg_idxs = [i for i in range(n_jet_classes) if i != b_idx]
    w = np.array([disc_bkg_weights[jet_class_names[i]] for i in bkg_idxs])
    w = w / w.sum()

    # discriminant = log( p_b / (w_c·p_c + w_light·p_light) )
    pb   = all_probs[:, b_idx]
    pbkg = all_probs[:, bkg_idxs] @ w
    disc = np.log(pb / (pbkg + 1e-10))
    finite = np.isfinite(disc)
    clip = np.percentile(np.abs(disc[finite]), 99)

    # -- discriminant distribution --
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(r"$\log(p_b\,/\,\sum_i w_i\,p_i)$ -- test sample",
                 fontweight="bold")
    for true_idx, true_name in enumerate(jet_class_names):
        mfin = (all_true == true_idx) & finite
        ax.hist(disc[mfin], bins=80, range=(-clip, clip), histtype="step",
                label=true_name, color=colours[true_name], linewidth=1.5,
                density=True)
    ax.set_xlabel(r"$\log(p_b\,/\,\sum_i w_i\,p_i)$")
    ax.set_ylabel("Density"); ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir + "discriminant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved discriminant.png")

    # -- ROC curves (b vs c, b vs light) --
    _roc_data = {}
    fig, axes = plt.subplots(1, len(bkg_idxs), figsize=(6 * len(bkg_idxs), 5))
    fig.suptitle(r"ROC curves — $\log(p_b\,/\,\sum_i w_i\,p_i)$",
                  fontweight="bold")
    ax_list = np.atleast_1d(axes)
    for ax, bkg_idx in zip(ax_list, bkg_idxs):
        bkg_name = jet_class_names[bkg_idx]
        mroc   = (all_true == b_idx) | (all_true == bkg_idx)
        labels = (all_true[mroc] == b_idx).astype(int)
        score  = disc[mroc]
        fin    = np.isfinite(score)
        fpr, tpr, _ = roc_curve(labels[fin], score[fin])
        _roc_data[bkg_name] = (fpr, tpr)
        ax.plot(tpr, fpr, color="#1f77b4", linewidth=1.5,
                label=f"AUC={auc(fpr,tpr):.3f}")
        ax.set_xlabel("b-jet efficiency (TPR)")
        ax.set_ylabel(f"{bkg_name} rate (FPR)")
        ax.set_title(f"b vs {bkg_name}"); ax.set_yscale("log")
        ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_dir + "roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved roc.png")

    # -- background rejection vs b-jet efficiency --
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(r"Background rejection vs $b$-jet efficiency",
                 fontweight="bold")
    _wp_eps = [0.65, 0.70, 0.77, 0.85, 0.90]
    print("\n=== b-tagging rejection rates ===")
    for ep in _wp_eps:
        _parts = []
        for bkg_name, (fpr, tpr) in _roc_data.items():
            _rej = np.interp(ep, tpr, 1.0 / np.clip(fpr, 1e-10, None))
            _bkg_short = bkg_name.replace("-jet", "")
            _parts.append(f"1/ε_{_bkg_short} = {_rej:.0f}")
        print(f"  ε_b={ep:.0%}:  " + "  ".join(_parts))
    for bkg_name, (fpr, tpr) in _roc_data.items():
        rej = 1.0 / np.clip(fpr, 1e-10, None)
        ax.plot(tpr, rej, linewidth=1.5, label=bkg_name,
                color=colours[bkg_name])
    for ep in _wp_eps:
        ax.axvline(ep, color="grey", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_xlabel(r"$b$-jet efficiency")
    ax.set_ylabel("Background rejection (1 / ε_bg)")
    ax.set_yscale("log")
    ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_dir + "rejection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved rejection.png")


# ===========================================================================
# plot_pair_vertexing — track-pair vertex compatibility evaluation.
#
# Left: ROC curve for classifying pairs as same-vertex vs different-vertex.
# Right: predicted same-vertex probability distributions by jet flavour.
#
# model: parallel_origin_vertex_jet
# ===========================================================================
def plot_pair_vertexing(pair_logits, pair_target, pair_mask,
                        jet_class_names, all_true, colours, plot_dir):
    B, K, _ = pair_logits.shape
    pair_prob    = 1.0 / (1.0 + np.exp(-pair_logits))  # sigmoid
    valid_pair   = pair_mask[:, :, None] & pair_mask[:, None, :]  # (B, K, K)
    pos_mask     = valid_pair & (pair_target == 1)     # same vertex
    neg_mask     = valid_pair & (pair_target == 0)     # different vertex
    true_classes = np.repeat(all_true, K * K)          # per-pair jet label

    def _hist(data, mask, label, style="-"):
        d = data[mask]
        if len(d) == 0:
            return
        clip = np.percentile(d, 99.9)
        bins = np.linspace(0, min(clip, 1), 51)
        plt.hist(d, bins=bins, histtype="step", label=label,
                 linestyle=style, linewidth=1.5, density=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Track-pair vertexing — parallel_origin_vertex_jet  (test sample)",
                 fontweight="bold")

    # -- ROC: same-vertex vs different-vertex binary classification --
    if pos_mask.any() and neg_mask.any():
        from sklearn.metrics import roc_curve, auc as _auc
        y_true  = np.concatenate([np.ones(pos_mask.sum(), dtype=int),
                                  np.zeros(neg_mask.sum(), dtype=int)])
        y_score = np.concatenate([pair_prob[pos_mask], pair_prob[neg_mask]])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax1.plot(tpr, fpr, color="#1f77b4", linewidth=1.5,
                 label=f"AUC={_auc(fpr,tpr):.3f}")
    ax1.set_xlabel("Same-vertex efficiency (TPR)")
    ax1.set_ylabel("Different-vertex rate (FPR)")
    ax1.set_title("Pair-vs-pair ROC"); ax1.set_yscale("log")
    ax1.legend(); ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    # -- score distributions split by jet flavour --
    for cls_idx, cls_name in enumerate(jet_class_names):
        m = true_classes == cls_idx
        _hist(pair_prob, pos_mask & m.reshape(B, K, K),
              f"{cls_name} (same vertex)", "--")
        _hist(pair_prob, neg_mask & m.reshape(B, K, K),
              f"{cls_name} (different)", ":")
    ax2.set_xlabel(r"Predicted $p$(same vertex)"); ax2.set_ylabel("Density")
    ax2.set_title("Pair-score distributions by jet flavour")
    ax2.legend(fontsize=7); ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir + "pair_vertexing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved pair_vertexing.png")


# ===========================================================================
# plot_c_discriminant_roc — c-tagging discriminant, ROC, and rejection.
# Symmetric to the b-tagging version, using c-jet as signal.
# ===========================================================================
def plot_c_discriminant_roc(all_probs, all_true, jet_class_names,
                            disc_bkg_weights, colours, plot_dir):
    n_jet_classes = len(jet_class_names)
    c_idx = jet_class_names.index("c-jet")
    bkg_idxs = [i for i in range(n_jet_classes) if i != c_idx]
    w = np.array([disc_bkg_weights[jet_class_names[i]] for i in bkg_idxs])
    w = w / w.sum()

    # discriminant = log( p_c / (w_b·p_b + w_light·p_light) )
    pc   = all_probs[:, c_idx]
    pbkg = all_probs[:, bkg_idxs] @ w
    disc = np.log(pc / (pbkg + 1e-10))
    finite = np.isfinite(disc)
    clip = np.percentile(np.abs(disc[finite]), 99)

    # -- discriminant distribution --
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(r"$\log(p_c\,/\,\sum_i w_i\,p_i)$ -- test sample",
                 fontweight="bold")
    for true_idx, true_name in enumerate(jet_class_names):
        mfin = (all_true == true_idx) & finite
        ax.hist(disc[mfin], bins=80, range=(-clip, clip), histtype="step",
                label=true_name, color=colours[true_name], linewidth=1.5,
                density=True)
    ax.set_xlabel(r"$\log(p_c\,/\,\sum_i w_i\,p_i)$")
    ax.set_ylabel("Density"); ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir + "c_discriminant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved c_discriminant.png")

    # -- ROC curves (c vs b, c vs light) --
    _roc_data = {}
    fig, axes = plt.subplots(1, len(bkg_idxs), figsize=(6 * len(bkg_idxs), 5))
    fig.suptitle(r"ROC curves — $\log(p_c\,/\,\sum_i w_i\,p_i)$",
                  fontweight="bold")
    ax_list = np.atleast_1d(axes)
    for ax, bkg_idx in zip(ax_list, bkg_idxs):
        bkg_name = jet_class_names[bkg_idx]
        mroc   = (all_true == c_idx) | (all_true == bkg_idx)
        labels = (all_true[mroc] == c_idx).astype(int)
        score  = disc[mroc]
        fin    = np.isfinite(score)
        fpr, tpr, _ = roc_curve(labels[fin], score[fin])
        _roc_data[bkg_name] = (fpr, tpr)
        ax.plot(tpr, fpr, color="#ff7f0e", linewidth=1.5,
                label=f"AUC={auc(fpr,tpr):.3f}")
        ax.set_xlabel("c-jet efficiency (TPR)")
        ax.set_ylabel(f"{bkg_name} rate (FPR)")
        ax.set_title(f"c vs {bkg_name}"); ax.set_yscale("log")
        ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_dir + "c_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved c_roc.png")

    # -- background rejection vs c-jet efficiency --
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(r"Background rejection vs $c$-jet efficiency",
                 fontweight="bold")
    _wp_eps = [0.20, 0.30, 0.40]
    print("\n=== c-tagging rejection rates ===")
    for ep in _wp_eps:
        _parts = []
        for bkg_name, (fpr, tpr) in _roc_data.items():
            _rej = np.interp(ep, tpr, 1.0 / np.clip(fpr, 1e-10, None))
            _bkg_short = bkg_name.replace("-jet", "")
            _parts.append(f"1/ε_{_bkg_short} = {_rej:.0f}")
        print(f"  ε_c={ep:.0%}:  " + "  ".join(_parts))
    for bkg_name, (fpr, tpr) in _roc_data.items():
        rej = 1.0 / np.clip(fpr, 1e-10, None)
        ax.plot(tpr, rej, linewidth=1.5, label=bkg_name,
                color=colours[bkg_name])
    for ep in _wp_eps:
        ax.axvline(ep, color="grey", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_xlabel(r"$c$-jet efficiency")
    ax.set_ylabel("Background rejection (1 / ε_bg)")
    ax.set_yscale("log")
    ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_dir + "c_rejection.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved c_rejection.png")


# ===========================================================================
# plot_track_vertex_assignment — per-leg track-to-vertex weight fidelity.
#
# For each vertex leg, plots the vtx_weight distribution for tracks whose
# true origin matches the leg's target classes vs. tracks from the same jet
# flavour with a different origin.  Also prints assignment efficiency at
# weight thresholds 0.5 and 0.8.
#
# model: staged_origin_vertex_jet
# ===========================================================================
def plot_track_vertex_assignment(vtx_weight, origin_full, mask_full, all_true,
                                 origin_class_names, vertex_leg_names,
                                 vertex_legs, n_vertex_legs, leg_owner_cls,
                                 jet_class_names, colours, plot_dir,
                                 leg_origin_probs=None, gate=None,
                                 refine=None):
    # origin_full: (N, K)  — true origin label per track (-1 = padding)
    # vtx_weight:  (N, K, L)
    # mask_full:   (N, K)  — valid track mask
    # all_true:    (N,)    — jet flavour label

    print("\n=== Track-to-vertex assignment efficiency ===")
    for leg in range(n_vertex_legs):
        leg_name = vertex_leg_names[leg]
        owner_cls = leg_owner_cls[leg_name]
        owner_name = jet_class_names[owner_cls]
        leg_origin_cls_names = vertex_legs[leg_name]
        if isinstance(leg_origin_cls_names, str):
            leg_origin_cls_names = [leg_origin_cls_names]
        leg_origin_ids = [origin_class_names.index(c) for c in leg_origin_cls_names]

        # select jets of the leg-owner flavour
        owner_mask = (all_true == owner_cls)
        if not owner_mask.any():
            print(f"  {leg_name}: no {owner_name} jets in test set")
            continue

        jet_vtx_w = vtx_weight[owner_mask, :, leg]    # (N_owner, K)
        jet_orig  = origin_full[owner_mask]             # (N_owner, K)
        jet_mask  = mask_full[owner_mask]               # (N_owner, K)

        # tracks whose true origin matches this leg
        match_mask = np.isin(jet_orig, leg_origin_ids) & jet_mask
        # tracks from same jet flavour but different origin
        other_mask = ~np.isin(jet_orig, leg_origin_ids) & jet_mask & (jet_orig >= 0)

        match_weights = jet_vtx_w[match_mask]
        other_weights = jet_vtx_w[other_mask]

        # -- print gate / leg_origin_probs diagnostics --
        _match_leg_p = None; _other_leg_p = None
        _match_gate  = None; _other_gate  = None
        if leg_origin_probs is not None:
            jet_leg_p  = leg_origin_probs[owner_mask, :, leg]   # (N_owner, K)
            _match_leg_p = jet_leg_p[match_mask]
            _other_leg_p = jet_leg_p[other_mask]
        if gate is not None:
            jet_gate   = gate[owner_mask, :, leg]               # (N_owner, K)
            _match_gate = jet_gate[match_mask]
            _other_gate = jet_gate[other_mask]
        if _match_leg_p is not None and len(_match_leg_p):
            print(f"  {leg_name}  P_leg match:  "
                  f"min={_match_leg_p.min():.4f}  mean={_match_leg_p.mean():.4f}  "
                  f"P25={np.percentile(_match_leg_p,25):.4f}  "
                  f"P50={np.percentile(_match_leg_p,50):.4f}  "
                  f"P75={np.percentile(_match_leg_p,75):.4f}  "
                  f"max={_match_leg_p.max():.4f}")
            print(f"  {leg_name}  P_leg other:  "
                  f"min={_other_leg_p.min():.4f}  mean={_other_leg_p.mean():.4f}  "
                  f"P50={np.percentile(_other_leg_p,50):.4f}  "
                  f"P90={np.percentile(_other_leg_p,90):.4f}  "
                  f"max={_other_leg_p.max():.4f}")
        if _match_gate is not None and len(_match_gate):
            print(f"  {leg_name}  gate  match:  "
                  f"min={_match_gate.min():.4f}  mean={_match_gate.mean():.4f}  "
                  f"P25={np.percentile(_match_gate,25):.4f}  "
                  f"P50={np.percentile(_match_gate,50):.4f}  "
                  f"P75={np.percentile(_match_gate,75):.4f}  "
                  f"max={_match_gate.max():.4f}")
            print(f"  {leg_name}  gate  other:  "
                  f"min={_other_gate.min():.4f}  mean={_other_gate.mean():.4f}  "
                  f"P50={np.percentile(_other_gate,50):.4f}  "
                  f"P90={np.percentile(_other_gate,90):.4f}  "
                  f"max={_other_gate.max():.4f}")

        _match_refine = None; _other_refine = None
        if refine is not None:
            jet_refine    = refine[owner_mask, :, leg]       # (N_owner, K)
            _match_refine = jet_refine[match_mask]
            _other_refine = jet_refine[other_mask]
        if _match_refine is not None and len(_match_refine):
            print(f"  {leg_name}  refine match:  "
                  f"min={_match_refine.min():.4f}  mean={_match_refine.mean():.4f}  "
                  f"P25={np.percentile(_match_refine,25):.4f}  "
                  f"P50={np.percentile(_match_refine,50):.4f}  "
                  f"P75={np.percentile(_match_refine,75):.4f}  "
                  f"max={_match_refine.max():.4f}")
            print(f"  {leg_name}  refine other:  "
                  f"min={_other_refine.min():.4f}  mean={_other_refine.mean():.4f}  "
                  f"P50={np.percentile(_other_refine,50):.4f}  "
                  f"P90={np.percentile(_other_refine,90):.4f}  "
                  f"max={_other_refine.max():.4f}")

        _parts = [w for w in [match_weights, other_weights] if len(w)]
        all_w = np.concatenate(_parts) if _parts else np.array([])
        if len(all_w):
            print(f"  {leg_name}  vtx_w range=[{all_w.min():.5f}, {all_w.max():.5f}]  "
                  f"mean={all_w.mean():.5f}  median={np.median(all_w):.5f}  "
                  f"P99={np.percentile(all_w, 99):.5f}")

        for thr in [0.5, 0.8]:
            eff = (match_weights > thr).mean() if len(match_weights) > 0 else 0.0
            fp  = (other_weights > thr).mean() if len(other_weights) > 0 else 0.0
            print(f"  {leg_name} (thr>{thr:.1f}): assignment={eff:.3f}  "
                  f"false-positive={fp:.4f}  n_match={len(match_weights)}")

        # -- histogram figure --
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        fig.suptitle(f"Track-to-vertex weight — {leg_name}  ({owner_name} jets only)",
                     fontweight="bold")
        if len(match_weights) > 0:
            ax.hist(match_weights, bins=40, range=(0, 1), histtype="step",
                    color=colours[owner_name], linewidth=1.5, density=True,
                    label=f"True {leg_name.replace('_','-')} origin  (n={len(match_weights)}")
        if len(other_weights) > 0:
            ax.hist(other_weights, bins=40, range=(0, 1), histtype="step",
                    color="grey", linestyle="--", linewidth=1.5, density=True,
                    label=f"Other origin  (n={len(other_weights)})")
        ax.set_xlabel("vtx_weight"); ax.set_ylabel("Density")
        ax.legend(fontsize=8); ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir + f"track_vtx_assignment_{leg_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved track_vtx_assignment_{leg_name}.png")


# ===========================================================================
# plot_refine_vtx_weight_history — per-epoch refine & vtx_weight curves.
# ===========================================================================
def plot_refine_vtx_weight_history(history, plot_dir):
    has_refine = any(k.startswith("val_") and "refine" in k
                     for k in history)
    if not has_refine:
        return

    epochs = list(range(1, len(history["val_loss"]) + 1))

    fig, (ax_r, ax_v) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("refine & vtx_weight per epoch", fontweight="bold")

    for ax, prefix in [(ax_r, "refine"), (ax_v, "vtx_weight")]:
        key_pairs = [
            (f"val_b_{prefix}_match_mean", f"val_b_{prefix}_other_mean", "b", "#1f77b4"),
            (f"val_c_{prefix}_match_mean", f"val_c_{prefix}_other_mean", "c", "#2ca02c"),
        ]
        for mkey, okey, label, color in key_pairs:
            if (mkey in history and okey in history and
                    len(history[mkey]) == len(epochs)):
                ax.plot(epochs, history[mkey], color=color, linestyle="-",
                        linewidth=1.5, label=f"{label} match")
                ax.plot(epochs, history[okey], color=color, linestyle="--",
                        linewidth=1.5, label=f"{label} other")
        tk = f"train_{prefix}_mean"
        if tk in history and len(history[tk]) == len(epochs):
            ax.plot(epochs, history[tk], color="grey", linestyle=":",
                    linewidth=1.0, label="train overall")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(prefix)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir + "refine_vtx_weight_history.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved refine_vtx_weight_history.png")


# ===========================================================================
# plot_vertex_metrics_history — per-epoch Lxy/dz MAE & Pearson r curves.
# ===========================================================================
def plot_vertex_metrics_history(history, plot_dir):
    has_data = any(k.startswith("val_") and ("mae" in k or "pearson" in k)
                   for k in history)
    if not has_data:
        return

    epochs = list(range(1, len(history["val_loss"]) + 1))

    leg_suffixes = set()
    for k in history:
        if k.startswith("val_") and (k.endswith("_mae") or k.endswith("_pearson")):
            _s = k.replace("val_", "").split("_")[0]
            leg_suffixes.add(_s)
    leg_suffixes = sorted(leg_suffixes)

    colors = {"b": "#1f77b4", "c": "#2ca02c"}

    panels = [
        ("lxy_mae",     "Lxy MAE [mm]"),
        ("dz_mae",      "dz MAE [mm]"),
        ("lxy_pearson", "Lxy Pearson r"),
        ("dz_pearson",  "dz Pearson r"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Vertex reconstruction metrics per epoch", fontweight="bold")

    for (metric, ylabel), ax in zip(panels, axes.ravel()):
        for s in leg_suffixes:
            key = f"val_{s}_{metric}"
            if key in history and len(history[key]) == len(epochs):
                ax.plot(epochs, history[key], color=colors.get(s, "grey"),
                        linewidth=1.5, label=f"{s}-vertex")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir + "vertex_metrics_history.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved vertex_metrics_history.png")


# ===========================================================================
# plot_vertex_loss_components — per-epoch Lxy/dz loss decomposition.
# ===========================================================================
def plot_vertex_loss_components(history, plot_dir):
    has_data = "train_lxy_loss" in history or "train_dz_loss" in history
    if not has_data:
        return

    epochs = list(range(1, len(history["val_loss"]) + 1))

    fig, (ax_lxy, ax_dz) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Vertex loss by coordinate (log1p space)", fontweight="bold")

    for ax, coord in [(ax_lxy, "lxy"), (ax_dz, "dz")]:
        tkey = f"train_{coord}_loss"
        vkey = f"val_{coord}_loss"
        if tkey in history and len(history[tkey]) == len(epochs):
            ax.plot(epochs, history[tkey], color="#1f77b4", linewidth=1.5,
                    label="train")
        if vkey in history and len(history[vkey]) == len(epochs):
            ax.plot(epochs, history[vkey], color="#d62728", linewidth=1.5,
                    label="val")
        ax.set_title(f"{coord} vertex loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"{coord} smooth-L1 (log1p)")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir + "vertex_loss_components.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved vertex_loss_components.png")


# ===========================================================================
# plot_gradient_diagnostics — per-task gradient norms & cosine conflicts.
# ===========================================================================
def plot_gradient_diagnostics(history, plot_dir):
    has_grad = any(k.startswith("grad_norm_shared_encoder_") for k in history)
    if not has_grad:
        return

    epochs = list(range(1, len(history["val_loss"]) + 1))

    fig, (ax_norm, ax_cos) = plt.subplots(2, 1, figsize=(12, 9))
    fig.suptitle("Per-task gradient diagnostics", fontweight="bold")

    # ── Row 1: all gradient norms ──────────────────────────────────────
    norm_keys = [
        ("grad_norm_shared_encoder_jet",     "#1f77b4", "-",  "shared_enc jet",      1.8),
        ("grad_norm_shared_encoder_origin",  "#d62728", "-",  "shared_enc origin",   1.8),
        ("grad_norm_shared_encoder_vertex",  "#2ca02c", "-",  "shared_enc vertex",   1.8),
        ("grad_norm_jet_encoder",            "#1f77b4", "--", "jet_enc jet",         1.0),
        ("grad_norm_vertex_encoder",         "#2ca02c", "--", "vertex_enc vertex",   1.0),
        ("grad_norm_head_jet",               "#1f77b4", ":",  "head jet",            0.8),
        ("grad_norm_head_origin",            "#d62728", ":",  "head origin",         0.8),
        ("grad_norm_head_vtxw",              "#2ca02c", ":",  "head vtxw",           0.8),
    ]
    for key, color, ls, label, lw in norm_keys:
        if key in history and len(history[key]) == len(epochs):
            ax_norm.plot(epochs, history[key], color=color, linestyle=ls,
                         linewidth=lw, label=label, alpha=0.85)
    ax_norm.set_yscale("log")
    ax_norm.set_ylabel("Gradient L2 norm (log scale)")
    ax_norm.set_xlabel("Epoch")
    ax_norm.legend(fontsize=7, ncol=2)
    ax_norm.grid(True, linestyle="--", alpha=0.3)
    ax_norm.set_title("Per-parameter-group gradient norms")

    # ── Row 2: cosine similarities on shared encoder ───────────────────
    cos_keys = [
        ("grad_cos_origin_vertex", "#d62728", "origin vs vertex"),
        ("grad_cos_origin_jet",    "#1f77b4", "origin vs jet"),
        ("grad_cos_vertex_jet",    "#2ca02c", "vertex vs jet"),
    ]
    for key, color, label in cos_keys:
        if key in history and len(history[key]) == len(epochs):
            ax_cos.plot(epochs, history[key], color=color, linewidth=1.5,
                        label=label, alpha=0.85)
    ax_cos.set_ylim(-1.0, 1.0)
    ax_cos.axhline(y=0, color="grey", linestyle=":", linewidth=0.8)
    ax_cos.set_ylabel("Cosine similarity")
    ax_cos.set_xlabel("Epoch")
    ax_cos.legend(fontsize=7)
    ax_cos.grid(True, linestyle="--", alpha=0.3)
    ax_cos.set_title("Gradient cosine similarity on shared encoder")

    plt.tight_layout()
    plt.savefig(plot_dir + "gradient_diagnostics.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved gradient_diagnostics.png")
