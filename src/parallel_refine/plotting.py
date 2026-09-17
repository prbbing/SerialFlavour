"""Reusable non-interactive plots for locked-Y evaluation artifacts."""

import json
from pathlib import Path

import numpy as np


_JET_CLASS_NAMES = ("b-jet", "c-jet", "light-jet")
_JET_COLOURS = {"b-jet": "#1f77b4", "c-jet": "#ff7f0e", "light-jet": "#2ca02c"}


def _roc_curve(labels, scores):
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order].astype(bool)
    positives = labels.sum()
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    return (np.r_[0, np.cumsum(~labels)] / negatives,
            np.r_[0, np.cumsum(labels)] / positives)


def _plot_discriminant(plt, probabilities, labels, signal, weights, output, stem):
    background = [index for index in range(3) if index != signal]
    denominator = probabilities[:, background] @ np.asarray(weights)
    score = np.log(np.clip(probabilities[:, signal], 1e-12, None)
                   / np.clip(denominator, 1e-12, None))
    figure, (distribution_axis, roc_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    figure.suptitle(f"{_JET_CLASS_NAMES[signal]} discriminant on locked Y", fontweight="bold")
    finite = np.isfinite(score)
    limit = max(float(np.percentile(np.abs(score[finite]), 99)), 1e-12)
    for index, name in enumerate(_JET_CLASS_NAMES):
        distribution_axis.hist(score[finite & (labels == index)], bins=80,
                               range=(-limit, limit), density=True, histtype="step",
                               linewidth=1.5, color=_JET_COLOURS[name], label=name)
    distribution_axis.set(xlabel="log(signal probability / weighted background)", ylabel="Density")
    distribution_axis.legend(fontsize=7)
    for index in background:
        selected = (labels == signal) | (labels == index)
        curve = _roc_curve(labels[selected] == signal, score[selected])
        if curve is not None:
            false_positive, true_positive = curve
            roc_axis.plot(true_positive, false_positive, linewidth=1.5,
                          color=_JET_COLOURS[_JET_CLASS_NAMES[index]],
                          label=f"vs {_JET_CLASS_NAMES[index]}")
    roc_axis.set(xlabel=f"{_JET_CLASS_NAMES[signal]} efficiency", ylabel="Background rate",
                 yscale="log", ylim=(1e-4, 1.0))
    roc_axis.legend(fontsize=8)
    roc_axis.grid(True, which="both", linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(Path(output) / f"{stem}_discriminant_roc.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_jet_evaluation(labels, probabilities, output_directory):
    """Write jet-probability, b-discriminant, and c-discriminant diagnostic plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory = Path(output_directory)
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    figure.suptitle("Jet output probabilities on locked Y", fontweight="bold")
    for predicted, axis in enumerate(axes):
        for truth, name in enumerate(_JET_CLASS_NAMES):
            axis.hist(probabilities[labels == truth, predicted], bins=50, range=(0, 1),
                      density=True, histtype="step", linewidth=1.5,
                      color=_JET_COLOURS[name], label=name)
        axis.set(title=f"P({_JET_CLASS_NAMES[predicted]})", xlabel="Probability", ylabel="Density")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_directory / "output_probabilities.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    _plot_discriminant(plt, probabilities, labels, 0, (0.2, 0.8), output_directory, "b")
    _plot_discriminant(plt, probabilities, labels, 1, (0.3, 0.7), output_directory, "c")

# Locked-Y rejection comparison and seed-aggregation plots.
from src.parallel_refine.metrics import b_discriminant, c_discriminant
from src.parallel_refine.config import write_json_atomic

_JET_CLASS_NAMES = ("b-jet", "c-jet", "light-jet")
_REJECTION_SPECS = (
    (0, (1, 2), np.round(np.linspace(0.60, 1.00, 81), 6), 0.70, b_discriminant),
    (1, (0, 2), np.round(np.linspace(0.10, 0.40, 61), 6), 0.30, c_discriminant),
)


def _finite_or_none(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _rejection_curve(labels, probabilities, signal, background, efficiencies,
                     discriminant):
    """Return rejection at fixed target signal efficiencies on one Y sample."""
    labels = np.asarray(labels, dtype=np.int64)
    score = np.asarray(discriminant(probabilities), dtype=np.float64)
    signal_scores = np.sort(score[labels == signal])
    background_scores = np.sort(score[labels == background])
    efficiencies = np.asarray(efficiencies, dtype=np.float64)
    rejection = np.full(len(efficiencies), np.nan, dtype=np.float64)
    passed = np.zeros(len(efficiencies), dtype=np.int64)
    actual_efficiency = np.full(len(efficiencies), np.nan, dtype=np.float64)
    thresholds = np.full(len(efficiencies), np.nan, dtype=np.float64)
    if not len(signal_scores) or not len(background_scores):
        return {
            "target_signal_efficiency": efficiencies,
            "actual_signal_efficiency": actual_efficiency,
            "threshold": thresholds,
            "background_pass": passed,
            "background_total": int(len(background_scores)),
            "rejection": rejection,
        }
    thresholds = np.quantile(signal_scores, 1.0 - efficiencies)
    signal_pass = len(signal_scores) - np.searchsorted(
        signal_scores, thresholds, side="left")
    actual_efficiency = signal_pass / len(signal_scores)
    passed = len(background_scores) - np.searchsorted(
        background_scores, thresholds, side="left")
    rejection = np.divide(
        float(len(background_scores)), passed,
        out=rejection, where=passed > 0)
    return {
        "target_signal_efficiency": efficiencies,
        "actual_signal_efficiency": actual_efficiency,
        "threshold": thresholds,
        "background_pass": passed,
        "background_total": int(len(background_scores)),
        "rejection": rejection,
    }


def comparison_curves(labels, parallel_probabilities, dnn_probabilities):
    """Build paired rejection curves for one Parallel/DNN seed combination."""
    curves = {}
    for signal, backgrounds, efficiencies, working_point, discriminant in _REJECTION_SPECS:
        for background in backgrounds:
            parallel = _rejection_curve(
                labels, parallel_probabilities, signal, background,
                efficiencies, discriminant)
            dnn = _rejection_curve(
                labels, dnn_probabilities, signal, background,
                efficiencies, discriminant)
            ratio = np.divide(
                dnn["rejection"], parallel["rejection"],
                out=np.full(len(efficiencies), np.nan),
                where=np.isfinite(dnn["rejection"]) & np.isfinite(parallel["rejection"])
                & (parallel["rejection"] != 0))
            curves[signal, background] = {
                "parallel": parallel, "dnn": dnn, "ratio": ratio,
                "working_point": working_point,
            }
    return curves


def _mean_comparison_curves(curve_sets):
    """Average rejection curves bin-by-bin; ratios derive from those means."""
    if not curve_sets:
        raise ValueError("cannot average an empty set of rejection curves")
    averaged = {}
    for key in curve_sets[0]:
        parallel = np.asarray(
            [curves[key]["parallel"]["rejection"] for curves in curve_sets],
            dtype=np.float64)
        dnn = np.asarray(
            [curves[key]["dnn"]["rejection"] for curves in curve_sets],
            dtype=np.float64)
        def finite_mean(values):
            count = np.isfinite(values).sum(axis=0)
            return np.divide(
                np.nansum(values, axis=0), count,
                out=np.full(values.shape[1], np.nan), where=count > 0)

        def sample_sd(values):
            count = np.isfinite(values).sum(axis=0)
            centered = values - finite_mean(values)
            return np.sqrt(np.divide(
                np.nansum(centered ** 2, axis=0), count - 1,
                out=np.full(values.shape[1], np.nan), where=count > 1))

        parallel_mean = finite_mean(parallel)
        dnn_mean = finite_mean(dnn)
        averaged[key] = {
            "parallel": {"rejection": parallel_mean,
                         "rejection_std": sample_sd(parallel)},
            "dnn": {"rejection": dnn_mean, "rejection_std": sample_sd(dnn)},
            "ratio": np.divide(
                dnn_mean, parallel_mean,
                out=np.full(len(parallel_mean), np.nan),
                where=np.isfinite(dnn_mean) & np.isfinite(parallel_mean)
                & (parallel_mean != 0)),
            "working_point": curve_sets[0][key]["working_point"],
        }
    return averaged


def _plot_rejection_curves(curves, output_directory, *, title, aggregation,
                           scales=("linear", "log"),
                           artifact_stem="rejection_comparison"):
    """Render one paired comparison, optionally from bin-wise seed averages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory = Path(output_directory)
    payload = {
        "definition": {
            "signal_threshold": "Each model uses its Y-test signal-score quantile at every target efficiency.",
            "rejection": "background_total / background_pass",
            "ratio": "DNN rejection / Parallel rejection at the same target efficiency",
            "zero_background_pass": "rejection and ratio are null rather than capped",
            "seed_aggregation": aggregation,
        },
        "curves": [],
    }
    for signal, backgrounds, efficiencies, _, _ in _REJECTION_SPECS:
        for background in backgrounds:
            item = curves[signal, background]
            parallel, dnn, ratio = item["parallel"], item["dnn"], item["ratio"]
            for index, efficiency in enumerate(efficiencies):
                row = {
                    "signal": _JET_CLASS_NAMES[signal],
                    "background": _JET_CLASS_NAMES[background],
                    "target_signal_efficiency": float(efficiency),
                    "parallel_rejection": _finite_or_none(parallel["rejection"][index]),
                    "dnn_rejection": _finite_or_none(dnn["rejection"][index]),
                    "dnn_to_parallel_ratio": _finite_or_none(ratio[index]),
                }
                if "rejection_std" in parallel:
                    row["parallel_rejection_sample_sd"] = _finite_or_none(
                        parallel["rejection_std"][index])
                    row["dnn_rejection_sample_sd"] = _finite_or_none(
                        dnn["rejection_std"][index])
                if "background_pass" in parallel:
                    row["parallel_background_pass"] = int(parallel["background_pass"][index])
                    row["dnn_background_pass"] = int(dnn["background_pass"][index])
                payload["curves"].append(row)

    images = []
    for scale in scales:
        figure = plt.figure(figsize=(13, 11.5))
        outer = figure.add_gridspec(
            2, 2, left=0.08, right=0.98, bottom=0.10, top=0.88,
            hspace=0.32, wspace=0.23)
        figure.suptitle(title, fontweight="bold", y=0.97)
        figure.text(0.08, 0.925, "Thresholds are set independently for each model at every target signal efficiency.", fontsize=10)
        first_axis = None
        for signal, backgrounds, efficiencies, _, _ in _REJECTION_SPECS:
            for column, background in enumerate(backgrounds):
                item = curves[signal, background]
                inner = outer[signal, column].subgridspec(2, 1, height_ratios=(3, 1), hspace=0.06)
                axis = figure.add_subplot(inner[0])
                ratio_axis = figure.add_subplot(inner[1], sharex=axis)
                if first_axis is None:
                    first_axis = axis
                for name, colour, style in (("Parallel", "#1f77b4", "-"), ("DNN", "#ff7f0e", "--")):
                    model = item[name.lower()]
                    rejection = model["rejection"]
                    axis.plot(efficiencies, rejection, color=colour,
                              linestyle=style, linewidth=1.7, label=name)
                    if "rejection_std" in model:
                        lower = np.maximum(1, rejection - model["rejection_std"])
                        axis.fill_between(
                            efficiencies, lower, rejection + model["rejection_std"],
                            color=colour, alpha=0.18, linewidth=0)
                axis.axvline(item["working_point"], color="#666666", linestyle=":", linewidth=0.9)
                axis.set(title=f"{_JET_CLASS_NAMES[signal]} tagging: {_JET_CLASS_NAMES[background]} rejection",
                         ylabel=rf"$R_{{{_JET_CLASS_NAMES[background]}}}=1/\epsilon$",
                         xlim=(efficiencies[0], efficiencies[-1]), yscale=scale)
                axis.set_ylim(bottom=1 if scale == "log" else 0)
                axis.tick_params(axis="x", labelbottom=False)
                axis.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.7)
                ratio_axis.plot(efficiencies, item["ratio"], color="#ff7f0e", linewidth=1.5)
                ratio_axis.axhline(1, color="#555555", linestyle="--", linewidth=1)
                ratio_axis.axvline(item["working_point"], color="#666666", linestyle=":", linewidth=0.9)
                ratio_axis.set(xlabel=rf"$\epsilon_{{{_JET_CLASS_NAMES[signal]}}}$ (signal efficiency)",
                               ylabel="DNN / Parallel", xlim=(efficiencies[0], efficiencies[-1]))
                ratio_axis.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.7)
                ratio_axis.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
                ratio_axis.margins(y=0.15)
        handles, names = first_axis.get_legend_handles_labels()
        figure.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=2, frameon=False)
        figure.text(0.08, 0.045, "Ratio > 1 means DNN has higher rejection. Missing points indicate zero background passing the threshold.", fontsize=9)
        figure.text(0.08, 0.022, "Vertical dotted lines: b70 and c30. Shading is seed sample SD (ddof=1); ratio is linear.", fontsize=9)
        filename = f"{artifact_stem}_{scale}.png"
        figure.savefig(output_directory / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
        images.append(filename)
    data_json = f"{artifact_stem}.json"
    write_json_atomic(output_directory / data_json, payload)
    return {f"{scale}_png": filename for scale, filename in zip(scales, images)} | {
        "data_json": data_json}


def plot_rejection_comparison(
        labels, parallel_probabilities, dnn_probabilities, output_directory):
    """Plot per-seed DNN/Parallel rejection and their fixed-efficiency ratio."""
    return _plot_rejection_curves(
        comparison_curves(labels, parallel_probabilities, dnn_probabilities),
        output_directory, title="Locked Y rejection: Parallel vs DNN",
        aggregation="none; this artifact compares one paired seed")


def _read_comparison_curves(path):
    """Load the rejection arrays needed to make a seed-mean comparison plot."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = {(row["signal"], row["background"]): [] for row in payload["curves"]}
    for row in payload["curves"]:
        rows[row["signal"], row["background"]].append(row)
    curves = {}
    for signal, backgrounds, efficiencies, working_point, _ in _REJECTION_SPECS:
        for background in backgrounds:
            values = rows.get((_JET_CLASS_NAMES[signal], _JET_CLASS_NAMES[background]), [])
            if len(values) != len(efficiencies):
                raise ValueError(f"{path}: incomplete rejection curve")
            if not np.allclose(
                    [row["target_signal_efficiency"] for row in values], efficiencies):
                raise ValueError(f"{path}: unexpected rejection efficiency grid")
            parallel = np.asarray(
                [np.nan if row["parallel_rejection"] is None else row["parallel_rejection"]
                 for row in values], dtype=np.float64)
            dnn = np.asarray(
                [np.nan if row["dnn_rejection"] is None else row["dnn_rejection"]
                 for row in values], dtype=np.float64)
            curves[signal, background] = {
                "parallel": {"rejection": parallel},
                "dnn": {"rejection": dnn},
                "ratio": np.divide(
                    dnn, parallel, out=np.full(len(parallel), np.nan),
                    where=np.isfinite(dnn) & np.isfinite(parallel) & (parallel != 0)),
                "working_point": working_point,
            }
    return curves


def write_dnn_seed_mean(run, recipe, directories, curve_sets):
    """Write the five-DNN-seed mean for one frozen Parallel seed and recipe."""
    output_directory = directories[0].parent
    return _plot_rejection_curves(
        _mean_comparison_curves(curve_sets), output_directory,
        title=f"Locked Y rejection: Parallel seed {run.seed} vs mean DNN",
        aggregation=("DNN rejection is the bin-wise mean over "
                     f"{len(curve_sets)} downstream seeds; Parallel is the paired seed."),
        scales=("log",), artifact_stem="rejection_comparison_dnn_seed_mean")


def write_parallel_seed_mean(study, recipes):
    """Write per-recipe means once every configured Parallel/DNN seed is present."""
    for recipe in recipes:
        per_parallel_seed = []
        for run in study.seeds:
            paths = [
                study.evaluation_directory(run, recipe, downstream_seed)
                / "rejection_comparison.json"
                for downstream_seed in study.downstream_seeds
            ]
            if not all(path.is_file() for path in paths):
                break
            per_parallel_seed.append(_mean_comparison_curves(
                [_read_comparison_curves(path) for path in paths]))
        if len(per_parallel_seed) != len(study.seeds):
            continue
        output_directory = study.evaluation_results_directory / recipe
        output_directory.mkdir(parents=True, exist_ok=True)
        _plot_rejection_curves(
            _mean_comparison_curves(per_parallel_seed), output_directory,
            title="Locked Y rejection: mean Parallel vs mean DNN",
            aggregation=("Parallel is the bin-wise mean over "
                         f"{len(study.seeds)} Parallel seeds; DNN is the equal-weight "
                         f"mean over their {len(study.downstream_seeds)} downstream seeds."),
            scales=("log",), artifact_stem="rejection_comparison_parallel_seed_mean")



