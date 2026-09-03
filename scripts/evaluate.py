#!/usr/bin/env python3
"""Evaluate locked Parallel and DNN predictions on cached Y."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.parallel_refine.cache import load_frozen_cache
from src.parallel_refine.config import (
    GRAPH_RECIPES, load_study_config, write_experiment_manifest, write_json_atomic)
from src.parallel_refine.downstream import create_tabular_loader, load_dnn
from src.parallel_refine.graph_cache import load_graph_cache
from src.parallel_refine.graph_refiner import (
    create_graph_loader, load_graph_refiner)
from src.parallel_refine.metrics import (
    b_discriminant, c_discriminant, write_prediction_result)


_JET_CLASS_NAMES = ("b-jet", "c-jet", "light-jet")
_JET_COLOURS = {"b-jet": "#1f77b4", "c-jet": "#ff7f0e", "light-jet": "#2ca02c"}
_REJECTION_SPECS = (
    (0, (1, 2), np.round(np.linspace(0.60, 1.00, 81), 6), 0.70,
     b_discriminant),
    (1, (0, 2), np.round(np.linspace(0.10, 0.40, 61), 6), 0.30,
     c_discriminant),
)


def _device(config):
    gpu_ids = config.get("gpu_ids", [-1])
    if torch.cuda.is_available() and gpu_ids != [-1]:
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parallel_probabilities(cache):
    names = cache.manifest["feature_names"]
    columns = [names.index(f"jet_prob_{name}") for name in ("b", "c", "light")]
    return np.asarray(cache.features[:, columns], dtype=np.float32)


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
    figure.suptitle(f"{_JET_CLASS_NAMES[signal]} discriminant on locked Y",
                     fontweight="bold")
    finite = np.isfinite(score)
    limit = max(float(np.percentile(np.abs(score[finite]), 99)), 1e-12)
    for index, name in enumerate(_JET_CLASS_NAMES):
        distribution_axis.hist(score[finite & (labels == index)], bins=80,
                               range=(-limit, limit), density=True,
                               histtype="step", linewidth=1.5,
                               color=_JET_COLOURS[name], label=name)
    distribution_axis.set(xlabel="log(signal probability / weighted background)",
                          ylabel="Density")
    distribution_axis.legend(fontsize=7)
    for index in background:
        selected = (labels == signal) | (labels == index)
        curve = _roc_curve(labels[selected] == signal, score[selected])
        if curve is not None:
            false_positive, true_positive = curve
            roc_axis.plot(true_positive, false_positive, linewidth=1.5,
                          color=_JET_COLOURS[_JET_CLASS_NAMES[index]],
                          label=f"vs {_JET_CLASS_NAMES[index]}")
    roc_axis.set(xlabel=f"{_JET_CLASS_NAMES[signal]} efficiency",
                 ylabel="Background rate", yscale="log", ylim=(1e-4, 1.0))
    roc_axis.legend(fontsize=8)
    roc_axis.grid(True, which="both", linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(Path(output) / f"{stem}_discriminant_roc.png", dpi=150,
                   bbox_inches="tight")
    plt.close(figure)


def _plot_jet_evaluation(labels, probabilities, output_directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_directory = Path(output_directory)
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    figure.suptitle("Jet output probabilities on locked Y", fontweight="bold")
    for predicted, axis in enumerate(axes):
        for truth, name in enumerate(_JET_CLASS_NAMES):
            axis.hist(probabilities[labels == truth, predicted], bins=50,
                      range=(0, 1), density=True, histtype="step", linewidth=1.5,
                      color=_JET_COLOURS[name], label=name)
        axis.set(title=f"P({_JET_CLASS_NAMES[predicted]})", xlabel="Probability",
                 ylabel="Density")
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output_directory / "output_probabilities.png", dpi=150,
                   bbox_inches="tight")
    plt.close(figure)
    _plot_discriminant(plt, probabilities, labels, 0, (0.2, 0.8), output_directory,
                       "b")
    _plot_discriminant(plt, probabilities, labels, 1, (0.3, 0.7), output_directory,
                       "c")


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


def _plot_rejection_comparison(
        labels, parallel_probabilities, dnn_probabilities, output_directory):
    """Plot per-seed DNN/Parallel rejection and their fixed-efficiency ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory = Path(output_directory)
    curves = {}
    payload = {
        "definition": {
            "signal_threshold": "Each model uses its Y-test signal-score quantile at every target efficiency.",
            "rejection": "background_total / background_pass",
            "ratio": "DNN rejection / Parallel rejection at the same target efficiency",
            "zero_background_pass": "rejection and ratio are null rather than capped",
            "seed_aggregation": "none; this artifact compares one paired seed",
        },
        "curves": [],
    }
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
            for index, efficiency in enumerate(efficiencies):
                payload["curves"].append({
                    "signal": _JET_CLASS_NAMES[signal],
                    "background": _JET_CLASS_NAMES[background],
                    "target_signal_efficiency": float(efficiency),
                    "parallel_rejection": _finite_or_none(parallel["rejection"][index]),
                    "dnn_rejection": _finite_or_none(dnn["rejection"][index]),
                    "dnn_to_parallel_ratio": _finite_or_none(ratio[index]),
                    "parallel_background_pass": int(parallel["background_pass"][index]),
                    "dnn_background_pass": int(dnn["background_pass"][index]),
                })

    images = []
    for scale in ("linear", "log"):
        figure = plt.figure(figsize=(13, 11.5))
        outer = figure.add_gridspec(
            2, 2, left=0.08, right=0.98, bottom=0.10, top=0.88,
            hspace=0.32, wspace=0.23)
        figure.suptitle("Locked Y rejection: Parallel vs DNN", fontweight="bold", y=0.97)
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
                    axis.plot(efficiencies, item[name.lower()]["rejection"], color=colour,
                              linestyle=style, linewidth=1.7, label=name)
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
        figure.text(0.08, 0.022, "Vertical dotted lines: b70 and c30. Ratio is always shown on a linear axis.", fontsize=9)
        filename = f"rejection_comparison_{scale}.png"
        figure.savefig(output_directory / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
        images.append(filename)
    write_json_atomic(output_directory / "rejection_comparison.json", payload)
    return {"linear_png": images[0], "log_png": images[1], "data_json": "rejection_comparison.json"}


@torch.no_grad()
def _dnn_probabilities(model, cache, columns, config, device):
    loader = create_tabular_loader(
        cache, columns, batch_size=config["batch_size"], shuffle=False,
        num_workers=config.get("num_workers", 0), seed=0)
    probabilities = []
    for values, _ in loader:
        probabilities.append(torch.softmax(
            model(values.to(device)), dim=-1).cpu())
    return torch.cat(probabilities).numpy()


@torch.no_grad()
def _graph_probabilities(model, table, graph, columns, recipe, config, device):
    loader = create_graph_loader(
        table, graph, columns, recipe, batch_size=config["batch_size"],
        shuffle=False, num_workers=config.get("num_workers", 0), seed=0)
    probabilities = []
    for batch in loader:
        values = {name: value.to(device) for name, value in batch.items()}
        probabilities.append(torch.softmax(model(
            values["context"], values["node_values"], values["pair_probs"],
            values["track_mask"]), dim=-1).cpu())
    return torch.cat(probabilities).numpy()


def _write_manifest(path, *, study, run, recipe, model, cache, result, checkpoint,
                    graph_cache=None):
    payload = {
        "evaluation_version": "parallel_refine_y_v1",
        "study_name": study.study_name,
        "experiment_config": str(study.path),
        "experiment_config_sha256": study.source_sha256,
        "experiment_markers": study.experiment_markers,
        "parallel_seed": run.seed,
        "downstream_seed": run.seed,
        "parallel_output_name": run.output_name,
        "parallel_checkpoint": str(study.checkpoint(run).resolve()),
        "downstream_model": model,
        "downstream_checkpoint": None if checkpoint is None else str(checkpoint.resolve()),
        "recipe": recipe,
        "split": "y_test",
        "feature_cache": cache.manifest,
        "result": result,
    }
    if graph_cache is not None:
        payload["graph_cache"] = graph_cache.manifest
    write_json_atomic(path, payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--recipe", action="append")
    parser.add_argument(
        "--model",
        choices=("parallel", "dnn", "parallel_dnn"),
        default="parallel_dnn")
    args = parser.parse_args(argv)
    study = load_study_config(args.config)
    print(f"experiment_manifest={write_experiment_manifest(study)}")
    recipes = args.recipe or study.refiners["recipes"]
    unknown = set(recipes) - set(study.refiners["recipes"])
    if unknown:
        raise ValueError(f"recipe(s) not enabled by config: {sorted(unknown)}")

    for run in study.selected_seeds(args.seed):
        cache = load_frozen_cache(study, run, "y_test")
        y = np.asarray(cache.labels)
        source_index = np.asarray(cache.source_index)
        event_number = np.asarray(cache.event_number)

        if args.model in {"parallel", "parallel_dnn"}:
            from src.parallel_refine.auxiliary_evaluation import (
                evaluate_parallel_auxiliary)

            directory = (
                study.output_directory / "refiners" / run.output_name
                / "parallel" / "evaluation" / "y_test" / "parallel")
            auxiliary = evaluate_parallel_auxiliary(
                study, run, cache, directory,
                _device(study.parallel.get("training", {})))
            result = write_prediction_result(
                directory, model_name="parallel", split="y_test", y=y,
                probabilities=_parallel_probabilities(cache),
                source_index=source_index, event_number=event_number,
                metadata={"parallel_seed": run.seed},
                auxiliary_metrics=auxiliary)
            _plot_jet_evaluation(y, _parallel_probabilities(cache), directory)
            _write_manifest(
                directory.parent / "evaluation_manifest.json",
                study=study, run=run, recipe=None, model="parallel",
                cache=cache, result=result, checkpoint=study.checkpoint(run))

        for recipe in recipes:
            if args.model in {"dnn", "parallel_dnn"}:
                if recipe in GRAPH_RECIPES:
                    columns = cache.recipe_columns("F1O")
                    graph = load_graph_cache(study, run, "y_test")
                    model_directory = study.refiner_directory(
                        run, recipe, "graph_dnn")
                    checkpoint = model_directory / "best_graph_refiner.pt"
                    if not checkpoint.is_file():
                        raise FileNotFoundError(
                            f"missing locked graph refiner: {checkpoint}")
                    device = _device(study.refiners["graph"])
                    dnn, description = load_graph_refiner(model_directory, device)
                    if (
                            description["recipe"] != recipe
                            or not np.array_equal(
                                columns, np.asarray(
                                    description["context_columns"], dtype=np.int64))):
                        raise ValueError("graph model/cache feature schema mismatch")
                    probabilities = _graph_probabilities(
                        dnn, cache, graph, columns, recipe,
                        study.refiners["graph"], device)
                    directory = model_directory / "evaluation" / "y_test" / "graph_dnn"
                    result = write_prediction_result(
                        directory, model_name="graph_dnn", split="y_test", y=y,
                        probabilities=probabilities, source_index=source_index,
                        event_number=event_number,
                        metadata={"parallel_seed": run.seed, "dnn_seed": run.seed,
                                  "recipe": recipe})
                    _plot_jet_evaluation(y, probabilities, directory)
                    result["comparison_artifacts"] = _plot_rejection_comparison(
                        y, _parallel_probabilities(cache), probabilities, directory)
                    _write_manifest(
                        directory.parent / "evaluation_manifest.json",
                        study=study, run=run, recipe=recipe, model="graph_dnn",
                        cache=cache, result=result, checkpoint=checkpoint,
                        graph_cache=graph)
                    continue
                columns = cache.recipe_columns(recipe)
                model_directory = study.refiner_directory(run, recipe, "dnn")
                checkpoint = model_directory / "best_dnn.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"missing locked DNN: {checkpoint}")
                device = _device(study.refiners["dnn"])
                dnn, description = load_dnn(model_directory, device)
                if description["recipe"] != recipe or not np.array_equal(
                        columns, np.asarray(description["columns"], dtype=np.int64)):
                    raise ValueError("DNN model/cache feature schema mismatch")
                probabilities = _dnn_probabilities(
                    dnn, cache, columns, study.refiners["dnn"], device)
                directory = model_directory / "evaluation" / "y_test" / "dnn"
                result = write_prediction_result(
                    directory, model_name="dnn", split="y_test", y=y,
                    probabilities=probabilities, source_index=source_index,
                    event_number=event_number,
                    metadata={"parallel_seed": run.seed, "dnn_seed": run.seed,
                              "recipe": recipe})
                _plot_jet_evaluation(y, probabilities, directory)
                result["comparison_artifacts"] = _plot_rejection_comparison(
                    y, _parallel_probabilities(cache), probabilities, directory)
                _write_manifest(
                    directory.parent / "evaluation_manifest.json",
                    study=study, run=run, recipe=recipe, model="dnn",
                    cache=cache, result=result, checkpoint=checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
