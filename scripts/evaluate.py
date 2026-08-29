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
from src.parallel_refine.config import load_study_config, write_json_atomic
from src.parallel_refine.downstream import create_tabular_loader, load_dnn
from src.parallel_refine.metrics import write_prediction_result


_JET_CLASS_NAMES = ("b-jet", "c-jet", "light-jet")
_JET_COLOURS = {"b-jet": "#1f77b4", "c-jet": "#ff7f0e", "light-jet": "#2ca02c"}


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


def _write_manifest(path, *, study, run, recipe, model, cache, result, checkpoint):
    write_json_atomic(path, {
        "evaluation_version": "parallel_refine_y_v1",
        "study_name": study.study_name,
        "experiment_config": str(study.path),
        "experiment_config_sha256": study.source_sha256,
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
    })


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
            columns = cache.recipe_columns(recipe)
            if args.model in {"dnn", "parallel_dnn"}:
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
                _write_manifest(
                    directory.parent / "evaluation_manifest.json",
                    study=study, run=run, recipe=recipe, model="dnn",
                    cache=cache, result=result, checkpoint=checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
