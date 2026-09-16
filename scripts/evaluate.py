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
    GRAPH_RECIPES, graph_context_recipe, load_study_config,
    write_experiment_manifest, write_json_atomic)
from src.parallel_refine.downstream import create_tabular_loader, load_dnn
from src.parallel_refine.graph_cache import load_graph_cache
from src.parallel_refine.graph_refiner import (
    create_graph_loader, load_graph_refiner)
from src.parallel_refine.metrics import write_prediction_result
from src.parallel_refine.plotting import (
    comparison_curves, plot_jet_evaluation, plot_rejection_comparison,
    write_dnn_seed_mean, write_parallel_seed_mean)


def _device(config):
    gpu_ids = config.get("gpu_ids", [-1])
    if torch.cuda.is_available() and gpu_ids != [-1]:
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parallel_probabilities(cache):
    names = cache.manifest["feature_names"]
    columns = [names.index(f"jet_prob_{name}") for name in ("b", "c", "light")]
    return np.asarray(cache.features[:, columns], dtype=np.float32)


@torch.no_grad()
def _dnn_probability_table(cache, columns, config, device, models):
    """Evaluate every model on one tabular pass over the cached split."""
    loader = create_tabular_loader(
        cache, columns, batch_size=config["batch_size"], shuffle=False,
        num_workers=config.get("num_workers", 0), seed=0)
    collected = [[] for _ in models]
    for values, _ in loader:
        values = values.to(device)
        for index, model in enumerate(models):
            collected[index].append(torch.softmax(
                model(values), dim=-1).cpu())
    return [torch.cat(parts).numpy() for parts in collected]


@torch.no_grad()
def _graph_probability_table(table, graph, columns, recipe, config, device,
                             models):
    """Evaluate every model on one graph pass over the cached split."""
    loader = create_graph_loader(
        table, graph, columns, recipe, batch_size=config["batch_size"],
        shuffle=False, num_workers=config.get("num_workers", 0), seed=0)
    collected = [[] for _ in models]
    for batch in loader:
        values = {name: value.to(device) for name, value in batch.items()}
        for index, model in enumerate(models):
            collected[index].append(torch.softmax(model(
                values["context"], values["node_values"], values["pair_probs"],
                values["track_mask"]), dim=-1).cpu())
    return [torch.cat(parts).numpy() for parts in collected]


def _write_manifest(path, *, study, run, recipe, downstream_seed, model, cache,
                    result, checkpoint, graph_cache=None):
    payload = {
        "evaluation_version": "parallel_refine_y_v1",
        "study_name": study.study_name,
        "experiment_config": str(study.path),
        "experiment_config_sha256": study.source_sha256,
        "experiment_markers": study.experiment_markers,
        "parallel_seed": run.seed,
        "downstream_seed": downstream_seed,
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
    parser.add_argument(
        "--downstream-seed", type=int, action="append",
        help="Evaluate only this refiner initialization seed; repeat as needed.")
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
    downstream_seeds = study.selected_downstream_seeds(args.downstream_seed)

    for run in study.selected_seeds(args.seed):
        cache = load_frozen_cache(study, run, "y_test")
        y = np.asarray(cache.labels)
        source_index = np.asarray(cache.source_index)
        event_number = np.asarray(cache.event_number)
        # A single read of the frozen jet posterior is shared by the direct
        # Parallel artifacts and every downstream rejection comparison.
        parallel_probabilities = _parallel_probabilities(cache)

        if args.model in {"parallel", "parallel_dnn"}:
            from src.parallel_refine.auxiliary_evaluation import (
                evaluate_parallel_auxiliary)

            directory = study.parallel_evaluation_directory(run)
            auxiliary = evaluate_parallel_auxiliary(
                study, run, cache, directory,
                _device(study.parallel.get("training", {})))
            result = write_prediction_result(
                directory, model_name="parallel", split="y_test", y=y,
                probabilities=parallel_probabilities,
                source_index=source_index, event_number=event_number,
                metadata={"parallel_seed": run.seed},
                auxiliary_metrics=auxiliary)
            plot_jet_evaluation(y, parallel_probabilities, directory)
            _write_manifest(
                directory / "evaluation_manifest.json",
                study=study, run=run, recipe=None, model="parallel",
                downstream_seed=None,
                cache=cache, result=result, checkpoint=study.checkpoint(run))

        if args.model not in {"dnn", "parallel_dnn"}:
            continue
        # The cached split does not depend on the refiner seed, so every
        # selected downstream seed is scored in a single data pass per recipe.
        seeds = list(downstream_seeds)
        for recipe in recipes:
            if recipe in GRAPH_RECIPES:
                device = _device(study.refiners["graph"])
                columns = cache.recipe_columns(graph_context_recipe(recipe))
                graph = load_graph_cache(study, run, "y_test")
                directories, checkpoints, models = [], [], []
                for downstream_seed in seeds:
                    model_directory = study.refiner_directory(
                        run, recipe, downstream_seed)
                    checkpoint = model_directory / "best_graph_refiner.pt"
                    if not checkpoint.is_file():
                        raise FileNotFoundError(
                            f"missing locked graph refiner: {checkpoint}")
                    dnn, description = load_graph_refiner(model_directory, device)
                    if (
                            description["recipe"] != recipe
                            or not np.array_equal(
                                columns, np.asarray(
                                    description["context_columns"],
                                    dtype=np.int64))):
                        raise ValueError(
                            "graph model/cache feature schema mismatch")
                    directories.append(study.evaluation_directory(
                        run, recipe, downstream_seed))
                    checkpoints.append(checkpoint)
                    models.append(dnn)
                tables = _graph_probability_table(
                    cache, graph, columns, recipe, study.refiners["graph"],
                    device, models)
                curve_sets = []
                for downstream_seed, directory, checkpoint, probabilities in zip(
                        seeds, directories, checkpoints, tables):
                    result = write_prediction_result(
                        directory, model_name="graph_dnn", split="y_test", y=y,
                        probabilities=probabilities, source_index=source_index,
                        event_number=event_number,
                        metadata={"parallel_seed": run.seed,
                                  "dnn_seed": downstream_seed,
                                  "recipe": recipe})
                    plot_jet_evaluation(y, probabilities, directory)
                    result["comparison_artifacts"] = plot_rejection_comparison(
                        y, parallel_probabilities, probabilities, directory)
                    curve_sets.append(comparison_curves(
                        y, parallel_probabilities, probabilities))
                    _write_manifest(
                        directory / "evaluation_manifest.json",
                        study=study, run=run, recipe=recipe,
                        model="graph_dnn", downstream_seed=downstream_seed,
                        cache=cache, result=result, checkpoint=checkpoint,
                        graph_cache=graph)
                if set(seeds) == set(study.downstream_seeds):
                    write_dnn_seed_mean(run, recipe, directories, curve_sets)
                continue
            device = _device(study.refiners["dnn"])
            columns = cache.recipe_columns(recipe)
            directories, checkpoints, models = [], [], []
            for downstream_seed in seeds:
                model_directory = study.refiner_directory(
                    run, recipe, downstream_seed)
                checkpoint = model_directory / "best_dnn.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"missing locked DNN: {checkpoint}")
                dnn, description = load_dnn(model_directory, device)
                if description["recipe"] != recipe or not np.array_equal(
                        columns, np.asarray(
                            description["columns"], dtype=np.int64)):
                    raise ValueError("DNN model/cache feature schema mismatch")
                directories.append(study.evaluation_directory(
                    run, recipe, downstream_seed))
                checkpoints.append(checkpoint)
                models.append(dnn)
            tables = _dnn_probability_table(
                cache, columns, study.refiners["dnn"], device, models)
            curve_sets = []
            for downstream_seed, directory, checkpoint, probabilities in zip(
                    seeds, directories, checkpoints, tables):
                result = write_prediction_result(
                    directory, model_name="dnn", split="y_test", y=y,
                    probabilities=probabilities, source_index=source_index,
                    event_number=event_number,
                    metadata={"parallel_seed": run.seed,
                              "dnn_seed": downstream_seed,
                              "recipe": recipe})
                plot_jet_evaluation(y, probabilities, directory)
                result["comparison_artifacts"] = plot_rejection_comparison(
                    y, parallel_probabilities, probabilities, directory)
                curve_sets.append(comparison_curves(
                    y, parallel_probabilities, probabilities))
                _write_manifest(
                    directory / "evaluation_manifest.json",
                    study=study, run=run, recipe=recipe, model="dnn",
                    downstream_seed=downstream_seed,
                    cache=cache, result=result, checkpoint=checkpoint)
            if set(seeds) == set(study.downstream_seeds):
                write_dnn_seed_mean(run, recipe, directories, curve_sets)
    if args.model in {"dnn", "parallel_dnn"}:
        write_parallel_seed_mean(study, recipes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
