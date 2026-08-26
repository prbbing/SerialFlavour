#!/usr/bin/env python3
"""Evaluate locked direct Parallel and DNN predictions on cached Y.

BDT evaluation remains available for reproducing earlier studies, but is not
part of the default workflow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from parallel_refine.src.cache import load_frozen_cache
from parallel_refine.src.config import load_study_config, write_json_atomic
from parallel_refine.src.downstream import create_tabular_loader, load_dnn
from parallel_refine.src.metrics import write_prediction_result
from parallel_refine.src.xgboost_utils import booster_probabilities


def _device(config):
    gpu_ids = config.get("gpu_ids", [-1])
    if torch.cuda.is_available() and gpu_ids != [-1]:
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _direct_probabilities(cache):
    names = cache.manifest["feature_names"]
    columns = [names.index(f"jet_prob_{name}") for name in ("b", "c", "light")]
    return np.asarray(cache.features[:, columns], dtype=np.float32)


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
        choices=("direct", "dnn", "direct_dnn", "bdt", "all"),
        default="direct_dnn")
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

        if args.model in {"direct", "direct_dnn", "all"}:
            from parallel_refine.src.auxiliary_evaluation import (
                evaluate_parallel_auxiliary)

            directory = (
                study.output_directory / "refiners" / run.output_name
                / "direct_parallel" / "evaluation" / "y_test" / "parallel")
            auxiliary = evaluate_parallel_auxiliary(
                study, run, cache, directory,
                _device(study.parallel.get("training", {})))
            result = write_prediction_result(
                directory, model_name="parallel", split="y_test", y=y,
                probabilities=_direct_probabilities(cache),
                source_index=source_index, event_number=event_number,
                metadata={"parallel_seed": run.seed},
                auxiliary_metrics=auxiliary)
            _write_manifest(
                directory.parent / "evaluation_manifest.json",
                study=study, run=run, recipe=None, model="direct_parallel",
                cache=cache, result=result, checkpoint=study.checkpoint(run))

        for recipe in recipes:
            columns = cache.recipe_columns(recipe)
            if args.model in {"dnn", "direct_dnn", "all"}:
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
                _write_manifest(
                    directory.parent / "evaluation_manifest.json",
                    study=study, run=run, recipe=recipe, model="dnn",
                    cache=cache, result=result, checkpoint=checkpoint)

            if args.model in {"bdt", "all"}:
                model_directory = study.refiner_directory(run, recipe, "bdt")
                checkpoint = model_directory / "best_bdt.json"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"missing locked BDT: {checkpoint}")
                try:
                    import xgboost
                except ModuleNotFoundError as error:
                    raise RuntimeError(
                        "BDT evaluation requires xgboost>=2.0; install it in the "
                        "active SerialFlavour environment") from error
                description = json.loads(
                    (model_directory / "model.json").read_text(encoding="utf-8"))
                if description.get("model_type") not in {
                        "xgboost_booster", "xgboost_xgbclassifier"}:
                    raise ValueError("locked BDT is not an XGBoost model")
                if description["recipe"] != recipe or not np.array_equal(
                        columns, np.asarray(description["columns"], dtype=np.int64)):
                    raise ValueError("BDT model/cache feature schema mismatch")
                booster = xgboost.Booster()
                booster.load_model(str(checkpoint))
                probabilities = booster_probabilities(
                    booster, cache.features[:, columns],
                    iteration_range=(0, description["best_iteration"] + 1))
                directory = model_directory / "evaluation" / "y_test" / "bdt"
                result = write_prediction_result(
                    directory, model_name="bdt", split="y_test", y=y,
                    probabilities=probabilities, source_index=source_index,
                    event_number=event_number,
                    metadata={"parallel_seed": run.seed, "bdt_seed": run.seed,
                              "recipe": recipe})
                _write_manifest(
                    directory.parent / "evaluation_manifest.json",
                    study=study, run=run, recipe=recipe, model="bdt",
                    cache=cache, result=result, checkpoint=checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
