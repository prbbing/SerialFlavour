#!/usr/bin/env python3
"""Train one same-seed XGBoost BDT per Parallel seed and feature recipe on B."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from parallel_refine.src.cache import load_frozen_cache
from parallel_refine.src.config import load_study_config, write_json_atomic
from parallel_refine.src.metrics import probability_metrics


def _train_one(study, run, recipe, *, skip_complete):
    try:
        import xgboost
        from xgboost import XGBClassifier
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "BDT training requires xgboost>=2.0; install it in the active "
            "SerialFlavour environment") from error

    config = study.refiners["bdt"]
    output = study.refiner_directory(run, recipe, "bdt")
    checkpoint = output / "best_bdt.json"
    if checkpoint.exists() and skip_complete:
        print(f"skip BDT seed={run.seed} recipe={recipe}: {checkpoint}")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite BDT output: {output}")
    output.mkdir(parents=True)

    train_cache = load_frozen_cache(study, run, "b_train")
    val_cache = load_frozen_cache(study, run, "b_val")
    columns = train_cache.recipe_columns(recipe)
    if not np.array_equal(columns, val_cache.recipe_columns(recipe)):
        raise ValueError("B-train/B-val feature schema mismatch")
    x_train = np.asarray(train_cache.features[:, columns], dtype=np.float32)
    y_train = np.asarray(train_cache.labels, dtype=np.int64)
    x_val = np.asarray(val_cache.features[:, columns], dtype=np.float32)
    y_val = np.asarray(val_cache.labels, dtype=np.int64)

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        gamma=config["gamma"],
        reg_alpha=config["reg_alpha"],
        reg_lambda=config["reg_lambda"],
        max_bin=config["max_bin"],
        early_stopping_rounds=config["early_stopping_rounds"],
        tree_method=config["tree_method"],
        device=config["device"],
        n_jobs=config["n_jobs"],
        verbosity=config["verbosity"],
        random_state=run.seed,
    )
    started = time.perf_counter()
    model.fit(
        x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    elapsed = time.perf_counter() - started
    best_iteration = int(model.best_iteration)
    probabilities = model.predict_proba(
        x_val, iteration_range=(0, best_iteration + 1))
    model.save_model(str(checkpoint))
    np.savez(
        output / "validation_predictions.npz", y=y_val,
        probabilities=probabilities,
        source_index=np.asarray(val_cache.source_index),
        event_number=np.asarray(val_cache.event_number))
    write_json_atomic(output / "validation_metrics.json", {
        "split": "b_val",
        "n_jets": int(len(y_val)),
        "metrics": probability_metrics(y_val, probabilities),
    })
    description = {
        "model_type": "xgboost_xgbclassifier",
        "xgboost_version": xgboost.__version__,
        "experiment_config": str(study.path),
        "experiment_config_sha256": study.source_sha256,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "recipe": recipe,
        "columns": columns.tolist(),
        "feature_names": train_cache.recipe_names(recipe),
        "input_dim": int(len(columns)),
        "parallel_seed": run.seed,
        "downstream_seed": run.seed,
        "best_iteration": best_iteration,
        "iterations_used": best_iteration + 1,
        "best_validation_mlogloss": float(model.best_score),
        "training_seconds": elapsed,
        "config": config,
    }
    write_json_atomic(output / "model.json", description)
    write_json_atomic(output / "run_manifest.json", {
        **description,
        "parallel_output_name": run.output_name,
        "parallel_checkpoint": str(study.checkpoint(run).resolve()),
        "train_cache": train_cache.manifest,
        "validation_cache": val_cache.manifest,
    })
    print(
        f"XGBoost seed={run.seed} recipe={recipe} "
        f"iterations={best_iteration + 1} "
        f"val_ce={probability_metrics(y_val, probabilities)['cross_entropy']:.6f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--recipe", action="append")
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args(argv)
    study = load_study_config(args.config)
    recipes = args.recipe or study.refiners["recipes"]
    unknown = set(recipes) - set(study.refiners["recipes"])
    if unknown:
        raise ValueError(f"recipe(s) not enabled by config: {sorted(unknown)}")
    for run in study.selected_seeds(args.seed):
        for recipe in recipes:
            _train_one(study, run, recipe, skip_complete=args.skip_complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
