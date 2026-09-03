#!/usr/bin/env python3
"""Train one same-seed tabular DNN per Parallel seed and feature recipe on B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.parallel_refine.cache import load_frozen_cache
from src.parallel_refine.config import (
    load_study_config, write_experiment_manifest, write_json_atomic)
from src.parallel_refine.downstream import (
    TabularDNN, create_tabular_loader, fit_normalization,
    save_dnn_description)
from src.parallel_refine.metrics import probability_metrics
from src.config import seed_everything
from src.training import (
    create_tensorboard_writer, log_tensorboard_scalars, save_history_csv)


def _device(config):
    gpu_ids = config.get("gpu_ids", [-1])
    if torch.cuda.is_available() and gpu_ids != [-1]:
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    total = correct = count = 0
    probabilities = []
    labels = []
    for values, target in loader:
        values = values.to(device)
        target = target.to(device)
        logits = model(values)
        loss = torch.nn.functional.cross_entropy(logits, target)
        total += float(loss) * len(target)
        correct += int((logits.argmax(-1) == target).sum())
        count += len(target)
        probabilities.append(torch.softmax(logits, dim=-1).cpu())
        labels.append(target.cpu())
    return {
        "loss": total / max(count, 1),
        "accuracy": correct / max(count, 1),
        "probabilities": torch.cat(probabilities).numpy(),
        "labels": torch.cat(labels).numpy(),
    }


def _train_one(study, run, recipe, *, skip_complete):
    config = study.refiners["dnn"]
    output = study.refiner_directory(run, recipe, "dnn")
    checkpoint = output / "best_dnn.pt"
    if checkpoint.exists() and skip_complete:
        print(f"skip DNN seed={run.seed} recipe={recipe}: {checkpoint}")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite DNN output: {output}")
    output.mkdir(parents=True)

    train_cache = load_frozen_cache(study, run, "b_train")
    val_cache = load_frozen_cache(study, run, "b_val")
    columns = train_cache.recipe_columns(recipe)
    if not np.array_equal(columns, val_cache.recipe_columns(recipe)):
        raise ValueError("B-train/B-val feature schema mismatch")
    mean, std = fit_normalization(train_cache, columns)
    np.savez(output / "normalization.npz", mean=mean, std=std)
    save_dnn_description(
        output / "model.json", recipe=recipe, columns=columns,
        feature_names=train_cache.recipe_names(recipe), config=config)

    device = _device(config)
    seed_everything(run.seed, [device.index] if device.type == "cuda" else ())
    train_loader = create_tabular_loader(
        train_cache, columns, batch_size=config["batch_size"], shuffle=True,
        num_workers=config.get("num_workers", 0), seed=run.seed)
    val_loader = create_tabular_loader(
        val_cache, columns, batch_size=config["batch_size"], shuffle=False,
        num_workers=config.get("num_workers", 0), seed=run.seed + 1000)
    model = TabularDNN(
        len(columns), config["hidden_dims"], config["dropout"], mean, std).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    print(f"DNN seed={run.seed} recipe={recipe} parameters={parameter_count:,}")
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0))

    history = []
    best = float("inf")
    stale = 0
    best_epoch = None
    tensorboard = config.get("tensorboard", {})
    tensorboard_enabled = tensorboard.get("enabled", True)
    tensorboard_directory = output / tensorboard.get("subdir", "tensorboard")
    writer = (
        create_tensorboard_writer(tensorboard_directory)
        if tensorboard_enabled else None)
    training_started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        total = correct = count = 0
        for values, target in train_loader:
            values = values.to(device)
            target = target.to(device)
            optimiser.zero_grad(set_to_none=True)
            logits = model(values)
            loss = torch.nn.functional.cross_entropy(logits, target)
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(target)
            correct += int((logits.argmax(-1) == target).sum())
            count += len(target)
        val = _evaluate(model, val_loader, device)
        improved = val["loss"] < best
        if improved:
            best = val["loss"]
            stale = 0
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        torch.save(model.state_dict(), output / "last_dnn.pt")
        history.append({
            "epoch": epoch,
            "epoch_seconds": time.perf_counter() - started,
            "train_samples": count,
            "train_cross_entropy": total / max(count, 1),
            "train_accuracy": correct / max(count, 1),
            "val_cross_entropy": val["loss"],
            "val_accuracy": val["accuracy"],
            "saved_best": improved,
            "best_epoch_so_far": best_epoch,
        })
        if writer is not None:
            log_tensorboard_scalars(writer, step=epoch, scalars={
                "loss/train/cross_entropy": history[-1]["train_cross_entropy"],
                "loss/validation/cross_entropy": history[-1]["val_cross_entropy"],
                "metrics/train/accuracy": history[-1]["train_accuracy"],
                "metrics/validation/accuracy": history[-1]["val_accuracy"],
                "optimizer/learning_rate": optimiser.param_groups[0]["lr"],
                "timing/epoch_seconds": history[-1]["epoch_seconds"],
            })
        save_history_csv(history, output / "training_history.csv")
        write_json_atomic(output / "training_history.json", {
            "history_version": "parallel_refine_dnn_v1",
            "parallel_seed": run.seed,
            "downstream_seed": run.seed,
            "recipe": recipe,
            "epochs": history,
        })
        print(
            f"DNN seed={run.seed} recipe={recipe} epoch={epoch} "
            f"train_ce={history[-1]['train_cross_entropy']:.6f} "
            f"val_ce={val['loss']:.6f}")
        if stale >= config["early_stopping_patience"]:
            break
    if writer is not None:
        writer.close()

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    selected = _evaluate(model, val_loader, device)
    np.savez(
        output / "validation_predictions.npz", y=selected["labels"],
        probabilities=selected["probabilities"],
        source_index=np.asarray(val_cache.source_index),
        event_number=np.asarray(val_cache.event_number))
    write_json_atomic(output / "validation_metrics.json", {
        "split": "b_val",
        "n_jets": int(len(selected["labels"])),
        "best_epoch": best_epoch,
        "metrics": probability_metrics(
            selected["labels"], selected["probabilities"]),
    })
    write_json_atomic(output / "run_manifest.json", {
        "model": "dnn",
        "experiment_config": str(study.path),
        "experiment_config_sha256": study.source_sha256,
        "experiment_markers": study.experiment_markers,
        "parallel_seed": run.seed,
        "downstream_seed": run.seed,
        "parallel_output_name": run.output_name,
        "parallel_checkpoint": str(study.checkpoint(run).resolve()),
        "recipe": recipe,
        "train_cache": train_cache.manifest,
        "validation_cache": val_cache.manifest,
        "parameters": parameter_count,
        "training_summary": {
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_cross_entropy": best,
            "wall_seconds": time.perf_counter() - training_started,
            "train_samples": int(len(train_cache.labels)),
            "validation_samples": int(len(val_cache.labels)),
        },
        "artifacts": {
            "training_history_json": "training_history.json",
            "training_history_csv": "training_history.csv",
            "validation_metrics": "validation_metrics.json",
            "tensorboard": (
                str(tensorboard_directory.relative_to(output))
                if tensorboard_enabled else None),
        },
        "config": config,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--recipe", action="append")
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args(argv)
    study = load_study_config(args.config)
    if study.data["sizes"]["b_train"] == 0:
        raise ValueError(
            "cannot train a DNN for a Transformer-only configuration with "
            "b_train=0")
    print(f"experiment_manifest={write_experiment_manifest(study)}")
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
