#!/usr/bin/env python3
"""Train one same-seed FG0/FG1/FG2 graph-DNN refiner on B."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.config import seed_everything
from src.parallel_refine.cache import load_frozen_cache
from src.parallel_refine.config import (
    GRAPH_RECIPES, load_study_config, write_experiment_manifest, write_json_atomic)
from src.parallel_refine.downstream import fit_normalization
from src.parallel_refine.graph_cache import load_graph_cache
from src.parallel_refine.graph_refiner import (
    GraphDNNRefiner, create_graph_loader, graph_node_values,
    resolve_graph_config, save_graph_description)
from src.parallel_refine.metrics import probability_metrics
from src.training import (
    create_tensorboard_writer, log_tensorboard_scalars, save_history_csv)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    total = correct = count = 0
    probabilities, labels = [], []
    for batch in loader:
        values = {name: value.to(device) for name, value in batch.items()}
        logits = model(values["context"], values["node_values"],
                       values["pair_probs"], values["track_mask"])
        loss = torch.nn.functional.cross_entropy(logits, values["y"])
        total += float(loss) * len(values["y"])
        correct += int((logits.argmax(-1) == values["y"]).sum())
        count += len(values["y"])
        probabilities.append(torch.softmax(logits, dim=-1).cpu())
        labels.append(values["y"].cpu())
    return {
        "loss": total / max(count, 1), "accuracy": correct / max(count, 1),
        "probabilities": torch.cat(probabilities).numpy(),
        "labels": torch.cat(labels).numpy(),
    }


def _train_one(study, run, recipe, *, skip_complete):
    requested_config = study.refiners["graph"]
    output = study.refiner_directory(run, recipe, "graph_dnn")
    checkpoint = output / "best_graph_refiner.pt"
    if checkpoint.exists() and skip_complete:
        print(f"skip graph refiner seed={run.seed} recipe={recipe}: {checkpoint}")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite graph-refiner output: {output}")
    output.mkdir(parents=True)

    train_table = load_frozen_cache(study, run, "b_train")
    val_table = load_frozen_cache(study, run, "b_val")
    train_graph = load_graph_cache(study, run, "b_train")
    val_graph = load_graph_cache(study, run, "b_val")
    config = resolve_graph_config(requested_config, train_graph)
    if train_graph.track_embedding.shape[-1] != val_graph.track_embedding.shape[-1]:
        raise ValueError("B-train/B-val graph embedding dimension mismatch")
    context_columns = train_table.recipe_columns("F1O")
    if not np.array_equal(context_columns, val_table.recipe_columns("F1O")):
        raise ValueError("B-train/B-val graph context schema mismatch")
    mean, std = fit_normalization(train_table, context_columns)
    np.savez(output / "normalization.npz", mean=mean, std=std)
    node_dim = graph_node_values(train_graph, recipe, 0).shape[-1]
    save_graph_description(
        output / "model.json", recipe=recipe, context_columns=context_columns,
        context_names=train_table.recipe_names("F1O"), node_dim=node_dim,
        graph_config=config)

    device = _device()
    seed_everything(run.seed, [device.index] if device.type == "cuda" else ())
    train_loader = create_graph_loader(
        train_table, train_graph, context_columns, recipe,
        batch_size=config["batch_size"], shuffle=True,
        num_workers=config.get("num_workers", 0), seed=run.seed)
    val_loader = create_graph_loader(
        val_table, val_graph, context_columns, recipe,
        batch_size=config["batch_size"], shuffle=False,
        num_workers=config.get("num_workers", 0), seed=run.seed + 1000)
    model = GraphDNNRefiner(
        len(context_columns), node_dim, config, mean, std).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    print(
        f"graph refiner seed={run.seed} recipe={recipe} "
        f"parameters={parameter_count:,}")
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"],
        weight_decay=config["weight_decay"])
    tensorboard = config.get("tensorboard", {})
    tensorboard_enabled = tensorboard.get("enabled", True)
    tensorboard_directory = output / tensorboard.get("subdir", "tensorboard")
    writer = create_tensorboard_writer(tensorboard_directory) if tensorboard_enabled else None
    history, best, stale, best_epoch = [], float("inf"), 0, None
    for epoch in range(1, config["epochs"] + 1):
        started = time.perf_counter()
        model.train()
        total = correct = count = 0
        for batch in train_loader:
            values = {name: value.to(device) for name, value in batch.items()}
            optimiser.zero_grad(set_to_none=True)
            logits = model(values["context"], values["node_values"],
                           values["pair_probs"], values["track_mask"])
            loss = torch.nn.functional.cross_entropy(logits, values["y"])
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(values["y"])
            correct += int((logits.argmax(-1) == values["y"]).sum())
            count += len(values["y"])
        val = _evaluate(model, val_loader, device)
        improved = val["loss"] < best
        if improved:
            best, stale, best_epoch = val["loss"], 0, epoch
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        torch.save(model.state_dict(), output / "last_graph_refiner.pt")
        row = {
            "epoch": epoch, "epoch_seconds": time.perf_counter() - started,
            "train_samples": count,
            "train_cross_entropy": total / max(count, 1),
            "train_accuracy": correct / max(count, 1),
            "val_cross_entropy": val["loss"], "val_accuracy": val["accuracy"],
            "saved_best": improved, "best_epoch_so_far": best_epoch,
        }
        history.append(row)
        if writer is not None:
            log_tensorboard_scalars(writer, step=epoch, scalars={
                "loss/train/cross_entropy": row["train_cross_entropy"],
                "loss/validation/cross_entropy": row["val_cross_entropy"],
                "metrics/train/accuracy": row["train_accuracy"],
                "metrics/validation/accuracy": row["val_accuracy"],
                "timing/epoch_seconds": row["epoch_seconds"],
            })
        save_history_csv(history, output / "training_history.csv")
        write_json_atomic(output / "training_history.json", {
            "history_version": "parallel_refine_graph_dnn_v1",
            "parallel_seed": run.seed, "downstream_seed": run.seed,
            "recipe": recipe, "epochs": history})
        print(f"graph seed={run.seed} recipe={recipe} epoch={epoch} "
              f"train_ce={row['train_cross_entropy']:.6f} val_ce={val['loss']:.6f}")
        if stale >= config["early_stopping_patience"]:
            break
    if writer is not None:
        writer.close()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    selected = _evaluate(model, val_loader, device)
    np.savez(output / "validation_predictions.npz", y=selected["labels"],
             probabilities=selected["probabilities"],
             source_index=np.asarray(val_table.source_index),
             event_number=np.asarray(val_table.event_number))
    write_json_atomic(output / "validation_metrics.json", {
        "split": "b_val", "n_jets": int(len(selected["labels"])),
        "best_epoch": best_epoch,
        "metrics": probability_metrics(selected["labels"], selected["probabilities"])})
    write_json_atomic(output / "run_manifest.json", {
        "model": "graph_dnn_refiner", "experiment_config": str(study.path),
        "experiment_config_sha256": study.source_sha256,
        "experiment_markers": study.experiment_markers,
        "parallel_seed": run.seed, "downstream_seed": run.seed,
        "parallel_output_name": run.output_name,
        "parallel_checkpoint": str(study.checkpoint(run).resolve()),
        "recipe": recipe, "parameters": parameter_count,
        "train_cache": train_table.manifest, "validation_cache": val_table.manifest,
        "graph_train_cache": train_graph.manifest,
        "graph_validation_cache": val_graph.manifest,
        "training_summary": {
            "epochs_completed": len(history), "best_epoch": best_epoch,
            "best_validation_cross_entropy": best,
            "wall_seconds": sum(row["epoch_seconds"] for row in history),
            "train_samples": int(len(train_table.labels)),
            "validation_samples": int(len(val_table.labels))},
        "requested_config": requested_config,
        "config": config})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--recipe", action="append")
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args(argv)
    study = load_study_config(args.config)
    print(f"experiment_manifest={write_experiment_manifest(study)}")
    recipes = args.recipe or [recipe for recipe in study.refiners["recipes"]
                              if recipe in GRAPH_RECIPES]
    unknown = set(recipes) - set(GRAPH_RECIPES)
    if unknown:
        raise ValueError(f"graph recipe(s) required, got: {sorted(unknown)}")
    for run in study.selected_seeds(args.seed):
        for recipe in recipes:
            _train_one(study, run, recipe, skip_complete=args.skip_complete)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
