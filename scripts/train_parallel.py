#!/usr/bin/env python3
"""Train configured Parallel seeds on A and validate on A-val."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.config import seed_everything
from src.parallel_refine.config import (
    active_parallel_config, load_study_config, materialize_parallel_config,
    write_experiment_manifest, write_json_atomic)
from src.parallel_refine.data import create_loader
from src.parallel_refine.upstream import build_parallel
from src.parallel_refine.splits import load_split_bundle
from src.training import (
    aggregate_losses, choose_device, create_tensorboard_writer, evaluate_loss,
    jet_class_weights, log_tensorboard_epoch, move_batch,
    origin_class_weights, parallel_losses, save_history)


def _plot_training_history(history, output_directory):
    if not history:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    epochs = range(1, len(history) + 1)
    figure, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
    figure.suptitle("Parallel training history", fontweight="bold")
    for prefix, colour, label in (("train", "#1f77b4", "train"),
                                  ("val", "#d62728", "validation")):
        for key, style, component in (("total", "-", "total"),
                                      ("jet", "--", "jet CE"),
                                      ("origin", ":", "origin CE"),
                                      ("pair", "-.", "pair BCE")):
            values = [row.get(f"{prefix}_{key}") for row in history]
            if all(value is not None for value in values):
                loss_axis.plot(epochs, values, color=colour, linestyle=style,
                               label=f"{label} {component}")
        values = [row.get(f"{prefix}_jet_accuracy") for row in history]
        if all(value is not None for value in values):
            accuracy_axis.plot(epochs, values, color=colour, label=label)
    loss_axis.set(xlabel="Epoch", ylabel="Loss", yscale="log")
    loss_axis.legend(fontsize=7)
    loss_axis.grid(True, which="both", linestyle="--", alpha=0.3)
    accuracy_axis.set(xlabel="Epoch", ylabel="Jet accuracy", ylim=(0, 1))
    accuracy_axis.legend(fontsize=8)
    accuracy_axis.grid(True, linestyle="--", alpha=0.3)
    figure.tight_layout()
    figure.savefig(Path(output_directory) / "training_history.png", dpi=150,
                   bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument(
        "--skip-complete", action="store_true",
        help="Skip a seed when its configured checkpoint already exists.")
    args = parser.parse_args(argv)
    study = load_study_config(args.config)
    print(f"experiment_manifest={write_experiment_manifest(study)}")
    selected = study.selected_seeds(args.seed)

    for run in selected:
        checkpoint = study.checkpoint(run)
        if checkpoint.exists():
            if args.skip_complete:
                print(f"skip seed={run.seed}: {checkpoint}")
                continue
            raise FileExistsError(
                f"configured checkpoint already exists: {checkpoint}; "
                "use --skip-complete or choose another output_name")
        output = study.parallel_directory(run)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"refusing to mix with partial output directory: {output}")
        resolved = materialize_parallel_config(study, run, stage="parallel")
        print(
            f"train Parallel seed={run.seed} output_name={run.output_name} "
            f"config={resolved}")
        config = active_parallel_config(study, run, stage="parallel")
        output.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output / "config.json", config._raw)
        bundle = load_split_bundle(config.split_dir, config=config)
        write_json_atomic(output / "split_manifest.json", bundle.summary)

        device = choose_device(config)
        seed_everything(
            config.seed, [device.index] if device.type == "cuda" else ())
        train_loader, _ = create_loader(
            config, "a_train", progress=True)
        val_loader, _ = create_loader(
            config, "a_val", shuffle=False, progress=True)
        model = build_parallel(config).to(device)
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=config.lr,
            weight_decay=config.weight_decay)
        origin_criterion = nn.CrossEntropyLoss(
            ignore_index=-1,
            weight=origin_class_weights(config, device))
        jet_criterion = nn.CrossEntropyLoss(
            weight=jet_class_weights(config, device))
        loss_fn = lambda model_output, batch: parallel_losses(
            model_output, batch, config, jet_criterion, origin_criterion)
        tensorboard_dir = output / config.tensorboard_subdir
        writer = (
            create_tensorboard_writer(tensorboard_dir)
            if config.tensorboard_enabled else None)
        history = []
        best_jet = float("inf")
        best_total = float("inf")
        best_jet_epoch = None
        best_total_epoch = None
        history_metadata = {
            "stage": "parallel",
            "experiment_name": study.experiment_name,
            "parallel_output_name": run.output_name,
            "seed": run.seed,
            "train_split": "a_train",
            "validation_split": "a_val",
            "model_parameters": parameter_count,
            "validation_samples": int(len(val_loader.dataset)),
            "checkpoint_policy": {
                "best_jet.pt": "minimum validation jet cross-entropy",
                "best_total.pt": "minimum validation total loss",
            },
        }
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            totals = {}
            count = 0
            jet_correct = 0
            for raw in train_loader:
                batch = move_batch(raw, device)
                optimiser.zero_grad(set_to_none=True)
                model_output = model(
                    batch["X"], batch["jet_X"], batch["mask"])
                losses = loss_fn(model_output, batch)
                losses["total"].backward()
                optimiser.step()
                aggregate_losses(totals, losses, len(batch["y"]))
                jet_correct += int((
                    model_output["jet_logits"].argmax(-1) == batch["y"]).sum())
                count += len(batch["y"])
            train = {
                name: value / max(count, 1) for name, value in totals.items()
            }
            train["jet_accuracy"] = jet_correct / max(count, 1)
            validation = evaluate_loss(model, val_loader, device, loss_fn)
            saved_best_jet = validation["jet"] < best_jet
            saved_best_total = validation["total"] < best_total
            if saved_best_jet:
                best_jet = validation["jet"]
                best_jet_epoch = epoch
                torch.save(model.state_dict(), output / "best_jet.pt")
            if saved_best_total:
                best_total = validation["total"]
                best_total_epoch = epoch
                torch.save(model.state_dict(), output / "best_total.pt")
            if epoch % config.checkpoint_interval == 0:
                torch.save(model.state_dict(), output / f"epoch_{epoch}.pt")
            torch.save(model.state_dict(), output / "last.pt")
            history.append({
                "lr": optimiser.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_start,
                "train_samples": count,
                **{f"train_{key}": value for key, value in train.items()},
                **{f"val_{key}": value for key, value in validation.items()},
                "best_jet_loss_so_far": best_jet,
                "best_jet_epoch_so_far": best_jet_epoch,
                "best_total_loss_so_far": best_total,
                "best_total_epoch_so_far": best_total_epoch,
                "saved_best_jet": saved_best_jet,
                "saved_best_total": saved_best_total,
            })
            if writer is not None:
                log_tensorboard_epoch(
                    writer,
                    epoch=epoch,
                    train=train,
                    validation=validation,
                    learning_rate=optimiser.param_groups[0]["lr"],
                    epoch_seconds=history[-1]["epoch_seconds"],
                )
            save_history(history, output, metadata=history_metadata)
            print(
                f"seed={run.seed} epoch={epoch} "
                f"train={train['total']:.6f} "
                f"val_jet={validation['jet']:.6f}")
        if writer is not None:
            writer.close()
        _plot_training_history(history, output)
        write_json_atomic(output / "run_manifest.json", {
            "model": "parallel",
            "experiment_config": str(study.path),
            "experiment_config_sha256": study.source_sha256,
            "experiment_markers": study.experiment_markers,
            "parallel_seed": run.seed,
            "parallel_output_name": run.output_name,
            "checkpoint_policy": history_metadata["checkpoint_policy"],
            "parameters": parameter_count,
            "training_summary": {
                "epochs_completed": len(history),
                "best_jet_epoch": best_jet_epoch,
                "best_jet_loss": best_jet,
                "best_total_epoch": best_total_epoch,
                "best_total_loss": best_total,
                "wall_seconds": sum(row["epoch_seconds"] for row in history),
                "train_samples": int(len(train_loader.dataset)),
                "validation_samples": int(len(val_loader.dataset)),
            },
            "artifacts": {
                "config": "config.json",
                "split_manifest": "split_manifest.json",
                "training_history_json": "training_history.json",
                "training_history_csv": "training_history.csv",
                "training_history_png": "training_history.png",
                "tensorboard": (
                    str(tensorboard_dir.relative_to(output))
                    if config.tensorboard_enabled else None),
                "best_jet_checkpoint": "best_jet.pt",
                "best_total_checkpoint": "best_total.pt",
                "last_checkpoint": "last.pt",
            },
            "config": config._raw,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
